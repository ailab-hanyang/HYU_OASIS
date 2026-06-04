#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
# run_vlm_server.sh — launch / stop / check the VLM (Qwen) vLLM server fleet.
#
# Serves the per-object vision-language classifier used by the atomic functions
# get_visual_actor / get_visual_behavior (refAV/atomic_functions.py ->
# _visual_filter in refAV/utils.py): one OpenAI-compatible vLLM replica per GPU
# on ports 8000.. — exactly the round-robin set the atoms read from
# REFAV_VLM_ENDPOINTS. One image per request, so the default vLLM mm-limit (1)
# is enough (no --limit-mm-per-prompt). See tools/vlm_server/README.md.
#
# Actions (first arg, default: start):
#   start    launch one replica per GPU, wait for readiness, then health-check
#   stop     stop the replicas this script started
#   restart  stop, then start
#   status   show which ports answer GET /v1/models
#   check    run tools/vlm_server/check_connection.py against the fleet
#
# Usage:
#   CKPT=/path/to/Qwen3.6-35B-A3B bash tools/scripts/run_vlm_server.sh
#   bash tools/scripts/run_vlm_server.sh status
#   bash tools/scripts/run_vlm_server.sh stop
#   GPUS="0 1" MODE=native CKPT=/weights bash tools/scripts/run_vlm_server.sh
#
# Optional env overrides:
#   CKPT=/path/to/weights          # local dir or HF id   (REQUIRED for start)
#   MODEL_NAME=qwen3.6-35b         # --served-model-name (== REFAV_VLM_MODEL)
#   GPUS="0 1 2 3"                 # one replica per listed GPU
#   BASE_PORT=8000                 # replica i -> BASE_PORT + i
#   MODE=docker|native             # default docker (official vllm/vllm-openai
#                                  #   image, auto-pulled — no Dockerfile/build);
#                                  #   native = `vllm serve` directly on the host
#                                  #   docker mode needs the nvidia-container-toolkit
#   TP=1                           # --tensor-parallel-size  (per replica)
#   GPU_MEM_UTIL=0.85              # --gpu-memory-utilization
#   MAX_MODEL_LEN=8192             # --max-model-len
#   STARTUP_TIMEOUT=900            # seconds to wait for replicas to load
#   DOCKER_IMAGE=vllm/vllm-openai:latest   # pin a version/digest for reproducible runs
#   HF_TOKEN / HF_HOME             # docker + HF-id CKPT: token for gated repos; host
#                                  #   HF cache mounted so weights download once
#   VLLM_BIN=vllm                  # native launcher binary
#   LOG_DIR=output/vlm_server      # native logs + pidfiles live here
#   PY=python
# ════════════════════════════════════════════════════════════════════════════
set -eo pipefail

# ── project root (relative to this script: tools/scripts -> ../..) ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# ── settings (override via env) ─────────────────────────────────────
CKPT="${CKPT:-}"                                   # weights path / HF id (start)
MODEL_NAME="${MODEL_NAME:-qwen3.6-35b}"            # == REFAV_VLM_MODEL
GPUS="${GPUS:-0 1 2 3}"                            # one replica per GPU
BASE_PORT="${BASE_PORT:-8000}"
MODE="${MODE:-docker}"                             # docker (default) | native
TP="${TP:-1}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"               # Mamba-hybrid (Qwen3.6-A3B): must be <= available mamba cache blocks
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-900}"
DOCKER_IMAGE="${DOCKER_IMAGE:-vllm/vllm-openai:latest}"
VLLM_BIN="${VLLM_BIN:-vllm}"
LOG_DIR="${LOG_DIR:-output/vlm_server}"
PY="${PY:-python}"

# ── derive ports + endpoint set from the GPU list ───────────────────
read -ra GPU_ARR <<< "${GPUS}"
if (( ${#GPU_ARR[@]} == 0 )); then echo "[ERROR] GPUS is empty."; exit 1; fi
PORTS=()
for i in "${!GPU_ARR[@]}"; do PORTS+=( $(( BASE_PORT + i )) ); done
ENDPOINTS=""
for p in "${PORTS[@]}"; do ENDPOINTS+="http://localhost:${p},"; done
ENDPOINTS="${ENDPOINTS%,}"

banner() { echo; echo "════════════════════════════════════════════════════════════"; echo "> $*"; echo "════════════════════════════════════════════════════════════"; }
endpoint_up() { curl -sf -m 2 "http://localhost:$1/v1/models" >/dev/null 2>&1; }

# ── status: which ports answer /v1/models ──────────────────────────
do_status() {
  echo "[run_vlm_server] status  mode=${MODE}  ports=${PORTS[*]}"
  for p in "${PORTS[@]}"; do
    if endpoint_up "$p"; then
      models=$(curl -sf -m 2 "http://localhost:$p/v1/models" \
        | "${PY}" -c 'import sys,json; print([m["id"] for m in json.load(sys.stdin).get("data",[])])' 2>/dev/null || echo "?")
      echo "  [UP]   localhost:${p}  models=${models}"
    else
      echo "  [DOWN] localhost:${p}"
    fi
  done
}

# ── check: reuse the canonical smoke test against this fleet ─────────
do_check() {
  banner "check — tools/vlm_server/check_connection.py"
  REFAV_VLM_ENDPOINTS="${ENDPOINTS}" REFAV_VLM_MODEL="${MODEL_NAME}" \
    "${PY}" tools/vlm_server/check_connection.py
}

# ── stop: kill native pidfiles, or remove docker containers ─────────
do_stop() {
  banner "stop — mode=${MODE}"
  if [[ "${MODE}" == "docker" ]]; then
    local cs; cs=$(docker ps -aq --filter "name=^vlm_" 2>/dev/null || true)
    if [[ -n "${cs}" ]]; then docker rm -f ${cs}; else echo "  no vlm_* containers"; fi
  else
    shopt -s nullglob
    local found=0 pid
    for pidf in "${LOG_DIR}"/vlm_*.pid; do
      found=1
      pid=$(cat "${pidf}" 2>/dev/null || true)
      if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
        echo "  [stop] pid ${pid} (${pidf##*/})"
        kill "${pid}" 2>/dev/null || true
      else
        echo "  [gone] ${pidf##*/}"
      fi
      rm -f "${pidf}"
    done
    (( found )) || echo "  no pidfiles in ${LOG_DIR} (nothing this script started)"
  fi
}

# ── wait until every expected port answers, or STARTUP_TIMEOUT ──────
wait_ready() {
  local deadline=$(( $(date +%s) + STARTUP_TIMEOUT ))
  local pending=("${PORTS[@]}")
  echo "[wait] up to ${STARTUP_TIMEOUT}s for ${#PORTS[@]} replica(s) to load..."
  while (( ${#pending[@]} )); do
    local still=()
    for p in "${pending[@]}"; do
      if endpoint_up "$p"; then echo "  [ready] localhost:${p}"; else still+=("$p"); fi
    done
    pending=("${still[@]}")
    (( ${#pending[@]} == 0 )) && return 0
    if (( $(date +%s) > deadline )); then
      echo "[warn] timeout; still loading: ${pending[*]}  (inspect logs in ${LOG_DIR})"
      return 1
    fi
    sleep 5
  done
}

# ── start: launch one replica per GPU ──────────────────────────────
do_start() {
  if [[ -z "${CKPT}" ]]; then
    echo "[ERROR] CKPT is required for start (local weights dir or HF id)."
    echo "        e.g. CKPT=/path/to/Qwen3.6-35B-A3B bash tools/scripts/run_vlm_server.sh"
    exit 1
  fi

  echo "[run_vlm_server] start  mode=${MODE}  model=${MODEL_NAME}"
  echo "  CKPT=${CKPT}"
  echo "  GPUs=[${GPUS}]  ->  ports=[${PORTS[*]}]  (TP=${TP}, mem=${GPU_MEM_UTIL}, max_len=${MAX_MODEL_LEN})"

  if [[ "${MODE}" == "docker" ]]; then
    command -v docker >/dev/null || { echo "[ERROR] docker not found."; exit 1; }
    banner "launch — docker (${DOCKER_IMAGE})"
    [[ "${DOCKER_IMAGE}" == *:latest ]] && echo "  [warn] DOCKER_IMAGE is unpinned (:latest) — pin a version/digest for reproducible runs."
    # Weights: a real local dir is mounted read-only. For a HF id, mount a persistent
    # HF cache so weights download ONCE and survive restarts (otherwise every replica
    # re-downloads into its ephemeral layer), and pass HF_TOKEN through for gated repos.
    local mount=() hfenv=()
    if [[ -d "${CKPT}" ]]; then
      mount=(-v "${CKPT}:${CKPT}:ro")
    else
      mount=(-v "${HF_HOME:-${HOME}/.cache/huggingface}:/root/.cache/huggingface")
      [[ -n "${HF_TOKEN:-}" ]] && hfenv+=(-e "HF_TOKEN=${HF_TOKEN}")
      [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]] && hfenv+=(-e "HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN}")
      echo "  [info] CKPT is not a local dir -> treating as HF id; HF cache: ${HF_HOME:-${HOME}/.cache/huggingface}"
    fi
    for i in "${!GPU_ARR[@]}"; do
      local gpu="${GPU_ARR[i]}" port="${PORTS[i]}"
      if endpoint_up "${port}"; then echo "  [skip] port ${port} already serving"; continue; fi
      echo "  [start] GPU ${gpu} -> container vlm_${port} -> host ${port}"
      docker run -d --name "vlm_${port}" --gpus all -e CUDA_VISIBLE_DEVICES="${gpu}" \
        --shm-size=16g --ipc=host -p "${port}:8000" "${mount[@]}" "${hfenv[@]}" \
        "${DOCKER_IMAGE}" \
        --model "${CKPT}" --served-model-name "${MODEL_NAME}" \
        --tensor-parallel-size "${TP}" --gpu-memory-utilization "${GPU_MEM_UTIL}" \
        --max-model-len "${MAX_MODEL_LEN}" --max-num-seqs "${MAX_NUM_SEQS}" --trust-remote-code --port 8000 >/dev/null
    done
  else
    command -v "${VLLM_BIN}" >/dev/null || { echo "[ERROR] '${VLLM_BIN}' not on PATH (run inside the vLLM env)."; exit 1; }
    mkdir -p "${LOG_DIR}"
    banner "launch — native (${VLLM_BIN} serve)"
    for i in "${!GPU_ARR[@]}"; do
      local gpu="${GPU_ARR[i]}" port="${PORTS[i]}"
      if endpoint_up "${port}"; then echo "  [skip] port ${port} already serving"; continue; fi
      local log="${LOG_DIR}/vlm_${port}.log" pidf="${LOG_DIR}/vlm_${port}.pid"
      echo "  [start] GPU ${gpu} -> port ${port}  (log: ${log})"
      CUDA_VISIBLE_DEVICES="${gpu}" nohup "${VLLM_BIN}" serve "${CKPT}" \
        --served-model-name "${MODEL_NAME}" \
        --tensor-parallel-size "${TP}" \
        --gpu-memory-utilization "${GPU_MEM_UTIL}" \
        --max-model-len "${MAX_MODEL_LEN}" \
        --max-num-seqs "${MAX_NUM_SEQS}" \
        --trust-remote-code \
        --port "${port}" > "${log}" 2>&1 &
      echo $! > "${pidf}"
    done
  fi

  wait_ready || true
  do_check
  banner "Done.  Export these so the atoms hit the same fleet:"
  echo "  export REFAV_VLM_ENDPOINTS=${ENDPOINTS}"
  echo "  export REFAV_VLM_MODEL=${MODEL_NAME}"
}

# ── dispatch ────────────────────────────────────────────────────────
ACTION="${1:-start}"
case "${ACTION}" in
  start)   do_start ;;
  stop)    do_stop ;;
  restart) do_stop; do_start ;;
  status)  do_status ;;
  check)   do_check ;;
  *) echo "usage: $0 {start|stop|restart|status|check}"; exit 1 ;;
esac
