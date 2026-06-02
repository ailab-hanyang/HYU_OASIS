#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
# run_layer1_context.sh — run the full layer1_context pipeline in one command.
#
#   (1) annotate     vLLM context annotation   (ring-camera frames -> raw JSON)
#   (2) postprocess  smoothing + dilation       (raw JSON -> *_processed JSON)
#   (3) validate     schema check (optional)     (per-log processed JSON)
#
# Output root: output/layer1_context/<split>{,_processed}/<log_id>/<ts>.json
#
# Usage:
#   bash tools/scripts/run_layer1_context.sh
#
# Optional env overrides:
#   SPLIT=val WORKERS=8 \
#   LOGS="3de5b5d6-...  02678d04-..."   # empty = whole split
#   CONFIG=tools/layer1_context/config/settings.yaml   # alt settings.yaml
#   SKIP_VALIDATE=1                     # skip the schema-validation stage
#   PY=python
#
# Prerequisite: run inside the vLLM container / on a GPU host (the annotate
#   stage imports vllm). See tools/layer1_context/docker/run_container.sh.
# ════════════════════════════════════════════════════════════════════════════
set -eo pipefail

# ── project root (relative to this script: tools/scripts -> ../..) ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# ── settings (override via env) ─────────────────────────────────────
SPLIT="${SPLIT:-val}"
WORKERS="${WORKERS:-8}"                 # postprocess worker processes
LOGS="${LOGS:-}"                        # specific logs (empty = whole split)
CONFIG="${CONFIG:-}"                    # alt settings.yaml for the annotate stage
SKIP_VALIDATE="${SKIP_VALIDATE:-0}"
PY="${PY:-python}"
ROOT="output/layer1_context"

# optional --log-ids / --config arguments
LOGS_ARG=()
if [[ -n "${LOGS}" ]]; then LOGS_ARG=(--log-ids ${LOGS}); fi
CONFIG_ARG=()
if [[ -n "${CONFIG}" ]]; then
  CONFIG_ARG=(--config "${CONFIG}")
  # annotate honors paths.output_dir from CONFIG; resolve it so postprocess +
  # validate operate on the SAME root (otherwise a custom output_dir would make
  # stage 1 write one place while stages 2/3 read the hardcoded default).
  _cfg_root="$(${PY} -c 'import sys; from tools.layer1_context.src.loader import load_config; print(load_config(sys.argv[1])["paths"]["output_dir"])' "${CONFIG}" 2>/dev/null || true)"
  ROOT="${_cfg_root:-${ROOT}}"
fi

banner() { echo; echo "════════════════════════════════════════════════════════════"; echo "> $*"; echo "════════════════════════════════════════════════════════════"; }

START=$(date +%s)
echo "[run_layer1_context]  split=${SPLIT}  workers=${WORKERS}  logs=${LOGS:-ALL}"
echo "  (1) annotate     -> ${ROOT}/${SPLIT}/"
echo "  (2) postprocess  -> ${ROOT}/${SPLIT}_processed/"
echo "  (3) validate     $( [[ "${SKIP_VALIDATE}" == 1 ]] && echo '(skip)' || echo '-> schema check' )"

# ── (1) vLLM annotation ────────────────────────────────────────────
banner "(1) annotate — tools.layer1_context.annotate"
${PY} -m tools.layer1_context.annotate \
  --split "${SPLIT}" "${CONFIG_ARG[@]}" "${LOGS_ARG[@]}"

# ── (2) post-processing (smoothing + confirmed-run dilation) ───────
banner "(2) postprocess — tools.layer1_context.postprocess"
${PY} -m tools.layer1_context.postprocess \
  --split "${SPLIT}" --workers "${WORKERS}" \
  --input-dir "${ROOT}/${SPLIT}" --output-dir "${ROOT}/${SPLIT}_processed" \
  "${LOGS_ARG[@]}"

# ── (3) schema validation (optional) ───────────────────────────────
if [[ "${SKIP_VALIDATE}" == "1" ]]; then
  banner "(3) validate — skip (SKIP_VALIDATE=1)"
else
  banner "(3) validate — tools.layer1_context.validate"
  PROC_ROOT="${ROOT}/${SPLIT}_processed"
  if [[ -n "${LOGS}" ]]; then
    VAL_DIRS=(); for lg in ${LOGS}; do VAL_DIRS+=("${PROC_ROOT}/${lg}"); done
  else
    VAL_DIRS=("${PROC_ROOT}"/*/)
  fi
  for d in "${VAL_DIRS[@]}"; do
    [[ -d "${d}" ]] || { echo "[warn] missing processed dir: ${d}"; continue; }
    ${PY} -m tools.layer1_context.validate "${d%/}"
  done
fi

ELAPSED=$(( $(date +%s) - START ))
banner "Done.  Output: ${ROOT}/${SPLIT}_processed/   (elapsed ${ELAPSED}s / $((ELAPSED/60))m)"
