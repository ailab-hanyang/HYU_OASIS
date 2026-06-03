#!/usr/bin/env bash
# Run the scene_context_extraction vLLM container (e.g. on an H100 server).
# Adjust REPO_ROOT / EXTRA_MOUNTS below for your environment.

set -euo pipefail

# Host path to this repo (defaults to the repo root this script lives in).
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"

WORKSPACE=/workspace/HYU_OASIS
IMAGE="${IMAGE:-hyu-oasis-vllm:latest}"
NAME="${NAME:-scene_context_vllm}"

# Add extra read-only mounts here if your Argoverse2 data or VLM weights live
# outside the repo, e.g.:
#   EXTRA_MOUNTS="-v /data/Argoverse2:/data/Argoverse2:ro -v /models/Qwen:/models/Qwen:ro"
EXTRA_MOUNTS="${EXTRA_MOUNTS:-}"

docker run --rm -it \
  --gpus all \
  --shm-size=32g \
  --ipc=host \
  -v "${REPO_ROOT}:${WORKSPACE}" \
  ${EXTRA_MOUNTS} \
  -w "${WORKSPACE}" \
  --name "${NAME}" \
  "${IMAGE}" \
  bash
