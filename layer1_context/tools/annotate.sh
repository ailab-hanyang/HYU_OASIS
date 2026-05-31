#!/usr/bin/env bash
# Full-split production run: vLLM annotation on every log of a split,
# followed by post-processing. Pair with tmux/nohup for long jobs.
#
# Usage (inside the docker container, from repo root):
#     bash layer1_context/tools/annotate.sh                # default: val
#     bash layer1_context/tools/annotate.sh test
#     bash layer1_context/tools/annotate.sh val 8          # postprocess workers

set -euo pipefail

SPLIT="${1:-val}"
PP_WORKERS="${2:-4}"

cd "$(dirname "$0")/../.."   # repo root
export PYTHONPATH="."

START=$(date +%s)
echo "============================================================"
echo "  Context Layer v4 — full run"
echo "  split:           ${SPLIT}"
echo "  postprocess workers: ${PP_WORKERS}"
echo "  started:         $(date -Iseconds)"
echo "============================================================"

echo
echo "[1/2] vLLM annotation (split=${SPLIT})"
python -m layer1_context.tools.annotate --split "${SPLIT}"

echo
echo "[2/2] post-processing (split=${SPLIT}, workers=${PP_WORKERS})"
python -m layer1_context.tools.postprocess --split "${SPLIT}" --workers "${PP_WORKERS}"

ELAPSED=$(( $(date +%s) - START ))
echo
echo "Done. Total elapsed: ${ELAPSED}s ($((ELAPSED/60))m)"
