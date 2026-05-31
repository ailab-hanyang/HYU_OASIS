#!/usr/bin/env bash
# Quick smoke test: run vLLM annotation + postprocess on a single log,
# then validate the output JSON schema.
#
# Usage (inside the docker container, from repo root):
#     bash layer1_context/tools/test.sh
#     bash layer1_context/tools/test.sh <LOG_ID>
#     bash layer1_context/tools/test.sh <LOG_ID> <SPLIT>

set -euo pipefail

LOG_ID="${1:-02678d04-cc9f-3148-9f95-1ba66347dff9}"
SPLIT="${2:-val}"
ROOT="output/context_annotations_v4"

cd "$(dirname "$0")/../.."   # repo root
export PYTHONPATH="."

echo "============================================================"
echo "  Context Layer v4 — smoke test"
echo "  log_id: ${LOG_ID}"
echo "  split:  ${SPLIT}"
echo "============================================================"

echo
echo "[1/3] vLLM annotation"
python -m layer1_context.tools.annotate --log-ids "${LOG_ID}" --split "${SPLIT}"

echo
echo "[2/3] post-processing"
python -m layer1_context.tools.postprocess --split "${SPLIT}" --log-ids "${LOG_ID}"

echo
echo "[3/3] schema validation"
python -m layer1_context.tools.validate "${ROOT}/${SPLIT}/${LOG_ID}"
python -m layer1_context.tools.validate "${ROOT}/${SPLIT}_processed/${LOG_ID}"

echo
echo "Done. Raw JSON:       ${ROOT}/${SPLIT}/${LOG_ID}/"
echo "      Processed JSON: ${ROOT}/${SPLIT}_processed/${LOG_ID}/"
