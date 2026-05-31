#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
# run_imm_smoothing.sh — run the full IMM smoothing pipeline in one command.
#
#   (1) ego offset      derive_ego_from_base   (Le3DE2E_Tracking -> _ego)
#   (2) yaw-fix         apply_yaw_correction   (_ego -> _ego_yawfix)
#   (3) IMM re-tracking multi_class_tracking   (_ego_yawfix -> IMM dst + sidecar)
#   (4) IMM smoothing   apply_rts_smoothing    (IMM dst -> smoothed)
#
# Usage:
#   bash tools/scripts/run_imm_smoothing.sh
#
# Optional env overrides:
#   SPLIT=val WORKERS=8 \
#   BASE=Le3DE2E_Tracking EGO=... YAWFIX=... TRACK=... SMOOTH=... \
#   LOGS="3de5b5d6-...  02678d04-..."   # empty = whole split
#   SKIP_EGO=1 SKIP_YAWFIX=1            # skip stages already produced
#
# Prerequisite: tracking.motion_model=imm in
#   tools/multi_class_tracking/config/config.yaml
#   (produces the imm_smooth_inputs.feather sidecar required by step (4)).
# ════════════════════════════════════════════════════════════════════════════
set -eo pipefail

# ── project root (relative to this script: tools/scripts -> ../..) ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# ── settings (override via env) ─────────────────────────────────────
SPLIT="${SPLIT:-val}"
WORKERS="${WORKERS:-8}"
BASE="${BASE:-Le3DE2E_Tracking}"                          # input base (raw detection)
EGO="${EGO:-${BASE}_ego}"                                 # (1) output
YAWFIX="${YAWFIX:-${EGO}_yawfix}"                         # (2) output
TRACK="${TRACK:-Le3DE2E_Tracking_ego_yawfix2_track6}"     # (3) IMM re-tracking output
SMOOTH="${SMOOTH:-${TRACK}_imm_smooth}"                   # (4) smoothing output
LOGS="${LOGS:-}"                                          # specific logs (empty = all)
SKIP_EGO="${SKIP_EGO:-0}"
SKIP_YAWFIX="${SKIP_YAWFIX:-0}"
PY="${PY:-python}"
MCT_CFG="tools/multi_class_tracking/config/config.yaml"

# --logs argument (empty LOGS => whole split)
LOGS_ARG=()
if [[ -n "${LOGS}" ]]; then LOGS_ARG=(--logs ${LOGS}); fi

banner() { echo; echo "════════════════════════════════════════════════════════════"; echo "> $*"; echo "════════════════════════════════════════════════════════════"; }

# ── precondition: re-tracking must be IMM so the sidecar is produced ──
if ! grep -qE "^[[:space:]]*motion_model:[[:space:]]*imm" "${MCT_CFG}"; then
  echo "[ERROR] tracking.motion_model in ${MCT_CFG} is not 'imm'."
  echo "        Set it to 'imm' so the sidecar (imm_smooth_inputs.feather) is produced, then rerun."
  exit 1
fi

echo "[run_imm_smoothing]  split=${SPLIT}  workers=${WORKERS}  logs=${LOGS:-ALL}"
echo "  (1) ${BASE}  ->  ${EGO}        $( [[ "${SKIP_EGO}" == 1 ]] && echo '(skip)')"
echo "  (2) ${EGO}  ->  ${YAWFIX}      $( [[ "${SKIP_YAWFIX}" == 1 ]] && echo '(skip)')"
echo "  (3) ${YAWFIX}  ->  ${TRACK}    (IMM re-tracking)"
echo "  (4) ${TRACK}  ->  ${SMOOTH}    (IMM smoothing)"

# ── (1) ego offset ─────────────────────────────────────────────────
if [[ "${SKIP_EGO}" == "1" ]]; then
  banner "(1) ego offset — skip (SKIP_EGO=1)"
else
  banner "(1) ego offset — derive_ego_from_base"
  ${PY} -m tools.rts_smoothing.derive_ego_from_base \
    --src_tracker "${BASE}" --dst_tracker "${EGO}" \
    --split "${SPLIT}" --workers "${WORKERS}" --force "${LOGS_ARG[@]}"
fi

# ── (2) yaw-fix ────────────────────────────────────────────────────
if [[ "${SKIP_YAWFIX}" == "1" ]]; then
  banner "(2) yaw-fix — skip (SKIP_YAWFIX=1)"
else
  banner "(2) yaw-fix — apply_yaw_correction"
  ${PY} -m tools.rts_smoothing.apply_yaw_correction \
    --src_tracker "${EGO}" --dst_tracker "${YAWFIX}" \
    --split "${SPLIT}" --workers "${WORKERS}" --force "${LOGS_ARG[@]}"
fi

# ── (3) IMM re-tracking (multi_class_tracking) ─────────────────────
banner "(3) IMM re-tracking — multi_class_tracking.apply_tracking"
if [[ -n "${LOGS}" ]]; then
  ${PY} -m tools.multi_class_tracking.apply_tracking \
    --src_tracker "${YAWFIX}" --dst_tracker "${TRACK}" \
    --split "${SPLIT}" --workers "${WORKERS}" --force "${LOGS_ARG[@]}"
else
  ${PY} -m tools.multi_class_tracking.apply_tracking \
    --src_tracker "${YAWFIX}" --dst_tracker "${TRACK}" \
    --split "${SPLIT}" --all --workers "${WORKERS}" --force
fi

# ── (4) IMM smoothing (apply_rts_smoothing) ────────────────────────
banner "(4) IMM smoothing — apply_rts_smoothing (smoother_mode=imm)"
${PY} -m tools.rts_smoothing.apply_rts_smoothing \
  --src_tracker "${TRACK}" --dst_tracker "${SMOOTH}" \
  --split "${SPLIT}" --smoother_mode imm --workers "${WORKERS}" --force "${LOGS_ARG[@]}"

banner "Done.  Final output: output/tracker_predictions/${SMOOTH}/${SPLIT}/"
