"""Per-group default mu/M for IMM + class → group → params mapping.

AV2 categories are grouped into 4 groups identically to _CLASS_GROUPS in association.py
(vehicle / two_wheeler / pedestrian / static), and IMM-MOT's NuScenes 7-class tuning values
are mapped to the closest class:

  AV2 group       NuScenes source            IMM used?
  ─────────────   ─────────────────────────   ──────────
  vehicle         car (idx=2)                 ✓
  two_wheeler     bicycle (idx=0)             ✓
  pedestrian      pedestrian (idx=4)*         ✓
  static          —                           ✗ (CV only)

  * IMM-MOT uses a single ExtendKalmanFilter(CTRA) for pedestrian, but per the user's
    decision we use IMM for pedestrian too. mu assumes acceleration/turning happens
    frequently, similar to car (conservative initial value).
"""

from __future__ import annotations

import numpy as np

from .motion_models import MotionModelParams
from .imm_filter import IMMParams


# ── sub-filter order common to all groups ────────────────────────────────────
SUB_MODELS = ["CV", "CA", "CTRV", "CTRA"]


# ── mu/M mapping from NuScenes IMM-MOT ─────────────────────────────────────
# The sub-filter order in the IMM-MOT config (tools/IMM-MOT/config/nusc_config.yaml:28-64) is
#   filter1=LinearKalmanFilter(CV), filter2=LinearKalmanFilter(CA),
#   filter3=ExtendKalmanFilter(CTRV), filter4=ExtendKalmanFilter(CTRA)
# same as our SUB_MODELS order [CV, CA, CTRV, CTRA].

# car (NuScenes idx=2) — vehicle group default
_MU_CAR = np.array([0.04, 0.04, 0.12, 0.80], dtype=np.float64)
_M_CAR = np.array([
    [0.95, 0.02, 0.02, 0.01],
    [0.02, 0.95, 0.01, 0.02],
    [0.02, 0.01, 0.95, 0.02],
    [0.01, 0.02, 0.02, 0.95],
], dtype=np.float64)

# bicycle (NuScenes idx=0) — two_wheeler group default
_MU_BICYCLE = np.array([0.01, 0.01, 0.01, 0.97], dtype=np.float64)
_M_BICYCLE = _M_CAR.copy()

# IMM tuning for pedestrian (NuScenes idx=4) is not in the original (pedestrian → single EKF).
# guess — pedestrian has frequent yaw changes and frequent accel/decel, so weight CTRV/CTRA high
#         and CV/CA low. mu_init is similar to car but with CTRV/CTRA equal.
_MU_PEDESTRIAN = np.array([0.05, 0.05, 0.45, 0.45], dtype=np.float64)
_M_PEDESTRIAN = np.array([
    [0.90, 0.05, 0.03, 0.02],
    [0.05, 0.90, 0.02, 0.03],
    [0.03, 0.02, 0.90, 0.05],
    [0.02, 0.03, 0.05, 0.90],
], dtype=np.float64)


# ── per-group MotionModelParams ──────────────────────────────────────────
# The default is the same scale as multi_class_tracking's 4D CV (Q.x=0.07, R.x=0.15, etc.) —
# overridable in config.yaml.
def _default_motion_params() -> MotionModelParams:
    return MotionModelParams(
        q_xy=0.07, q_v=0.35, q_a=0.5,
        q_yaw=0.05, q_yaw_rate=0.1,
        r_xy=0.15, r_yaw=0.1,
        p0_xy=1.0, p0_v=100.0, p0_a=100.0,
        p0_yaw=1.0, p0_yaw_rate=10.0,
    )


# ── group → IMMParams ──────────────────────────────────────────────────
def default_imm_params(group: str) -> IMMParams:
    """group name → default IMMParams. Used only when config does not override."""
    mp = _default_motion_params()
    if group == "vehicle":
        return IMMParams(mu_init=_MU_CAR.copy(), M=_M_CAR.copy(),
                         sub_models=SUB_MODELS, motion_params=mp)
    if group == "two_wheeler":
        return IMMParams(mu_init=_MU_BICYCLE.copy(), M=_M_BICYCLE.copy(),
                         sub_models=SUB_MODELS, motion_params=mp)
    if group == "pedestrian":
        return IMMParams(mu_init=_MU_PEDESTRIAN.copy(), M=_M_PEDESTRIAN.copy(),
                         sub_models=SUB_MODELS, motion_params=mp)
    raise ValueError(f"no IMM defaults for group {group!r} (static uses CV, not IMM)")


# ── config dict → IMMParams override ──────────────────────────────────
def _merge_motion_params(base: MotionModelParams, *override_dicts: dict) -> MotionModelParams:
    """Overwrite override_dicts onto base in order. None / empty dict is skipped.
    Each dict maps MotionModelParams field names → float. Missing keys keep base.
    """
    kwargs = {k: getattr(base, k) for k in base.__dataclass_fields__}
    for od in override_dicts:
        if not od:
            continue
        for k, v in od.items():
            if k in kwargs and v is not None:
                kwargs[k] = float(v)
    return MotionModelParams(**kwargs)


def imm_params_from_config(group: str, cfg_imm: dict,
                           t_cfg: dict | None = None) -> IMMParams:
    """Build per-group IMMParams from the tracking.imm.* structure of config.yaml.

    Structure (no cascade — simple):
      - mu_by_group[group]  : per-group mode prior
      - M_by_group[group]   : per-group transition
      - imm.<MODEL>         : per-model Q/R/P0 (common to all groups). e.g. imm.CV.q_xy
                              missing keys use the MotionModelParams() code default.

    If t_cfg (the whole tracking.*) is given, read the same direction_skew /
    init_cov_yaw_aligned / conf_low_* as the CV-only KF and inject them commonly into all
    sub-filters (model-independent shared params).
    """
    base = default_imm_params(group)
    if not cfg_imm:
        return base

    # mu override (per-group)
    mu_by_group = (cfg_imm.get("mu_by_group") or {})
    if group in mu_by_group:
        mu = np.array(mu_by_group[group], dtype=np.float64)
        assert mu.shape == (len(SUB_MODELS),), \
            f"mu_by_group[{group}] must be length {len(SUB_MODELS)}"
        s = float(mu.sum())
        assert s > 0, f"mu_by_group[{group}] sum must be > 0"
        base.mu_init = (mu / s)

    # M override (per-group)
    M_by_group = (cfg_imm.get("M_by_group") or {})
    if group in M_by_group:
        M = np.array(M_by_group[group], dtype=np.float64)
        N = len(SUB_MODELS)
        assert M.shape == (N, N), f"M_by_group[{group}] must be {N}x{N}"
        for i in range(N):
            s = float(M[i].sum())
            if s > 0:
                M[i] = M[i] / s
        base.M = M

    # ── direction_skew / init_cov_yaw_aligned / conf_low_* — read from tracking.* and
    # inject commonly into all sub-filters (reuse the same keys as the CV-only KF, model-independent) ──
    tc = t_cfg or {}
    skew = tc.get("direction_skew", {}) or {}
    init_cov = tc.get("init_cov_yaw_aligned", {}) or {}
    skew_enabled = bool(skew.get("enabled", False))
    skew_pose = float(skew.get("lon_pose_factor", 1.0))
    skew_vel = float(skew.get("lon_velocity_factor", 1.0))
    init_enabled = bool(init_cov.get("enabled", False))
    init_long = float(init_cov.get("long_factor", 1.0))
    init_lat = float(init_cov.get("lat_factor", 1.0))
    conf_low_thr = float(tc.get("conf_low_threshold", 0.0))
    conf_low_scale = float(tc.get("conf_low_R_scale", 1.0))
    static_r_yaw_scale = float(tc.get("static_r_yaw_scale", 1.0))

    # ── per-model Q/R/P0 — directly from imm.<MODEL> (common to all groups) ──
    # Overwrite the per-model values from config onto the code default (MotionModelParams()).
    sub_mp: dict[str, MotionModelParams] = {}
    for m in SUB_MODELS:
        md = cfg_imm.get(m) or {}
        mp = _merge_motion_params(MotionModelParams(), md)
        mp.skew_enabled = skew_enabled
        mp.skew_lon_pose_factor = skew_pose
        mp.skew_lon_velocity_factor = skew_vel
        mp.init_cov_yaw_enabled = init_enabled
        mp.init_cov_long_factor = init_long
        mp.init_cov_lat_factor = init_lat
        mp.conf_low_threshold = conf_low_thr
        mp.conf_low_R_scale = conf_low_scale
        mp.static_r_yaw_scale = static_r_yaw_scale
        sub_mp[m] = mp
    base.sub_motion_params = sub_mp
    # motion_params is the fallback (rarely used since every model is specified) — set to the CV value
    base.motion_params = sub_mp.get("CV", MotionModelParams())

    return base
