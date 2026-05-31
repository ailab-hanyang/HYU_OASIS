"""IMM RTS Smoother — reads the forward IMM sidecar (imm_smooth_inputs.feather)
and produces a smoothed state via per-model RTS backward + forward μ re-combination (stage 1).

Reference: Yadav et al., "IMM Forward Filtering and Backward Smoothing"
(IEEE TAES 2012). This stage 1 performs that paper's state smoothing (eqs. 24–26) per model,
and combines modes using the forward filtered mode probability μ_{k|k} (backward mode mixing = stage 2, deferred).

Input sidecar schema (multi_class_tracking.SmoothInputRecorder):
  track_uuid | timestamp_ns | stage(init|predict|update) | model | model_idx |
  dim | mu | likelihood | dt | state(list) | cov(flat) | F(flat)
  - stage="predict" : prior  x_{k|k-1}, P_{k|k-1}, F of this step
  - stage="update"/"init" : posterior x_{k|k}, P_{k|k}
  - first frame has no predict → init(=posterior) only. F_all[0]=I placeholder.

coords/yaw: each model's native state. backward runs in native dim (F is defined natively).
per-model smoothed native → standard 8D (world) via motion_models conversion → circular μ-weighted
combination → smoothed mixed (x, y, yaw). Only the RTS output's position/yaw is used (upper apply replaces sm_annotations).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Reuse multi_class_tracking's model/standard-form utils (native→standard, circular combination).
import sys as _sys
_MCT = Path(__file__).resolve().parents[3]
if str(_MCT) not in _sys.path:
    _sys.path.insert(0, str(_MCT))
from tools.multi_class_tracking.src.motion_models import (  # noqa: E402
    init_from_xy_yaw, MotionModelParams, STD_X, STD_Y, STD_YAW, _wrap,
)
from tools.multi_class_tracking.src.imm_filter import (  # noqa: E402
    _circular_weighted_state,
)

# Per-model native yaw index (same as motion_models' YAW_STATE_IDX).
_YAW_IDX = {"CV": 4, "CA": 6, "CTRV": 3, "CTRA": 4}


@dataclass
class IMMSmoothParams:
    n_min: int = 2          # frame count < n_min → skip smoothing (pass through original)
    freeze_initial: bool = False  # True → each model's backward does not smooth first frame (k=0)
    # For a static frame (T), inflate the T+1 prior cov (backward gain information matrix) to weaken the gain.
    static_inflate_enabled: bool = False
    static_inflate_mult: float = 10.0
    static_inflate_vstatic: float = 0.25   # m/s

    @classmethod
    def from_config(cls, cfg: dict) -> "IMMSmoothParams":
        sm = (cfg or {}).get("imm_smoothing", {}) or {}
        sci = (sm.get("static_cov_inflate") or {})
        return cls(n_min=int(sm.get("n_min", 2)),
                   freeze_initial=bool(sm.get("freeze_initial", True)),
                   static_inflate_enabled=bool(sci.get("enabled", True)),
                   static_inflate_mult=float(sci.get("multiplier", 10.0)),
                   static_inflate_vstatic=float(sci.get("v_static", 0.25)))


# ── single-model native RTS backward ──────────────────────────────────────
def _rts_backward_native(x_pred, P_pred, x_post, P_post, F_all, yaw_idx,
                         freeze_initial=False, static_mask=None, static_mult=10.0):
    """Single-model native RTS backward. (yaw-idx generalized version of rts_smoother._backward_pass)

    x_pred[k], P_pred[k] : prior  (k=0 is placeholder, not used)
    x_post[k], P_post[k] : posterior
    F_all[k]             : x_{k-1}→x_k transition (k=0 is I)
    boundary: x_{N-1|N}=x_{N-1|N-1}. k=N-2..0.
    freeze_initial=True → don't smooth first frame (k=0) (keeps forward posterior). Prevents
    a large initial P0 from making the backward gain too large and dragging the initial box toward the next measurement.
    static_mask[k]=True → frame k(T) static → inflate the gain's P_pred[k+1] (T+1 prior) by static_mult
    to weaken the backward gain (= more trust in forward/R). Corrects the dilution of a static vehicle's large Q.
    Returns: x_sm (N, dim), P_sm (N, dim, dim).
    """
    N, dim = x_post.shape
    x_sm = x_post.copy()   # non-smoothed frames (incl. frozen first frame) keep forward posterior
    P_sm = P_post.copy()
    k_stop = 0 if freeze_initial else -1
    for k in range(N - 2, k_stop, -1):
        F_next = F_all[k + 1]                       # x_k → x_{k+1}
        Pkk_FT = P_post[k] @ F_next.T
        P_pred_k1 = P_pred[k + 1]
        if static_mask is not None and bool(static_mask[k]):
            P_pred_k1 = static_mult * P_pred_k1     # static → inflate T+1 prior cov to weaken gain
        try:
            C = np.linalg.solve(P_pred_k1.T, Pkk_FT.T).T
        except np.linalg.LinAlgError:
            C = Pkk_FT @ np.linalg.pinv(P_pred_k1)
        dx = x_sm[k + 1] - x_pred[k + 1]
        if yaw_idx >= 0:
            dx[yaw_idx] = _wrap(float(dx[yaw_idx]))
        x_sm[k] = x_post[k] + C @ dx
        if yaw_idx >= 0:
            x_sm[k, yaw_idx] = _wrap(float(x_sm[k, yaw_idx]))
        dP = P_sm[k + 1] - P_pred[k + 1]
        P_sm[k] = P_post[k] + C @ dP @ C.T
        P_sm[k] = 0.5 * (P_sm[k] + P_sm[k].T)
    return x_sm, P_sm


def _dummy_params() -> MotionModelParams:
    # Used only for native→standard conversion, so noise values don't matter. For model instantiation.
    return MotionModelParams()


def smooth_one_track(side: pd.DataFrame, params: IMMSmoothParams):
    """sidecar rows of one track_uuid → smoothed (ts, x_world, y_world, yaw_world).

    side: rows of this uuid (multiple model × multiple ts × stage). Returning None means skip (pass through original).
    """
    ts_all = np.array(sorted(side["timestamp_ns"].unique()), dtype=np.int64)
    N = ts_all.size
    if N < params.n_min:
        return None
    ts_index = {int(t): i for i, t in enumerate(ts_all)}
    models = sorted(side["model"].unique())          # usually [CA, CTRA, CTRV, CV]

    # Build per-model time-series tensors + native RTS. Then per-model smoothed → standard 8D.
    per_model_std = {}     # model → (N, 8) standard state (input to circular yaw combination)
    mu_post = np.zeros((N, len(models)), dtype=np.float64)  # forward filtered μ (at update time)

    for mj, model in enumerate(models):
        sm = side[side["model"] == model]
        dim = int(sm["dim"].iloc[0])
        yaw_idx = _YAW_IDX.get(model, -1)
        x_pred = np.zeros((N, dim)); P_pred = np.zeros((N, dim, dim))
        x_post = np.zeros((N, dim)); P_post = np.zeros((N, dim, dim))
        F_all = np.repeat(np.eye(dim)[None], N, axis=0)
        have_post = np.zeros(N, dtype=bool)
        # fill posterior (update/init) and prior (predict)
        for _, r in sm.iterrows():
            k = ts_index[int(r["timestamp_ns"])]
            st = np.asarray(r["state"], dtype=np.float64)
            cv = np.asarray(r["cov"], dtype=np.float64).reshape(dim, dim)
            if r["stage"] == "predict":
                x_pred[k] = st; P_pred[k] = cv
                F_all[k] = np.asarray(r["F"], dtype=np.float64).reshape(dim, dim)
            else:  # update or init → posterior
                x_post[k] = st; P_post[k] = cv
                have_post[k] = True
                mu_post[k, mj] = float(r["mu"])
        # for frames without posterior (rare), fill with prior (continuity). At k=0, prior=posterior(init).
        for k in range(N):
            if not have_post[k]:
                x_post[k] = x_pred[k]; P_post[k] = P_pred[k]
        # k=0 prior placeholder = posterior
        x_pred[0] = x_post[0]; P_pred[0] = P_post[0]

        # static frame mask — native state position ([0,1]) frame-to-frame speed < v_static.
        # (per-model positions are nearly identical, so computing with the current model's x_post is consistent)
        static_mask = None
        if params.static_inflate_enabled and N >= 2:
            pxy = x_post[:, :2]
            spd = np.zeros(N)
            for kk in range(N - 1):
                dt = max((float(ts_all[kk + 1]) - float(ts_all[kk])) * 1e-9, 1e-3)
                spd[kk] = np.linalg.norm(pxy[kk + 1] - pxy[kk]) / dt
            static_mask = spd < params.static_inflate_vstatic

        x_sm, P_sm = _rts_backward_native(x_pred, P_pred, x_post, P_post, F_all, yaw_idx,
                                          freeze_initial=params.freeze_initial,
                                          static_mask=static_mask,
                                          static_mult=params.static_inflate_mult)

        # per-model smoothed native → standard 8D (reuse motion_models conversion).
        f = init_from_xy_yaw(0.0, 0.0, 0.0, _dummy_params(), model)
        std_states = np.zeros((N, 8), dtype=np.float64)
        for k in range(N):
            f.state = x_sm[k].copy()
            f.cov = P_sm[k].copy()
            s_std, _ = f.to_standard()
            std_states[k] = s_std
        per_model_std[model] = std_states

    # mode combination — forward μ_{k|k} (uniform if absent). circular yaw mean.
    out = np.zeros((N, 3), dtype=np.float64)   # x, y, yaw (world)
    for k in range(N):
        w = mu_post[k].copy()
        if w.sum() <= 1e-9:
            w = np.ones(len(models)) / len(models)
        else:
            w = w / w.sum()
        states_k = [per_model_std[m][k] for m in models]
        mixed = _circular_weighted_state(states_k, w)
        out[k, 0] = mixed[STD_X]; out[k, 1] = mixed[STD_Y]
        out[k, 2] = _wrap(float(mixed[STD_YAW]))
    return ts_all, out


def smooth_log(sidecar_path: Path, params: IMMSmoothParams) -> dict:
    """sidecar feather → {uuid: (ts_arr, out[N,3] world x/y/yaw)}.

    A uuid not in the sidecar (= CV single static track) is absent from the result dict → apply passes through original.
    """
    if not sidecar_path.exists():
        return {}
    df = pd.read_feather(sidecar_path)
    result = {}
    for uuid, side in df.groupby("track_uuid"):
        r = smooth_one_track(side, params)
        if r is not None:
            result[str(uuid)] = r
    return result


def world_to_ego(wx: float, wy: float, wyaw: float,
                 cx: float, cy: float, cyaw: float):
    """smoothed world (x, y, yaw) → ego frame (ex, ey, eyaw). ego pose=(cx,cy,cyaw)."""
    c, s = np.cos(cyaw), np.sin(cyaw)
    dx, dy = wx - cx, wy - cy
    ex = c * dx + s * dy
    ey = -s * dx + c * dy
    eyaw = _wrap(wyaw - cyaw)
    return ex, ey, eyaw


def smooth_log_to_ego(sidecar_path: Path, params: IMMSmoothParams,
                      ego_ts: np.ndarray, ego_xyz: np.ndarray, ego_yaw: np.ndarray) -> dict:
    """Return a replace dict converting smooth_log results (world) to ego frame.

    Returns: {(uuid, ts): (ex, ey, eyaw)} — used by apply to replace tx_m/ty_m/qw/qz in sm_annotations.
    ego_* are Tracks' ego_poses_ts/xyz/yaw (city_SE3_ego). Nearest-ts matching.
    """
    world = smooth_log(sidecar_path, params)
    if not world:
        return {}
    ego_ts = np.asarray(ego_ts, dtype=np.int64)
    order = np.argsort(ego_ts)
    ets = ego_ts[order]; exyz = np.asarray(ego_xyz)[order]; eyaw = np.asarray(ego_yaw)[order]

    def _ego_at(t: int):
        i = int(np.searchsorted(ets, t))
        if i < len(ets) and int(ets[i]) == int(t):
            pass
        elif i == 0:
            i = 0
        elif i >= len(ets):
            i = len(ets) - 1
        else:
            i = i - 1 if (t - ets[i - 1]) <= (ets[i] - t) else i
        return float(exyz[i, 0]), float(exyz[i, 1]), float(eyaw[i])

    out = {}
    for uuid, (ts_arr, w) in world.items():
        for k in range(len(ts_arr)):
            cx, cy, cyaw = _ego_at(int(ts_arr[k]))
            ex, ey, ez_yaw = world_to_ego(w[k, 0], w[k, 1], w[k, 2], cx, cy, cyaw)
            out[(str(uuid), int(ts_arr[k]))] = (ex, ey, ez_yaw)
    return out
