"""RTS Smoother — forward Kalman filter + backward RTS pass.

State : T = [x, y, z, θ, l, w, h, s, vx, vy, vz]^T   (11)
Meas  : z = [x, y, z, θ, l, w, h, s]^T               (8)

Pipeline (smooth_track)
─────────────────────────────────────────────────────────────────────────
   TrackState (uuid, ts, t, yaw, lwh, score, quat)
        │
        ▼
   [1] _build_measurements   →  Z (N, 8)            yaw made continuous via unwrap
        │                       (timestamps_ns → dts [s])
        │
        ▼
   [2] _initial_state        →  x_0 (11), P_0 = diag(P0_diag)
        │                       velocity from first two frames finite-difference
        │
        ▼
   [3] _forward_pass (Kalman Filter, k = 0..N-1)
        │       k = 0  : predict skip (initial as prior) → update(Z[0])
        │       k ≥ 1  : predict(Δt_k) → update(Z[k])
        │       store: x_pred, P_pred, x_post, P_post, F  (all length N)
        │
        ▼
   [4] _backward_pass (RTS, k = N-2..0)
        │       C_k = P_{k|k} F_{k+1}^T P_{k+1|k}^{-1}
        │       x_{k|N} = x_{k|k} + C_k · (x_{k+1|N} − x_{k+1|k})  (θ wrap)
        │       P_{k|N} = P_{k|k} + C_k · (P_{k+1|N} − P_{k+1|k}) · C_k^T
        │       boundary: x_{N-1|N} = x_{N-1|N-1}, P_{N-1|N} = P_{N-1|N-1}
        │
        ▼
   [5] _to_track_state  →  smoothed TrackState
        │       yaw → quat re-synthesis, score [0,1] clip, lwh ≥ 1e-3
        ▼
   (info, new_track)

Forward equations (kalman_predict.predict + kalman_update.update)
    x_{k|k-1} = F_k · x_{k-1|k-1}
    P_{k|k-1} = F_k · P_{k-1|k-1} · F_k^T + Q
    y_k       = z_k − H · x_{k|k-1}        (θ innovation wrap)
    S_k       = H · P_{k|k-1} · H^T + R
    K_k       = P_{k|k-1} · H^T · S_k^{-1}
    x_{k|k}   = x_{k|k-1} + K_k · y_k
    P_{k|k}   = (I − K_k H) P_{k|k-1} (I − K_k H)^T + K_k R K_k^T  [Joseph]

Backward RTS
    C_k     = P_{k|k} F_{k+1}^T P_{k+1|k}^{-1}
    Δx      = x_{k+1|N} − x_{k+1|k}        (θ diff wrap)
    x_{k|N} = x_{k|k} + C_k · Δx
    P_{k|N} = P_{k|k} + C_k · (P_{k+1|N} − P_{k+1|k}) · C_k^T
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .track_state import TrackState
from .kalman_predict import (
    PredictParams, predict, STATE_DIM, STATE_NAMES,
    IDX_X, IDX_Y, IDX_Z, IDX_THETA,
    IDX_L, IDX_W, IDX_H, IDX_S,
    IDX_VX, IDX_VY, IDX_VZ,
)
from .kalman_update import UpdateParams, update, MEAS_DIM


# ── parameter / result containers ─────────────────────────────────────────
@dataclass
class RTSParams:
    predict: PredictParams      # Q (11×11 diag)
    update: UpdateParams        # R (8×8 diag)
    P0_diag: np.ndarray         # (11,) — initial covariance diagonal
    n_min: int = 2              # frame count < n_min → skip smoothing
    q_aligned: dict = None      # direction/class-aware process noise (optional)
    freeze_initial: bool = False  # True → backward does not smooth first frame (k=0) (keeps forward value)
    # For a static frame (T), inflate the T+1 prior cov (= information matrix of backward gain) to weaken the gain
    # → trust the forward (measurement/R) estimate more. Corrects the dilution of a static vehicle's large Q through backward.
    static_inflate_enabled: bool = False
    static_inflate_mult: float = 10.0
    static_inflate_vstatic: float = 0.25   # m/s — speed below this is considered static

    @classmethod
    def from_config(cls, cfg: dict) -> "RTSParams":
        rts = cfg.get("rts_smoothing", {}) or {}
        P0_diag = np.array([float(rts["P0"][n]) for n in STATE_NAMES],
                           dtype=np.float64)
        sci = (rts.get("static_cov_inflate") or {})
        return cls(
            predict=PredictParams.from_config(rts["Q"]),
            update=UpdateParams.from_config(rts["R"]),
            P0_diag=P0_diag,
            n_min=int(rts.get("n_min", 2)),
            q_aligned=rts.get("Q_aligned"),
            freeze_initial=bool(rts.get("freeze_initial", True)),
            static_inflate_enabled=bool(sci.get("enabled", True)),
            static_inflate_mult=float(sci.get("multiplier", 10.0)),
            static_inflate_vstatic=float(sci.get("v_static", 0.25)),
        )


@dataclass
class TrackSmoothInfo:
    uuid: str
    category: str
    n_frames: int
    status: str                            # smoothed | skip_short
    mean_innovation: Optional[float] = None  # mean ‖y_xy‖  (forward, k≥1)


# ── helpers ───────────────────────────────────────────────────────────────
def _wrap(a):
    """Wrap to [-π, π]."""
    return np.arctan2(np.sin(a), np.cos(a))



def _class_longlat(category, qa):
    """category → (q_long, q_lat) or None."""
    bc = (qa.get("by_class") or {})
    e = bc.get(category, qa.get("default"))
    if not e:
        return None
    return float(e["long"]), float(e["lat"])

def _yaw_z_to_quat(yaw: np.ndarray) -> np.ndarray:
    """yaw (z-axis only) → (qw, qx, qy, qz). qw ≥ 0 canonical."""
    half = yaw * 0.5
    out = np.stack(
        [np.cos(half), np.zeros_like(yaw), np.zeros_like(yaw), np.sin(half)],
        axis=-1,
    )
    neg = out[:, 0] < 0
    if neg.any():
        out[neg] *= -1.0
    return out


def _build_measurements(track: TrackState) -> np.ndarray:
    """TrackState → Z (N, 8) = [x, y, z, θ, l, w, h, s].

    yaw is unwrapped over time (prevents ±π jumps inside the filter).
    """
    N = len(track)
    Z = np.empty((N, MEAS_DIM), dtype=np.float64)
    Z[:, 0:3] = track.translations_m
    Z[:, 3]   = np.unwrap(track.yaws_rad.astype(np.float64))
    Z[:, 4:7] = track.sizes_m
    Z[:, 7]   = track.scores
    return Z


def _initial_state(Z: np.ndarray, dt0: float) -> np.ndarray:
    """Build x_0 from the first two measurements. velocity via finite difference."""
    x0 = np.zeros(STATE_DIM, dtype=np.float64)
    x0[:MEAS_DIM] = Z[0]
    if Z.shape[0] >= 2 and dt0 > 0:
        x0[IDX_VX] = (Z[1, 0] - Z[0, 0]) / dt0
        x0[IDX_VY] = (Z[1, 1] - Z[0, 1]) / dt0
        x0[IDX_VZ] = (Z[1, 2] - Z[0, 2]) / dt0
    return x0


# ── [3] Forward pass ──────────────────────────────────────────────────────
def _forward_pass(Z: np.ndarray,
                  dts: np.ndarray,
                  x0: np.ndarray,
                  P0: np.ndarray,
                  params: RTSParams,
                  Q_steps=None):
    """Kalman filter forward pass — store prior/posterior of every step.

    k = 0     : no predict (initial = prior). Start with update(Z[0]).
                F_all[0] is a placeholder I since it is not used in RTS.
    k = 1..N-1: predict(Δt_k = dts[k-1]) → update(Z[k])

    Returns
    -------
    x_pred (N,11)  P_pred (N,11,11)
    x_post (N,11)  P_post (N,11,11)
    F      (N,11,11)
    innov_xy : list[float] — √(y_x² + y_y²) for k ≥ 1
    """
    N = Z.shape[0]
    x_pred = np.zeros((N, STATE_DIM))
    P_pred = np.zeros((N, STATE_DIM, STATE_DIM))
    x_post = np.zeros((N, STATE_DIM))
    P_post = np.zeros((N, STATE_DIM, STATE_DIM))
    F_all  = np.zeros((N, STATE_DIM, STATE_DIM))

    # k = 0
    F_all[0] = np.eye(STATE_DIM)
    x_pred[0] = x0
    P_pred[0] = P0
    x_k, P_k, _y, _S = update(x0, P0, Z[0], params.update)
    x_post[0] = x_k
    P_post[0] = P_k

    innov_xy: list[float] = []

    # k = 1..N-1
    for k in range(1, N):
        dt = float(dts[k - 1])
        x_p, P_p, F_k = predict(x_k, P_k, dt, params.predict,
                                Q_override=(None if Q_steps is None else Q_steps[k]))
        x_k, P_k, y, _S = update(x_p, P_p, Z[k], params.update)

        x_pred[k] = x_p
        P_pred[k] = P_p
        F_all[k]  = F_k
        x_post[k] = x_k
        P_post[k] = P_k
        innov_xy.append(float(np.hypot(y[0], y[1])))

    return x_pred, P_pred, x_post, P_post, F_all, innov_xy


# ── [4] Backward pass ─────────────────────────────────────────────────────
def _backward_pass(x_pred: np.ndarray,
                   P_pred: np.ndarray,
                   x_post: np.ndarray,
                   P_post: np.ndarray,
                   F_all: np.ndarray,
                   freeze_initial: bool = False,
                   static_mask: np.ndarray = None,
                   static_mult: float = 10.0):
    """RTS smoother backward pass.

    boundary: x_{N-1|N} = x_{N-1|N-1}, P_{N-1|N} = P_{N-1|N-1}.
    Smoothing in reverse order k = N-2..0.

    freeze_initial=True → the first frame (k=0) is not smoothed and keeps its
    forward value (x_post[0]). This prevents a large initial P0 from making the
    backward gain C_0 too large and dragging the initial box toward the next
    measurement (symmetric with anchoring the last frame).

    static_mask[k]=True → frame k(T) is static → inflate P_pred[k+1] (T+1 prior,
    information matrix) in the gain computation by static_mult to weaken the
    backward gain (= more trust in forward/R). Corrects the dilution of a static
    vehicle's large Q through backward. (the cov update's dP uses the original prior)
    """
    N = x_pred.shape[0]
    x_sm = np.zeros_like(x_post)
    P_sm = np.zeros_like(P_post)
    x_sm[-1] = x_post[-1]
    P_sm[-1] = P_post[-1]

    k_stop = 0 if freeze_initial else -1   # when freeze, k=0 not smoothed (loop stops at 1)
    for k in range(N - 2, k_stop, -1):
        # C_k = P_{k|k} · F_{k+1}^T · P_{k+1|k}^{-1}
        # Information matirx : P_{k+1|k}^{-1}
        # stabilization: solve with transpose of P_{k+1|k}, then restore transpose
        Pkk_FT = P_post[k] @ F_all[k + 1].T
        # static frame → inflate T+1 prior cov (information matrix) to weaken gain
        P_pred_k1 = P_pred[k + 1]
        if static_mask is not None and bool(static_mask[k]):
            P_pred_k1 = static_mult * P_pred_k1
        try:
            C = np.linalg.solve(P_pred_k1.T, Pkk_FT.T).T
        except np.linalg.LinAlgError:
            C = Pkk_FT @ np.linalg.pinv(P_pred_k1)

        dx = x_sm[k + 1] - x_pred[k + 1]
        dx[IDX_THETA] = _wrap(dx[IDX_THETA])

        x_sm[k] = x_post[k] + C @ dx
        x_sm[k, IDX_THETA] = _wrap(x_sm[k, IDX_THETA])

        dP = P_sm[k + 1] - P_pred[k + 1]
        P_sm[k] = P_post[k] + C @ dP @ C.T
        P_sm[k] = 0.5 * (P_sm[k] + P_sm[k].T)   # symmetrize

    if freeze_initial and N >= 1:
        # first frame keeps its forward value (loop skips k=0)
        x_sm[0] = x_post[0]
        P_sm[0] = P_post[0]

    return x_sm, P_sm


# ── [5] state matrix → TrackState ─────────────────────────────────────────
def _to_track_state(track: TrackState, x_sm: np.ndarray) -> TrackState:
    """smoothed state (N, 11) → new TrackState.

    score is clipped to [0, 1], l/w/h floored at 1e-3 m (guarantees positivity),
    quaternion re-synthesized from yaw (z-axis only).
    """
    yaw = _wrap(x_sm[:, IDX_THETA]).astype(np.float64)
    lwh = np.maximum(x_sm[:, [IDX_L, IDX_W, IDX_H]], 1e-3).astype(np.float32)
    score = np.clip(x_sm[:, IDX_S], 0.0, 1.0).astype(np.float32)
    return TrackState(
        uuid=track.uuid,
        category=track.category,
        timestamps_ns=track.timestamps_ns,
        translations_m=x_sm[:, [IDX_X, IDX_Y, IDX_Z]].astype(np.float64),
        yaws_rad=yaw,
        sizes_m=lwh,
        scores=score,
        quaternions=_yaw_z_to_quat(yaw),
    )


# ── core: single track ────────────────────────────────────────────────────
def smooth_track(track: TrackState,
                 params: RTSParams) -> tuple[TrackSmoothInfo, TrackState]:
    """Apply forward Kalman + backward RTS to a single track. Returns (info, new_track)."""
    N = len(track)
    if N < params.n_min:
        return (TrackSmoothInfo(uuid=track.uuid, category=track.category,
                                n_frames=N, status="skip_short"),
                track)

    # [1][2] measurement / dt / initial
    Z = _build_measurements(track)
    ts_s = track.timestamps_ns.astype(np.float64) * 1e-9
    dts = np.diff(ts_s) # 2Hz so 0.5s                                         # (N-1,)
    x0 = _initial_state(Z, float(dts[0]) if N >= 2 else 0.0)
    P0 = np.diag(params.P0_diag).copy()

    # [3] forward  (direction/class-aware Q option)
    Q_steps = None
    qa = params.q_aligned
    if qa and qa.get("enabled"):
        ll = _class_longlat(track.category, qa)
        if ll is not None:
            q_long, q_lat = ll
            baseQ = params.predict.Q
            Q_steps = np.repeat(baseQ[None, :, :], N, axis=0).copy()
            yaws = Z[:, IDX_THETA]
            D = np.diag([q_long, q_lat])
            for k in range(N):
                c, sn = np.cos(yaws[k]), np.sin(yaws[k])
                R2 = np.array([[c, -sn], [sn, c]])
                Q_steps[k, 0:2, 0:2] = R2 @ D @ R2.T
    x_pred, P_pred, x_post, P_post, F_all, innov_xy = _forward_pass(
        Z, dts, x0, P0, params, Q_steps=Q_steps,
    )

    # static frame mask (forward posterior speed ‖vx,vy‖ < v_static). 11D state: vx=8, vy=9.
    static_mask = None
    if params.static_inflate_enabled:
        spd = np.hypot(x_post[:, 8], x_post[:, 9])
        static_mask = spd < params.static_inflate_vstatic

    # [4] backward (freeze_initial → don't smooth first frame, static → inflate T+1 prior cov)
    x_sm, _P_sm = _backward_pass(x_pred, P_pred, x_post, P_post, F_all,
                                 freeze_initial=params.freeze_initial,
                                 static_mask=static_mask,
                                 static_mult=params.static_inflate_mult)

    # [5] state → TrackState
    new_track = _to_track_state(track, x_sm)

    info = TrackSmoothInfo(
        uuid=track.uuid, category=track.category, n_frames=N, status="smoothed",
        mean_innovation=(float(np.mean(innov_xy)) if innov_xy else None),
    )
    return info, new_track


# ── core: all tracks in a Tracks ──────────────────────────────────────────
def smooth_all(tracks, params: RTSParams):
    """Apply RTS smoothing to all Tracks. Returns (new_tracks_dict, summary).

    Returns only (dict[uuid → TrackState], summary) instead of a Tracks object —
    the caller re-packages the ego pose, etc. (same pattern as apply_yaw_correction).
    """
    new_map: dict[str, TrackState] = {}
    infos: list[TrackSmoothInfo] = []
    by_status: dict[str, int] = {}

    for uuid, tr in tracks.tracks.items():
        info, new_tr = smooth_track(tr, params)
        new_map[uuid] = new_tr
        infos.append(info)
        by_status[info.status] = by_status.get(info.status, 0) + 1

    summary = {
        "total": len(infos),
        "by_status": by_status,
        "tracks": [
            {"uuid": i.uuid, "category": i.category,
             "n_frames": i.n_frames, "status": i.status,
             "mean_innovation": i.mean_innovation}
            for i in infos
        ],
    }
    return new_map, summary
