"""Kalman Filter — 4D CV state (x, y, vx, vy), 2D measurement (x, y).

State transition F(Δt) (4×4):
    F = [[1, 0, Δt, 0 ],
         [0, 1, 0,  Δt],
         [0, 0, 1,  0 ],
         [0, 0, 0,  1 ]]

Measurement matrix H (2×4):
    H = [[1, 0, 0, 0],
         [0, 1, 0, 0]]

Process noise Q (4×4)
    base = diag(σ_x², σ_y², σ_vx², σ_vy²)
    For dynamic objects (‖v‖ ≥ v_static), the xy 2×2 block is rotated and scaled
    along the **velocity heading direction**
    (`max(‖v‖·scale, longitudinal_min_factor)` × σ_x², σ_y² × 1).
    Same trick as C++ multi_class_object_tracking.

Measurement noise R (2×2)
    diag(σ_x², σ_y²). If detection_confidence is low, scaled by conf_low_R_scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .track import STATE_DIM, MEAS_DIM, S_X, S_Y, S_VX, S_VY


# ── parameters ──────────────────────────────────────────────────────
@dataclass
class KFParams:
    Q_diag: np.ndarray              # (4,)
    R_diag: np.ndarray              # (2,)
    P0_diag: np.ndarray             # (4,)
    skew_enabled: bool
    skew_lon_pose_factor: float     # pos block longitudinal-axis factor (constant)
    skew_lon_velocity_factor: float # vel block longitudinal-axis factor (constant)
    v_static: float                 # threshold to force velocity to 0 (m/s)
    conf_low_threshold: float       # default 0.5
    conf_low_R_scale: float         # default 10.0
    # ── Measurement noise R is also heading-aligned anisotropic — based on measurement.yaw_world.
    r_skew_enabled: bool
    r_lon_factor: float             # R longitudinal-axis factor (constant). 1.0 means isotropic.
    # ── Init covariance — yaw-aligned anisotropy
    # At init, v=0 so direction_skew (velocity-based) cannot be used. Instead the detection's yaw
    # is used as a prior to stretch the xy / velocity blocks of P0 along the heading direction.
    # Mahalanobis cost becomes direction-sensitive from the first frame → blocks swaps.
    init_cov_yaw_enabled: bool
    init_cov_long_factor: float     # longitudinal-axis variance = P0_diag · long_factor
    init_cov_lat_factor: float      # lateral-axis variance = P0_diag · lat_factor

    @classmethod
    def from_config(cls, t_cfg: dict) -> "KFParams":
        Q = t_cfg["Q"]
        R = t_cfg["R"]
        P0 = t_cfg["P0"]
        skew = t_cfg.get("direction_skew", {})
        init_cov = t_cfg.get("init_cov_yaw_aligned", {}) or {}
        return cls(
            Q_diag=np.array([Q["x"], Q["y"], Q["vx"], Q["vy"]], dtype=np.float64),
            R_diag=np.array([R["x"], R["y"]], dtype=np.float64),
            P0_diag=np.array([P0["x"], P0["y"], P0["vx"], P0["vy"]], dtype=np.float64),
            skew_enabled=bool(skew.get("enabled", True)),
            skew_lon_pose_factor=float(skew.get("lon_pose_factor", 5.0)),
            skew_lon_velocity_factor=float(skew.get("lon_velocity_factor", 5.0)),
            v_static=float(t_cfg.get("v_static", 0.25)),
            conf_low_threshold=float(t_cfg.get("conf_low_threshold", 0.5)),
            conf_low_R_scale=float(t_cfg.get("conf_low_R_scale", 10.0)),
            r_skew_enabled=bool((t_cfg.get("measurement_skew") or {}).get("enabled", False)),
            r_lon_factor=float((t_cfg.get("measurement_skew") or {}).get("lon_factor", 1.0)),
            init_cov_yaw_enabled=bool(init_cov.get("enabled", False)),
            init_cov_long_factor=float(init_cov.get("long_factor", 5.0)),
            init_cov_lat_factor=float(init_cov.get("lat_factor", 0.2)),
        )


# ── matrix builders ─────────────────────────────────────────────────
def build_F(dt: float) -> np.ndarray:
    """4×4 CV transition matrix."""
    F = np.eye(STATE_DIM, dtype=np.float64)
    F[S_X, S_VX] = dt
    F[S_Y, S_VY] = dt
    return F


def build_H() -> np.ndarray:
    """2×4 measurement matrix — observes xy only."""
    H = np.zeros((MEAS_DIM, STATE_DIM), dtype=np.float64)
    H[0, S_X] = 1.0
    H[1, S_Y] = 1.0
    return H


H_MATRIX = build_H()


def build_P0(yaw_world: float | None, params: KFParams,
             is_static: bool = False) -> np.ndarray:
    """Build initial covariance P0 (4×4).

    If init_cov_yaw_enabled is False, yaw_world is None, or is_static (fixed class),
    use isotropic diag(P0_diag). Fixed objects (BOLLARD/CONE/BARREL/SIGN etc.) have no
    meaningful heading, so there is no reason to inflate the longitudinal variance; an
    isotropic P0 is used.

    When Enabled (and non-static), it is anisotropic along the detection's yaw direction:
      · longitudinal-axis (yaw direction) variance = P0_diag · long_factor
      · lateral-axis (perpendicular to yaw) variance = P0_diag · lat_factor
    Both the Position (xy) block and the Velocity (vx, vy) block are rotated.
    The cross-block (xy ↔ vxvy) is left at 0.

    This anisotropic P0 provides first-frame direction sensitivity for the Mahalanobis cost:
      · measurement along θ → low cost (absorbed by the large longitudinal variance)
      · measurement perpendicular to θ → high cost (small lateral variance)
    """
    P0 = np.diag(params.P0_diag).copy()

    if not params.init_cov_yaw_enabled or yaw_world is None or is_static:
        return P0   # static: skip longitudinal anisotropy → isotropic

    c, s = float(np.cos(yaw_world)), float(np.sin(yaw_world))
    R = np.array([[c, -s], [s, c]], dtype=np.float64)

    long_f = params.init_cov_long_factor
    lat_f = params.init_cov_lat_factor

    # Position block — based on P0_diag[S_X, S_Y]
    pos_aligned = np.diag([
        params.P0_diag[S_X] * long_f,
        params.P0_diag[S_Y] * lat_f,
    ])
    P0[:2, :2] = R @ pos_aligned @ R.T

    # Velocity block — based on P0_diag[S_VX, S_VY]
    vel_aligned = np.diag([
        params.P0_diag[S_VX] * long_f,
        params.P0_diag[S_VY] * lat_f,
    ])
    P0[2:4, 2:4] = R @ vel_aligned @ R.T

    return P0


def build_Q(state: np.ndarray, params: KFParams,
            yaw_hint: float | None = None) -> np.ndarray:
    """Q (4×4). Applies anisotropic skew when heading information is available.

    Direction decision priority:
      1. ‖v‖ ≥ v_static  → KF velocity direction
      2. yaw_hint present → measured yaw direction (last_yaw)
      3. neither         → isotropic Q as-is

    The Position xy block is grown along the longitudinal axis by lon_pose_factor, and the
    Velocity vxvy block by lon_velocity_factor. The lateral-axis variance stays as-is (Q_diag[lat]).
    """
    Q = np.diag(params.Q_diag).copy()

    if not params.skew_enabled:
        return Q

    vx, vy = float(state[S_VX]), float(state[S_VY])
    speed = float(np.hypot(vx, vy))

    if speed >= params.v_static:
        angle = float(np.arctan2(vy, vx))
    elif yaw_hint is not None:
        angle = float(yaw_hint)
    else:
        return Q   # direction unknown → isotropic

    c, s = float(np.cos(angle)), float(np.sin(angle))
    R = np.array([[c, -s], [s, c]], dtype=np.float64)

    # Position xy block — lon_pose_factor times larger along the longitudinal axis
    Q_xy_aligned = np.diag([params.Q_diag[S_X] * params.skew_lon_pose_factor,
                            params.Q_diag[S_Y]])
    Q[:2, :2] = R @ Q_xy_aligned @ R.T

    # Velocity vxvy block — lon_velocity_factor times larger along the longitudinal axis
    Q_v_aligned = np.diag([params.Q_diag[S_VX] * params.skew_lon_velocity_factor,
                           params.Q_diag[S_VY]])
    Q[2:4, 2:4] = R @ Q_v_aligned @ R.T

    return Q


def build_R(params: KFParams, detection_confidence: float = 1.0,
            yaw_hint: float | None = None) -> np.ndarray:
    """2×2 measurement noise.

    - for a low-confidence detection, scaled by conf_low_R_scale (isotropic scale up)
    - if r_skew_enabled and yaw_hint is present, rotated anisotropically along the longitudinal axis:
        longitudinal-axis variance = R_diag · r_lon_factor, lateral-axis variance = R_diag (as-is)
      → the vehicle's longitudinal (length-axis) position noise is larger — the Mahalanobis
        cost becomes lenient toward measurements in the heading direction.
    """
    R = np.diag(params.R_diag).copy()
    if detection_confidence < params.conf_low_threshold:
        R = R * params.conf_low_R_scale

    if params.r_skew_enabled and yaw_hint is not None and params.r_lon_factor != 1.0:
        c, s = float(np.cos(yaw_hint)), float(np.sin(yaw_hint))
        Rot = np.array([[c, -s], [s, c]], dtype=np.float64)
        # R_aligned: longitudinal axis = R_diag[0] · lon_factor, lateral axis = R_diag[1] (as-is)
        # the low-conf scale effect is already applied to R above → use R[0,0], R[1,1]
        R_aligned = np.diag([R[0, 0] * params.r_lon_factor, R[1, 1]])
        R = Rot @ R_aligned @ Rot.T
    return R


# ── Predict / Update ────────────────────────────────────────────────
def predict(state: np.ndarray, cov: np.ndarray, dt: float,
            params: KFParams,
            yaw_hint: float | None = None
            ) -> tuple[np.ndarray, np.ndarray]:
    """KF predict step. Returns: (x_pred, P_pred).

    yaw_hint: for static / just-initialized tracks — the last measurement yaw (world frame).
    When velocity is too small for the KF to determine direction, build_Q rotates using this.
    """
    F = build_F(dt)
    Q = build_Q(state, params, yaw_hint=yaw_hint)
    x_pred = F @ state
    P_pred = F @ cov @ F.T + Q
    P_pred = 0.5 * (P_pred + P_pred.T)
    return x_pred, P_pred


def update(state: np.ndarray, cov: np.ndarray, z: np.ndarray,
           params: KFParams, detection_confidence: float = 1.0,
           yaw_hint: float | None = None,
           ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """KF update step.

    Parameters
    ----------
    state : (4,)   x_{k|k-1}
    cov   : (4,4) P_{k|k-1}
    z     : (2,)   measurement [x, y]
    params, detection_confidence
    yaw_hint : float — the measurement's world-frame yaw (used for R skew).

    Returns
    -------
    x_post : (4,)   x_{k|k}
    P_post : (4,4) P_{k|k}
    y      : (2,)   innovation
    S      : (2,2) innovation covariance
    """
    H = H_MATRIX
    R = build_R(params, detection_confidence, yaw_hint=yaw_hint)

    y = z - H @ state
    S = H @ cov @ H.T + R
    # K = P·H^T·S^{-1} → solved safely via solve
    K = np.linalg.solve(S.T, (cov @ H.T).T).T   # (4, 2)

    x_post = state + K @ y

    I = np.eye(STATE_DIM, dtype=np.float64)
    A = I - K @ H
    P_post = A @ cov @ A.T + K @ R @ K.T
    P_post = 0.5 * (P_post + P_post.T)

    return x_post, P_post, y, S
