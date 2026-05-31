"""4 motion models for IMM — CV / CA / CTRV / CTRA.

Each model provides conversion between its native state space and the standard
8D state space. IMM mixing is performed in the standard form, while predict/update
is performed in the native space.

standard state (8D): [x, y, vx, vy, ax, ay, yaw, yaw_rate]

native state:
  CV   (4D): [x, y, vx, vy]
  CA   (6D): [x, y, vx, vy, ax, ay]
  CTRV (5D): [x, y, v, yaw, yaw_rate]
  CTRA (6D): [x, y, v, a, yaw, yaw_rate]

measurement (3D): [x, y, yaw]
  CV/CA   has a 2-row H (xy only); the yaw measurement is ignored (no yaw in state).
  CTRV/CTRA has a 3-row H (xy + yaw).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ── standard-form constants ────────────────────────────────────────────────────────
STD_DIM = 8
STD_X, STD_Y, STD_VX, STD_VY, STD_AX, STD_AY, STD_YAW, STD_YR = 0, 1, 2, 3, 4, 5, 6, 7

# unmodeled diag value for expand_matrix (same as IMM-MOT). Too large diverges after
# mixing, too small buries the signal of other models — kept at 1000.0 as IMM-MOT uses.
_UNMODELED_DIAG = 1000.0


def _wrap(a: float) -> float:
    return float(np.arctan2(np.sin(a), np.cos(a)))


def _wrap_arr(a: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(a), np.cos(a))


def _expand_matrix(P: np.ndarray, dim: int, missing_idx: list[int],
                   diag_value: float = _UNMODELED_DIAG) -> np.ndarray:
    """Expand native P to the standard dim. missing_idx are the dimensions this model
    does not estimate in the standard form — those slots are filled with diag_value to
    mark large uncertainty."""
    out = np.zeros((dim, dim), dtype=np.float64)
    keep = [i for i in range(dim) if i not in missing_idx]
    # copy native P into the keep positions — ValueError if sizes mismatch
    out[np.ix_(keep, keep)] = P
    for i in missing_idx:
        out[i, i] = diag_value
    return out


# ── base ────────────────────────────────────────────────────────────
@dataclass
class MotionModelParams:
    """σ² of Q/R (diagonal). Only the base scale differs per model and can be overridden externally."""
    q_xy: float = 0.07      # position process noise (same level as Q.x in multi_class_tracking)
    q_v:  float = 0.35      # velocity process noise
    q_a:  float = 0.5       # acceleration process noise (CA/CTRA only)
    q_yaw: float = 0.05     # yaw process noise (CTRV/CTRA only)
    q_yaw_rate: float = 0.1 # yaw_rate process noise
    r_xy: float = 0.15      # measurement noise xy
    r_yaw: float = 0.1      # measurement noise yaw (CTRV/CTRA only)
    p0_xy: float = 1.0
    p0_v:  float = 100.0
    p0_a:  float = 100.0
    p0_yaw: float = 1.0
    p0_yaw_rate: float = 10.0
    # ── direction skew (make Q's pos/vel block anisotropic along the state yaw direction) ──
    # Same trick as kalman.build_Q in the CV-only KF. Shared by all sub-filters (config tracking.direction_skew).
    skew_enabled: bool = False
    skew_lon_pose_factor: float = 1.0      # pos longitudinal-axis factor
    skew_lon_velocity_factor: float = 1.0  # vel longitudinal-axis factor (CV/CA only — cartesian vx/vy)
    # ── init covariance yaw-aligned anisotropy (P0's pos/vel block) ──
    # Same as kalman.build_P0. longitudinal axis = base·long_factor, lateral axis = base·lat_factor.
    init_cov_yaw_enabled: bool = False
    init_cov_long_factor: float = 1.0
    init_cov_lat_factor: float = 1.0
    # ── low-confidence measurement R scale ──
    # If detection_confidence < conf_low_threshold, scale R by conf_low_R_scale (same as CV KF).
    conf_low_threshold: float = 0.0
    conf_low_R_scale: float = 1.0
    # ── distrust yaw measurement of static objects — when static, scale R's yaw component ──
    # For static objects the detection yaw is ill-defined / heavily jittery, so following the
    # measurement makes yaw wobble. When static, increase r_yaw to reduce the effect of the
    # measured yaw and hold yaw (the counterpart of the CV-only static yaw stabilization).
    static_r_yaw_scale: float = 1.0
    # ── when static, CTRV/CTRA yaw-related noise override (suppress momentum spin after accel→stop) ──
    # If judged stopped (‖v‖<v_static), use the values below instead of q_yaw/q_yaw_rate/r_yaw.
    static_q_yaw: float = 0.05
    static_q_yaw_rate: float = 0.1
    static_r_yaw: float = 0.01


class BaseMotionModel(abc.ABC):
    """Common interface for all sub-filters.

    Internally holds native state + cov. predict/update run in native space.
    IMM mixing round-trips to/from the standard 8D form via to_standard / from_standard.
    """

    SD: int = -1
    name: str = "BASE"
    # indices of the dimensions this model does not estimate in the standard 8D form
    MISSING_STD_IDX: tuple[int, ...] = ()
    # ── native state indices for direction skew ──
    # POS_IDX : cartesian (x, y) indices (same for all models).
    # VEL_IDX : cartesian (vx, vy) indices — CV/CA only. Polar-v models (CTRV/CTRA) use an empty tuple.
    # YAW_STATE_IDX : yaw index in the native state (-1 disables skew).
    POS_IDX: tuple[int, int] = (0, 1)
    VEL_IDX: tuple[int, ...] = ()
    YAW_STATE_IDX: int = -1

    def __init__(self, state: np.ndarray, cov: np.ndarray,
                 params: MotionModelParams):
        assert state.shape == (self.SD,), f"{self.name}: state shape {state.shape} != ({self.SD},)"
        assert cov.shape == (self.SD, self.SD)
        self.state = state.astype(np.float64).copy()
        self.cov = cov.astype(np.float64).copy()
        self.p = params
        self.last_likelihood: float = 1.0
        self.last_innovation: Optional[np.ndarray] = None
        # static mode — refreshed every frame by update(static=). CTRV/CTRA's Q()/R()
        # read this value and apply the static-only yaw noise (static_q_yaw/q_yaw_rate/r_yaw).
        self.static_mode: bool = False
        # for RTS smoothing — last predict's transition matrix/dt (identity right after init).
        self.last_F: np.ndarray = np.eye(self.SD, dtype=np.float64)
        self.last_dt: float = 0.0

    # ── yaw-related noise when static (used only by CTRV/CTRA) ──
    def _eff_q_yaw(self) -> float:
        return self.p.static_q_yaw if self.static_mode else self.p.q_yaw

    def _eff_q_yaw_rate(self) -> float:
        return self.p.static_q_yaw_rate if self.static_mode else self.p.q_yaw_rate

    def _eff_r_yaw(self) -> float:
        return self.p.static_r_yaw if self.static_mode else self.p.r_yaw

    # ── F, Q, H, R ──
    @abc.abstractmethod
    def F(self, dt: float) -> np.ndarray: ...
    @abc.abstractmethod
    def Q(self, dt: float) -> np.ndarray: ...
    @abc.abstractmethod
    def H(self) -> np.ndarray: ...
    @abc.abstractmethod
    def R(self) -> np.ndarray: ...

    # ── direction skew ──
    def _skew_Q(self, Q: np.ndarray) -> np.ndarray:
        """Anisotropically rotate/scale Q's position (always) and velocity (CV/CA) blocks
        along the heading axis (state yaw). Only the longitudinal-axis variance is scaled by
        factor, the lateral axis is unchanged. Same trick as kalman.build_Q in the CV-only KF.
        Called by each model's Q(dt)."""
        if not self.p.skew_enabled or self.YAW_STATE_IDX < 0:
            return Q
        yaw = float(self.state[self.YAW_STATE_IDX])
        c, s = float(np.cos(yaw)), float(np.sin(yaw))
        R = np.array([[c, -s], [s, c]], dtype=np.float64)
        i, j = self.POS_IDX
        pos = np.diag([Q[i, i] * self.p.skew_lon_pose_factor, Q[j, j]])
        Q[np.ix_([i, j], [i, j])] = R @ pos @ R.T
        if self.VEL_IDX:
            a, b = self.VEL_IDX
            vel = np.diag([Q[a, a] * self.p.skew_lon_velocity_factor, Q[b, b]])
            Q[np.ix_([a, b], [a, b])] = R @ vel @ R.T
        return Q

    # ── state transition (override if non-linear) ──
    def state_transition(self, dt: float) -> np.ndarray:
        return self.F(dt) @ self.state

    # ── measurement projection (override if non-linear) ──
    def state_to_measurement(self) -> np.ndarray:
        return self.H() @ self.state

    # ── predict / update ──
    def predict(self, dt: float) -> None:
        F = self.F(dt)
        Q = self.Q(dt)
        self.state = self.state_transition(dt)
        self.cov = F @ self.cov @ F.T + Q
        self.cov = 0.5 * (self.cov + self.cov.T)
        self._wrap_state_yaw()
        # for the RTS smoothing backward gain — this step's transition matrix (EKF: Jacobian at predict time).
        # Reused as F_{k+1} in the smoother's C_k = P_{k|k}·F_{k+1}^T·P_{k+1|k}^{-1}.
        self.last_F = F
        self.last_dt = float(dt)

    def update(self, z: np.ndarray, detection_confidence: float = 1.0,
               static: bool = False) -> None:
        """z = (x, y) or (x, y, yaw) — the model slices it with H.

        If detection_confidence < conf_low_threshold, scale R by conf_low_R_scale to reduce
        the effect of low-confidence measurements (same as build_R in the CV-only KF).
        If static=True, scale R's yaw component (last row) by static_r_yaw_scale to suppress
        following the measured yaw jitter of static objects (yaw hold)."""
        # refresh static mode — this update's R() and the next frame's predict Q() use the static-only values.
        self.static_mode = bool(static)
        H = self.H()
        R = self.R()
        if detection_confidence < self.p.conf_low_threshold:
            R = R * self.p.conf_low_R_scale
        if static and self.p.static_r_yaw_scale != 1.0 and H.shape[0] == 3:
            R = R.copy()
            R[2, 2] = R[2, 2] * self.p.static_r_yaw_scale   # distrust the yaw measurement only
        # use only the dimensions each model needs (CV/CA use z[:2])
        z_used = z[: H.shape[0]].astype(np.float64)

        h_x = self.state_to_measurement()
        y = z_used - h_x
        # if a yaw measurement is included, wrap the last row
        if H.shape[0] == 3:
            y[2] = _wrap(y[2])

        S = H @ self.cov @ H.T + R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            self.last_likelihood = 1e-300
            return
        K = self.cov @ H.T @ S_inv
        self.state = self.state + K @ y
        I = np.eye(self.SD, dtype=np.float64)
        A = I - K @ H
        self.cov = A @ self.cov @ A.T + K @ R @ K.T
        self.cov = 0.5 * (self.cov + self.cov.T)
        self._wrap_state_yaw()

        # likelihood — multivariate gaussian density
        try:
            det = np.linalg.det(S)
            if det <= 0:
                self.last_likelihood = 1e-300
            else:
                d = y.shape[0]
                exponent = -0.5 * float(y @ S_inv @ y)
                norm = 1.0 / np.sqrt((2 * np.pi) ** d * det)
                self.last_likelihood = max(float(norm * np.exp(exponent)), 1e-300)
        except np.linalg.LinAlgError:
            self.last_likelihood = 1e-300
        self.last_innovation = y

    # ── standard form round-trip ──
    @abc.abstractmethod
    def to_standard(self) -> tuple[np.ndarray, np.ndarray]:
        """native state, cov → (state_std[8], cov_std[8,8])."""
    @abc.abstractmethod
    def set_from_standard(self, state_std: np.ndarray, cov_std: np.ndarray) -> None:
        """Convert standard 8D state, cov → native state, cov and set in place."""

    # ── yaw wrap (override only in models that carry yaw in their state) ──
    def _wrap_state_yaw(self) -> None:
        pass


# ── CV ────────────────────────────────────────────────────────────────
class CVModel(BaseMotionModel):
    """Constant Velocity + constant yaw. native = [x, y, vx, vy, yaw] (5D).

    Like IMM-MOT's LinearKalmanFilter(CV), yaw is included in the state — during IMM
    mixing it is weighted-averaged in the same dimension as the yaw of other sub-filters
    (CTRV/CTRA), preventing the death-spiral where yaw is dragged toward 0. The CV assumption
    is that yaw itself stays constant over time (F's yaw row = identity). When a yaw
    measurement (H's 3rd row) arrives, it is used for the update.
    """
    SD = 5
    name = "CV"
    MISSING_STD_IDX = (STD_AX, STD_AY, STD_YR)
    POS_IDX = (0, 1)
    VEL_IDX = (2, 3)
    YAW_STATE_IDX = 4

    def F(self, dt: float) -> np.ndarray:
        F = np.eye(5, dtype=np.float64)
        F[0, 2] = dt
        F[1, 3] = dt
        # F[4, 4] = 1.0 (already identity) — yaw constant
        return F

    def Q(self, dt: float) -> np.ndarray:
        return self._skew_Q(np.diag([self.p.q_xy, self.p.q_xy,
                        self.p.q_v, self.p.q_v,
                        self.p.q_yaw]).astype(np.float64))

    def H(self) -> np.ndarray:
        H = np.zeros((3, 5), dtype=np.float64)
        H[0, 0] = 1.0; H[1, 1] = 1.0; H[2, 4] = 1.0
        return H

    def R(self) -> np.ndarray:
        return np.diag([self.p.r_xy, self.p.r_xy, self.p.r_yaw]).astype(np.float64)

    def state_to_measurement(self) -> np.ndarray:
        x, y, _, _, yaw = self.state.tolist()
        return np.array([x, y, yaw], dtype=np.float64)

    def to_standard(self) -> tuple[np.ndarray, np.ndarray]:
        s = np.zeros(STD_DIM, dtype=np.float64)
        s[STD_X] = self.state[0]; s[STD_Y] = self.state[1]
        s[STD_VX] = self.state[2]; s[STD_VY] = self.state[3]
        s[STD_YAW] = self.state[4]
        P_std = _expand_matrix(self.cov, STD_DIM, list(self.MISSING_STD_IDX))
        return s, P_std

    def set_from_standard(self, state_std: np.ndarray, cov_std: np.ndarray) -> None:
        # extract the (x, y, vx, vy, yaw) block from standard
        keep = [STD_X, STD_Y, STD_VX, STD_VY, STD_YAW]
        self.state = state_std[keep].copy()
        self.state[4] = _wrap(float(self.state[4]))
        self.cov = cov_std[np.ix_(keep, keep)].copy()

    def _wrap_state_yaw(self) -> None:
        self.state[4] = _wrap(float(self.state[4]))


# ── CA ────────────────────────────────────────────────────────────────
class CAModel(BaseMotionModel):
    """Constant Acceleration + constant yaw. native = [x, y, vx, vy, ax, ay, yaw] (7D).

    Same motivation as CV: during IMM mixing it must fill its own value into the standard
    form yaw dimension so that CTRV/CTRA's yaw is not dragged down in the weighted average.
    """
    SD = 7
    name = "CA"
    MISSING_STD_IDX = (STD_YR,)
    POS_IDX = (0, 1)
    VEL_IDX = (2, 3)
    YAW_STATE_IDX = 6

    def F(self, dt: float) -> np.ndarray:
        F = np.eye(7, dtype=np.float64)
        F[0, 2] = dt; F[0, 4] = 0.5 * dt * dt
        F[1, 3] = dt; F[1, 5] = 0.5 * dt * dt
        F[2, 4] = dt
        F[3, 5] = dt
        # F[6, 6] = 1.0 (already identity) — yaw constant
        return F

    def Q(self, dt: float) -> np.ndarray:
        return self._skew_Q(np.diag([self.p.q_xy, self.p.q_xy,
                        self.p.q_v, self.p.q_v,
                        self.p.q_a, self.p.q_a,
                        self.p.q_yaw]).astype(np.float64))

    def H(self) -> np.ndarray:
        H = np.zeros((3, 7), dtype=np.float64)
        H[0, 0] = 1.0; H[1, 1] = 1.0; H[2, 6] = 1.0
        return H

    def R(self) -> np.ndarray:
        return np.diag([self.p.r_xy, self.p.r_xy, self.p.r_yaw]).astype(np.float64)

    def state_to_measurement(self) -> np.ndarray:
        x, y, _, _, _, _, yaw = self.state.tolist()
        return np.array([x, y, yaw], dtype=np.float64)

    def to_standard(self) -> tuple[np.ndarray, np.ndarray]:
        s = np.zeros(STD_DIM, dtype=np.float64)
        s[STD_X] = self.state[0]; s[STD_Y] = self.state[1]
        s[STD_VX] = self.state[2]; s[STD_VY] = self.state[3]
        s[STD_AX] = self.state[4]; s[STD_AY] = self.state[5]
        s[STD_YAW] = self.state[6]
        P_std = _expand_matrix(self.cov, STD_DIM, list(self.MISSING_STD_IDX))
        return s, P_std

    def set_from_standard(self, state_std: np.ndarray, cov_std: np.ndarray) -> None:
        keep = [STD_X, STD_Y, STD_VX, STD_VY, STD_AX, STD_AY, STD_YAW]
        self.state = state_std[keep].copy()
        self.state[6] = _wrap(float(self.state[6]))
        self.cov = cov_std[np.ix_(keep, keep)].copy()

    def _wrap_state_yaw(self) -> None:
        self.state[6] = _wrap(float(self.state[6]))


# ── CTRV ──────────────────────────────────────────────────────────────
class CTRVModel(BaseMotionModel):
    """Constant Turn Rate & Velocity. native = [x, y, v, yaw, yaw_rate] (5D)."""
    SD = 5
    name = "CTRV"
    MISSING_STD_IDX = (STD_AX, STD_AY)
    POS_IDX = (0, 1)
    VEL_IDX = ()            # speed v is polar (yaw axis) — not subject to cartesian skew
    YAW_STATE_IDX = 3

    def state_transition(self, dt: float) -> np.ndarray:
        x, y, v, yaw, w = self.state.tolist()
        if abs(w) < 1e-3:
            # straight-line approximation (same as IMM-MOT motion_model.py:653-659) — avoids divide-by-zero.
            # 1st-order curvature term omitted (0th order) → aligned with the IMM-MOT reference.
            nx = x + v * np.cos(yaw) * dt
            ny = y + v * np.sin(yaw) * dt
            nyaw = yaw + w * dt
        else:
            nyaw = yaw + w * dt
            nx = x + (v / w) * (np.sin(nyaw) - np.sin(yaw))
            ny = y + (v / w) * (-np.cos(nyaw) + np.cos(yaw))
        return np.array([nx, ny, v, nyaw, w], dtype=np.float64)

    def F(self, dt: float) -> np.ndarray:
        """Exact Jacobian ∂f/∂x of state_transition.

        |w|≥1e-2 uses the rotation model's full Jacobian; |w|<1e-2 uses the exact Jacobian
        of the 1st-order Taylor branch (including the ∂pos/∂w curvature term). The straight-line
        approximation does not drop ∂pos/∂w.
        """
        x, y, v, yaw, w = self.state.tolist()
        F = np.eye(5, dtype=np.float64)
        c, s = np.cos(yaw), np.sin(yaw)
        if abs(w) < 1e-3:
            # 0th-order straight-line approximation — same as IMM-MOT motion_model.py:709-710.
            # ∂pos/∂yaw only, ∂pos/∂ω = 0 (curvature term omitted).
            F[0, 2] = dt * c
            F[0, 3] = -v * dt * s
            F[1, 2] = dt * s
            F[1, 3] = v * dt * c
        else:
            nyaw = yaw + w * dt
            cn, sn = np.cos(nyaw), np.sin(nyaw)
            F[0, 2] = (sn - s) / w
            F[0, 3] = (v / w) * (cn - c)
            F[0, 4] = (v * dt / w) * cn - (v / (w * w)) * (sn - s)
            F[1, 2] = (-cn + c) / w
            F[1, 3] = (v / w) * (sn - s)
            F[1, 4] = (v * dt / w) * sn - (v / (w * w)) * (-cn + c)
        F[3, 4] = dt
        return F

    def Q(self, dt: float) -> np.ndarray:
        return self._skew_Q(np.diag([self.p.q_xy, self.p.q_xy,
                        self.p.q_v,
                        self._eff_q_yaw(), self._eff_q_yaw_rate()]).astype(np.float64))

    def state_to_measurement(self) -> np.ndarray:
        x, y, _, yaw, _ = self.state.tolist()
        return np.array([x, y, yaw], dtype=np.float64)

    def H(self) -> np.ndarray:
        H = np.zeros((3, 5), dtype=np.float64)
        H[0, 0] = 1.0; H[1, 1] = 1.0; H[2, 3] = 1.0
        return H

    def R(self) -> np.ndarray:
        return np.diag([self.p.r_xy, self.p.r_xy, self._eff_r_yaw()]).astype(np.float64)

    def to_standard(self) -> tuple[np.ndarray, np.ndarray]:
        x, y, v, yaw, w = self.state.tolist()
        vx, vy = v * np.cos(yaw), v * np.sin(yaw)
        s = np.zeros(STD_DIM, dtype=np.float64)
        s[STD_X] = x; s[STD_Y] = y
        s[STD_VX] = vx; s[STD_VY] = vy
        s[STD_YAW] = yaw; s[STD_YR] = w
        # native [x, y, v, yaw, w] → standard [x, y, vx, vy, ax, ay, yaw, yaw_rate]
        # vx, vy are nonlinear functions of (v, yaw) — propagate cov via the Jacobian.
        # Approximation: distribute v's own cov equally onto both the vx/vy diagonals. Conservative but safe.
        P_std = np.zeros((STD_DIM, STD_DIM), dtype=np.float64)
        # direct keep-idx mapping
        # native idx: 0=x, 1=y, 2=v, 3=yaw, 4=w
        # standard idx for keep: x→0, y→1, v→(vx=2, vy=3) split, yaw→6, w→7
        P_std[STD_X, STD_X] = self.cov[0, 0]
        P_std[STD_Y, STD_Y] = self.cov[1, 1]
        P_std[STD_X, STD_Y] = self.cov[0, 1]; P_std[STD_Y, STD_X] = self.cov[1, 0]
        # v → vx, vy equal distribution
        P_std[STD_VX, STD_VX] = self.cov[2, 2]
        P_std[STD_VY, STD_VY] = self.cov[2, 2]
        P_std[STD_YAW, STD_YAW] = self.cov[3, 3]
        P_std[STD_YR, STD_YR] = self.cov[4, 4]
        P_std[STD_YAW, STD_YR] = self.cov[3, 4]; P_std[STD_YR, STD_YAW] = self.cov[4, 3]
        # unobserved dimensions (ax, ay) get large variance
        P_std[STD_AX, STD_AX] = _UNMODELED_DIAG
        P_std[STD_AY, STD_AY] = _UNMODELED_DIAG
        return s, P_std

    def set_from_standard(self, state_std: np.ndarray, cov_std: np.ndarray) -> None:
        x = state_std[STD_X]; y = state_std[STD_Y]
        vx = state_std[STD_VX]; vy = state_std[STD_VY]
        yaw = _wrap(float(state_std[STD_YAW]))
        w = float(state_std[STD_YR])
        v = float(np.hypot(vx, vy))
        self.state = np.array([x, y, v, yaw, w], dtype=np.float64)
        # cov — retrieve only the relevant blocks of the standard form. v cov takes the larger of vx/vy (conservative).
        P = np.zeros((5, 5), dtype=np.float64)
        P[0, 0] = cov_std[STD_X, STD_X]
        P[1, 1] = cov_std[STD_Y, STD_Y]
        P[0, 1] = cov_std[STD_X, STD_Y]; P[1, 0] = cov_std[STD_Y, STD_X]
        P[2, 2] = max(cov_std[STD_VX, STD_VX], cov_std[STD_VY, STD_VY])
        P[3, 3] = cov_std[STD_YAW, STD_YAW]
        P[4, 4] = cov_std[STD_YR, STD_YR]
        P[3, 4] = cov_std[STD_YAW, STD_YR]; P[4, 3] = cov_std[STD_YR, STD_YAW]
        self.cov = P

    def _wrap_state_yaw(self) -> None:
        self.state[3] = _wrap(float(self.state[3]))


# ── CTRA ──────────────────────────────────────────────────────────────
class CTRAModel(BaseMotionModel):
    """Constant Turn Rate & Acceleration. native = [x, y, v, a, yaw, yaw_rate] (6D)."""
    SD = 6
    name = "CTRA"
    MISSING_STD_IDX = ()   # can estimate all 8 standard dimensions (ax/ay = a·cos/sin(yaw))
    POS_IDX = (0, 1)
    VEL_IDX = ()            # speed v is polar (yaw axis) — not subject to cartesian skew
    YAW_STATE_IDX = 4

    def state_transition(self, dt: float) -> np.ndarray:
        x, y, v, a, yaw, w = self.state.tolist()
        nv = v + a * dt
        nyaw = yaw + w * dt
        if abs(w) < 1e-3:
            # straight-line approximation (same as IMM-MOT motion_model.py:429-435) — 1st-order curvature term omitted (0th order).
            disp = v * dt + 0.5 * a * dt * dt
            nx = x + disp * np.cos(yaw)
            ny = y + disp * np.sin(yaw)
        else:
            inv_w2 = 1.0 / (w * w)
            nx = x + inv_w2 * (nv * w * np.sin(nyaw) + a * np.cos(nyaw)
                               - v * w * np.sin(yaw) - a * np.cos(yaw))
            ny = y + inv_w2 * (-nv * w * np.cos(nyaw) + a * np.sin(nyaw)
                               + v * w * np.cos(yaw) - a * np.sin(yaw))
        return np.array([nx, ny, nv, a, nyaw, w], dtype=np.float64)

    def F(self, dt: float) -> np.ndarray:
        """Exact Jacobian ∂f/∂x of state_transition.

        |w|≥1e-3 uses the rotation model's full Jacobian (including curvature terms). |w|<1e-3
        is a straight-line approximation but uses the w→0 limit (including ∂pos/∂w) so it is
        continuous at the threshold boundary. It does not unify on the straight-line approximation
        — during rotation it preserves the ∂x/∂w·∂x/∂yaw curvature information.
        """
        x, y, v, a, yaw, w = self.state.tolist()
        F = np.eye(6, dtype=np.float64)
        c, s = np.cos(yaw), np.sin(yaw)
        if abs(w) < 1e-3:
            # 0th-order straight-line approximation — same as IMM-MOT motion_model.py:489-490. ∂pos/∂ω = 0.
            disp = v * dt + 0.5 * a * dt * dt
            F[0, 2] = dt * c
            F[0, 3] = 0.5 * dt * dt * c
            F[0, 4] = -disp * s
            F[1, 2] = dt * s
            F[1, 3] = 0.5 * dt * dt * s
            F[1, 4] = disp * c
        else:
            nv = v + a * dt
            yawp = yaw + w * dt
            cp, sp = np.cos(yawp), np.sin(yawp)
            w2 = w * w
            # position displacement (same expression as state_transition) — used to simplify ∂/∂yaw, ∂/∂w
            dx = (nv * w * sp + a * cp - v * w * s - a * c) / w2
            dy = (-nv * w * cp + a * sp + v * w * c - a * s) / w2
            # ∂x/∂{v, a, yaw, w}
            F[0, 2] = (sp - s) / w
            F[0, 3] = (dt * w * sp + cp - c) / w2
            F[0, 4] = -dy
            F[0, 5] = (nv * sp + nv * w * dt * cp - a * dt * sp - v * s) / w2 - 2.0 * dx / w
            # ∂y/∂{v, a, yaw, w}
            F[1, 2] = (c - cp) / w
            F[1, 3] = (-dt * w * cp + sp - s) / w2
            F[1, 4] = dx
            F[1, 5] = (-nv * cp + nv * w * dt * sp + a * dt * cp + v * c) / w2 - 2.0 * dy / w
        F[2, 3] = dt   # v ← a
        F[4, 5] = dt   # yaw ← w
        return F

    def Q(self, dt: float) -> np.ndarray:
        return self._skew_Q(np.diag([self.p.q_xy, self.p.q_xy,
                        self.p.q_v, self.p.q_a,
                        self._eff_q_yaw(), self._eff_q_yaw_rate()]).astype(np.float64))

    def state_to_measurement(self) -> np.ndarray:
        x, y, _, _, yaw, _ = self.state.tolist()
        return np.array([x, y, yaw], dtype=np.float64)

    def H(self) -> np.ndarray:
        H = np.zeros((3, 6), dtype=np.float64)
        H[0, 0] = 1.0; H[1, 1] = 1.0; H[2, 4] = 1.0
        return H

    def R(self) -> np.ndarray:
        return np.diag([self.p.r_xy, self.p.r_xy, self._eff_r_yaw()]).astype(np.float64)

    def to_standard(self) -> tuple[np.ndarray, np.ndarray]:
        x, y, v, a, yaw, w = self.state.tolist()
        s = np.zeros(STD_DIM, dtype=np.float64)
        s[STD_X] = x; s[STD_Y] = y
        s[STD_VX] = v * np.cos(yaw); s[STD_VY] = v * np.sin(yaw)
        s[STD_AX] = a * np.cos(yaw); s[STD_AY] = a * np.sin(yaw)
        s[STD_YAW] = yaw; s[STD_YR] = w
        P_std = np.zeros((STD_DIM, STD_DIM), dtype=np.float64)
        P_std[STD_X, STD_X] = self.cov[0, 0]
        P_std[STD_Y, STD_Y] = self.cov[1, 1]
        P_std[STD_X, STD_Y] = self.cov[0, 1]; P_std[STD_Y, STD_X] = self.cov[1, 0]
        # distribute the cov of v, a onto both vx/vy and ax/ay (conservative)
        P_std[STD_VX, STD_VX] = self.cov[2, 2]
        P_std[STD_VY, STD_VY] = self.cov[2, 2]
        P_std[STD_AX, STD_AX] = self.cov[3, 3]
        P_std[STD_AY, STD_AY] = self.cov[3, 3]
        P_std[STD_YAW, STD_YAW] = self.cov[4, 4]
        P_std[STD_YR, STD_YR] = self.cov[5, 5]
        P_std[STD_YAW, STD_YR] = self.cov[4, 5]; P_std[STD_YR, STD_YAW] = self.cov[5, 4]
        return s, P_std

    def set_from_standard(self, state_std: np.ndarray, cov_std: np.ndarray) -> None:
        x = state_std[STD_X]; y = state_std[STD_Y]
        vx = state_std[STD_VX]; vy = state_std[STD_VY]
        ax = state_std[STD_AX]; ay = state_std[STD_AY]
        yaw = _wrap(float(state_std[STD_YAW]))
        w = float(state_std[STD_YR])
        v = float(np.hypot(vx, vy))
        a = float(np.hypot(ax, ay))
        self.state = np.array([x, y, v, a, yaw, w], dtype=np.float64)
        P = np.zeros((6, 6), dtype=np.float64)
        P[0, 0] = cov_std[STD_X, STD_X]
        P[1, 1] = cov_std[STD_Y, STD_Y]
        P[0, 1] = cov_std[STD_X, STD_Y]; P[1, 0] = cov_std[STD_Y, STD_X]
        P[2, 2] = max(cov_std[STD_VX, STD_VX], cov_std[STD_VY, STD_VY])
        P[3, 3] = max(cov_std[STD_AX, STD_AX], cov_std[STD_AY, STD_AY])
        P[4, 4] = cov_std[STD_YAW, STD_YAW]
        P[5, 5] = cov_std[STD_YR, STD_YR]
        P[4, 5] = cov_std[STD_YAW, STD_YR]; P[5, 4] = cov_std[STD_YR, STD_YAW]
        self.cov = P

    def _wrap_state_yaw(self) -> None:
        self.state[4] = _wrap(float(self.state[4]))


# ── initialization helpers ────────────────────────────────────────────────────────
def _apply_init_cov_skew(P: np.ndarray, yaw: float, params: MotionModelParams,
                         pos_idx: tuple[int, int],
                         vel_idx: tuple[int, ...]) -> np.ndarray:
    """Anisotropically rotate the position (always) and velocity (CV/CA) blocks of init P0
    along the yaw direction. longitudinal axis = base·long_factor (larger), lateral axis =
    base·lat_factor (smaller). Same as kalman.build_P0."""
    if not params.init_cov_yaw_enabled:
        return P
    c, s = float(np.cos(yaw)), float(np.sin(yaw))
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    long_f, lat_f = params.init_cov_long_factor, params.init_cov_lat_factor
    i, j = pos_idx
    pos = np.diag([P[i, i] * long_f, P[j, j] * lat_f])
    P[np.ix_([i, j], [i, j])] = R @ pos @ R.T
    if vel_idx:
        a, b = vel_idx
        vel = np.diag([P[a, a] * long_f, P[b, b] * lat_f])
        P[np.ix_([a, b], [a, b])] = R @ vel @ R.T
    return P


def init_from_xy_yaw(x: float, y: float, yaw: float,
                     params: MotionModelParams,
                     model: str) -> BaseMotionModel:
    """Initialize a sub-filter from measurement (x, y, yaw). v/a/yaw_rate are 0.
    CV/CA also carry yaw in their state, so at init it is set to the detection yaw.
    If init_cov_yaw_enabled, anisotropically rotate P0's pos/vel block along the yaw direction."""
    yaw_w = _wrap(float(yaw))
    if model == "CV":
        s = np.array([x, y, 0.0, 0.0, yaw_w], dtype=np.float64)
        P = np.diag([params.p0_xy, params.p0_xy,
                     params.p0_v, params.p0_v,
                     params.p0_yaw]).astype(np.float64)
        P = _apply_init_cov_skew(P, yaw_w, params, (0, 1), (2, 3))
        return CVModel(s, P, params)
    if model == "CA":
        s = np.array([x, y, 0.0, 0.0, 0.0, 0.0, yaw_w], dtype=np.float64)
        P = np.diag([params.p0_xy, params.p0_xy,
                     params.p0_v, params.p0_v,
                     params.p0_a, params.p0_a,
                     params.p0_yaw]).astype(np.float64)
        P = _apply_init_cov_skew(P, yaw_w, params, (0, 1), (2, 3))
        return CAModel(s, P, params)
    if model == "CTRV":
        s = np.array([x, y, 0.0, yaw_w, 0.0], dtype=np.float64)
        P = np.diag([params.p0_xy, params.p0_xy, params.p0_v,
                     params.p0_yaw, params.p0_yaw_rate]).astype(np.float64)
        P = _apply_init_cov_skew(P, yaw_w, params, (0, 1), ())
        return CTRVModel(s, P, params)
    if model == "CTRA":
        s = np.array([x, y, 0.0, 0.0, yaw_w, 0.0], dtype=np.float64)
        P = np.diag([params.p0_xy, params.p0_xy, params.p0_v, params.p0_a,
                     params.p0_yaw, params.p0_yaw_rate]).astype(np.float64)
        P = _apply_init_cov_skew(P, yaw_w, params, (0, 1), ())
        return CTRAModel(s, P, params)
    raise ValueError(f"unknown model: {model!r}")
