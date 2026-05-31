"""Kalman Filter — Prediction model.

State (11-D):
    T = [x, y, z, θ, l, w, h, s, vx, vy, vz]^T
            0  1  2  3  4  5  6  7   8    9   10

Constant-velocity (CV) for (x, y, z) — random walk for (θ, l, w, h, s, vx, vy, vz).

State transition F(Δt) (11×11):

        ┌  1  0  0  0  0  0  0  0  Δt  0   0  ┐
        │  0  1  0  0  0  0  0  0  0   Δt  0  │
        │  0  0  1  0  0  0  0  0  0   0   Δt │
        │  0  0  0  1  0  0  0  0  0   0   0  │
        │  0  0  0  0  1  0  0  0  0   0   0  │
    F = │  0  0  0  0  0  1  0  0  0   0   0  │
        │  0  0  0  0  0  0  1  0  0   0   0  │
        │  0  0  0  0  0  0  0  1  0   0   0  │
        │  0  0  0  0  0  0  0  0  1   0   0  │
        │  0  0  0  0  0  0  0  0  0   1   0  │
        └  0  0  0  0  0  0  0  0  0   0   1  ┘

Process noise Q (11×11 diagonal). Uses the config's Q entries as the diagonal.
config Q is interpreted as "1 step" variance (no Δt scaling applied).

Predict equations:
    x_{k|k-1} = F(Δt_k) · x_{k-1|k-1}
    P_{k|k-1} = F(Δt_k) · P_{k-1|k-1} · F^T + Q
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ── State indices (shared with other modules) ─────────────────────────────
STATE_DIM = 11
IDX_X, IDX_Y, IDX_Z = 0, 1, 2
IDX_THETA = 3
IDX_L, IDX_W, IDX_H = 4, 5, 6
IDX_S = 7
IDX_VX, IDX_VY, IDX_VZ = 8, 9, 10

STATE_NAMES = ("x", "y", "z", "theta", "l", "w", "h", "s", "vx", "vy", "vz")


# ── parameter container ───────────────────────────────────────────────────
@dataclass
class PredictParams:
    """Q (11×11 diag). Built from the config's dict."""
    Q_diag: np.ndarray   # (11,)

    @property
    def Q(self) -> np.ndarray:
        return np.diag(self.Q_diag)

    @classmethod
    def from_config(cls, q_cfg: dict) -> "PredictParams":
        """config['rts_smoothing']['Q'] dict → PredictParams."""
        diag = np.array([float(q_cfg[name]) for name in STATE_NAMES], dtype=np.float64)
        return cls(Q_diag=diag)


# ── F(Δt) builder ──────────────────────────────────────────────────────────
def build_F(dt: float) -> np.ndarray:
    """Constant-velocity transition matrix (11×11). dt unit is second."""
    F = np.eye(STATE_DIM, dtype=np.float64)
    F[IDX_X, IDX_VX] = dt
    F[IDX_Y, IDX_VY] = dt
    F[IDX_Z, IDX_VZ] = dt
    return F


# ── Predict step ───────────────────────────────────────────────────────────
def predict(x_prev: np.ndarray,
            P_prev: np.ndarray,
            dt: float,
            params: PredictParams,
            Q_override=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prediction of one step.

    Parameters
    ----------
    x_prev : (11,)   x_{k-1|k-1}
    P_prev : (11,11) P_{k-1|k-1}
    dt     : float   Δt = t_k − t_{k-1} [s]
    params : PredictParams

    Returns
    -------
    x_pred : (11,)   x_{k|k-1}
    P_pred : (11,11) P_{k|k-1}
    F      : (11,11) F(Δt) — reused in RTS backward
    """
    F = build_F(dt) # build F matrix considering dt
    x_pred = F @ x_prev
    _Q = params.Q if Q_override is None else Q_override
    P_pred = F @ P_prev @ F.T + _Q
    # symmetrize (numerical stability)
    P_pred = 0.5 * (P_pred + P_pred.T)
    return x_pred, P_pred, F
