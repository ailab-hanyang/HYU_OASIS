"""Kalman Filter — Update (measurement) model.

Measurement (8-D):
    z = [x, y, z, θ, l, w, h, s]^T
          0  1  2  3  4  5  6  7

Only the first 8 dimensions of the state are directly observed. velocity (vx, vy, vz) is unobserved.

Measurement matrix H (8×11):

        ┌  1  0  0  0  0  0  0  0  | 0  0  0  ┐
        │  0  1  0  0  0  0  0  0  | 0  0  0  │
        │  0  0  1  0  0  0  0  0  | 0  0  0  │
        │  0  0  0  1  0  0  0  0  | 0  0  0  │
    H = │  0  0  0  0  1  0  0  0  | 0  0  0  │
        │  0  0  0  0  0  1  0  0  | 0  0  0  │
        │  0  0  0  0  0  0  1  0  | 0  0  0  │
        └  0  0  0  0  0  0  0  1  | 0  0  0  ┘
              I_8 (observed dims)      vel (unobserved)

Measurement noise R (8×8 diagonal). config's R entries as the diagonal.

Update equations:
    y_k = z_k − H · x_{k|k-1}            (theta innovation is wrapped to [-π, π])
    S_k = H · P_{k|k-1} · H^T + R
    K_k = P_{k|k-1} · H^T · S_k^{-1}
    x_{k|k} = x_{k|k-1} + K_k · y_k
    P_{k|k} = (I − K_k · H) · P_{k|k-1}    (symmetrized via Joseph form)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .kalman_predict import STATE_DIM


# ── measurement indices / names ───────────────────────────────────────────
MEAS_DIM = 8
MEAS_NAMES = ("x", "y", "z", "theta", "l", "w", "h", "s")
MEAS_THETA_IDX = 3   # wrap applied to this in innovation


# ── H builder (constant) ──────────────────────────────────────────────────
def build_H() -> np.ndarray:
    """Measurement matrix (8×11). first 8 dims = identity, remaining 3 = 0."""
    H = np.zeros((MEAS_DIM, STATE_DIM), dtype=np.float64)
    H[:MEAS_DIM, :MEAS_DIM] = np.eye(MEAS_DIM, dtype=np.float64)
    return H


H_MATRIX = build_H()


# ── parameter container ───────────────────────────────────────────────────
@dataclass
class UpdateParams:
    """R (8×8 diag)."""
    R_diag: np.ndarray   # (8,)

    @property
    def R(self) -> np.ndarray:
        return np.diag(self.R_diag)

    @classmethod
    def from_config(cls, r_cfg: dict) -> "UpdateParams":
        diag = np.array([float(r_cfg[name]) for name in MEAS_NAMES], dtype=np.float64)
        return cls(R_diag=diag)


# ── helpers ───────────────────────────────────────────────────────────────
def _wrap(a: np.ndarray | float) -> np.ndarray | float:
    """Wrap to [-π, π]."""
    return np.arctan2(np.sin(a), np.cos(a))


# ── Update step ────────────────────────────────────────────────────────────
def update(x_pred: np.ndarray,
           P_pred: np.ndarray,
           z: np.ndarray,
           params: UpdateParams) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Measurement update of one step.

    Parameters
    ----------
    x_pred : (11,)   x_{k|k-1}
    P_pred : (11,11) P_{k|k-1}
    z      : (8,)    measurement [x, y, z, θ, l, w, h, s]
    params : UpdateParams

    Returns
    -------
    x_post : (11,)   x_{k|k}
    P_post : (11,11) P_{k|k}
    y      : (8,)    innovation (theta wrapped)
    S      : (8,8)   innovation covariance
    """
    H = H_MATRIX
    R = params.R

    # innovation — theta is wrapped
    y = z - H @ x_pred
    y[MEAS_THETA_IDX] = _wrap(y[MEAS_THETA_IDX])

    S = H @ P_pred @ H.T + R
    # K = P · H^T · S^{-1} → solve stably
    '''
    (K · S)ᵀ = (P·Hᵀ)ᵀ
    ↓ apply (A·B)ᵀ = Bᵀ·Aᵀ to the left side
    Sᵀ · Kᵀ = (P·Hᵀ)ᵀ
    │     │       │
    A     x       b      ← standard A·x = b form!
    '''
    K = np.linalg.solve(S.T, (P_pred @ H.T).T).T   # (11, 8)

    x_post = x_pred + K @ y
    # wrap theta itself to [-π, π] too (prevents accumulated drift)
    x_post[3] = _wrap(x_post[3])

    # Joseph form for numerical stability
    I = np.eye(STATE_DIM, dtype=np.float64)
    A = I - K @ H
    P_post = A @ P_pred @ A.T + K @ R @ K.T
    P_post = 0.5 * (P_post + P_post.T)

    return x_post, P_post, y, S
