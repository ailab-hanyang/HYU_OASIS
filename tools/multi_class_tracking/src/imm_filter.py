"""Interacting Multiple Model (IMM) filter — 4 sub-filter mixing.

Algorithm order (same as IROS 2025 IMM-MOT, compatible with multi_class_tracking 4D):

predict(dt):
    1. cbar = mu @ M
    2. omega[i,j] = M[i,j] * mu[i] / cbar[j]
    3. inject the mixing initial condition into each sub-filter j (standard 8D mixing then back to native)
    4. f_j.predict(dt) for each sub-filter
    5. IMM mixed state/cov = Σ_j mu[j] · standard(f_j)

update(z=(x,y,yaw)):
    1. f_j.update(z) → likelihood_j for each sub-filter
    2. mu_j ← cbar_j · likelihood_j, normalize
    3. recompute the mixing probabilities
    4. re-synthesize the IMM mixed state/cov
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .motion_models import (
    BaseMotionModel, MotionModelParams,
    STD_DIM, STD_X, STD_Y, STD_VX, STD_VY, STD_YAW, STD_YR,
    init_from_xy_yaw, _wrap,
)


# ── weighted mean/covariance accounting for circular yaw ─────────────────────────────
# The yaw dimension (STD_YAW) of the standard 8D state is a circular quantity, so a linear
# weighted average diverges at the ±π boundary (e.g. the mean of +3.0 and -3.0 = 0). Handle
# only yaw with the circular mean (atan2(Σw·sin, Σw·cos)); the rest stay linear.
def _circular_weighted_state(std_states: list[np.ndarray],
                             weights: np.ndarray) -> np.ndarray:
    """Weighted mean of the state_std list. Only the yaw dimension uses the circular mean."""
    mixed = np.zeros(STD_DIM, dtype=np.float64)
    for w, s in zip(weights, std_states):
        mixed += w * s
    # overwrite the yaw dimension — circular mean
    sin_sum = float(sum(w * np.sin(s[STD_YAW]) for w, s in zip(weights, std_states)))
    cos_sum = float(sum(w * np.cos(s[STD_YAW]) for w, s in zip(weights, std_states)))
    mixed[STD_YAW] = float(np.arctan2(sin_sum, cos_sum))
    return mixed


def _circular_weighted_cov(std_states: list[np.ndarray],
                           std_covs: list[np.ndarray],
                           weights: np.ndarray,
                           mixed: np.ndarray) -> np.ndarray:
    """Weighted covariance Σ w·(P + d·dᵀ). The yaw deviation is wrapped to [-π,π]."""
    P = np.zeros((STD_DIM, STD_DIM), dtype=np.float64)
    for w, s, Pj in zip(weights, std_states, std_covs):
        d = (s - mixed).copy()
        d[STD_YAW] = _wrap(float(d[STD_YAW]))   # yaw deviation circular
        d = d.reshape(-1, 1)
        P += w * (Pj + d @ d.T)
    return 0.5 * (P + P.T)


# ── common state dimensions — the dimensions every sub-filter models ──────────
# IMM mixing mixes only this common block across all filters (alternative A). Model-specific
# dimensions (ax,ay=CA/CTRA, yaw_rate=CTRV/CTRA) are excluded from mixing and each filter keeps
# its own posterior → eliminates at the root the problem where the unmodeled-dimension
# placeholder (_UNMODELED_DIAG) contaminated the modeled dimensions. Since position/velocity/yaw
# are common, the values used for likelihood and visualization are all normalized.
_COMMON_STD = [STD_X, STD_Y, STD_VX, STD_VY, STD_YAW]


def _nearest_psd(P: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Return the nearest PSD matrix by symmetrizing then clipping eigenvalues.
    Overwriting the common block with mixed (enlarged) while keeping own for cross/private can
    rarely break PSD, so this is applied as a safety net (negligible cost at 8×8)."""
    P = 0.5 * (P + P.T)
    w, V = np.linalg.eigh(P)
    w = np.clip(w, eps, None)
    return (V * w) @ V.T


# ── IMM parameters ──────────────────────────────────────────────────────
@dataclass
class IMMParams:
    """Per-group mu/M and the sub-filters' motion model params.

    `motion_params` is the common default for all sub-filters.
    If `sub_motion_params[model]` exists, only that sub-filter is overridden with its value
    (models without it use motion_params). This cascade is determined in imm_config.py while
    processing the yaml by_model / by_group_model overrides.
    """
    mu_init: np.ndarray            # (N,) initial mode probabilities
    M: np.ndarray                  # (N, N) Markov transition matrix
    sub_models: list[str]          # ["CV", "CA", "CTRV", "CTRA"] order
    motion_params: MotionModelParams                              # default (common to all subs)
    sub_motion_params: dict = None                                # model_name → MotionModelParams (optional)

    def __post_init__(self):
        assert self.mu_init.shape == (len(self.sub_models),)
        assert self.M.shape == (len(self.sub_models), len(self.sub_models))
        assert abs(float(self.mu_init.sum()) - 1.0) < 1e-6, "mu_init must sum to 1"
        if self.sub_motion_params is None:
            self.sub_motion_params = {}

    def params_for(self, model: str) -> MotionModelParams:
        """Return that sub-filter's motion_params (default if no override)."""
        return self.sub_motion_params.get(model, self.motion_params)


# ── IMMFilter ─────────────────────────────────────────────────────────
class IMMFilter:
    """Responsible for a track's motion estimation. Provides an interface compatible with
    multi_class_tracking's KF (predict/update + exposing 4D xy/vxvy externally)."""

    def __init__(self, init_x: float, init_y: float, init_yaw: float,
                 params: IMMParams):
        self.params = params
        self.N = len(params.sub_models)
        self.mu = params.mu_init.astype(np.float64).copy()
        self.M = params.M.astype(np.float64).copy()

        # each sub-filter starts from the same init state (v=a=yaw_rate=0).
        # If sub_motion_params exists, init only that model with its params.
        self.filters: list[BaseMotionModel] = [
            init_from_xy_yaw(init_x, init_y, init_yaw, params.params_for(m), m)
            for m in params.sub_models
        ]
        # mixed state/cov — standard 8D
        self.mixed_state_std = np.zeros(STD_DIM, dtype=np.float64)
        self.mixed_cov_std = np.zeros((STD_DIM, STD_DIM), dtype=np.float64)
        self._compute_mixed_state()
        self.last_likelihoods: np.ndarray = np.ones(self.N, dtype=np.float64)
        # previous step's mixing matrix — reused by the debug recorder and the mu update formula
        self._cbar = self.mu @ self.M
        self._omega = self._compute_omega()

    # ── helpers ──
    def _compute_omega(self) -> np.ndarray:
        omega = np.zeros((self.N, self.N), dtype=np.float64)
        for j in range(self.N):
            cb = self._cbar[j]
            if cb < 1e-300:
                # nearly 0 — fall back to uniform
                omega[:, j] = 1.0 / self.N
            else:
                for i in range(self.N):
                    omega[i, j] = self.M[i, j] * self.mu[i] / cb
        return omega

    def _compute_mixed_state(self) -> None:
        """Update mixed state/cov from the current mu and each sub-filter's standard form.
        The yaw dimension is handled with circular mean / circular deviation (avoids ±π divergence)."""
        s_list, P_list = [], []
        for f in self.filters:
            s, P = f.to_standard()
            s_list.append(s); P_list.append(P)
        mixed_s = _circular_weighted_state(s_list, self.mu)
        mixed_P = _circular_weighted_cov(s_list, P_list, self.mu, mixed_s)
        self.mixed_state_std = mixed_s
        self.mixed_cov_std = mixed_P

    # ── predict ──
    def predict(self, dt: float) -> None:
        # 1, 2. mixing matrices
        self._cbar = self.mu @ self.M
        self._omega = self._compute_omega()

        # 3. inject the mixing initial condition — mix only the common block (x,y,vx,vy,yaw)
        #    across all filters, while each filter keeps its own posterior for the model-specific
        #    dimensions (ax,ay / yaw_rate) and their cross terms.
        #    (alternative A) removes the problem where the unmodeled-dimension placeholder contaminated the modeled dimensions.
        std_states, std_covs = [], []
        for f in self.filters:
            s, P = f.to_standard()
            std_states.append(s); std_covs.append(P)
        cidx = _COMMON_STD
        for j, f in enumerate(self.filters):
            # mixing weights omega[:, j]. yaw circular mean + deviation wrap.
            wj = self._omega[:, j]
            # common-block mixed mean/cov (all filters contribute). The private dimensions are not used.
            full_mix_s = _circular_weighted_state(std_states, wj)
            full_mix_P = _circular_weighted_cov(std_states, std_covs, wj, full_mix_s)
            # copy j's own standard, then overwrite only the common block with mixed.
            mixed_s = std_states[j].copy()
            mixed_P = std_covs[j].copy()
            mixed_s[cidx] = full_mix_s[cidx]
            mixed_P[np.ix_(cidx, cidx)] = full_mix_P[np.ix_(cidx, cidx)]
            mixed_P = _nearest_psd(mixed_P)
            f.set_from_standard(mixed_s, mixed_P)

        # 4. predict — each sub-filter keeps yaw in [-π,π] via its own _wrap_state_yaw
        for f in self.filters:
            f.predict(dt)

        # 5. mixed state (circular yaw mean)
        self._compute_mixed_state()

    # ── update ──
    def update(self, z: np.ndarray, detection_confidence: float = 1.0,
               static: bool = False) -> None:
        """z = (x, y, yaw). All sub-filters (CV/CA/CTRV/CTRA) use the yaw measurement (H 3rd row).

        Trust the measurement (yawfix detection) as-is — no corrections like flip/velocity alignment.
        The measured yaw is fed to the sub-filters after only [-π,π] normalization (wrap).
        (the residual yaw / post-update state yaw wrap is handled by each sub-filter)

        detection_confidence is passed to each sub-filter — R scale on low-confidence measurements (same as CV KF).
        If static=True (predicted speed < v_static), distrust the yaw measurement (R_yaw↑) to suppress static-object yaw jitter.
        """
        z = np.asarray(z, dtype=np.float64).copy()
        if z.shape[0] >= 3:
            z[2] = _wrap(float(z[2]))   # measurement yaw normalization (before feeding in)
        for f in self.filters:
            f.update(z, detection_confidence, static=static)
        self.last_likelihoods = np.array([f.last_likelihood for f in self.filters],
                                         dtype=np.float64)
        # mu_j = cbar_j · likelihood_j → normalize
        new_mu = self._cbar * self.last_likelihoods
        s = float(new_mu.sum())
        if s < 1e-300:
            # all filters have likelihood 0 — keep mu
            pass
        else:
            self.mu = new_mu / s
        self._compute_mixed_state()

    # ── external exposure (compatible with multi_class_tracking's KF) ──
    @property
    def xy(self) -> tuple[float, float]:
        return float(self.mixed_state_std[STD_X]), float(self.mixed_state_std[STD_Y])

    @property
    def vxvy(self) -> tuple[float, float]:
        return float(self.mixed_state_std[STD_VX]), float(self.mixed_state_std[STD_VY])

    @property
    def yaw(self) -> float:
        return float(self.mixed_state_std[STD_YAW])

    @property
    def cov_xy_block(self) -> np.ndarray:
        """xy 2×2 cov block (world frame). The value for the
        cov_xx_world/cov_xy_world/cov_yy_world columns of sm_annotations.feather."""
        return self.mixed_cov_std[np.ix_([STD_X, STD_Y], [STD_X, STD_Y])]

    def set_velocity_zero(self) -> None:
        """Below v_static → force static. Set speed v + yaw_rate ω to 0 for the mixed state and all sub-filters.

        Forcing yaw_rate to 0 is the key — if ω remains while static, predict keeps drifting/oscillating
        heading via yaw += ω·dt (the IMM counterpart of CV-only zeroing only velocity and fixing yaw to
        the measurement). Suppressing following the measured yaw is handled by update's static_r_yaw_scale."""
        # update the mixed state — what external IMM users see
        self.mixed_state_std[STD_VX] = 0.0
        self.mixed_state_std[STD_VY] = 0.0
        self.mixed_state_std[STD_YR] = 0.0
        # force sub-filter native v + yaw_rate to 0 — indices differ per model
        for f in self.filters:
            if f.name == "CV":
                f.state[2] = 0.0; f.state[3] = 0.0
            elif f.name == "CA":
                f.state[2] = 0.0; f.state[3] = 0.0
                # keep acceleration — even when static, if detection wobbles it survives into the next step
            elif f.name == "CTRV":
                f.state[2] = 0.0      # v
                f.state[4] = 0.0      # yaw_rate ω
            elif f.name == "CTRA":
                f.state[2] = 0.0      # v
                f.state[3] = 0.0      # a
                f.state[5] = 0.0      # yaw_rate ω
