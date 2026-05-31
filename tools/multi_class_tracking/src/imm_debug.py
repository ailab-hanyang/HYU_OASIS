"""Save the IMM (and CV-only) sub-filter state time series to imm_debug.feather.

In long-format (long DataFrame) so server.py's qualitative-check / tuning panel can read it easily:

  track_uuid | timestamp_ns | stage   | model | mu  | likelihood | x | y | vx | vy | v | a | yaw | yaw_rate | cov_xx_world | cov_xy_world | cov_yy_world

  stage  ∈ {"predict", "update", "init"}
  model  ∈ {"CV", "CA", "CTRV", "CTRA", "IMM"}  (the "IMM" row is the mixed state)

For tracks that do not use IMM (CV-only, e.g. static) it also emits a model="CV" row + a
model="IMM" mirror row so server.py can read it with the same schema.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .motion_models import (
    BaseMotionModel,
    STD_X, STD_Y, STD_VX, STD_VY, STD_AX, STD_AY, STD_YAW, STD_YR,
)
from .imm_filter import IMMFilter


# ── helper that builds one row ────────────────────────────────────────────────
def _row_from_native(track_uuid: str, ts: int, stage: str,
                     f: BaseMotionModel, mu: float, likelihood: float,
                     cov_xy_2x2: Optional[np.ndarray] = None) -> dict:
    """Serialize one frame of a sub-filter's native state into one row."""
    # go through one native → standard conversion to fill all fields with consistent values
    s_std, P_std = f.to_standard()
    if cov_xy_2x2 is None:
        cov_xy_2x2 = P_std[np.ix_([STD_X, STD_Y], [STD_X, STD_Y])]
    # native-specific fields (NaN if absent)
    v = a = yaw_rate = np.nan
    if f.name == "CTRV":
        v = float(f.state[2])
        yaw_rate = float(f.state[4])
    elif f.name == "CTRA":
        v = float(f.state[2])
        a = float(f.state[3])
        yaw_rate = float(f.state[5])
    elif f.name == "CA":
        a = float(np.hypot(f.state[4], f.state[5]))
    return {
        "track_uuid":   track_uuid,
        "timestamp_ns": int(ts),
        "stage":        stage,
        "model":        f.name,
        "mu":           float(mu),
        "likelihood":   float(likelihood),
        "x":            float(s_std[STD_X]),
        "y":            float(s_std[STD_Y]),
        "vx":           float(s_std[STD_VX]),
        "vy":           float(s_std[STD_VY]),
        "v":            v,
        "a":            a,
        "yaw":          float(s_std[STD_YAW]),
        "yaw_rate":     yaw_rate if not np.isnan(yaw_rate) else float(s_std[STD_YR]),
        "cov_xx_world": float(cov_xy_2x2[0, 0]),
        "cov_xy_world": float(cov_xy_2x2[0, 1]),
        "cov_yy_world": float(cov_xy_2x2[1, 1]),
    }


def _row_from_imm(track_uuid: str, ts: int, stage: str,
                  imm: IMMFilter) -> dict:
    """Serialize one frame of the IMM mixed state into one row."""
    s_std = imm.mixed_state_std
    P_std = imm.mixed_cov_std
    cov_xy = P_std[np.ix_([STD_X, STD_Y], [STD_X, STD_Y])]
    return {
        "track_uuid":   track_uuid,
        "timestamp_ns": int(ts),
        "stage":        stage,
        "model":        "IMM",
        "mu":           1.0,
        "likelihood":   float(np.nan),
        "x":            float(s_std[STD_X]),
        "y":            float(s_std[STD_Y]),
        "vx":           float(s_std[STD_VX]),
        "vy":           float(s_std[STD_VY]),
        "v":            float(np.hypot(s_std[STD_VX], s_std[STD_VY])),
        "a":            float(np.hypot(s_std[STD_AX], s_std[STD_AY])),
        "yaw":          float(s_std[STD_YAW]),
        "yaw_rate":     float(s_std[STD_YR]),
        "cov_xx_world": float(cov_xy[0, 0]),
        "cov_xy_world": float(cov_xy[0, 1]),
        "cov_yy_world": float(cov_xy[1, 1]),
    }


def _row_from_cv_4d(track_uuid: str, ts: int, stage: str,
                    state_4d: np.ndarray, cov_4d: np.ndarray,
                    last_yaw: Optional[float]) -> tuple[dict, dict]:
    """Serialize multi_class_tracking's existing 4D CV state into two rows (CV row, IMM mirror row)
    — to keep the same schema as IMM tracks."""
    x, y, vx, vy = float(state_4d[0]), float(state_4d[1]), float(state_4d[2]), float(state_4d[3])
    yaw = float(last_yaw) if last_yaw is not None else float(np.arctan2(vy, vx))
    cov_xx = float(cov_4d[0, 0]); cov_xy = float(cov_4d[0, 1]); cov_yy = float(cov_4d[1, 1])
    cv_row = {
        "track_uuid":   track_uuid,
        "timestamp_ns": int(ts),
        "stage":        stage,
        "model":        "CV",
        "mu":           1.0,
        "likelihood":   float(np.nan),
        "x": x, "y": y, "vx": vx, "vy": vy,
        "v": float(np.hypot(vx, vy)),
        "a": float(np.nan),
        "yaw": yaw, "yaw_rate": float(np.nan),
        "cov_xx_world": cov_xx, "cov_xy_world": cov_xy, "cov_yy_world": cov_yy,
    }
    imm_row = dict(cv_row)
    imm_row["model"] = "IMM"
    return cv_row, imm_row


# ── Recorder ──────────────────────────────────────────────────────────
@dataclass
class DebugRecorder:
    """Call record_* every frame of tracker.step() to accumulate, flush at finalize."""
    enabled: bool = True
    rows: list[dict] = field(default_factory=list)
    # per-track summary accumulation (mu_mean, dominant, etc.)
    _mu_sum: dict[str, np.ndarray] = field(default_factory=dict)
    _mu_count: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _mu_last: dict[str, np.ndarray] = field(default_factory=dict)
    _dom_hist: dict[str, dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    _first_dom: dict[str, str] = field(default_factory=dict)

    def record_imm(self, track_uuid: str, ts: int, stage: str,
                   imm: IMMFilter) -> None:
        if not self.enabled:
            return
        # one row per sub-filter
        for j, f in enumerate(imm.filters):
            self.rows.append(_row_from_native(
                track_uuid, ts, stage, f,
                mu=float(imm.mu[j]),
                likelihood=float(imm.last_likelihoods[j]),
            ))
        # IMM mixed row
        self.rows.append(_row_from_imm(track_uuid, ts, stage, imm))
        # summary accumulation — only the mu of the update stage (predict's cbar is the same value,
        # but count only one stage to avoid double-counting). The init stage is ignored.
        if stage == "update":
            mu = imm.mu.astype(np.float64).copy()
            cur = self._mu_sum.get(track_uuid)
            if cur is None:
                self._mu_sum[track_uuid] = mu.copy()
            else:
                cur += mu
            self._mu_count[track_uuid] += 1
            self._mu_last[track_uuid] = mu
            dom_idx = int(np.argmax(mu))
            dom_name = imm.filters[dom_idx].name
            self._dom_hist[track_uuid][dom_name] += 1
            if track_uuid not in self._first_dom:
                self._first_dom[track_uuid] = dom_name

    def record_cv_4d(self, track_uuid: str, ts: int, stage: str,
                     state_4d: np.ndarray, cov_4d: np.ndarray,
                     last_yaw: Optional[float] = None) -> None:
        if not self.enabled:
            return
        cv_row, imm_row = _row_from_cv_4d(track_uuid, ts, stage,
                                          state_4d, cov_4d, last_yaw)
        self.rows.append(cv_row)
        self.rows.append(imm_row)
        # summary of a CV-only track — fill IMM mu with [0,0,0,0] (server.py can tell
        # it is CV-only just by looking at the length-4 array)
        if stage == "update":
            self._mu_last[track_uuid] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            self._mu_count[track_uuid] += 1
            self._dom_hist[track_uuid]["CV"] += 1
            if track_uuid not in self._first_dom:
                self._first_dom[track_uuid] = "CV"

    # ── flush ────────────────────────────────────────────────────
    def to_dataframe(self) -> pd.DataFrame:
        if not self.rows:
            return pd.DataFrame(columns=[
                "track_uuid", "timestamp_ns", "stage", "model",
                "mu", "likelihood",
                "x", "y", "vx", "vy", "v", "a", "yaw", "yaw_rate",
                "cov_xx_world", "cov_xy_world", "cov_yy_world",
            ])
        return pd.DataFrame(self.rows)

    def save_feather(self, path: Path) -> Path:
        df = self.to_dataframe()
        path.parent.mkdir(parents=True, exist_ok=True)
        df.reset_index(drop=True).to_feather(path)
        return path

    def summarize_per_track(self) -> dict[str, dict]:
        """The dict that goes into the per_track.<uuid>.imm section of tracking_summary.json."""
        out: dict[str, dict] = {}
        for uuid, count in self._mu_count.items():
            if count == 0:
                continue
            mu_mean = (self._mu_sum.get(uuid, np.zeros(4)) / max(1, count)).tolist()
            mu_last = self._mu_last.get(uuid, np.zeros(4)).tolist()
            out[uuid] = {
                "n_frames":             int(count),
                "mu_mean":              [float(x) for x in mu_mean],
                "mu_final":             [float(x) for x in mu_last],
                "dominant_model_hist":  dict(self._dom_hist.get(uuid, {})),
                "first_dominant_model": self._first_dom.get(uuid, ""),
            }
        return out


# ── IMM RTS smoothing sidecar recorder ───────────────────────────────────
# Accumulate the per-model full native state/cov/F/μ/Λ needed for the backward (per-model RTS)
# pass in long-format → imm_smooth_inputs.feather. (separate from imm_debug: heavy and unused by eval/viewer)
#
#  track_uuid | timestamp_ns | stage(predict|update|init) | model | model_idx |
#  dim | mu | likelihood | dt | state(list) | cov(flat list, dim*dim) | F(flat list, dim*dim)
#
# stage="predict" = prior (x_{k|k-1}, P_{k|k-1}, this step's F),
# stage="update"  = posterior (x_{k|k}, P_{k|k}).  init is the first frame's posterior.
# The backward pass pairs predict↔update by ts in each (uuid, model) time series and performs RTS.
def _smooth_row(track_uuid: str, ts: int, stage: str, model_idx: int,
                f: BaseMotionModel, mu: float, likelihood: float) -> dict:
    return {
        "track_uuid":   track_uuid,
        "timestamp_ns": int(ts),
        "stage":        stage,
        "model":        f.name,
        "model_idx":    int(model_idx),
        "dim":          int(f.SD),
        "mu":           float(mu),
        "likelihood":   float(likelihood),
        "dt":           float(getattr(f, "last_dt", 0.0)),
        "state":        [float(v) for v in np.asarray(f.state).ravel()],
        "cov":          [float(v) for v in np.asarray(f.cov).ravel()],
        "F":            [float(v) for v in np.asarray(getattr(f, "last_F", np.eye(f.SD))).ravel()],
    }


@dataclass
class SmoothInputRecorder:
    """IMM sidecar — accumulate per-model native state at the predict(prior)/update(posterior) times.
    Records IMM tracks only (CV-only tracks have no backward pass → not recorded)."""
    enabled: bool = True
    rows: list[dict] = field(default_factory=list)

    def record(self, track_uuid: str, ts: int, stage: str, imm: "IMMFilter") -> None:
        if not self.enabled:
            return
        for j, f in enumerate(imm.filters):
            self.rows.append(_smooth_row(
                track_uuid, ts, stage, j, f,
                mu=float(imm.mu[j]),
                likelihood=float(imm.last_likelihoods[j]),
            ))

    def to_dataframe(self) -> pd.DataFrame:
        cols = ["track_uuid", "timestamp_ns", "stage", "model", "model_idx",
                "dim", "mu", "likelihood", "dt", "state", "cov", "F"]
        if not self.rows:
            return pd.DataFrame(columns=cols)
        return pd.DataFrame(self.rows)[cols]

    def save_feather(self, path: Path) -> Path:
        df = self.to_dataframe()
        path.parent.mkdir(parents=True, exist_ok=True)
        df.reset_index(drop=True).to_feather(path)
        return path
