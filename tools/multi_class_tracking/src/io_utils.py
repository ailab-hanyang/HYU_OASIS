"""sm_annotations.feather I/O + ego pose transform helpers.

Le3DE2E tracker prediction directory structure:
    output/tracker_predictions/<src_tracker>/<split>/<log_id>/sm_annotations.feather
ego pose:
    data/datasets/sensor/<split>/<log_id>/city_SE3_egovehicle.feather

Coordinate frame flow
---------------------
sm_annotations holds ego frame coordinates (ego is the origin of every frame). For
the KF to work correctly with the CV model it must track in the **world (city) frame** —
otherwise ego motion is misinterpreted as object motion and association breaks.

Therefore load_frames also performs the **ego→world transform**, and the output stage
(post_process.build_output_df) inverse-transforms world→ego again.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .track import Frame, Measurement


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def get_log_dir(log_id: str, split: str, src_tracker: str) -> Path:
    return (PROJECT_ROOT / "output" / "tracker_predictions" / src_tracker
            / split / log_id)


def get_dst_log_dir(log_id: str, split: str, dst_tracker: str) -> Path:
    return (PROJECT_ROOT / "output" / "tracker_predictions" / dst_tracker
            / split / log_id)


def get_ego_pose_path(log_id: str, split: str) -> Path:
    return (PROJECT_ROOT / "data" / "datasets" / "sensor" / split / log_id
            / "city_SE3_egovehicle.feather")


def _quat_to_yaw_z(qw, qz) -> np.ndarray:
    """ego frame box yaw — assumes z-axis rotation only (same assumption as rts_smoothing)."""
    return 2.0 * np.arctan2(qz, qw)


# ── ego pose lookup ────────────────────────────────────────────────
@dataclass
class EgoPoseLookup:
    """timestamp_ns → (cx, cy, cyaw), matched to the nearest frame."""
    ts: np.ndarray         # (T,) int64
    xyz: np.ndarray        # (T, 3)
    yaw: np.ndarray        # (T,) — z-axis only

    def __call__(self, t_ns: int) -> tuple[float, float, float]:
        if len(self.ts) == 0:
            return 0.0, 0.0, 0.0
        idx = int(np.searchsorted(self.ts, t_ns))
        if idx < len(self.ts) and int(self.ts[idx]) == int(t_ns):
            return float(self.xyz[idx, 0]), float(self.xyz[idx, 1]), float(self.yaw[idx])
        if idx == 0:
            return float(self.xyz[0, 0]), float(self.xyz[0, 1]), float(self.yaw[0])
        if idx >= len(self.ts):
            i = len(self.ts) - 1
            return float(self.xyz[i, 0]), float(self.xyz[i, 1]), float(self.yaw[i])
        before, after = int(self.ts[idx - 1]), int(self.ts[idx])
        i = idx - 1 if (t_ns - before) <= (after - t_ns) else idx
        return float(self.xyz[i, 0]), float(self.xyz[i, 1]), float(self.yaw[i])


def load_ego_pose(log_id: str, split: str) -> EgoPoseLookup:
    p = get_ego_pose_path(log_id, split)
    if not p.exists():
        return EgoPoseLookup(
            ts=np.zeros(0, dtype=np.int64),
            xyz=np.zeros((0, 3), dtype=np.float64),
            yaw=np.zeros(0, dtype=np.float64),
        )
    df = pd.read_feather(p)
    ts = df["timestamp_ns"].to_numpy(dtype=np.int64)
    order = np.argsort(ts)
    ts = ts[order]
    xyz = df[["tx_m", "ty_m", "tz_m"]].to_numpy(dtype=np.float64)[order]
    yaw = _quat_to_yaw_z(
        df["qw"].to_numpy(dtype=np.float64)[order],
        df["qz"].to_numpy(dtype=np.float64)[order],
    )
    return EgoPoseLookup(ts=ts, xyz=xyz, yaw=yaw)


# ── ego→world transform ─────────────────────────────────────────────
def ego_to_world(tx: np.ndarray, ty: np.ndarray,
                 cx: float, cy: float, cyaw: float
                 ) -> tuple[np.ndarray, np.ndarray]:
    """ego frame (tx, ty) → world frame (wx, wy)."""
    c, s = math.cos(cyaw), math.sin(cyaw)
    wx = cx + c * tx - s * ty
    wy = cy + s * tx + c * ty
    return wx, wy


# ── frame loader ───────────────────────────────────────────────────
def load_frames(log_id: str, split: str, src_tracker: str,
                score_threshold: float = 0.0
                ) -> tuple[list[Frame], pd.DataFrame]:
    """sm_annotations.feather + ego pose → (Frame list, EGO pass-through df).

    Each Measurement's (tx, ty) is in the **world frame**, yaw stays in the **ego frame**.
    The Frame itself holds (ego_x, ego_y, ego_yaw) so it can be inverse-transformed on output.

    If score_threshold > 0, detections with a confidence score at or below the threshold
    are removed. If the feather has no score column (assumed all 1.0), filtering has no effect.

    EGO_VEHICLE handling (option A):
      · EGO is not included in frames so the tracker does not handle it (not tracked)
      · The original EGO rows are returned as a separate DataFrame → apply_tracking concats
        them as-is on output → preserves src's EGO uuid ('ego' etc.) and 32-frame coverage
      · score_threshold is also not applied to EGO (always passes)

    Returns:
      frames: list[Frame]              — non-EGO measurements only (tracking targets)
      ego_passthrough_df: pd.DataFrame — original EGO rows as-is (for output concat)
    """
    fea = get_log_dir(log_id, split, src_tracker) / "sm_annotations.feather"
    if not fea.exists():
        raise FileNotFoundError(f"sm_annotations.feather not found: {fea}")

    df = pd.read_feather(fea)
    df = df.sort_values(["timestamp_ns"]).reset_index(drop=True)

    # ── Separate out the EGO pass-through (before filtering, preserved as-is)
    ego_mask = df["category"] == "EGO_VEHICLE"
    ego_passthrough_df = df[ego_mask].copy().reset_index(drop=True)
    df = df[~ego_mask].reset_index(drop=True)   # below handles non-EGO only

    # Pre-filtering — remove rows with score ≤ threshold (1.0 if no score column).
    # EGO is already excluded above so it is unaffected.
    if score_threshold > 0.0 and "score" in df.columns:
        n_before = len(df)
        df = df[df["score"] > score_threshold].reset_index(drop=True)
        n_after = len(df)
        if n_before > n_after:
            # for diagnostics — introduce a logger to print this in verbose mode
            pass

    yaws_ego = _quat_to_yaw_z(
        df["qw"].to_numpy(dtype=np.float64),
        df["qz"].to_numpy(dtype=np.float64),
    )

    ego_at = load_ego_pose(log_id, split)

    has_score = "score" in df.columns
    frames: list[Frame] = []

    for ts, sub in df.groupby("timestamp_ns", sort=True):
        cx, cy, cyaw = ego_at(int(ts))
        idx = sub.index.to_numpy()

        tx_ego = sub["tx_m"].to_numpy(dtype=np.float64)
        ty_ego = sub["ty_m"].to_numpy(dtype=np.float64)
        tz = sub["tz_m"].to_numpy(dtype=np.float64)
        l_arr = sub["length_m"].to_numpy(dtype=np.float64)
        w_arr = sub["width_m"].to_numpy(dtype=np.float64)
        h_arr = sub["height_m"].to_numpy(dtype=np.float64)
        cat_arr = sub["category"].astype(str).to_numpy()
        if has_score:
            s_arr = sub["score"].to_numpy(dtype=np.float64)
        else:
            s_arr = np.ones(len(sub), dtype=np.float64)
        yaw_arr = yaws_ego[idx]

        # ego→world transform (vectorized)
        wx, wy = ego_to_world(tx_ego, ty_ego, cx, cy, cyaw)

        meas: list[Measurement] = []
        for i in range(len(sub)):
            yaw_e = float(yaw_arr[i])
            meas.append(Measurement(
                tx=float(wx[i]), ty=float(wy[i]),       # world frame
                tz=float(tz[i]),                         # ego frame z as-is
                yaw=yaw_e,                               # ego frame yaw
                length=float(l_arr[i]), width=float(w_arr[i]), height=float(h_arr[i]),
                score=float(s_arr[i]), category=str(cat_arr[i]),
                yaw_world=yaw_e + float(cyaw),           # for KF/cost computation (world frame)
            ))
        frames.append(Frame(
            timestamp_ns=int(ts),
            detections=meas,
            ego_x=float(cx), ego_y=float(cy), ego_yaw=float(cyaw),
        ))
    return frames, ego_passthrough_df


def save_output_feather(df: pd.DataFrame, log_id: str, split: str,
                        dst_tracker: str) -> Path:
    out_dir = get_dst_log_dir(log_id, split, dst_tracker)
    out_dir.mkdir(parents=True, exist_ok=True)
    fea_out = out_dir / "sm_annotations.feather"
    df.reset_index(drop=True).to_feather(fea_out)
    return fea_out
