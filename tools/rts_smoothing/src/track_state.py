"""Track state container — for holding per-uuid time-series state.

Stores ego frame coordinates as-is (same as sm_annotations.feather). Later used
in the yaw correction and RTS smoothing stages via in-place modification or new instance creation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class TrackState:
    """Time-series state of a single track_uuid (all ego frame; ego differs per frame).

    All field lengths are N (= number of frames of this uuid) and are assumed sorted by timestamp.
    """
    uuid: str
    category: str
    timestamps_ns: np.ndarray   # (N,) int64 — timestamp (ns)
    translations_m: np.ndarray  # (N, 3) — ego frame box center (tx, ty, tz)
    yaws_rad: np.ndarray        # (N,)  — ego frame yaw (rad)
    sizes_m: np.ndarray         # (N, 3) — length, width, height
    scores: np.ndarray          # (N,)  — confidence
    quaternions: Optional[np.ndarray] = None  # (N, 4) qw qx qy qz (ego frame). Held when needed
    # KF posterior xy cov in WORLD frame (only when available, multi_class_tracking output).
    # shape (N, 3): cov_xx, cov_xy, cov_yy. For visualization (covariance ellipse).
    cov_xy_world: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return len(self.timestamps_ns)

    @property
    def n_frames(self) -> int:
        return len(self.timestamps_ns)


@dataclass
class Tracks:
    """Collection of all tracks of one log + ego pose (city_SE3_ego) side data."""
    log_id: str
    split: str
    tracks: dict[str, TrackState] = field(default_factory=dict)
    # ego pose — timestamp_ns → 4x4 city_SE3_ego matrix (or (tx, ty, tz, qw, qx, qy, qz))
    # a numpy lookup is sufficient rather than managing a separate container
    ego_poses_ts: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))   # (T,)
    ego_poses_xyz: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))             # (T, 3)
    ego_poses_yaw: np.ndarray = field(default_factory=lambda: np.zeros(0))                  # (T,) — z-axis only (planar)

    def n_tracks(self) -> int:
        return len(self.tracks)

    def n_timestamps(self) -> int:
        return len(self.ego_poses_ts)

    def get_track(self, uuid: str) -> Optional[TrackState]:
        return self.tracks.get(uuid)
