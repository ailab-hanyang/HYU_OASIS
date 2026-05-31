"""Track / FrameRecord / Measurement / Frame data classes.

The basic units for all time-series / matching data handled by the re-tracker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ── State indices (KF 4D) ──────────────────────────────────────────────
S_X, S_Y, S_VX, S_VY = 0, 1, 2, 3
STATE_DIM = 4
MEAS_DIM = 2


@dataclass
class Measurement:
    """Single detection (one object in one frame).

    Coordinate frame note — after the ego→world transform in io_utils, (tx, ty)
    is in the **city/world** frame, while yaw is kept in the **ego** frame (the
    yaw of the original quaternion). yaw_world is the pre-computed world frame
    yaw (= yaw + ego_yaw) for KF math. z is kept as-is (the KF does not track z).
    """
    tx: float            # world frame x (city)
    ty: float            # world frame y (city)
    tz: float            # original ego frame z (as-is)
    yaw: float           # ego frame yaw (rad)
    length: float
    width: float
    height: float
    score: float
    category: str
    yaw_world: float = 0.0   # world frame yaw — filled in by io_utils (for KF/cost computation)


@dataclass
class Frame:
    """All detections of one timestamp + the ego pose of that timestamp."""
    timestamp_ns: int
    detections: list[Measurement] = field(default_factory=list)
    # ego pose at this frame (city_SE3_ego). Used for the world→ego inverse transform on output.
    ego_x: float = 0.0
    ego_y: float = 0.0
    ego_yaw: float = 0.0   # ego heading in city (rad, z-axis only)


@dataclass
class FrameRecord:
    """Per-frame record of a track — corresponds to one row of the output feather.

    The KF state is stored in the **world frame** (the tracker operates in world).
    On output it is inverse-transformed to the ego frame using the ego pose.
    Only matched frames are recorded (predicted-only frames are not recorded).
    """
    timestamp_ns: int
    # KF posterior — WORLD frame
    kf_x: float
    kf_y: float
    kf_vx: float
    kf_vy: float
    # original measurement (preserved outside the KF)
    z_meas: float                # ego frame z as-is
    yaw_meas: float              # ego frame yaw (for static-object output)
    length_meas: float
    width_meas: float
    height_meas: float
    score_meas: float
    category_meas: str
    # post-processing hint
    is_static: bool              # ‖v_world‖ < v_static (after update)
    # ego pose of this frame (for the world→ego inverse transform on output)
    ego_x: float = 0.0
    ego_y: float = 0.0
    ego_yaw: float = 0.0
    # KF posterior covariance — xy 2x2 block in the WORLD frame (for visualization).
    # Emitted to feather as-is in post_process; rotated to anchor_ego on the server.
    cov_xx: float = 0.0
    cov_xy: float = 0.0
    cov_yy: float = 0.0
    # original measurement position (WORLD frame) — for outputting the raw position instead of
    # the KF posterior in relabel_only mode. Keeping DetA identical to the baseline requires
    # the raw measurement position without KF smoothing.
    meas_x_world: float = 0.0
    meas_y_world: float = 0.0
    # IMM mixed posterior yaw (world frame) — filled only for IMM-active tracks. None means CV-only.
    # When set, the relabel_only branch emits the IMM posterior yaw instead of the raw yaw_meas.
    imm_yaw_world: Optional[float] = None


@dataclass
class Track:
    """Single track — KF state + class EMA + lifecycle + per-frame history."""
    track_id: int

    # KF
    state: np.ndarray = field(default_factory=lambda: np.zeros(STATE_DIM, dtype=np.float64))
    cov: np.ndarray = field(default_factory=lambda: np.zeros((STATE_DIM, STATE_DIM), dtype=np.float64))

    # Class EMA — key is the category string, value is the score in (0,1)
    class_scores: dict[str, float] = field(default_factory=dict)

    # Lifecycle
    age: int = 0                          # accumulated over all frames (associated/not)
    detection_arr: list[bool] = field(default_factory=list)   # recent max_history (idx 0 = newest)
    is_init: bool = False
    is_confirmed: bool = False            # stays True once confirmed even once
    is_associated_this_frame: bool = False

    # Per-frame history (associated frames only)
    frame_records: list[FrameRecord] = field(default_factory=list)

    # measurement yaw received from the last match (world frame). Used as build_Q's yaw_hint —
    # lets static / just-initialized tracks also apply heading-aligned anisotropic Q.
    last_yaw: Optional[float] = None

    # IMM motion estimator — None means the existing 4D CV (state, cov fields) is used.
    # The vehicle / two_wheeler / pedestrian groups hold an IMMFilter instance,
    # the static group is None (CV).
    imm: object = None   # imm_filter.IMMFilter | None (object to avoid circular import)

    # ── helpers ────────────────────────────────────────────────────
    @property
    def velocity_norm(self) -> float:
        return float(np.hypot(self.state[S_VX], self.state[S_VY]))

    def push_detection(self, associated: bool, max_history: int) -> None:
        """Push the new frame result to the front of detection_arr (trimming old ones)."""
        self.detection_arr.insert(0, associated)
        if len(self.detection_arr) > max_history:
            self.detection_arr = self.detection_arr[:max_history]

    def count_recent_detections(self) -> int:
        return sum(1 for d in self.detection_arr if d)

    def get_rep_class(self) -> Optional[str]:
        if not self.class_scores:
            return None
        return max(self.class_scores.items(), key=lambda kv: kv[1])[0]

    def update_class_scores(self, cur_class: str, alpha: float) -> None:
        """Same as C++ updateClassScore — current class only (1·α + (1−α)·old), the rest (1−α)·old."""
        # new class can be added — unknown categories are admitted naturally too
        if cur_class not in self.class_scores:
            self.class_scores[cur_class] = 0.0
        for k in list(self.class_scores.keys()):
            if k == cur_class:
                self.class_scores[k] = alpha + (1.0 - alpha) * self.class_scores[k]
            else:
                self.class_scores[k] = (1.0 - alpha) * self.class_scores[k]
