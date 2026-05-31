"""MultiClassTracker — frame loop + lifecycle management.

Receives all frames of a single log in time order and generates a new track ID
sequence. Takes Le3DE2E detection results (sm_annotations.feather) as input;
the output is produced by post_process.build_output_df.

Flow (per step)
─────────────────────────────────────────────────────────────────────
   Frame (timestamp, [Measurement,...])
        │
        ├─ [1] Predict — apply dt to all active tracks (update state, cov)
        │
        ├─ [2] Cost matrix + gating + Hungarian
        │      L2 / Mahalanobis branch + PEDESTRIAN class isolation
        │
        ├─ [3] Matched pairs → track update (KF + class EMA + frame record)
        │      ‖v‖ < v_static → static, force vx=vy=0
        │
        ├─ [4] Unmatched measurement → init new track
        │
        ├─ [5] Unmatched track → age++, detection_arr push False, outdated check
        │
        └─ (next frame)

On finalize() call, move all active tracks to completed.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .track import (
    Track, Measurement, Frame, FrameRecord,
    STATE_DIM, MEAS_DIM, S_X, S_Y, S_VX, S_VY,
)
from .kalman import KFParams, predict, update, build_P0
from .association import (
    AssociationParams, build_cost_matrix, match_pairs, STATIC_CATEGORIES,
)
from .imm_filter import IMMFilter, IMMParams
from .imm_config import imm_params_from_config
from .imm_debug import DebugRecorder, SmoothInputRecorder


# ── AV2 category → IMM group mapping ──────────────────────────────────
# Same definition as _CLASS_GROUPS in association.py. Duplicated here because
# the tracker needs to know only the groups without an extra dependency on the
# association module.
_GROUPS = {
    "vehicle": {"REGULAR_VEHICLE", "LARGE_VEHICLE", "BOX_TRUCK", "TRUCK", "TRUCK_CAB",
                "VEHICULAR_TRAILER", "BUS", "ARTICULATED_BUS", "SCHOOL_BUS",
                "RAILED_VEHICLE", "MESSAGE_BOARD_TRAILER", "TRAFFIC_LIGHT_TRAILER"},
    "two_wheeler": {"BICYCLE", "BICYCLIST", "MOTORCYCLE", "MOTORCYCLIST",
                    "WHEELED_DEVICE", "WHEELED_RIDER"},
    "pedestrian": {"PEDESTRIAN", "STROLLER", "DOG", "OFFICIAL_SIGNALER"},
    "static": {"BOLLARD", "STOP_SIGN", "SIGN", "MOBILE_PEDESTRIAN_CROSSING_SIGN",
               "CONSTRUCTION_CONE", "CONSTRUCTION_BARREL"},
}
_CAT_TO_GROUP = {c: g for g, cs in _GROUPS.items() for c in cs}
# IMM-applied groups (user decision: all groups except static use IMM)
_IMM_GROUPS = frozenset({"vehicle", "two_wheeler", "pedestrian"})


def _group_of(category: str) -> str:
    """category → group. Unregistered categories fall back to vehicle (the most
    common group, conservative)."""
    return _CAT_TO_GROUP.get(category, "vehicle")


# ── Combined parameters ────────────────────────────────────────────
@dataclass
class TrackerParams:
    kf: KFParams
    assoc: AssociationParams
    # Lifecycle
    max_history: int
    max_history_for_outdated: int
    static_max_history_for_outdated: int   # static classes only (larger → longer survival)
    confirmed_age_min: int
    confirmed_detect_min: int
    # Class EMA
    class_alpha: float
    # static/dynamic
    v_static: float
    # ── IMM ──
    motion_model: str = "cv"               # "cv" | "imm"
    imm_params_by_group: dict[str, IMMParams] | None = None   # active groups only
    save_imm_debug: bool = False
    save_smooth_inputs: bool = False       # generate IMM RTS smoothing sidecar (imm_smooth_inputs.feather)

    @classmethod
    def from_config(cls, t_cfg: dict) -> "TrackerParams":
        motion_model = str(t_cfg.get("motion_model", "cv")).lower()
        imm_cfg = (t_cfg.get("imm") or {})
        save_debug = bool(imm_cfg.get("save_debug", False))
        save_smooth = bool(t_cfg.get("save_smooth_inputs", True))
        imm_params_by_group: dict[str, IMMParams] | None = None
        if motion_model == "imm":
            imm_params_by_group = {
                g: imm_params_from_config(g, imm_cfg, t_cfg) for g in _IMM_GROUPS
            }
        return cls(
            kf=KFParams.from_config(t_cfg),
            assoc=AssociationParams.from_config(t_cfg),
            max_history=int(t_cfg.get("max_history", 15)),
            max_history_for_outdated=int(t_cfg.get("max_history_for_outdated", 12)),
            static_max_history_for_outdated=int(t_cfg.get(
                "static_max_history_for_outdated",
                t_cfg.get("max_history_for_outdated", 12))),
            confirmed_age_min=int(t_cfg.get("confirmed_age_min", 3)),
            confirmed_detect_min=int(t_cfg.get("confirmed_detect_min", 2)),
            class_alpha=float(t_cfg.get("class_alpha", 0.2)),
            v_static=float(t_cfg.get("v_static", 0.25)),
            motion_model=motion_model,
            imm_params_by_group=imm_params_by_group,
            save_imm_debug=save_debug,
            save_smooth_inputs=save_smooth,
        )


# ── Tracker ────────────────────────────────────────────────────────
class MultiClassTracker:
    def __init__(self, params: TrackerParams):
        self.p = params
        self.active_tracks: list[Track] = []
        self.completed_tracks: list[Track] = []
        self._next_track_id: int = 0
        self._previous_ts: Optional[int] = None
        # IMM debug recorder — enabled only when save_imm_debug=True. Even in
        # CV-only mode, if save_imm_debug=True the CV state time series is recorded
        # with the same schema (user decision Q7).
        self.debug_recorder: DebugRecorder = DebugRecorder(enabled=self.p.save_imm_debug)
        # IMM RTS smoothing sidecar recorder — accumulates per-model prior/posterior
        # of IMM tracks.
        self.smooth_recorder: SmoothInputRecorder = SmoothInputRecorder(
            enabled=self.p.save_smooth_inputs)
        # Until just before uuid assignment (post_process), only track_id (int) is known.
        # The debug recorder's track_uuid key is filled with str(track_id) for now, then
        # rewritten with uuid_map in apply_tracking.
        self._uuid_for_debug: dict[int, str] = {}

    # ── frame loop ─────────────────────────────────────────────
    def step(self, frame: Frame) -> None:
        ts = int(frame.timestamp_ns)

        # [1] dt → predict
        if self._previous_ts is not None:
            dt = (ts - self._previous_ts) * 1e-9
        else:
            dt = 0.0
        if dt > 0.0 and self.active_tracks:
            for tr in self.active_tracks:
                if tr.imm is not None:
                    # IMM predict — sub-filter mixing + each filter predicts
                    tr.imm.predict(dt)
                    # sync with multi_class_tracking external interface (track.state)
                    x, y = tr.imm.xy
                    vx, vy = tr.imm.vxvy
                    tr.state = np.array([x, y, vx, vy], dtype=np.float64)
                    # sync only the xy block of cov. Fill the velocity block from the
                    # vx/vy variance of the IMM standard, with cross-blocks set to 0
                    # (conservative). Sufficient because association's Mahalanobis only
                    # looks at the xy block.
                    cov_xy = tr.imm.cov_xy_block
                    new_cov = np.zeros((STATE_DIM, STATE_DIM), dtype=np.float64)
                    new_cov[:2, :2] = cov_xy
                    from .motion_models import STD_VX, STD_VY
                    new_cov[2, 2] = float(tr.imm.mixed_cov_std[STD_VX, STD_VX])
                    new_cov[3, 3] = float(tr.imm.mixed_cov_std[STD_VY, STD_VY])
                    tr.cov = new_cov
                    # debug — record the predict step
                    _du = self._uuid_for_debug.get(tr.track_id, str(tr.track_id))
                    self.debug_recorder.record_imm(_du, ts, "predict", tr.imm)
                    self.smooth_recorder.record(_du, ts, "predict", tr.imm)
                else:
                    # use last_yaw as yaw_hint — apply heading-axis anisotropic Q even
                    # to static/just-initialized tracks
                    tr.state, tr.cov = predict(
                        tr.state, tr.cov, dt, self.p.kf,
                        yaw_hint=tr.last_yaw,
                    )
                    # debug — record single CV (static group) with the same schema too
                    self.debug_recorder.record_cv_4d(
                        self._uuid_for_debug.get(tr.track_id, str(tr.track_id)),
                        ts, "predict", tr.state, tr.cov, tr.last_yaw,
                    )
                tr.is_associated_this_frame = False

        # initialize all to unassociated (including the predict-skip case)
        for tr in self.active_tracks:
            tr.is_associated_this_frame = False

        meas = frame.detections
        n_meas = len(meas)
        n_tracks = len(self.active_tracks)

        # [2] cost matrix + matching (greedy or hungarian — config branch)
        if n_meas > 0 and n_tracks > 0:
            cost, l2_arr = build_cost_matrix(
                meas, self.active_tracks, self.p.assoc, self.p.kf,
            )
            matches = match_pairs(
                cost, l2_arr,
                self.p.assoc.max_association_dist_m,
                algorithm=self.p.assoc.match_algorithm,
            )
        else:
            matches = []

        matched_meas_idx = {m for m, _ in matches}
        matched_track_idx = {t for _, t in matches}

        # [3] matched: update track
        for meas_idx, track_idx in matches:
            self._update_track(self.active_tracks[track_idx], meas[meas_idx], frame)

        # [4] unmatched measurements → init new track
        for r in range(n_meas):
            if r in matched_meas_idx:
                continue
            new_tr = self._init_track(meas[r], frame)
            self.active_tracks.append(new_tr)

        # [5] unmatched tracks → age++, outdated check → move to completed on expiry
        survivors: list[Track] = []
        for c, tr in enumerate(self.active_tracks):
            if c < n_tracks and c in matched_track_idx:
                # keep matched track as is
                survivors.append(tr)
                continue
            if c >= n_tracks:
                # track newly initialized this frame — keep as is
                survivors.append(tr)
                continue
            # unmatched track
            tr.age += 1
            tr.push_detection(False, self.p.max_history)
            if tr.is_init and self._is_outdated(tr):
                self.completed_tracks.append(tr)
            else:
                survivors.append(tr)
        self.active_tracks = survivors

        self._previous_ts = ts

    def finalize(self) -> None:
        """Move all remaining active tracks to completed."""
        self.completed_tracks.extend(self.active_tracks)
        self.active_tracks = []

    # ── internal helpers ──────────────────────────────────────
    def _new_track_id(self) -> int:
        i = self._next_track_id
        self._next_track_id += 1
        return i

    def _init_track(self, m: Measurement, frame: Frame) -> Track:
        tr = Track(track_id=self._new_track_id())
        tr.state = np.array([m.tx, m.ty, 0.0, 0.0], dtype=np.float64)
        # P0 — anisotropic along the detection's yaw direction (optional)
        # m.yaw is in ego frame, so convert to world frame (KF state is world frame).
        yaw_world = float(m.yaw) + float(frame.ego_yaw)
        # Static classes (BOLLARD/CONE/BARREL/SIGN, etc.) have no meaningful heading,
        # so use an isotropic P0 without applying longitudinal init_cov anisotropy.
        tr.cov = build_P0(yaw_world, self.p.kf,
                          is_static=(m.category in STATIC_CATEGORIES))
        tr.last_yaw = yaw_world   # used as yaw_hint for next frame's predict

        # whether IMM is active — motion_model='imm' AND group is an IMM target
        group = _group_of(m.category)
        if self.p.motion_model == "imm" and group in _IMM_GROUPS \
                and self.p.imm_params_by_group is not None:
            ip = self.p.imm_params_by_group.get(group)
            if ip is not None:
                tr.imm = IMMFilter(
                    init_x=float(m.tx), init_y=float(m.ty), init_yaw=yaw_world,
                    params=ip,
                )

        tr.class_scores[m.category] = 1.0
        tr.age = 1
        tr.is_init = True
        tr.is_associated_this_frame = True
        tr.push_detection(True, self.p.max_history)
        # confirmed check (usually not at age=1)
        self._maybe_confirm(tr)
        # debug — record at init time
        debug_uuid = str(tr.track_id)
        self._uuid_for_debug[tr.track_id] = debug_uuid
        if tr.imm is not None:
            self.debug_recorder.record_imm(debug_uuid, frame.timestamp_ns,
                                           "init", tr.imm)
            self.smooth_recorder.record(debug_uuid, frame.timestamp_ns, "init", tr.imm)
        else:
            self.debug_recorder.record_cv_4d(debug_uuid, frame.timestamp_ns,
                                             "init", tr.state, tr.cov, tr.last_yaw)
        # frame record (store ego pose too — used for world→ego inverse transform)
        is_static = tr.velocity_norm < self.p.v_static
        tr.frame_records.append(FrameRecord(
            timestamp_ns=frame.timestamp_ns,
            kf_x=float(tr.state[S_X]), kf_y=float(tr.state[S_Y]),
            kf_vx=float(tr.state[S_VX]), kf_vy=float(tr.state[S_VY]),
            z_meas=m.tz, yaw_meas=m.yaw,
            length_meas=m.length, width_meas=m.width, height_meas=m.height,
            score_meas=m.score, category_meas=m.category,
            meas_x_world=float(m.tx), meas_y_world=float(m.ty),
            is_static=is_static,
            ego_x=frame.ego_x, ego_y=frame.ego_y, ego_yaw=frame.ego_yaw,
            cov_xx=float(tr.cov[S_X, S_X]),
            cov_xy=float(tr.cov[S_X, S_Y]),
            cov_yy=float(tr.cov[S_Y, S_Y]),
            imm_yaw_world=(float(tr.imm.yaw) if tr.imm is not None else None),
        ))
        return tr

    def _update_track(self, tr: Track, m: Measurement, frame: Frame) -> None:
        if tr.imm is not None:
            # IMM update — z=(x, y, yaw_world). CV/CA have a 2-row H, so yaw is
            # ignored automatically. Determine static using the predicted (pre-update)
            # velocity — if static, distrust the measured yaw (R_yaw↑) to suppress
            # tracking of detection yaw jitter. tr.state is the predicted state synced
            # after predict.
            pred_static = tr.velocity_norm < self.p.v_static
            z = np.array([float(m.tx), float(m.ty), float(m.yaw_world)],
                         dtype=np.float64)
            tr.imm.update(z, detection_confidence=float(m.score), static=pred_static)
            # sync with multi_class_tracking external interface (track.state)
            x, y = tr.imm.xy
            vx, vy = tr.imm.vxvy
            tr.state = np.array([x, y, vx, vy], dtype=np.float64)
            cov_xy = tr.imm.cov_xy_block
            new_cov = np.zeros((STATE_DIM, STATE_DIM), dtype=np.float64)
            new_cov[:2, :2] = cov_xy
            from .motion_models import STD_VX, STD_VY
            new_cov[2, 2] = float(tr.imm.mixed_cov_std[STD_VX, STD_VX])
            new_cov[3, 3] = float(tr.imm.mixed_cov_std[STD_VY, STD_VY])
            tr.cov = new_cov
        else:
            z = np.array([m.tx, m.ty], dtype=np.float64)
            tr.state, tr.cov, _y, _S = update(
                tr.state, tr.cov, z, self.p.kf, m.score,
                yaw_hint=float(m.yaw_world),   # for R skew — measurement's world yaw
            )

        # static object — force velocity to 0 (requirement c)
        v = tr.velocity_norm
        is_static = v < self.p.v_static
        if is_static:
            tr.state[S_VX] = 0.0
            tr.state[S_VY] = 0.0
            if tr.imm is not None:
                tr.imm.set_velocity_zero()

        # update last_yaw — used as yaw_hint for next frame's predict (world frame)
        tr.last_yaw = float(m.yaw) + float(frame.ego_yaw)

        # debug — record the update step
        debug_uuid = self._uuid_for_debug.get(tr.track_id, str(tr.track_id))
        if tr.imm is not None:
            self.debug_recorder.record_imm(debug_uuid, frame.timestamp_ns,
                                           "update", tr.imm)
            self.smooth_recorder.record(debug_uuid, frame.timestamp_ns, "update", tr.imm)
        else:
            self.debug_recorder.record_cv_4d(debug_uuid, frame.timestamp_ns,
                                             "update", tr.state, tr.cov, tr.last_yaw)

        # class EMA
        tr.update_class_scores(m.category, self.p.class_alpha)

        # lifecycle bookkeeping
        tr.age += 1
        tr.push_detection(True, self.p.max_history)
        tr.is_associated_this_frame = True
        self._maybe_confirm(tr)

        # frame record (store ego pose too)
        tr.frame_records.append(FrameRecord(
            timestamp_ns=frame.timestamp_ns,
            kf_x=float(tr.state[S_X]), kf_y=float(tr.state[S_Y]),
            kf_vx=float(tr.state[S_VX]), kf_vy=float(tr.state[S_VY]),
            z_meas=m.tz, yaw_meas=m.yaw,
            length_meas=m.length, width_meas=m.width, height_meas=m.height,
            score_meas=m.score, category_meas=m.category,
            meas_x_world=float(m.tx), meas_y_world=float(m.ty),
            is_static=is_static,
            ego_x=frame.ego_x, ego_y=frame.ego_y, ego_yaw=frame.ego_yaw,
            cov_xx=float(tr.cov[S_X, S_X]),
            cov_xy=float(tr.cov[S_X, S_Y]),
            cov_yy=float(tr.cov[S_Y, S_Y]),
            imm_yaw_world=(float(tr.imm.yaw) if tr.imm is not None else None),
        ))

    def _maybe_confirm(self, tr: Track) -> None:
        if tr.is_confirmed:
            return
        if tr.age >= self.p.confirmed_age_min and \
                tr.count_recent_detections() >= self.p.confirmed_detect_min:
            tr.is_confirmed = True

    def _is_outdated(self, tr: Track) -> bool:
        """Same as C++ TrackStruct::isOutdated.

        Expires if (recent detection count) < min(age, max_history) − max_history_for_outdated.

        Static classes (BOLLARD/CONE/BARREL/SIGN, etc.) use the larger
        static_max_history_for_outdated to lower the threshold → keep tracks alive
        longer even through long occlusion/missed detections.
        """
        mho = self.p.max_history_for_outdated
        if tr.get_rep_class() in STATIC_CATEGORIES:
            mho = self.p.static_max_history_for_outdated
        det_count = tr.count_recent_detections()
        threshold = min(tr.age, self.p.max_history) - mho
        return det_count < threshold

    # ── results ───────────────────────────────────────────────
    def all_tracks(self) -> list[Track]:
        """Return active + completed combined (recommended to call after finalize)."""
        return list(self.completed_tracks) + list(self.active_tracks)
