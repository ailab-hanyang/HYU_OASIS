"""Read sm_annotations.feather + city_SE3_ego.feather and convert to a Tracks object."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from .track_state import TrackState, Tracks

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "tools" / "rts_smoothing" / "config" / "config.yaml"


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    return yaml.safe_load(Path(config_path).read_text())


def get_log_dir(log_id: str, split: str = "val", src_tracker: str = "Le3DE2E_Tracking_ego") -> Path:
    """tracker_predictions/<src_tracker>/<split>/<log_id>/ path."""
    return PROJECT_ROOT / "output" / "tracker_predictions" / src_tracker / split / log_id


def _quat_xyzw_to_yaw_z(qw: np.ndarray, qz: np.ndarray) -> np.ndarray:
    """Assume the ego frame box's yaw is a planar (z) rotation about the rear axle (qx, qy ≈ 0).

    yaw = 2 * atan2(qz, qw) — stable version. (q = [qw, qx, qy, qz], scalar-first)
    """
    return 2.0 * np.arctan2(qz, qw)


def load_tracks(
    log_id: str,
    split: str = "val",
    src_tracker: str = "Le3DE2E_Tracking_ego",
    score_threshold: float = 0.0,
) -> Tracks:
    """Read a single log's sm_annotations.feather and ego pose, return Tracks.

    sm_annotations is in [tracker_predictions/<src_tracker>/<split>/<log>/] and
    the ego pose is in [data/datasets/sensor/<split>/<log>/city_SE3_egovehicle.feather].

    If score_threshold > 0, detections with confidence score at or below the threshold are removed.
    EGO_VEHICLE always passes regardless of score (downstream evaluation compatibility).
    """
    log_dir = get_log_dir(log_id, split, src_tracker)
    fea = log_dir / "sm_annotations.feather"
    if not fea.exists():
        raise FileNotFoundError(f"sm_annotations.feather not found: {fea}")

    df = pd.read_feather(fea)

    # pre score filtering — remove detections at or below threshold (EGO always passes).
    if score_threshold > 0.0 and "score" in df.columns:
        keep_mask = (df["score"] > score_threshold) | (df["category"] == "EGO_VEHICLE")
        df = df[keep_mask].reset_index(drop=True)

    # stable sort by timestamp, then group by uuid
    df = df.sort_values(["track_uuid", "timestamp_ns"]).reset_index(drop=True)

    # box yaw — z-axis only assumption
    yaws = _quat_xyzw_to_yaw_z(df["qw"].to_numpy(), df["qz"].to_numpy())

    has_cov = all(c in df.columns for c in
                  ("cov_xx_world", "cov_xy_world", "cov_yy_world"))

    tracks: dict[str, TrackState] = {}
    for uuid, sub in df.groupby("track_uuid", sort=False):
        idx = sub.index.to_numpy()
        ts_arr = sub["timestamp_ns"].to_numpy(dtype=np.int64)
        order = np.argsort(ts_arr)
        ts_arr = ts_arr[order]
        translations = sub[["tx_m", "ty_m", "tz_m"]].to_numpy(dtype=np.float64)[order]
        sizes = sub[["length_m", "width_m", "height_m"]].to_numpy(dtype=np.float32)[order]
        quats = sub[["qw", "qx", "qy", "qz"]].to_numpy(dtype=np.float64)[order]
        yaws_uuid = yaws[idx][order]
        scores = (sub["score"].to_numpy(dtype=np.float32)[order]
                  if "score" in sub.columns else np.ones(len(sub), dtype=np.float32))
        category = str(sub["category"].iloc[0])
        cov_xy_world = None
        if has_cov:
            cov_xy_world = sub[["cov_xx_world", "cov_xy_world", "cov_yy_world"]].to_numpy(
                dtype=np.float64
            )[order]
        tracks[str(uuid)] = TrackState(
            uuid=str(uuid),
            category=category,
            timestamps_ns=ts_arr,
            translations_m=translations,
            yaws_rad=yaws_uuid,
            sizes_m=sizes,
            scores=scores,
            quaternions=quats,
            cov_xy_world=cov_xy_world,
        )

    # ego poses — directly from the sensor directory
    ego_pose_p = (
        PROJECT_ROOT / "data" / "datasets" / "sensor" / split / log_id
        / "city_SE3_egovehicle.feather"
    )
    if ego_pose_p.exists():
        ep = pd.read_feather(ego_pose_p)
        ts = ep["timestamp_ns"].to_numpy(dtype=np.int64)
        order = np.argsort(ts)
        ts = ts[order]
        xyz = ep[["tx_m", "ty_m", "tz_m"]].to_numpy(dtype=np.float64)[order]
        ego_yaw = _quat_xyzw_to_yaw_z(ep["qw"].to_numpy()[order], ep["qz"].to_numpy()[order])
    else:
        ts = np.zeros(0, dtype=np.int64)
        xyz = np.zeros((0, 3))
        ego_yaw = np.zeros(0)

    return Tracks(
        log_id=log_id,
        split=split,
        tracks=tracks,
        ego_poses_ts=ts,
        ego_poses_xyz=xyz,
        ego_poses_yaw=ego_yaw,
    )
