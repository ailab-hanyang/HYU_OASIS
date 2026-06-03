"""Annotation pipeline — enumerate logs, run VLM, save JSON per timestamp.

VLM runs at 2Hz on the tracker feather's timestamps so that downstream
atomic-function lookups hit exactly. For each tracker timestamp, the nearest
ring-camera frame is loaded; the output JSON keys/filenames use the tracker
timestamp (not the camera timestamp).
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from tools.scene_context_extraction.src.engine import VLLMAnnotator
from tools.scene_context_extraction.src.schema import CAMERA_NAMES, EGO_CAMERA_NAMES

logger = logging.getLogger(__name__)


def _enumerate_logs(config: Dict[str, Any]) -> List[Tuple[str, str]]:
    data_dir = Path(config["dataset"]["data_dir"])
    splits = config["dataset"]["splits"]
    filter_ids = config["dataset"].get("log_ids")

    logs: List[Tuple[str, str]] = []
    for split in splits:
        split_dir = data_dir / split
        if not split_dir.is_dir():
            logger.warning("Split dir not found: %s", split_dir)
            continue
        for log_path in sorted(split_dir.iterdir()):
            if not log_path.is_dir():
                continue
            log_id = log_path.name
            if filter_ids and log_id not in filter_ids:
                continue
            logs.append((log_id, split))
    return logs


def _camera_timestamps(log_dir: Path) -> Dict[str, np.ndarray]:
    out = {}
    for cam in CAMERA_NAMES:
        cam_dir = log_dir / "sensors" / "cameras" / cam
        files = sorted(cam_dir.glob("*.jpg"))
        out[cam] = np.array([int(f.stem) for f in files], dtype=np.int64)
    return out


def _read_tracker_timestamps(tracker_dir: Path, split: str, log_id: str) -> List[int]:
    """Read 2Hz tracker timestamps from sm_annotations.feather."""
    feather_path = tracker_dir / split / log_id / "sm_annotations.feather"
    df = pd.read_feather(feather_path, columns=["timestamp_ns"])
    return sorted(int(t) for t in df["timestamp_ns"].unique())


def _match_closest(timestamps: np.ndarray, target: int) -> int:
    return int(timestamps[np.argmin(np.abs(timestamps - target))])


def _process_log(
    annotator: VLLMAnnotator,
    log_id: str,
    split: str,
    log_index: int,
    total_logs: int,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    data_dir = Path(config["dataset"]["data_dir"])
    tracker_dir = Path(config["dataset"]["tracker_dir"])
    log_dir = data_dir / split / log_id
    output_dir = Path(config["paths"]["output_dir"]) / split / log_id

    feather_path = tracker_dir / split / log_id / "sm_annotations.feather"
    if not feather_path.exists():
        logger.warning("Skipping %s/%s: tracker feather not found at %s", split, log_id, feather_path)
        return {
            "log_id": log_id, "split": split,
            "timestamps": 0, "images": 0, "elapsed": 0.0,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    camera_ts = _camera_timestamps(log_dir)
    empty_cams = [cam for cam, ts in camera_ts.items() if ts.size == 0]
    if empty_cams:
        logger.warning(
            "Skipping %s/%s: no camera frames found for %s",
            split, log_id, ", ".join(empty_cams),
        )
        return {
            "log_id": log_id, "split": split,
            "timestamps": 0, "images": 0, "elapsed": 0.0,
        }
    tracker_timestamps = _read_tracker_timestamps(tracker_dir, split, log_id)

    existing = {int(f.stem) for f in output_dir.glob("*.json")}
    remaining = [ts for ts in tracker_timestamps if ts not in existing]

    if not remaining:
        print(f"[{log_index}/{total_logs}] {split}/{log_id}: all done (skipped)")
        return {
            "log_id": log_id, "split": split,
            "timestamps": len(tracker_timestamps), "images": 0, "elapsed": 0.0,
        }

    start = time.time()
    # batch_ts: timestamps per llm.chat() call. Defaults to whole log
    # (vLLM handles concurrency internally).
    batch_ts = config["inference"].get("batch_ts") or len(remaining)

    for batch_start in range(0, len(remaining), batch_ts):
        batch = remaining[batch_start : batch_start + batch_ts]

        # (A) Per-camera: 7 single-image conversations per timestamp.
        per_cam_paths: List[Path] = []
        # (ts_idx, cam_name, image_relative_path, matched_camera_ts)
        per_cam_meta: List[Tuple[int, str, str, int]] = []
        # (B) Ego: one multi-image conversation per timestamp, in EGO_CAMERA_NAMES order.
        ego_groups: List[List[Path]] = []
        for ts_idx, tracker_ts in enumerate(batch):
            for cam in CAMERA_NAMES:
                matched_cam_ts = _match_closest(camera_ts[cam], tracker_ts)
                rel = f"sensors/cameras/{cam}/{matched_cam_ts}.jpg"
                per_cam_paths.append(log_dir / rel)
                per_cam_meta.append((ts_idx, cam, rel, matched_cam_ts))
            ego_group = [
                log_dir / f"sensors/cameras/{cam}/{_match_closest(camera_ts[cam], tracker_ts)}.jpg"
                for cam in EGO_CAMERA_NAMES
            ]
            ego_groups.append(ego_group)

        per_cam_results = annotator.annotate_batch(per_cam_paths)
        ego_results = annotator.annotate_ego_batch(ego_groups)

        for ts_idx, tracker_ts in enumerate(batch):
            cameras_result = {}
            for (mi, cam, rel, matched_cam_ts), result in zip(per_cam_meta, per_cam_results):
                if mi == ts_idx:
                    cameras_result[cam] = {
                        "image_path": rel,
                        "camera_timestamp_ns": matched_cam_ts,
                        **result,
                    }
            out_data = {
                "log_id": log_id,
                "timestamp_ns": int(tracker_ts),
                "per_camera": cameras_result,
                "ego": ego_results[ts_idx],
            }
            with open(output_dir / f"{tracker_ts}.json", "w") as f:
                json.dump(out_data, f, indent=2)

        done = min(batch_start + batch_ts, len(remaining))
        imgs = done * (len(CAMERA_NAMES) + len(EGO_CAMERA_NAMES))
        rate = imgs / (time.time() - start)
        print(f"  [{done}/{len(remaining)}] {imgs} imgs, {rate:.1f} img/s")

    elapsed = time.time() - start
    images = len(remaining) * (len(CAMERA_NAMES) + len(EGO_CAMERA_NAMES))
    rate = images / elapsed if elapsed > 0 else 0.0
    print(
        f"[{log_index}/{total_logs}] {split}/{log_id}: "
        f"{len(tracker_timestamps)} frames, {elapsed:.1f}s | {images} imgs, {rate:.1f} img/s"
    )

    return {
        "log_id": log_id, "split": split,
        "timestamps": len(tracker_timestamps), "images": images, "elapsed": elapsed,
    }


def _write_metadata(stats: List[Dict], elapsed: float, config: Dict[str, Any]) -> None:
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model": config["vlm"]["model_path"],
        "thinking_mode": config["inference"].get("enable_thinking", False),
        "tensor_parallel_size": config["vlm"].get("tensor_parallel_size", 4),
        "timestamp_source": "tracker_feather_2hz",
        "tracker_dir": config["dataset"]["tracker_dir"],
        "splits": config["dataset"]["splits"],
        "total_logs": len(stats),
        "total_timestamps": sum(s["timestamps"] for s in stats),
        "total_images": sum(s["images"] for s in stats),
        "total_time_seconds": round(elapsed, 1),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nDone! {len(stats)} logs in {elapsed:.1f}s — metadata saved.")


def build_annotations(config: Dict[str, Any]) -> None:
    logs = _enumerate_logs(config)
    output_dir = Path(config["paths"]["output_dir"])
    total = len(logs)

    print("=" * 60)
    print("  Scene Context vLLM Annotation Pipeline")
    print(f"  Model: {config['vlm']['model_path']}")
    print(f"  Logs: {total}, Splits: {config['dataset']['splits']}")
    print(f"  Timestamps: tracker feather (2Hz)")
    print(f"  Tracker dir: {config['dataset']['tracker_dir']}")
    print(f"  Output: {output_dir}")
    print("=" * 60)
    print()

    if config.get("dry_run", False):
        for i, (log_id, split) in enumerate(logs, 1):
            print(f"  [{i}/{total}] {split}/{log_id}")
        return

    annotator = VLLMAnnotator(config)
    start = time.time()
    stats: List[Dict[str, Any]] = []
    for i, (log_id, split) in enumerate(logs, 1):
        stats.append(_process_log(annotator, log_id, split, i, total, config))
    _write_metadata(stats, time.time() - start, config)
