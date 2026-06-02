"""Post-processing pipeline (smoothing + confirmed-run dilation).

For each log, applies two independent operations on the 2Hz tracker-aligned
annotation timeline. The annotation has two parts:
  - per-camera (infra/weather/time_of_day): smoothing+dilation per (camera, item)
  - ego (15 items, no camera dimension): smoothing+dilation per (item)

Operations:
  1. Symmetric majority-vote smoothing — recovers isolated FNs, removes short
     FP bursts.
  2. Confirmed-run dilation — extends every True run of length >= min_run by
     `dilation_step` frames on each side, compensating VLM's conservative
     entry/exit boundaries.

Inputs:  output/layer1_context/<split>/<log_id>/<ts>.json
Outputs: output/layer1_context/<split>_processed/<log_id>/<ts>.json
Originals are preserved.
"""

import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

from tools.layer1_context.src.smoothing import (
    dilate_confirmed_runs,
    majority_vote_smoothing,
)
from tools.layer1_context.src.schema import CAMERA_NAMES, CONTEXT_SCHEMA

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def postprocess_log(
    log_id: str,
    input_split_dir: Path,
    output_split_dir: Path,
    mv_window_size: int = 3,
    mv_threshold: float = 0.5,
    dilation_min_run: int = 3,
    dilation_step: int = 1,
) -> dict:
    """Run the smoothing + dilation pipeline on a single log.

    Returns a summary dict: {"log_id", "num_timestamps", "status"}.
    """
    raw_dir = Path(input_split_dir) / log_id
    out_dir = Path(output_split_dir) / log_id

    if not raw_dir.exists():
        return {"log_id": log_id, "num_timestamps": 0, "status": "missing_raw"}

    annotations = _load_log_annotations(raw_dir)
    if not annotations:
        return {"log_id": log_id, "num_timestamps": 0, "status": "empty"}

    per_camera_items = _flatten_per_camera_items()
    ego_items = list(CONTEXT_SCHEMA["ego"])

    # --- per-camera: (cam x category x item) timelines ---
    for cam in CAMERA_NAMES:
        for category, item in per_camera_items:
            values = [_get_cam_item_value(ts, cam, category, item) for ts in annotations]
            smoothed = majority_vote_smoothing(
                values, window_size=mv_window_size, threshold=mv_threshold
            )
            dilated = dilate_confirmed_runs(
                smoothed, min_run_length=dilation_min_run, dilation=dilation_step
            )
            for ts_data, new_val in zip(annotations, dilated):
                _set_cam_item_value(ts_data, cam, category, item, bool(new_val))

    # --- ego: (item) timelines, no camera dimension ---
    for item in ego_items:
        values = [_get_ego_item_value(ts, item) for ts in annotations]
        smoothed = majority_vote_smoothing(
            values, window_size=mv_window_size, threshold=mv_threshold
        )
        dilated = dilate_confirmed_runs(
            smoothed, min_run_length=dilation_min_run, dilation=dilation_step
        )
        for ts_data, new_val in zip(annotations, dilated):
            _set_ego_item_value(ts_data, item, bool(new_val))

    _save_log_annotations(out_dir, annotations)
    return {"log_id": log_id, "num_timestamps": len(annotations), "status": "ok"}


def postprocess_split(
    input_split_dir: Path,
    output_split_dir: Path,
    mv_window_size: int = 3,
    mv_threshold: float = 0.5,
    dilation_min_run: int = 3,
    dilation_step: int = 1,
    log_ids: List[str] = None,
    num_workers: int = 1,
) -> List[dict]:
    """Post-process all logs under `input_split_dir` (or the given subset)."""
    input_split_dir = Path(input_split_dir)
    output_split_dir = Path(output_split_dir)
    targets = log_ids if log_ids else _discover_logs(input_split_dir)
    if not targets:
        logger.warning(f"No logs found under {input_split_dir}")
        return []

    print(
        f"Post-processing {input_split_dir.name} → {output_split_dir.name}: "
        f"{len(targets)} logs, window_size={mv_window_size}, "
        f"threshold={mv_threshold}, dilation_min_run={dilation_min_run}, "
        f"dilation_step={dilation_step}, workers={num_workers}"
    )

    kwargs = dict(
        mv_window_size=mv_window_size,
        mv_threshold=mv_threshold,
        dilation_min_run=dilation_min_run,
        dilation_step=dilation_step,
    )
    results: List[dict] = []

    # Use ProcessPoolExecutor for parallel processing of logs
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = {
            ex.submit(postprocess_log, log_id, input_split_dir, output_split_dir, **kwargs): log_id
            for log_id in targets
        }
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            results.append(r)
            print(f"  [{i}/{len(targets)}] {r['log_id']}: {r['num_timestamps']} ts ({r['status']})")

    ok = sum(1 for r in results if r["status"] == "ok")
    print(f"Done: {ok}/{len(results)} logs processed successfully.")
    return results


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

_PER_CAMERA_CATEGORIES = ("infra", "weather", "time_of_day")


def _flatten_per_camera_items() -> List[Tuple[str, str]]:
    """Flatten the per-camera categories into (category, item) pairs."""
    return [
        (category, item)
        for category in _PER_CAMERA_CATEGORIES
        for item in CONTEXT_SCHEMA[category]
    ]


def _discover_logs(split_dir: Path) -> List[str]:
    if not split_dir.exists():
        return []
    return sorted(p.name for p in split_dir.iterdir() if p.is_dir())


def _load_log_annotations(log_dir: Path) -> List[dict]:
    """Load per-timestamp JSONs under `log_dir`, sorted by timestamp."""
    files = sorted(log_dir.glob("*.json"), key=lambda p: int(p.stem))
    annotations: List[dict] = []
    for f in files:
        try:
            with open(f, "r") as fp:
                annotations.append(json.load(fp))
        except Exception as e:
            logger.warning(f"Failed to load {f}: {e}")
    return annotations


def _save_log_annotations(out_dir: Path, annotations: List[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ts_data in annotations:
        ts = ts_data["timestamp_ns"]
        with open(out_dir / f"{ts}.json", "w") as fp:
            json.dump(ts_data, fp, indent=2)


def _get_cam_item_value(ts_data: dict, cam: str, category: str, item: str) -> bool:
    return bool(
        ts_data.get("per_camera", {}).get(cam, {}).get(category, {}).get(item, False)
    )


def _set_cam_item_value(ts_data: dict, cam: str, category: str, item: str, value: bool) -> None:
    ts_data.setdefault("per_camera", {}).setdefault(cam, {}).setdefault(category, {})[
        item
    ] = value


def _get_ego_item_value(ts_data: dict, item: str) -> bool:
    return bool(ts_data.get("ego", {}).get(item, False))


def _set_ego_item_value(ts_data: dict, item: str, value: bool) -> None:
    ts_data.setdefault("ego", {})[item] = value
