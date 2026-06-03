"""Validate v4.x context annotation JSONs against CONTEXT_SCHEMA.

Walks a log dir (raw or processed) and checks every per-timestamp JSON has:
  - top-level keys = {log_id, timestamp_ns, per_camera, ego}
  - per_camera has all 7 ring cameras
  - each per_camera entry has infra/weather/time_of_day with the exact key sets
  - ego has the exact 15 key set, all bool

Usage:
    PYTHONPATH=. python -m tools.scene_context_extraction.validate <log_dir>
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.scene_context_extraction.src.schema import CAMERA_NAMES, CONTEXT_SCHEMA


_PER_CAMERA_CATEGORIES = ("infra", "weather", "time_of_day")


def _check_one(data: dict) -> list[str]:
    errs: list[str] = []
    top = set(data.keys())
    expected_top = {"log_id", "timestamp_ns", "per_camera", "ego"}
    missing = expected_top - top
    if missing:
        errs.append(f"missing top-level keys: {sorted(missing)}")

    per_cam = data.get("per_camera", {})
    if set(per_cam.keys()) != set(CAMERA_NAMES):
        errs.append(
            f"per_camera cams mismatch: got={sorted(per_cam.keys())}"
            f" expected={sorted(CAMERA_NAMES)}"
        )
    for cam in CAMERA_NAMES:
        block = per_cam.get(cam, {})
        for cat in _PER_CAMERA_CATEGORIES:
            expected = set(CONTEXT_SCHEMA[cat])
            actual = set(block.get(cat, {}).keys())
            if actual != expected:
                errs.append(
                    f"{cam}.{cat} keys mismatch: extra={sorted(actual - expected)}"
                    f" missing={sorted(expected - actual)}"
                )
            for k, v in block.get(cat, {}).items():
                if not isinstance(v, bool):
                    errs.append(f"{cam}.{cat}.{k} is not bool: {type(v).__name__}")

    ego = data.get("ego", {})
    expected_ego = set(CONTEXT_SCHEMA["ego"])
    if set(ego.keys()) != expected_ego:
        errs.append(
            f"ego keys mismatch: extra={sorted(set(ego.keys()) - expected_ego)}"
            f" missing={sorted(expected_ego - set(ego.keys()))}"
        )
    for k, v in ego.items():
        if not isinstance(v, bool):
            errs.append(f"ego.{k} is not bool: {type(v).__name__}")
    return errs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", type=Path, help="log dir or split dir of *.json files")
    ap.add_argument("--max-errors", type=int, default=10)
    args = ap.parse_args()

    files = sorted(args.path.rglob("*.json"))
    if not files:
        print(f"[FAIL] no *.json under {args.path}")
        sys.exit(1)

    bad = 0
    shown = 0
    for f in files:
        try:
            data = json.loads(f.read_text())
        except json.JSONDecodeError as e:
            bad += 1
            if shown < args.max_errors:
                print(f"[FAIL] {f}: invalid JSON ({e})")
                shown += 1
            continue
        errs = _check_one(data)
        if errs:
            bad += 1
            if shown < args.max_errors:
                print(f"[FAIL] {f}")
                for e in errs:
                    print(f"       - {e}")
                shown += 1

    total = len(files)
    ok = total - bad
    print(f"\n{'PASS' if bad == 0 else 'FAIL'}: {ok}/{total} files valid")
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
