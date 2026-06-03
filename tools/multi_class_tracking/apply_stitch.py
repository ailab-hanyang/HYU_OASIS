"""Apply heading-anisotropic tracklet stitching (SOTA tracking post-process) to a tracker.

Reads the src tracker's sm_annotations.feather (per-frame boxes + track_uuid) and creates
a new tracker (dst) that reconnects only tracklet fragments broken by an occlusion gap to
the same ID. No re-tracking/interpolation.
Algorithm/rationale: see tools/multi_class_tracking/src/stitch.py.

Default parameters = SOTA (sub20 official HOTA-Temporal +0.00526, TBA +0.0118 vs baseline).

Usage:
  python -m tools.multi_class_tracking.apply_stitch \
      --src_tracker Le3DE2E_Tracking_ego_yawfix \
      --dst_tracker Le3DE2E_Tracking_ego_yawfix_stitch \
      --split val [--logs <id> ...] [--workers 10]

Afterwards evaluate dst_tracker with run/run_experiment.py → judge with 'HOTA-Temporal'.
"""
from __future__ import annotations
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tools.multi_class_tracking.src.stitch import stitch_log, SOTA, TR


def _all_logs(src_tracker, split):
    d = TR / src_tracker / split
    return sorted(p.name for p in d.iterdir() if p.is_dir()) if d.exists() else []


def main():
    ap = argparse.ArgumentParser(description="SOTA tracklet stitching (heading-anisotropic).")
    ap.add_argument("--src_tracker", required=True)
    ap.add_argument("--dst_tracker", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--logs", nargs="*", default=None, help="default: all logs of the src tracker")
    ap.add_argument("--workers", type=int, default=10)
    # parameters (default = SOTA). For isotropic, use --max_lat 0
    ap.add_argument("--max_gap", type=int, default=SOTA["max_gap"])
    ap.add_argument("--max_dist", type=float, default=SOTA["max_dist_m"])
    ap.add_argument("--max_lat", type=float, default=SOTA["max_lat_m"])
    ap.add_argument("--max_lon", type=float, default=SOTA["max_lon_m"])
    ap.add_argument("--min_len", type=int, default=SOTA["min_anchor_len"])
    a = ap.parse_args()

    logs = a.logs or _all_logs(a.src_tracker, a.split)
    if not logs:
        raise SystemExit(f"no logs under {TR/a.src_tracker/a.split}")
    print(f"[stitch] {a.src_tracker} -> {a.dst_tracker} | {len(logs)} logs | "
          f"gap<={a.max_gap} lat<={a.max_lat} lon<={a.max_lon} (SOTA={a.max_lat>0})", flush=True)

    tot = 0; done = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(stitch_log, lg, a.src_tracker, a.dst_tracker, a.split,
                          a.max_gap, a.max_dist, SOTA["max_dist_static_m"], SOTA["size_ratio"],
                          a.max_lat, a.max_lon, a.min_len, a.min_len) for lg in logs]
        for f in futs:
            r = f.result(); tot += r["n_merge"]; done += 1
            if done % 25 == 0: print(f"  {done}/{len(logs)} logs, merges={tot}", flush=True)
    print(f"[stitch] DONE {done} logs, total tracklet merges = {tot}", flush=True)


if __name__ == "__main__":
    main()
