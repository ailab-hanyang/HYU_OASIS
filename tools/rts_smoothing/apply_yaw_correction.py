"""Apply stage-1 yaw flip correction to every log's sm_annotations.feather and
create a new tracker prediction directory.

Usage examples
--------------
# default (val, src=Le3DE2E_Tracking_ego, dst=<src>_yawfix)
python -m tools.rts_smoothing.apply_yaw_correction

# explicit options
python -m tools.rts_smoothing.apply_yaw_correction \
    --src_tracker Le3DE2E_Tracking_ego \
    --dst_tracker Le3DE2E_Tracking_ego_yawfix \
    --split val --workers 8

# single log only (for verification)
python -m tools.rts_smoothing.apply_yaw_correction \
    --logs 02678d04-cc9f-3148-9f95-1ba66347dff9 --workers 1

Cache policy
------------
The source directory's cache/ is not copied. The caches of atomic functions
(has_objects_in_relative_direction, etc.) depend on yaw, so they become invalid
when yaw changes. They are rebuilt automatically on the first inference.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.rts_smoothing.src import (   # noqa: E402
    YawCorrectionParams, load_tracks, correct_yaws_stage1,
)


# ────────────────────────────────────────────────────────────────────────
def _params_from_config() -> YawCorrectionParams:
    cfg_path = PROJECT_ROOT / "tools" / "rts_smoothing" / "config" / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    yc = (cfg or {}).get("yaw_correction") or {}
    d = YawCorrectionParams()
    return YawCorrectionParams(
        tau_static=float(yc.get("tau_static", d.tau_static)),
        tau_motion=float(yc.get("tau_motion", d.tau_motion)),
        n_min=int(yc.get("n_min", d.n_min)),
        skip_categories=tuple(yc.get("skip_categories", list(d.skip_categories))),
        static_yaw_stabilize=bool(yc.get("static_yaw_stabilize", d.static_yaw_stabilize)),
    )


def _yaw_z_to_quat_xyzw_scalarfirst(yaw: float):
    """yaw (rad, z-axis only) → (qw, qx, qy, qz). Same as load_tracks' assumption."""
    h = float(yaw) / 2.0
    return math.cos(h), 0.0, 0.0, math.sin(h)


# ────────────────────────────────────────────────────────────────────────
def process_log(args_tuple) -> dict:
    """Process a single log. Tuple arg so it is ProcessPool-serializable."""
    src_root, dst_root, log_id, split, params, force = args_tuple
    src_log = src_root / split / log_id
    dst_log = dst_root / split / log_id
    fea_in = src_log / "sm_annotations.feather"
    fea_out = dst_log / "sm_annotations.feather"

    if not fea_in.exists():
        return {"log_id": log_id, "status": "missing_input", "n_tracks": 0,
                "n_flipped": 0, "by_status": {}}
    if fea_out.exists() and not force:
        return {"log_id": log_id, "status": "skipped_existing", "n_tracks": 0,
                "n_flipped": 0, "by_status": {}}

    try:
        # 1) load tracks (sm_annotations.feather + ego pose)
        tracks = load_tracks(log_id, split=split, src_tracker=src_root.name)

        # 2) apply correction
        new_tracks, summary = correct_yaws_stage1(tracks, params)

        # 3) read original feather, replace quaternion columns
        df = pd.read_feather(fea_in)

        # build (uuid, ts) → (qw, qx, qy, qz)
        new_q: dict[tuple[str, int], tuple[float, float, float, float]] = {}
        for uuid, ts_obj in new_tracks.tracks.items():
            quats = ts_obj.quaternions
            ts_arr = ts_obj.timestamps_ns
            for i in range(len(ts_arr)):
                key = (str(uuid), int(ts_arr[i]))
                if quats is not None:
                    q = quats[i]
                    new_q[key] = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
                else:
                    new_q[key] = _yaw_z_to_quat_xyzw_scalarfirst(float(ts_obj.yaws_rad[i]))

        # vectorized lookup
        uuids = df["track_uuid"].astype(str).to_numpy()
        tss = df["timestamp_ns"].astype(np.int64).to_numpy()
        new_arr = np.empty((len(df), 4), dtype=np.float64)
        for i, (u, t) in enumerate(zip(uuids, tss)):
            q = new_q.get((u, int(t)))
            if q is None:
                # safety net — keep original (should not happen in theory)
                new_arr[i] = (df.at[i, "qw"], df.at[i, "qx"],
                              df.at[i, "qy"], df.at[i, "qz"])
            else:
                new_arr[i] = q

        df["qw"] = new_arr[:, 0]
        df["qx"] = new_arr[:, 1]
        df["qy"] = new_arr[:, 2]
        df["qz"] = new_arr[:, 3]

        # 4) write output
        dst_log.mkdir(parents=True, exist_ok=True)
        df.reset_index(drop=True).to_feather(fea_out)

        # per-log summary json (includes per-track diagnostics)
        with open(dst_log / "yaw_correction_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        n_flipped = sum(int(v) for v in summary["by_status_flipped"].values())
        return {
            "log_id": log_id,
            "status": "ok",
            "n_tracks": int(summary["total"]),
            "n_flipped": int(n_flipped),
            "by_status": summary["by_status"],
        }
    except Exception as e:
        return {
            "log_id": log_id, "status": "error",
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
            "n_tracks": 0, "n_flipped": 0, "by_status": {},
        }


# ────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src_tracker", default="Le3DE2E_Tracking_ego")
    p.add_argument("--dst_tracker", default=None,
                   help="default: <src_tracker>_yawfix")
    p.add_argument("--split", default="val")
    p.add_argument("--logs", nargs="*", default=None,
                   help="List of log_ids to process. If empty, all logs in the split.")
    p.add_argument("--workers", type=int, default=4,
                   help="Number of parallel processes (default 4). 1 means serial.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite feathers already present in dst (default: skip).")
    args = p.parse_args()

    if args.dst_tracker is None:
        args.dst_tracker = f"{args.src_tracker}_yawfix"

    src_root = PROJECT_ROOT / "output" / "tracker_predictions" / args.src_tracker
    dst_root = PROJECT_ROOT / "output" / "tracker_predictions" / args.dst_tracker

    if not src_root.exists():
        raise FileNotFoundError(f"src tracker dir not found: {src_root}")

    src_split = src_root / args.split
    if not src_split.exists():
        raise FileNotFoundError(f"src split dir not found: {src_split}")

    if args.logs:
        log_ids = list(args.logs)
    else:
        log_ids = sorted([p.name for p in src_split.iterdir() if p.is_dir()])

    params = _params_from_config()

    print(f"[apply_yaw_correction]")
    print(f"  src     : {src_root}")
    print(f"  dst     : {dst_root}")
    print(f"  split   : {args.split}")
    print(f"  logs    : {len(log_ids)}")
    print(f"  workers : {args.workers}")
    print(f"  params  : {params}")
    print()

    # process
    work = [(src_root, dst_root, log_id, args.split, params, args.force)
            for log_id in log_ids]

    results = []
    if args.workers <= 1:
        for w in work:
            r = process_log(w)
            results.append(r)
            _print_log_result(r)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            fut_to_log = {ex.submit(process_log, w): w[2] for w in work}
            done = 0
            total = len(fut_to_log)
            for fut in as_completed(fut_to_log):
                r = fut.result()
                results.append(r)
                done += 1
                _print_log_result(r, prefix=f"[{done:>3}/{total}] ")

    # aggregate
    print()
    print("=== Aggregate ===")
    agg_by_status: dict[str, int] = {}
    total_tracks = 0
    total_flipped = 0
    n_ok, n_err, n_skip, n_missing = 0, 0, 0, 0
    for r in results:
        if r["status"] == "ok":
            n_ok += 1
            total_tracks += r["n_tracks"]
            total_flipped += r["n_flipped"]
            for s, c in r["by_status"].items():
                agg_by_status[s] = agg_by_status.get(s, 0) + c
        elif r["status"] == "error":
            n_err += 1
        elif r["status"] == "skipped_existing":
            n_skip += 1
        elif r["status"] == "missing_input":
            n_missing += 1

    print(f"  logs ok: {n_ok}, skipped: {n_skip}, missing: {n_missing}, error: {n_err}")
    print(f"  total tracks processed: {total_tracks}")
    print(f"  total frames flipped:   {total_flipped}")
    print(f"  by_status: {agg_by_status}")

    # save aggregate
    dst_split = dst_root / args.split
    dst_split.mkdir(parents=True, exist_ok=True)
    agg_path = dst_split / "_aggregate_summary.json"
    with open(agg_path, "w") as f:
        json.dump({
            "src_tracker": args.src_tracker,
            "dst_tracker": args.dst_tracker,
            "split": args.split,
            "n_logs": len(results),
            "n_ok": n_ok, "n_skipped": n_skip,
            "n_missing": n_missing, "n_error": n_err,
            "total_tracks": total_tracks,
            "total_flipped_frames": total_flipped,
            "by_status": agg_by_status,
            "params": {
                "tau_static": params.tau_static,
                "tau_motion": params.tau_motion,
                "n_min": params.n_min,
                "skip_categories": list(params.skip_categories),
            },
        }, f, indent=2)
    print(f"  aggregate saved: {agg_path}")

    # error logs
    errs = [r for r in results if r["status"] == "error"]
    if errs:
        print()
        print(f"=== {len(errs)} ERRORS ===")
        for r in errs[:10]:
            print(f"  {r['log_id']}: {r['error']}")


def _print_log_result(r: dict, prefix: str = ""):
    if r["status"] == "ok":
        bs = r["by_status"]
        # short summary
        keys = ["corrected_dynamic", "corrected_dynamic_anchor",
                "skip_static", "skip_short",
                "skip_ego_vehicle", "skip_pedestrian", "skip_no_motion"]
        parts = [f"{k.replace('corrected_', 'cd_').replace('skip_', 'sk_')}={bs[k]}"
                 for k in keys if k in bs]
        print(f"{prefix}{r['log_id'][:8]}  flipped={r['n_flipped']:>4}  "
              f"tracks={r['n_tracks']:>3}  ({', '.join(parts)})")
    elif r["status"] == "skipped_existing":
        print(f"{prefix}{r['log_id'][:8]}  [skipped: dst exists]")
    elif r["status"] == "missing_input":
        print(f"{prefix}{r['log_id'][:8]}  [missing input feather]")
    elif r["status"] == "error":
        print(f"{prefix}{r['log_id'][:8]}  [ERROR] {r['error']}")


if __name__ == "__main__":
    main()
