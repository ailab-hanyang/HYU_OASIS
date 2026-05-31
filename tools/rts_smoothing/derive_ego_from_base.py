"""Create an _ego variant directory from the base tracker feather, changing only the EGO offset.

The only difference between base and _ego is the EGO_VEHICLE row's (tx_m, ty_m, tz_m):
  base : EGO_REAR_AXLE_OFFSET   (0.0, 0.0, 0.0)
  _ego : EGO_BODY_CENTER_OFFSET (1.422, 0.0, 0.25)
All other track boxes are identical, so the cache (color_cache + crops) is symlinked to base.

Normally `refAV.dataset_conversion.pickle_to_feather` generates the _ego variant from the
raw .pkl, but this is provided as a separate entry point so the same result can be
reproduced from the base feather alone, even in environments where the raw pkl is gone.

Usage examples
--------------
# generate the _ego variant for the test split
python -m tools.rts_smoothing.derive_ego_from_base \
    --src_tracker Le3DE2E_Tracking \
    --dst_tracker Le3DE2E_Tracking_ego \
    --split test --workers 8

# single log only (for verification)
python -m tools.rts_smoothing.derive_ego_from_base \
    --logs 0c6e62d7-bdfa-3061-8d3d-03b13aa21f68 --workers 1
"""
from __future__ import annotations

import argparse
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from refAV.dataset_conversion import EGO_BODY_CENTER_OFFSET  # noqa: E402


def process_log(args_tuple) -> dict:
    src_root, dst_root, log_id, split, force = args_tuple
    src_log = src_root / split / log_id
    dst_log = dst_root / split / log_id
    fea_in = src_log / "sm_annotations.feather"
    fea_out = dst_log / "sm_annotations.feather"

    if not fea_in.exists():
        return {"log_id": log_id, "status": "missing_input"}
    if fea_out.exists() and not force:
        return {"log_id": log_id, "status": "skipped_existing"}

    try:
        df = pd.read_feather(fea_in)
        mask = df["track_uuid"].astype(str) == "ego"
        n_ego = int(mask.sum())
        if n_ego == 0:
            # abnormal case where base feather has no ego — copy as-is anyway
            pass
        else:
            df.loc[mask, "tx_m"] = float(EGO_BODY_CENTER_OFFSET[0])
            df.loc[mask, "ty_m"] = float(EGO_BODY_CENTER_OFFSET[1])
            df.loc[mask, "tz_m"] = float(EGO_BODY_CENTER_OFFSET[2])

        dst_log.mkdir(parents=True, exist_ok=True)
        df.reset_index(drop=True).to_feather(fea_out)

        # cache symlink (absolute path) — reuse base's SigLIP color cache + track crops as-is.
        src_cache = src_log / "cache"
        dst_cache = dst_log / "cache"
        cache_status = "no_src_cache"
        if src_cache.exists() and not (dst_cache.exists() or dst_cache.is_symlink()):
            dst_cache.symlink_to(src_cache.resolve())
            cache_status = "linked"
        elif dst_cache.exists() or dst_cache.is_symlink():
            cache_status = "kept_existing"

        return {"log_id": log_id, "status": "ok", "n_ego_rows": n_ego, "cache": cache_status}
    except Exception as e:
        return {
            "log_id": log_id, "status": "error",
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }


def _print(r: dict, prefix: str = ""):
    if r["status"] == "ok":
        print(f"{prefix}{r['log_id'][:8]}  ego_rows={r['n_ego_rows']:>3}  cache={r['cache']}")
    elif r["status"] == "skipped_existing":
        print(f"{prefix}{r['log_id'][:8]}  [skipped: dst exists]")
    elif r["status"] == "missing_input":
        print(f"{prefix}{r['log_id'][:8]}  [missing input feather]")
    elif r["status"] == "error":
        print(f"{prefix}{r['log_id'][:8]}  [ERROR] {r['error']}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src_tracker", default="Le3DE2E_Tracking")
    p.add_argument("--dst_tracker", default=None,
                   help="default: <src_tracker>_ego")
    p.add_argument("--split", default="test")
    p.add_argument("--logs", nargs="*", default=None,
                   help="List of log_ids to process. If empty, all logs in the split.")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if args.dst_tracker is None:
        args.dst_tracker = f"{args.src_tracker}_ego"

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

    print(f"[derive_ego_from_base]")
    print(f"  src     : {src_root}")
    print(f"  dst     : {dst_root}")
    print(f"  split   : {args.split}")
    print(f"  logs    : {len(log_ids)}")
    print(f"  workers : {args.workers}")
    print(f"  ego off : {EGO_BODY_CENTER_OFFSET}")
    print()

    work = [(src_root, dst_root, lid, args.split, args.force) for lid in log_ids]
    results: list[dict] = []
    if args.workers <= 1:
        for w in work:
            r = process_log(w)
            results.append(r)
            _print(r)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_log, w): w[2] for w in work}
            done, total = 0, len(futs)
            for fut in as_completed(futs):
                r = fut.result()
                results.append(r)
                done += 1
                _print(r, prefix=f"[{done:>3}/{total}] ")

    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_skip = sum(1 for r in results if r["status"] == "skipped_existing")
    n_miss = sum(1 for r in results if r["status"] == "missing_input")
    n_err = sum(1 for r in results if r["status"] == "error")
    print()
    print(f"=== Aggregate: ok={n_ok} skipped={n_skip} missing={n_miss} error={n_err} ===")

    if n_err:
        print()
        for r in results:
            if r["status"] == "error":
                print(f"  {r['log_id']}: {r['error']}")


if __name__ == "__main__":
    main()
