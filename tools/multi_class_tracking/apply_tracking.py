"""RefAV rule-based re-tracking — per-log CLI.

Preserves the detection/class of the existing Le3DE2E sm_annotations.feather and
replaces only track_uuid and the KF post-processed (tx, ty, yaw, l, w, h, s, category).

Usage examples
--------------
# single log from config.yaml
python -m tools.multi_class_tracking.apply_tracking

# option override
python -m tools.multi_class_tracking.apply_tracking \\
    --src_tracker Le3DE2E_Tracking_ego \\
    --dst_tracker Le3DE2E_Tracking_ego_retrack \\
    --logs 0b5142c1-420b-3fea-9e98-b87327ae22c6 \\
    --split val --workers 1

# entire split
python -m tools.multi_class_tracking.apply_tracking --workers 4
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.multi_class_tracking.src import (   # noqa: E402
    MultiClassTracker, TrackerParams, PostProcessParams,
    load_frames, save_output_feather,
    build_output_df, summarize_tracks, get_log_dir, get_dst_log_dir,
)


CONFIG_PATH = PROJECT_ROOT / "tools" / "multi_class_tracking" / "config" / "config.yaml"


def _load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def _params_from_config(cfg: dict) -> tuple[TrackerParams, PostProcessParams]:
    return (
        TrackerParams.from_config(cfg.get("tracking", {}) or {}),
        PostProcessParams.from_config(cfg),
    )


def process_log(args_tuple) -> dict:
    """Process a single log. tuple argument for ProcessPool serializability."""
    log_id, split, src_tracker, dst_tracker, force = args_tuple

    src_dir = get_log_dir(log_id, split, src_tracker)
    dst_dir = get_dst_log_dir(log_id, split, dst_tracker)
    fea_in = src_dir / "sm_annotations.feather"
    fea_out = dst_dir / "sm_annotations.feather"

    if not fea_in.exists():
        return {"log_id": log_id, "status": "missing_input"}
    if fea_out.exists() and not force:
        return {"log_id": log_id, "status": "skipped_existing"}

    try:
        import pandas as pd

        cfg = _load_config()
        tp, pp = _params_from_config(cfg)
        score_thr = float(cfg.get("tracking", {}).get("detection_score_threshold", 0.0))

        # 1) load frames + EGO pass-through (includes score threshold pre-filtering)
        # EGO_VEHICLE is excluded from frames and not handled by the tracker — ego_df preserved as is.
        frames, ego_df = load_frames(log_id, split, src_tracker, score_threshold=score_thr)
        if not frames and len(ego_df) == 0:
            return {"log_id": log_id, "status": "empty_input"}

        # 2) run forward tracking (only EGO-excluded frames)
        tracker = MultiClassTracker(tp)
        for fr in frames:
            tracker.step(fr)
        tracker.finalize()

        # 3) post-process — tracker output df + uuid map (track_id → new_uuid)
        all_tracks = tracker.all_tracks()
        # interpolation must fill only actual log frame timestamps (fake ts → eval KeyError).
        # union of all timestamps in frames + ego_df = the log's actual frame set.
        log_ts = set(int(fr.timestamp_ns) for fr in frames)
        if len(ego_df) > 0 and "timestamp_ns" in ego_df.columns:
            log_ts |= set(int(t) for t in ego_df["timestamp_ns"].to_numpy())
        # map for STOP_SIGN yaw → lane heading correction. Loaded only if enabled (AV2 original map dir).
        avm = None
        if pp.stop_sign_yaw_to_lane:
            try:
                import refAV.paths as _paths
                from refAV.utils import get_map as _get_map
                _map_log_dir = PROJECT_ROOT / _paths.AV2_DATA_DIR / split / log_id
                avm = _get_map(_map_log_dir)
            except Exception as _e:
                print(f"  [warn] {log_id[:8]} map load failed (skip stop_sign yaw correction): {_e}")
                avm = None
        df_tracked, uuid_map = build_output_df(
            all_tracks, pp, log_timestamps=sorted(log_ts), avm=avm)

        # 4) EGO pass-through concat — merge src's EGO rows into the output as is.
        # column alignment: union of tracked and ego, missing columns are NaN. Keep src format such as uuid 'ego'.
        # NaN cov produces a 'NaN' token in the viewer JSON response that the browser rejects → patch with 0.0.
        if len(ego_df) > 0:
            for c in df_tracked.columns:
                if c not in ego_df.columns:
                    ego_df[c] = 0.0
            for c in ("cov_xx_world", "cov_xy_world", "cov_yy_world"):
                if c in ego_df.columns:
                    ego_df[c] = ego_df[c].fillna(0.0)
            df_out = pd.concat([df_tracked, ego_df], ignore_index=True, sort=False)
            df_out = df_out.sort_values(["timestamp_ns", "track_uuid"]).reset_index(drop=True)
        else:
            df_out = df_tracked

        # 5) save
        out_path = save_output_feather(df_out, log_id, split, dst_tracker)

        # 5b) imm_debug.feather — remap DebugRecorder's rows from the track_id-based
        # temporary uuid → the new_uuid assigned by build_output_df, then save. Skip if disabled.
        if tracker.debug_recorder.enabled and tracker.debug_recorder.rows:
            import pandas as _pd
            dbg_df = tracker.debug_recorder.to_dataframe()
            # track_id (str) → new_uuid mapping. uuid_map's key is int track_id, value is str.
            # tracker's _uuid_for_debug is int → str(int), so convert to str keys identically.
            tid_to_new = {str(int(tid)): u for tid, u in uuid_map.items()}
            dbg_df["track_uuid"] = dbg_df["track_uuid"].map(
                lambda s: tid_to_new.get(str(s), str(s))
            )
            # tracks not in uuid_map (unconfirmed, etc.) — keep the temporary id as is (can be
            # ignored by the server if needed).
            dbg_df.reset_index(drop=True).to_feather(
                dst_dir / "imm_debug.feather"
            )

        # 5c) imm_smooth_inputs.feather — IMM RTS smoothing sidecar (per-model prior/posterior
        # native state·cov·F·μ·Λ). Same as debug: remap track_id → new_uuid then save.
        if tracker.smooth_recorder.enabled and tracker.smooth_recorder.rows:
            import pandas as _pd
            sm_df = tracker.smooth_recorder.to_dataframe()
            tid_to_new = {str(int(tid)): u for tid, u in uuid_map.items()}
            sm_df["track_uuid"] = sm_df["track_uuid"].map(
                lambda s: tid_to_new.get(str(s), str(s))
            )
            sm_df.reset_index(drop=True).to_feather(
                dst_dir / "imm_smooth_inputs.feather"
            )

        # 6) summary json (includes per-track section — matched via uuid_map).
        # If an IMM summary exists, merge into per_track[new_uuid].imm.
        imm_summary = tracker.debug_recorder.summarize_per_track() \
            if tracker.debug_recorder.enabled else None
        # tracker._uuid_for_debug: int track_id → str debug uuid.
        debug_uuid_for_tid = dict(tracker._uuid_for_debug)
        summary = summarize_tracks(
            all_tracks, uuid_map=uuid_map,
            imm_summary_by_debug_uuid=imm_summary,
            debug_uuid_for_track_id=debug_uuid_for_tid,
        )
        summary["n_input_frames"] = len(frames)
        summary["n_output_rows"] = len(df_out)
        summary["n_ego_rows_passthrough"] = len(ego_df)
        summary["motion_model"] = tp.motion_model
        if imm_summary is not None:
            summary["n_imm_debug_rows"] = len(tracker.debug_recorder.rows)
        with open(dst_dir / "tracking_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        return {
            "log_id": log_id,
            "status": "ok",
            "n_input_frames": len(frames),
            "n_input_rows": sum(len(f.detections) for f in frames),
            "n_output_rows": len(df_out),
            "n_ego_passthrough": len(ego_df),
            "n_tracks_total": summary["n_tracks_total"],
            "n_tracks_confirmed": summary["n_tracks_confirmed"],
            "out_path": str(out_path),
        }

    except Exception as e:
        return {
            "log_id": log_id, "status": "error",
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }


def _print_log_result(r: dict, prefix: str = ""):
    if r["status"] == "ok":
        print(f"{prefix}{r['log_id'][:8]}  in={r['n_input_rows']} out={r['n_output_rows']}  "
              f"tracks={r['n_tracks_confirmed']}/{r['n_tracks_total']} (conf/total)")
    elif r["status"] == "skipped_existing":
        print(f"{prefix}{r['log_id'][:8]}  [skipped: dst exists]")
    elif r["status"] == "missing_input":
        print(f"{prefix}{r['log_id'][:8]}  [missing input]")
    elif r["status"] == "empty_input":
        print(f"{prefix}{r['log_id'][:8]}  [empty input]")
    elif r["status"] == "error":
        print(f"{prefix}{r['log_id'][:8]}  [ERROR] {r['error']}")


def main():
    cfg = _load_config()

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src_tracker", default=cfg.get("src_tracker", "Le3DE2E_Tracking_ego"))
    p.add_argument("--dst_tracker", default=None,
                   help="default: cfg.dst_tracker or <src>_retrack")
    p.add_argument("--split", default=cfg.get("split", "val"))
    p.add_argument("--logs", nargs="*", default=None,
                   help="list of log_ids to process. Default is the single log_id in config.yaml.")
    p.add_argument("--all", action="store_true",
                   help="process all logs in the split (ignore config.log_id)")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--force", action="store_true",
                   help="overwrite even if dst feather already exists")
    args = p.parse_args()

    dst_tracker = args.dst_tracker or cfg.get("dst_tracker") or f"{args.src_tracker}_retrack"

    src_split = (PROJECT_ROOT / "output" / "tracker_predictions"
                 / args.src_tracker / args.split)
    if not src_split.exists():
        raise FileNotFoundError(f"src split dir not found: {src_split}")

    if args.logs:
        log_ids = list(args.logs)
    elif args.all:
        log_ids = sorted([p.name for p in src_split.iterdir() if p.is_dir()])
    else:
        single = cfg.get("log_id")
        if not single:
            raise ValueError("no config.log_id and no --logs / --all")
        log_ids = [single]

    print("[apply_tracking]")
    print(f"  src_tracker : {args.src_tracker}")
    print(f"  dst_tracker : {dst_tracker}")
    print(f"  split       : {args.split}")
    print(f"  logs        : {len(log_ids)} ({log_ids[0][:8]}{'...' if len(log_ids) > 1 else ''})")
    print(f"  workers     : {args.workers}")
    print(f"  force       : {args.force}")
    print(f"  cost_metric : {cfg.get('tracking', {}).get('cost_metric', 'maha')}")
    print(f"  match_algo  : {cfg.get('tracking', {}).get('match_algorithm', 'greedy')}")
    print(f"  max_dist    : {cfg.get('tracking', {}).get('max_association_dist_m', 3.0)} m")
    print(f"  v_static    : {cfg.get('tracking', {}).get('v_static', 0.25)} m/s")
    print(f"  score_thr   : {cfg.get('tracking', {}).get('detection_score_threshold', 0.0)} "
          f"(detection pre-filter)")
    print()

    work = [(log_id, args.split, args.src_tracker, dst_tracker, args.force)
            for log_id in log_ids]

    results = []
    if args.workers <= 1:
        for w in work:
            r = process_log(w)
            results.append(r)
            _print_log_result(r)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            fut_to_log = {ex.submit(process_log, w): w[0] for w in work}
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
    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_err = sum(1 for r in results if r["status"] == "error")
    n_skip = sum(1 for r in results if r["status"] == "skipped_existing")
    n_miss = sum(1 for r in results if r["status"] == "missing_input")
    in_rows = sum(r.get("n_input_rows", 0) for r in results if r["status"] == "ok")
    out_rows = sum(r.get("n_output_rows", 0) for r in results if r["status"] == "ok")
    n_tracks_conf = sum(r.get("n_tracks_confirmed", 0) for r in results if r["status"] == "ok")

    print(f"  logs ok={n_ok}, skipped={n_skip}, missing={n_miss}, error={n_err}")
    print(f"  total input rows : {in_rows}")
    print(f"  total output rows: {out_rows}  (drop = {in_rows - out_rows})")
    print(f"  total confirmed tracks: {n_tracks_conf}")

    errs = [r for r in results if r["status"] == "error"]
    if errs:
        print()
        print(f"=== {len(errs)} ERRORS ===")
        for r in errs[:5]:
            print(f"  {r['log_id']}: {r['error']}")


if __name__ == "__main__":
    main()
