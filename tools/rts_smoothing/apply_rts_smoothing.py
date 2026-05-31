"""Apply RTS smoothing to every log's sm_annotations.feather and
create a new tracker prediction directory.

State : T = [x, y, z, θ, l, w, h, s, vx, vy, vz]   (11-D)
Meas  : z = [x, y, z, θ, l, w, h, s]               (8-D)

Usage examples
--------------
# default (val, src=Le3DE2E_Tracking_ego_yawfix → dst=<src>_rts)
python -m tools.rts_smoothing.apply_rts_smoothing \\
    --src_tracker Le3DE2E_Tracking_ego_yawfix

# explicit options
python -m tools.rts_smoothing.apply_rts_smoothing \\
    --src_tracker Le3DE2E_Tracking_ego_yawfix \\
    --dst_tracker Le3DE2E_Tracking_ego_yawfix_rts \\
    --split val --workers 8

# single log only
python -m tools.rts_smoothing.apply_rts_smoothing \\
    --src_tracker Le3DE2E_Tracking_ego_yawfix \\
    --logs 02678d04-cc9f-3148-9f95-1ba66347dff9 --workers 1
"""

from __future__ import annotations

import argparse
import json
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
    load_tracks, RTSParams, smooth_all,
)


def _params_from_config() -> RTSParams:
    cfg_path = PROJECT_ROOT / "tools" / "rts_smoothing" / "config" / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    return RTSParams.from_config(cfg)


def _score_threshold_from_config() -> float:
    """Use only rts_smoothing.detection_score_threshold (no effect on yaw_correction)."""
    cfg_path = PROJECT_ROOT / "tools" / "rts_smoothing" / "config" / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    rts = (cfg or {}).get("rts_smoothing") or {}
    return float(rts.get("detection_score_threshold", 0.0))


def process_log(args_tuple) -> dict:
    src_root, dst_root, log_id, split, params, score_thr, force = args_tuple
    src_log = src_root / split / log_id
    dst_log = dst_root / split / log_id
    fea_in = src_log / "sm_annotations.feather"
    fea_out = dst_log / "sm_annotations.feather"

    if not fea_in.exists():
        return {"log_id": log_id, "status": "missing_input",
                "n_tracks": 0, "by_status": {}}
    if fea_out.exists() and not force:
        return {"log_id": log_id, "status": "skipped_existing",
                "n_tracks": 0, "by_status": {}}

    try:
        # 1) load (pre-filter by score threshold — EGO always passes)
        tracks = load_tracks(log_id, split=split, src_tracker=src_root.name,
                             score_threshold=score_thr)

        # 2) RTS smoothing
        new_map, summary = smooth_all(tracks, params)

        # 3) replace df columns — (uuid, ts) lookup.
        # Apply the same score filter to the output feather (drop low-confidence rows, EGO excepted).
        df = pd.read_feather(fea_in)
        if score_thr > 0.0 and "score" in df.columns:
            n_before = len(df)
            keep = (df["score"] > score_thr) | (df["category"] == "EGO_VEHICLE")
            df = df[keep].reset_index(drop=True)
            n_filtered = n_before - len(df)
        else:
            n_filtered = 0

        # for yaw_lock_to_track — preserve pre-smoothing tracking-output ego yaw
        df["track_out_yaw_ego"] = np.arctan2(2.0 * df["qz"] * df["qw"],
                                             1.0 - 2.0 * df["qz"] * df["qz"])

        replace = {}
        # key: (uuid, ts) → tuple(tx, ty, tz, qw, qx, qy, qz, l, w, h, score)
        for uuid, ts_obj in new_map.items():
            ts_arr = ts_obj.timestamps_ns
            xyz = ts_obj.translations_m
            quats = ts_obj.quaternions
            sizes = ts_obj.sizes_m
            scores = ts_obj.scores
            for i in range(len(ts_arr)):
                key = (str(uuid), int(ts_arr[i]))
                q = quats[i] if quats is not None else (1.0, 0.0, 0.0, 0.0)
                replace[key] = (
                    float(xyz[i, 0]), float(xyz[i, 1]), float(xyz[i, 2]),
                    float(q[0]), float(q[1]), float(q[2]), float(q[3]),
                    float(sizes[i, 0]), float(sizes[i, 1]), float(sizes[i, 2]),
                    float(scores[i]),
                )

        uuids = df["track_uuid"].astype(str).to_numpy()
        tss = df["timestamp_ns"].astype(np.int64).to_numpy()
        # column-by-column replacement
        new_tx = df["tx_m"].to_numpy(dtype=np.float64).copy()
        new_ty = df["ty_m"].to_numpy(dtype=np.float64).copy()
        new_tz = df["tz_m"].to_numpy(dtype=np.float64).copy()
        new_qw = df["qw"].to_numpy(dtype=np.float64).copy()
        new_qx = df["qx"].to_numpy(dtype=np.float64).copy()
        new_qy = df["qy"].to_numpy(dtype=np.float64).copy()
        new_qz = df["qz"].to_numpy(dtype=np.float64).copy()
        new_l = df["length_m"].to_numpy(dtype=np.float32).copy()
        new_w = df["width_m"].to_numpy(dtype=np.float32).copy()
        new_h = df["height_m"].to_numpy(dtype=np.float32).copy()
        if "score" in df.columns:
            new_s = df["score"].to_numpy(dtype=np.float32).copy()
        else:
            new_s = None

        for i, (u, t) in enumerate(zip(uuids, tss)):
            v = replace.get((u, int(t)))
            if v is None:
                continue
            new_tx[i], new_ty[i], new_tz[i] = v[0], v[1], v[2]
            new_qw[i], new_qx[i], new_qy[i], new_qz[i] = v[3], v[4], v[5], v[6]
            new_l[i], new_w[i], new_h[i] = v[7], v[8], v[9]
            if new_s is not None:
                new_s[i] = v[10]

        df["tx_m"] = new_tx
        df["ty_m"] = new_ty
        df["tz_m"] = new_tz
        df["qw"] = new_qw
        df["qx"] = new_qx
        df["qy"] = new_qy
        df["qz"] = new_qz
        df["length_m"] = new_l
        df["width_m"] = new_w
        df["height_m"] = new_h
        if new_s is not None:
            df["score"] = new_s

        # 3.5) reapply yaw post-processing — reapply tracking's yaw post-processing
        # (stop_sign_yaw_to_lane / jitter_freeze / dynamic_velocity_yaw) onto the yaw the smoother overwrote, to unify.
        try:
            from tools.rts_smoothing.src.yaw_postproc import (
                load_pp_and_avm, reapply_yaw_postproc)
            _en, _pp, _avm, _syjh, _ylt = load_pp_and_avm(log_id, split)
            if _en and _pp is not None:
                df = reapply_yaw_postproc(df, tracks.ego_poses_ts,
                                          tracks.ego_poses_yaw, _avm, _pp,
                                          syjh=_syjh, ylt=_ylt)
        except Exception as _e:
            print(f"  [warn] {log_id[:8]} yaw_postproc skip: {_e}")

        # 4) write
        dst_log.mkdir(parents=True, exist_ok=True)
        df.reset_index(drop=True).to_feather(fea_out)

        # per-log summary
        with open(dst_log / "rts_smoothing_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        return {
            "log_id": log_id, "status": "ok",
            "n_tracks": int(summary["total"]),
            "by_status": summary["by_status"],
        }
    except Exception as e:
        return {
            "log_id": log_id, "status": "error",
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
            "n_tracks": 0, "by_status": {},
        }


def _yaw_to_quat(yaw: float):
    """yaw (z-axis) → (qw, qx, qy, qz), qw≥0."""
    half = float(yaw) * 0.5
    qw = float(np.cos(half)); qz = float(np.sin(half))
    if qw < 0.0:
        qw, qz = -qw, -qz
    return qw, 0.0, 0.0, qz


def process_log_imm(args_tuple) -> dict:
    """IMM smoother — per-model RTS from imm_smooth_inputs.feather (sidecar) + ego pose →
    smoothed world (x,y,yaw) → ego frame → replace tx_m/ty_m/qw/qz in sm_annotations.
    Smooth position/yaw only (keep original tz/lwh/score)."""
    src_root, dst_root, log_id, split, imm_params, force = args_tuple
    src_log = src_root / split / log_id
    dst_log = dst_root / split / log_id
    fea_in = src_log / "sm_annotations.feather"
    fea_out = dst_log / "sm_annotations.feather"
    sidecar = src_log / "imm_smooth_inputs.feather"

    if not fea_in.exists():
        return {"log_id": log_id, "status": "missing_input", "n_tracks": 0, "by_status": {}}
    if not sidecar.exists():
        return {"log_id": log_id, "status": "missing_sidecar", "n_tracks": 0, "by_status": {}}
    if fea_out.exists() and not force:
        return {"log_id": log_id, "status": "skipped_existing", "n_tracks": 0, "by_status": {}}

    try:
        from tools.rts_smoothing.src import load_tracks
        from tools.rts_smoothing.src.imm_smoother import smooth_log_to_ego
        # obtain ego pose (for world→ego transform). load_tracks fills Tracks.ego_poses_*.
        T = load_tracks(log_id, split=split, src_tracker=src_root.name)
        replace = smooth_log_to_ego(
            sidecar, imm_params,
            T.ego_poses_ts, T.ego_poses_xyz, T.ego_poses_yaw,
        )   # {(uuid, ts): (ex, ey, eyaw)}

        df = pd.read_feather(fea_in)
        # for yaw_lock_to_track — preserve pre-smoothing tracking-output ego yaw
        df["track_out_yaw_ego"] = np.arctan2(2.0 * df["qz"] * df["qw"],
                                             1.0 - 2.0 * df["qz"] * df["qz"])
        uuids = df["track_uuid"].astype(str).to_numpy()
        tss = df["timestamp_ns"].astype(np.int64).to_numpy()
        new_tx = df["tx_m"].to_numpy(dtype=np.float64).copy()
        new_ty = df["ty_m"].to_numpy(dtype=np.float64).copy()
        new_qw = df["qw"].to_numpy(dtype=np.float64).copy()
        new_qx = df["qx"].to_numpy(dtype=np.float64).copy()
        new_qy = df["qy"].to_numpy(dtype=np.float64).copy()
        new_qz = df["qz"].to_numpy(dtype=np.float64).copy()
        n_rep = 0
        for i, (u, t) in enumerate(zip(uuids, tss)):
            v = replace.get((u, int(t)))
            if v is None:
                continue
            ex, ey, eyaw = v
            qw, qx, qy, qz = _yaw_to_quat(eyaw)
            new_tx[i], new_ty[i] = ex, ey
            new_qw[i], new_qx[i], new_qy[i], new_qz[i] = qw, qx, qy, qz
            n_rep += 1
        df["tx_m"] = new_tx; df["ty_m"] = new_ty
        df["qw"] = new_qw; df["qx"] = new_qx; df["qy"] = new_qy; df["qz"] = new_qz

        # reapply yaw post-processing — the imm smoother overwrites IMM-track yaw with the
        # raw sidecar values, so reapply tracking's yaw post-processing (stop_sign/jitter_freeze/dynamic_velocity).
        try:
            from tools.rts_smoothing.src.yaw_postproc import (
                load_pp_and_avm, reapply_yaw_postproc)
            _en, _pp, _avm, _syjh, _ylt = load_pp_and_avm(log_id, split)
            if _en and _pp is not None:
                df = reapply_yaw_postproc(df, T.ego_poses_ts, T.ego_poses_yaw,
                                          _avm, _pp, syjh=_syjh, ylt=_ylt)
        except Exception as _e:
            print(f"  [warn] {log_id[:8]} yaw_postproc skip: {_e}")

        dst_log.mkdir(parents=True, exist_ok=True)
        df.reset_index(drop=True).to_feather(fea_out)
        return {"log_id": log_id, "status": "ok",
                "n_tracks": len(set(k[0] for k in replace)),
                "by_status": {"smoothed_rows": n_rep}}
    except Exception as e:
        return {"log_id": log_id, "status": "error",
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
                "n_tracks": 0, "by_status": {}}


def _print_log_result(r: dict, prefix: str = ""):
    if r["status"] == "ok":
        bs = r["by_status"]
        parts = [f"{k}={v}" for k, v in bs.items()]
        print(f"{prefix}{r['log_id'][:8]}  tracks={r['n_tracks']:>3}  ({', '.join(parts)})")
    elif r["status"] == "skipped_existing":
        print(f"{prefix}{r['log_id'][:8]}  [skipped: dst exists]")
    elif r["status"] == "missing_input":
        print(f"{prefix}{r['log_id'][:8]}  [missing input feather]")
    elif r["status"] == "error":
        print(f"{prefix}{r['log_id'][:8]}  [ERROR] {r['error']}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src_tracker", default=None,
                   help="default: src_tracker from config")
    p.add_argument("--dst_tracker", default=None,
                   help="default: dst_tracker from config (if null, <src_tracker>_<imm_smooth|rts>)")
    p.add_argument("--split", default=None, help="default: split from config")
    p.add_argument("--logs", nargs="*", default=None,
                   help="List of log_ids to process. If empty, all logs in the split.")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--force", action="store_true")
    p.add_argument("--smoother_mode", default=None,
                   help="rts | imm. If unset, smoother_mode from config (default rts).")
    args = p.parse_args()

    # priority CLI > config > default
    _cfg_path = PROJECT_ROOT / "tools" / "rts_smoothing" / "config" / "config.yaml"
    _cfg = yaml.safe_load(_cfg_path.read_text())
    smoother_mode = (args.smoother_mode or _cfg.get("smoother_mode", "rts")).lower()
    args.src_tracker = args.src_tracker or _cfg.get("src_tracker") or "Le3DE2E_Tracking_ego_yawfix"
    args.split = args.split or _cfg.get("split", "val")

    if args.dst_tracker is None:
        # dst_tracker from config (if null, auto: <src>_<imm_smooth|rts>)
        args.dst_tracker = (_cfg.get("dst_tracker")
                            or f"{args.src_tracker}_{'imm_smooth' if smoother_mode == 'imm' else 'rts'}")

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

    print(f"[apply_rts_smoothing]  mode={smoother_mode}")
    print(f"  src       : {src_root}")
    print(f"  dst       : {dst_root}")
    print(f"  split     : {args.split}")
    print(f"  logs      : {len(log_ids)}")
    print(f"  workers   : {args.workers}")

    if smoother_mode == "imm":
        from tools.rts_smoothing.src import IMMSmoothParams
        imm_params = IMMSmoothParams.from_config(_cfg)
        print(f"  imm n_min : {imm_params.n_min}  (based on sidecar imm_smooth_inputs.feather)")
        print()
        _proc = process_log_imm
        work = [(src_root, dst_root, log_id, args.split, imm_params, args.force)
                for log_id in log_ids]
    else:
        params = _params_from_config()
        score_thr = _score_threshold_from_config()
        print(f"  score_thr : {score_thr}  (0=off, EGO always passes)")
        print(f"  Q diag    : {params.predict.Q_diag.tolist()}")
        print(f"  R diag    : {params.update.R_diag.tolist()}")
        print(f"  P0 diag   : {params.P0_diag.tolist()}")
        print()
        _proc = process_log
        work = [(src_root, dst_root, log_id, args.split, params, score_thr, args.force)
                for log_id in log_ids]

    results = []
    if args.workers <= 1:
        for w in work:
            r = _proc(w)
            results.append(r)
            _print_log_result(r)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            fut_to_log = {ex.submit(_proc, w): w[2] for w in work}
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
    n_ok, n_err, n_skip, n_missing = 0, 0, 0, 0
    for r in results:
        if r["status"] == "ok":
            n_ok += 1
            total_tracks += r["n_tracks"]
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
    print(f"  by_status: {agg_by_status}")

    dst_split = dst_root / args.split
    dst_split.mkdir(parents=True, exist_ok=True)
    agg_path = dst_split / "_aggregate_summary.json"
    _agg = {
        "src_tracker": args.src_tracker,
        "dst_tracker": args.dst_tracker,
        "split": args.split,
        "smoother_mode": smoother_mode,
        "n_logs": len(results),
        "n_ok": n_ok, "n_skipped": n_skip,
        "n_missing": n_missing, "n_error": n_err,
        "total_tracks": total_tracks,
        "by_status": agg_by_status,
    }
    if smoother_mode != "imm":
        _agg["Q_diag"] = params.predict.Q_diag.tolist()
        _agg["R_diag"] = params.update.R_diag.tolist()
        _agg["P0_diag"] = params.P0_diag.tolist()
    with open(agg_path, "w") as f:
        json.dump(_agg, f, indent=2)
    print(f"  aggregate saved: {agg_path}")

    errs = [r for r in results if r["status"] == "error"]
    if errs:
        print()
        print(f"=== {len(errs)} ERRORS ===")
        for r in errs[:10]:
            print(f"  {r['log_id']}: {r['error']}")


if __name__ == "__main__":
    main()
