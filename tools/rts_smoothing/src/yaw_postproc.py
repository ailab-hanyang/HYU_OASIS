"""Re-apply post-smoothing yaw post-processing — identical to tracking build_output_df's yaw chain.

When the smoother (rts/imm) overwrites yaw (quat) during smoothing, the yaw
corrections applied in tracking post-processing (stop_sign_yaw_to_lane · jitter_freeze ·
dynamic_velocity_yaw) get lost (especially on IMM tracks in imm mode); this unifies them by
re-applying the **same functions** to the smoothed output.

input df = sm_annotations (after smoothing). Column convention:
  · tx_m/ty_m/qw..qz : smoothed (ego frame)
  · meas_x_world/meas_y_world/meas_yaw_world/kf_vx/kf_vy : tracking original preserved (world)
  · category, track_uuid, timestamp_ns
yaw post-processing is performed in world frame, then returned to ego frame (quat) updating only qw..qz.
(position/size/score are not touched — smoothing result preserved)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse post_process's post-processing functions (single source — guarantees logic identical to tracking)
_PR = Path(__file__).resolve().parents[3]
if str(_PR) not in sys.path:
    sys.path.insert(0, str(_PR))
from tools.multi_class_tracking.src.post_process import (   # noqa: E402
    _wrap, _yaw_to_quat, jitter_freeze_yaw, dynamic_velocity_yaw,
    _stop_sign_lane_yaw_world,
)


def _quat_to_ego_yaw(qw: np.ndarray, qz: np.ndarray) -> np.ndarray:
    return np.arctan2(2.0 * qz * qw, 1.0 - 2.0 * qz * qz)


def load_pp_and_avm(log_id: str, split: str):
    """Load multi_class_tracking config's PostProcessParams + AV2 map (avm).

    Returns: (enabled: bool, pp: PostProcessParams|None, avm|None, syjh: dict|None)
    syjh = static_yaw_jump_hold parameters {enabled, thr_rad, v_static} (smoothing-only).
    enabled=False if yaw_postproc is off or dependency loading fails.
    """
    import yaml
    rts_cfg_path = _PR / "tools" / "rts_smoothing" / "config" / "config.yaml"
    try:
        rts_cfg = yaml.safe_load(rts_cfg_path.read_text())
    except Exception:
        return False, None, None, None, None
    yp = (rts_cfg.get("yaw_postproc") or {})
    if not bool(yp.get("enabled", False)):
        return False, None, None, None, None

    from tools.multi_class_tracking.apply_tracking import _load_config as _mct_cfg
    from tools.multi_class_tracking.src.post_process import PostProcessParams
    pp = PostProcessParams.from_config(_mct_cfg())

    avm = None
    if pp.stop_sign_yaw_to_lane:
        try:
            import refAV.paths as _paths
            from refAV.utils import get_map as _get_map
            avm = _get_map(_PR / _paths.AV2_DATA_DIR / split / log_id)
        except Exception:
            avm = None

    _sj = (yp.get("static_yaw_jump_hold") or {})
    syjh = {
        "enabled": bool(_sj.get("enabled", True)),
        "thr_rad": float(np.radians(float(_sj.get("thr_deg", 20.0)))),
        "v_static": float(_sj.get("v_static", 0.25)),
    }
    _yl = (yp.get("yaw_lock_to_track") or {})
    ylt = {
        "enabled": bool(_yl.get("enabled", True)),
        "thr_rad": float(np.radians(float(_yl.get("thr_deg", 30.0)))),
    }
    return True, pp, avm, syjh, ylt


def reapply_yaw_postproc(df: pd.DataFrame, ego_ts, ego_yaw, avm, pp,
                         syjh=None, ylt=None) -> pd.DataFrame:
    """Re-apply tracking yaw post-processing to the smoothed df → return df with qw/qx/qy/qz updated.

    syjh = static_yaw_jump_hold {enabled, thr_rad, v_static} (smoothing-only) or None.
    ylt  = yaw_lock_to_track {enabled, thr_rad} — if the smoothed yaw diverges from the tracking
           output yaw by thr or more, replace it with the tracking yaw (requires df["track_out_yaw_ego"] column, preserved by the caller before smoothing).
    Returns the original df unchanged if required columns are missing (safe).
    """
    need = {"track_uuid", "timestamp_ns", "qw", "qz", "category"}
    if not need.issubset(df.columns):
        return df
    has_meas = {"meas_x_world", "meas_y_world", "meas_yaw_world"}.issubset(df.columns)
    has_vel = {"kf_vx", "kf_vy"}.issubset(df.columns)

    df = df.reset_index(drop=True)
    ego_map = {int(t): float(y) for t, y in zip(np.asarray(ego_ts), np.asarray(ego_yaw))}

    out_qw = df["qw"].to_numpy(np.float64).copy()
    out_qx = df["qx"].to_numpy(np.float64).copy()
    out_qy = df["qy"].to_numpy(np.float64).copy()
    out_qz = df["qz"].to_numpy(np.float64).copy()

    for uuid, sub in df.groupby("track_uuid", sort=False):
        cat = str(sub["category"].iloc[0])
        if cat == "EGO_VEHICLE":
            continue
        pos = sub.index.to_numpy()
        ts = sub["timestamp_ns"].to_numpy(np.int64)
        n = len(ts)
        egy = np.array([ego_map.get(int(t), 0.0) for t in ts], dtype=np.float64)
        sm_ego_yaw = _quat_to_ego_yaw(sub["qw"].to_numpy(np.float64),
                                      sub["qz"].to_numpy(np.float64))
        world_yaw = np.array([_wrap(sm_ego_yaw[i] + egy[i]) for i in range(n)],
                             dtype=np.float64)

        mx = sub["meas_x_world"].to_numpy(np.float64) if has_meas else None
        my = sub["meas_y_world"].to_numpy(np.float64) if has_meas else None
        meas_yaw_world = sub["meas_yaw_world"].to_numpy(np.float64) if has_meas else None

        frozen = False
        # (1) STOP_SIGN → opposite of the nearest lane travel direction
        if pp.stop_sign_yaw_to_lane and avm is not None and cat == "STOP_SIGN" and has_meas:
            yw = _stop_sign_lane_yaw_world(avm, float(np.median(mx)), float(np.median(my)))
            if yw is not None:
                world_yaw = np.full(n, yw, dtype=np.float64)

        # (2) jitter_freeze — static spinner (gate/axis is MEASUREMENT world yaw, same as tracking)
        if (pp.yaw_freeze_enabled and has_meas and cat != "STOP_SIGN"):
            theta = jitter_freeze_yaw([_wrap(float(v)) for v in meas_yaw_world], mx, my,
                                      pp.yaw_freeze_tau_total, pp.yaw_freeze_tau_net)
            if theta is not None:
                world_yaw = np.full(n, theta, dtype=np.float64)
                frozen = True

        # (3) dynamic_velocity_yaw — flip a dynamic track's yaw toward the velocity direction
        if (pp.dyn_yaw_enabled and not frozen and has_vel and has_meas and cat != "STOP_SIGN"):
            vx = sub["kf_vx"].to_numpy(np.float64)
            vy = sub["kf_vy"].to_numpy(np.float64)
            wyd = dynamic_velocity_yaw(world_yaw, vx, vy, mx, my,
                                       pp.dyn_yaw_tau_dyn, pp.dyn_yaw_vmin)
            if wyd is not None:
                world_yaw = np.asarray(wyd, dtype=np.float64)

        # (4) static yaw jump hold — in static frames (‖v‖<v_static), if yaw jumps by thr or
        # more vs the previous, keep the previous yaw (sequential propagate). Suppresses momentary jumps/flips of static object yaw.
        if (syjh is not None and syjh.get("enabled") and has_vel and n >= 2):
            vx = sub["kf_vx"].to_numpy(np.float64)
            vy = sub["kf_vy"].to_numpy(np.float64)
            spd = np.hypot(vx, vy)
            thr = float(syjh["thr_rad"]); vstat = float(syjh["v_static"])
            for i in range(1, n):
                if spd[i] < vstat and abs(_wrap(world_yaw[i] - world_yaw[i - 1])) >= thr:
                    world_yaw[i] = world_yaw[i - 1]

        # (5) yaw lock to tracking output — if the smoothed yaw diverges from the tracking output yaw (flip resolved)
        # by thr or more, replace it with the tracking yaw (regardless of speed). Corrects the smoother's ±180° flip smear.
        if (ylt is not None and ylt.get("enabled") and "track_out_yaw_ego" in df.columns):
            tye = sub["track_out_yaw_ego"].to_numpy(np.float64)
            lthr = float(ylt["thr_rad"])
            for i in range(n):
                track_world = _wrap(float(tye[i]) + egy[i])
                if abs(_wrap(world_yaw[i] - track_world)) > lthr:
                    world_yaw[i] = track_world

        # write back: world → ego → quat
        for i in range(n):
            ego_y = _wrap(float(world_yaw[i]) - egy[i])
            qw, qx, qy, qz = _yaw_to_quat(ego_y)
            out_qw[pos[i]] = qw; out_qx[pos[i]] = qx
            out_qy[pos[i]] = qy; out_qz[pos[i]] = qz

    df["qw"] = out_qw; df["qx"] = out_qx; df["qy"] = out_qy; df["qz"] = out_qz
    return df


def postproc_yaw_map_for_imm(sm_feather_path, replace, ego_ts, ego_yaw, avm, pp, syjh=None, ylt=None) -> dict:
    """For server stage3 (IMM-S) — build the smoothed df from the original sm_annotations + imm replace →
    re-apply yaw post-processing → return {(uuid, ts): final_ego_yaw_rad}.

    Since the server serializes the response with TrackState.yaws_rad (ego frame),
    overwriting each frame's yaw with this map makes the IMM-S visualization identical to the offline pipeline.
    replace: {(uuid, ts): (ex, ey, eyaw)} — output of smooth_log_to_ego.
    """
    df = pd.read_feather(sm_feather_path)
    need = {"track_uuid", "timestamp_ns", "qw", "qx", "qy", "qz"}
    if not need.issubset(df.columns):
        return {}
    uuids = df["track_uuid"].astype(str).to_numpy()
    tss = df["timestamp_ns"].astype(np.int64).to_numpy()
    qw = df["qw"].to_numpy(np.float64).copy()
    qx = df["qx"].to_numpy(np.float64).copy()
    qy = df["qy"].to_numpy(np.float64).copy()
    qz = df["qz"].to_numpy(np.float64).copy()
    has_xy = {"tx_m", "ty_m"}.issubset(df.columns)
    if has_xy:
        tx = df["tx_m"].to_numpy(np.float64).copy()
        ty = df["ty_m"].to_numpy(np.float64).copy()
    # for yaw_lock_to_track — preserve the tracking output ego yaw before the smoothed replace
    df["track_out_yaw_ego"] = np.arctan2(2.0 * qz * qw, 1.0 - 2.0 * qz * qz)
    # reflect the imm smoothing result (eyaw, ex, ey) into df (IMM tracks only; static etc. keep original)
    for i in range(len(uuids)):
        v = replace.get((uuids[i], int(tss[i])))
        if v is None:
            continue
        ex, ey, eyaw = v
        a, b, c, d = _yaw_to_quat(float(eyaw))
        qw[i], qx[i], qy[i], qz[i] = a, b, c, d
        if has_xy:
            tx[i], ty[i] = float(ex), float(ey)
    df["qw"] = qw; df["qx"] = qx; df["qy"] = qy; df["qz"] = qz
    if has_xy:
        df["tx_m"] = tx; df["ty_m"] = ty

    df = reapply_yaw_postproc(df, ego_ts, ego_yaw, avm, pp, syjh=syjh, ylt=ylt)

    out = {}
    fu = df["track_uuid"].astype(str).to_numpy()
    ft = df["timestamp_ns"].astype(np.int64).to_numpy()
    fqw = df["qw"].to_numpy(np.float64)
    fqz = df["qz"].to_numpy(np.float64)
    for i in range(len(df)):
        out[(fu[i], int(ft[i]))] = float(np.arctan2(2.0 * fqz[i] * fqw[i],
                                                    1.0 - 2.0 * fqz[i] * fqz[i]))
    return out
