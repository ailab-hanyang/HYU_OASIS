"""Heading-anisotropic tracklet stitching  —  SOTA tracking post-process.

**Preserves as is** the per-frame association of the base tracker (e.g.
Le3DE2E_Tracking_ego_yawfix) (IDSW is already low) and conservatively reconnects
only tracklet fragments broken by an occlusion gap to the same ID. Since boxes are
not interpolated/generated (= gaps are left empty), there is no phantom risk, and it
does not newly create the adjacent-swap (mismatching neighboring objects) that MODT
KF re-tracking used to cause.

Root cause: MODT re-tracking reduces occlusion-gap switches but increases
adjacent-swaps by +43%, a net loss. → stitch fixes only the gap fragmentation.

Algorithm:
  1) Group the src tracker feather by track_uuid and summarize tracklets in world coords
     (first/last ts·xy, last-2-frame velocity vel, category, size, n).
  2) end→start greedy matching. For an ending tracklet A, pick the best B that satisfies all of:
       - same category
       - 1 <= (B.first_ts - A.last_ts)/dt <= max_gap   (occlusion duration frames)
       - A constant-velocity extrapolation pred = A.last_xy + A.vel*gap ; decompose the
         error vs B.first_xy into heading lon/lat → |lat| <= max_lat_m (strict, blocks lane crossing)
                              & |lon| <= max_lon_m (lenient, allows forward motion along heading).
         (for stopped/single-frame tracklets, constant velocity is meaningless → isotropic max_dist_m fallback)
       - size ratio <= size_ratio
  3) After 1:1 chain merging via union-find, assign the root uuid to all detections of the merged tracklet.

SOTA (sub20, official run_experiment HOTA-Temporal): baseline 0.26864 → stitch_aniso **0.27390
(+0.00526)**, TBA 0.7938 → **0.8056 (+0.0118)**. The default parameters below are that SOTA setting.
Always judge with 'HOTA-Temporal' in run/run_experiment.py (custom breakdowns go off on merged tracks).

Apply: python -m tools.multi_class_tracking.apply_stitch --src_tracker <base> --dst_tracker <out>
"""
from __future__ import annotations
import uuid as _uuid
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
import pandas as pd

from .io_utils import load_ego_pose, ego_to_world

ROOT = Path(__file__).resolve().parents[3]
TR = ROOT / "output" / "tracker_predictions"

# ── SOTA default parameters (heading-anisotropic) ─────────────────────────
SOTA = dict(max_gap=5, max_dist_m=2.0, max_dist_static_m=1.5,
            size_ratio=2.0, max_lat_m=0.7, max_lon_m=5.0,
            min_anchor_len=1, min_partner_len=1)


class _UF:
    """union-find (path-compression)."""
    def __init__(self): self.p = {}
    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.p[rb] = ra


def stitch_log(log, src_tracker, dst_tracker, split="val",
               max_gap=SOTA["max_gap"], max_dist_m=SOTA["max_dist_m"],
               max_dist_static_m=SOTA["max_dist_static_m"], size_ratio=SOTA["size_ratio"],
               max_lat_m=SOTA["max_lat_m"], max_lon_m=SOTA["max_lon_m"],
               min_anchor_len=SOTA["min_anchor_len"], min_partner_len=SOTA["min_partner_len"],
               static_only_speed=0.0, write=True):
    """Stitch a single log's src tracker feather and save to dst_tracker. Returns a stats dict.

    If max_lat_m>0, use the heading lon/lat anisotropic gate (SOTA). If 0, isotropic max_dist_m.
    """
    fp = TR / src_tracker / split / log / "sm_annotations.feather"
    df = pd.read_feather(fp)
    ego_mask = df.category == "EGO_VEHICLE"
    ego_df = df[ego_mask].copy()
    obj = df[~ego_mask].copy()
    if len(obj) == 0:
        if write: _write(df, dst_tracker, split, log)
        return dict(log=log[:8], n_track=0, n_merge=0)

    ego = load_ego_pose(log, split)
    ts_arr = obj.timestamp_ns.to_numpy()
    txw = np.empty(len(obj)); tyw = np.empty(len(obj))
    for ts in np.unique(ts_arr):
        m = ts_arr == ts
        cx, cy, cyaw = ego(int(ts))
        wx, wy = ego_to_world(obj.tx_m.to_numpy(float)[m], obj.ty_m.to_numpy(float)[m], cx, cy, cyaw)
        txw[m] = wx; tyw[m] = wy
    obj = obj.assign(_wx=txw, _wy=tyw)

    uts = np.sort(obj.timestamp_ns.unique())
    dt = float(np.median(np.diff(uts))) if len(uts) > 1 else 1e8

    tracks = {}
    for uu, sub in obj.groupby("track_uuid", sort=False):
        sub = sub.sort_values("timestamp_ns")
        t = sub.timestamp_ns.to_numpy(); xs = sub._wx.to_numpy(); ys = sub._wy.to_numpy()
        cat = sub.category.iloc[0]
        sz = float(np.median(sub.length_m.to_numpy())) * float(np.median(sub.width_m.to_numpy()))
        if len(t) >= 2:
            v = np.array([(xs[-1] - xs[-2]) / max(t[-1] - t[-2], 1) * dt,
                          (ys[-1] - ys[-2]) / max(t[-1] - t[-2], 1) * dt])
        else:
            v = np.array([0.0, 0.0])
        tracks[str(uu)] = dict(first_ts=int(t[0]), last_ts=int(t[-1]), cat=str(cat),
                               first_xy=np.array([xs[0], ys[0]]), last_xy=np.array([xs[-1], ys[-1]]),
                               vel=v, size=sz, n=len(t))

    starts_by_cat = defaultdict(list)
    for uu, tk in tracks.items():
        starts_by_cat[tk["cat"]].append(uu)
    uf = _UF(); n_merge = 0; used_start = set()
    for ua in sorted(tracks, key=lambda u: tracks[u]["last_ts"]):
        A = tracks[ua]
        if A["n"] < min_anchor_len: continue
        best = None; bestd = 1e9
        for ub in starts_by_cat[A["cat"]]:
            if ub == ua or ub in used_start: continue
            B = tracks[ub]
            if B["n"] < min_partner_len: continue
            gap_frames = (B["first_ts"] - A["last_ts"]) / dt
            if gap_frames < 0.5 or gap_frames > max_gap + 0.5:
                continue
            steps = round(gap_frames)
            pred = A["last_xy"] + A["vel"] * steps
            disp = B["first_xy"] - pred
            d = float(np.hypot(*disp))
            speed = float(np.hypot(*A["vel"]))
            # static_only_speed>0: merge only static tracks (speed below the value) (skip dynamic objects)
            if static_only_speed > 0 and speed > static_only_speed: continue
            if max_lat_m > 0 and A["n"] >= 2 and speed > 0.3:
                hx, hy = A["vel"] / speed
                lon = abs(disp[0] * hx + disp[1] * hy)   # along heading (lenient)
                lat = abs(-disp[0] * hy + disp[1] * hx)  # perpendicular (strict)
                if lat > max_lat_m or lon > max_lon_m: continue
                d = lat
            else:
                thr = max_dist_m if A["n"] >= 2 else max_dist_static_m
                if d > thr: continue
            if A["size"] > 0 and B["size"] > 0:
                r = A["size"] / B["size"]; r = r if r >= 1 else 1 / r
                if r > size_ratio: continue
            if d < bestd: bestd = d; best = ub
        if best is not None:
            uf.union(ua, best); used_start.add(best); n_merge += 1

    if n_merge > 0:
        # Assign a 'new uuid' to merged groups. Reason: refAV's scenario_lanes disk cache
        # is keyed by log_id only (GLOBAL_CACHE_PATH/<log_id>/scenario_lanes.json) and is
        # shared across trackers. If a merged track reuses a fragment's original uuid, it
        # collides with that uuid (= fewer timestamps) cached by the baseline → get_scenario_lanes
        # returns only stale ts → KeyError in the atomic function → triggers eval's LLM
        # auto-fix (API). With a new uuid, the cache does not exist, so it is rebuilt with the
        # full merged timestamps → no collision. The identity structure is the same → HOTA unchanged.
        roots = {u: uf.find(u) for u in tracks}
        rootcount = Counter(roots.values())
        new_for_root = {r: str(_uuid.uuid4()) for r, c in rootcount.items() if c >= 2}
        def _remap(u):
            r = roots.get(str(u), str(u))
            return new_for_root.get(r, str(u))   # merged group→new uuid, single track→original uuid
        obj = obj.assign(track_uuid=obj.track_uuid.astype(str).map(_remap))

    out = pd.concat([obj.drop(columns=["_wx", "_wy"]), ego_df], ignore_index=True)
    out = out.sort_values(["timestamp_ns", "track_uuid"]).reset_index(drop=True)
    if write: _write(out, dst_tracker, split, log)
    return dict(log=log[:8], n_track=len(tracks), n_merge=n_merge)


def _write(df, dst_tracker, split, log):
    d = TR / dst_tracker / split / log
    d.mkdir(parents=True, exist_ok=True)
    df.to_feather(d / "sm_annotations.feather")
