"""Track post-processing — called once after forward tracking finishes.

Coordinate frame — the KF tracked in the world (city) frame, so a **world→ego
inverse transform** is required on output. This frame's ego pose is stored
alongside in FrameRecord.

Requirements (re-specified)
- yaw  : dynamic frames (‖v_world‖ ≥ v_static) recompute ego frame as atan2(vy, vx) − ego_yaw
         static frames keep the detection's yaw_meas as is (already ego frame)
- l/w/h/s : score-weighted average within a track → per-track constant
            l/w/h are each floored by min_l/min_w/min_h
- category: argmax of the track's class EMA (per-track constant)
- track_uuid: new UUID (per-track constant)
- tx, ty : KF posterior (world) → ego frame transform (per frame)
- tz     : detection original as is (per frame, no KF effect)
"""

from __future__ import annotations

import bisect
import uuid
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .track import Track


@dataclass
class PostProcessParams:
    dynamic_yaw_threshold: float       # m/s
    min_l: float
    min_w: float
    min_h: float
    per_frame_score: bool = False      # [ABLATION lever1] if True, output per-frame raw score (score-tuning granularity↑)
    per_frame_size: bool = False       # [ABLATION lever1] if True, output per-frame raw l/w/h
    relabel_only: bool = False         # [ABLATION lever2] if True, keep detection/position/score/size/category
                                       # raw as is and replace only track_uuid with the MODT-consistent ID.
                                       # Output all tracks (incl. unconfirmed) and all frames → keep DetA
                                       # identical to baseline and try to gain only AssA (ID consistency).
    relabel_confirmed_only: bool = False  # with relabel_only — output confirmed tracks only (remove single-frame
                                          # junk → try to suppress program over-prediction/FP). Keep raw geometry.
    interpolate_max_gap: int = 0          # if >0, fill the detection gap (missed-detection span) of a relabel track up to this many frames
                                          # via linear interpolation (same uuid) → recover missed detections → DetRe/DetA↑.
                                          # Does not increase IDSW (same track). For occlusion/momentary-miss recovery.
    # ── IMM track output yaw flip correction (ported from rts_smoothing yaw_correction) ──
    # Per-track static decision: world bbox diagonal d_max < yaw_tau_static → static → unify majority direction to ±π,
    # dynamic → resolve flip based on motion direction (tau_motion). IMM mixed yaw is the base (value preserved, ±π flip only).
    yaw_corr_enabled: bool = False
    yaw_tau_static: float = 4.0
    yaw_tau_motion: float = 0.2
    # ── static object detection (IMM posterior position based, per-track) — for qualitative validation/visualization ──
    # static = (track posterior position bbox diagonal < static_tau_static) AND
    #          (max inter-frame displacement < static_tau_motion). Both position based (independent of velocity-zero).
    static_tau_static: float = 4.0
    static_tau_motion: float = 0.3
    # ── jitter-static yaw freeze — fix the output yaw of a pinwheel (in-place rotation) track to the dominant direction ──
    # Gate: yaw_total (cumulative Σ|Δyaw| of output world yaw) ≥ tau_total AND net_disp(robust) < tau_net.
    # Aimed at reducing downstream turning atomic function FP. OFF by default (turn on after server live validation).
    yaw_freeze_enabled: bool = False
    yaw_freeze_tau_total: float = 450.0  # deg (cumulative total rotation Σ|Δyaw|)
    yaw_freeze_tau_net: float = 4.0      # m (handles large-vehicle center wander)
    # ── IoU clean — among bbox-overlapping tracks in one frame, remove the shorter (smaller history) one ──
    iou_clean_enabled: bool = False
    iou_clean_thr: float = 0.2           # rotated BEV IoU threshold
    iou_clean_protect_both: int = 10     # if both have history ≥ this value, keep both
    # ── dynamic velocity yaw flip — if a dynamic track's yaw is opposite to the motion direction, flip by +180 ──
    dyn_yaw_enabled: bool = False
    dyn_yaw_tau_dyn: float = 2.0         # m — if robust net movement ≥ this value, dynamic track
    dyn_yaw_vmin: float = 1.5            # m/s — trusted speed of vdir used as flip reference (high-speed frames only,
                                         #        avoids noisy low-speed velocity at acceleration onset)
    # ── fix STOP_SIGN yaw → opposite (anti-parallel) of the nearest lane's heading ──
    # To satisfy the at_stop_sign atomic's pass condition (vehicle yaw≈180° in the sign's frame),
    # the sign heading must be opposite to the lane's heading. Replace a static sign's raw detection
    # yaw error with the map centerline → aimed at reducing at_stop_sign FN. build_output_df needs avm.
    stop_sign_yaw_to_lane: bool = False

    @classmethod
    def from_config(cls, cfg: dict) -> "PostProcessParams":
        pp = cfg.get("post_processing", {}) or {}
        min_lwh = pp.get("min_lwh", {})
        yc = pp.get("yaw_correction", {}) or {}
        sd = pp.get("static_detection", {}) or {}
        jf = pp.get("yaw_jitter_freeze", {}) or {}
        ic = pp.get("iou_clean", {}) or {}
        dy = pp.get("dynamic_velocity_yaw", {}) or {}
        return cls(
            dynamic_yaw_threshold=float(pp.get("dynamic_yaw_threshold", 0.25)),
            min_l=float(min_lwh.get("l", 0.3)),
            min_w=float(min_lwh.get("w", 0.3)),
            min_h=float(min_lwh.get("h", 0.5)),
            per_frame_score=bool(pp.get("per_frame_score", False)),
            per_frame_size=bool(pp.get("per_frame_size", False)),
            relabel_only=bool(pp.get("relabel_only", False)),
            relabel_confirmed_only=bool(pp.get("relabel_confirmed_only", False)),
            interpolate_max_gap=int(pp.get("interpolate_max_gap", 0)),
            yaw_corr_enabled=bool(yc.get("enabled", False)),
            yaw_tau_static=float(yc.get("tau_static", 4.0)),
            yaw_tau_motion=float(yc.get("tau_motion", 0.2)),
            static_tau_static=float(sd.get("tau_static", 4.0)),
            static_tau_motion=float(sd.get("tau_motion", 0.3)),
            yaw_freeze_enabled=bool(jf.get("enabled", False)),
            yaw_freeze_tau_total=float(jf.get("tau_total", 450.0)),
            yaw_freeze_tau_net=float(jf.get("tau_net", 2.0)),
            iou_clean_enabled=bool(ic.get("enabled", False)),
            iou_clean_thr=float(ic.get("iou_thr", 0.2)),
            iou_clean_protect_both=int(ic.get("protect_both_min", 10)),
            dyn_yaw_enabled=bool(dy.get("enabled", False)),
            dyn_yaw_tau_dyn=float(dy.get("tau_dyn", 2.0)),
            dyn_yaw_vmin=float(dy.get("v_min", 1.5)),
            stop_sign_yaw_to_lane=bool(
                (pp.get("stop_sign_yaw_to_lane", {}) or {}).get("enabled", False)),
        )


# ── helpers ───────────────────────────────────────────────────────
def _yaw_to_quat(yaw: float) -> tuple[float, float, float, float]:
    """yaw (z-axis only) → (qw, qx, qy, qz). qw ≥ 0 canonical."""
    half = yaw * 0.5
    qw = float(np.cos(half))
    qz = float(np.sin(half))
    if qw < 0.0:
        qw = -qw
        qz = -qz
    return qw, 0.0, 0.0, qz


def _wrap(a: float) -> float:
    return float(np.arctan2(np.sin(a), np.cos(a)))


def classify_static(px, py, tau_static: float, tau_motion: float) -> bool:
    """Determine static from IMM posterior position (px, py: world frame, per-track time series).

    static = (position bbox diagonal d_max < tau_static)   # small overall track displacement
             OR  (max inter-frame displacement < tau_motion)  # no frame moves much
    OR (single condition) — static if either holds. Compared to AND, robust to
    single-frame noise spikes (stays static when d_max is small), but may also
    catch slow constant-velocity movers (small step).
    Both are position based, so it can be tuned cleanly regardless of the static
    velocity=0 forcing. Split into a module function so the server's live
    visualization and postprocess use the same logic.
    """
    px = np.asarray(px, dtype=np.float64)
    py = np.asarray(py, dtype=np.float64)
    if px.size == 0:
        return False
    d_max = float(np.hypot(px.max() - px.min(), py.max() - py.min()))
    if px.size >= 2:
        step = float(np.hypot(np.diff(px), np.diff(py)).max())
    else:
        step = 0.0
    return bool(d_max < tau_static or step < tau_motion)


def jitter_freeze_yaw(yaws_world, px, py,
                      tau_total_deg: float, tau_net: float,
                      min_n: int = 8) -> float | None:
    """Compute the dominant yaw of a static track that spins in place (pinwheel) due to detection noise.

    Gate (both must hold to be a freeze target):
      yaw_total = Σ|Δyaw_world|  ≥  tau_total_deg   # large cumulative track rotation (heavily spun track)
      net_disp = ‖median(last 3) - median(first 3)‖  <  tau_net  # small robust start↔end movement (in place)
    On pass: RANSAC axis consensus — in axis (mod180) space, find the inlier axis with
    the most points within ±TH, remove outliers (flip/jitter spikes) → inlier line fit
    (2θ mean) → front majority vote → return a single world yaw. Returns None if not
    passed (no freeze).

    yaws_world : per-frame world yaw of the track (rad). px/py : world positions (measurement recommended).
    """
    yw = np.asarray(yaws_world, dtype=np.float64)
    n = yw.size
    if n < int(min_n):
        return None
    dy = np.arctan2(np.sin(np.diff(yw)), np.cos(np.diff(yw)))
    yaw_total = float(np.degrees(np.abs(dy).sum()))
    if yaw_total < float(tau_total_deg):
        return None
    qx = np.asarray(px, dtype=np.float64)
    qy = np.asarray(py, dtype=np.float64)
    K = 3
    p0 = np.array([np.median(qx[:K]), np.median(qy[:K])])
    p1 = np.array([np.median(qx[-K:]), np.median(qy[-K:])])
    if float(np.hypot(p1[0] - p0[0], p1[1] - p0[1])) >= float(tau_net):
        return None
    # RANSAC axis consensus — find the inlier with the most points on the line
    # passing through the direction vector (axis, ±180 symmetric) and remove outliers
    # (flip/jitter spikes). Axis space (mod180) matches box symmetry.
    TH = np.radians(20.0)   # axis-distance inlier threshold
    # Exhaustive candidate-axis search (N≤32): set each frame's yaw as the axis and pick the one with the most inliers.
    best_cnt, best_axis = -1, float(yw[0])
    for i in range(n):
        # axis distance = 0.5·wrap(2·Δyaw) (±90°, ignore flip)
        ad = np.arctan2(np.sin(2.0 * (yw - yw[i])), np.cos(2.0 * (yw - yw[i]))) / 2.0
        cnt = int(np.sum(np.abs(ad) <= TH))
        if cnt > best_cnt:
            best_cnt, best_axis = cnt, float(yw[i])
    # refit axis with inliers (direction-vector line fit = inlier 2θ circular mean)
    ad = np.arctan2(np.sin(2.0 * (yw - best_axis)), np.cos(2.0 * (yw - best_axis))) / 2.0
    inl = yw[np.abs(ad) <= TH]
    axis = 0.5 * float(np.arctan2(float(np.sin(2.0 * inl).sum()),
                                  float(np.cos(2.0 * inl).sum())))
    # front direction: if inliers cluster more on the axis(±90°) side, use axis, else axis+π
    dd = np.arctan2(np.sin(inl - axis), np.cos(inl - axis))
    if int(np.sum(np.abs(dd) <= np.pi / 2.0)) < inl.size / 2.0:
        axis = axis + np.pi
    return _wrap(axis)


# ── IoU clean — among bbox-overlapping tracks in one frame, remove the shorter (smaller age) one as noise ──
def _box_poly(cx: float, cy: float, l: float, w: float, yaw: float):
    """oriented BEV rect → shapely Polygon (for rotated IoU)."""
    from shapely.geometry import Polygon
    c, s = np.cos(yaw), np.sin(yaw)
    hl, hw = l / 2.0, w / 2.0
    corners = [(hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)]
    return Polygon([(cx + c * x - s * y, cy + s * x + c * y) for x, y in corners])


def _bev_iou(pa, pb) -> float:
    if not pa.intersects(pb):
        return 0.0
    inter = pa.intersection(pb).area
    union = pa.area + pb.area - inter
    return float(inter / union) if union > 0 else 0.0


def iou_clean_removed(tracks: list[dict], iou_thr: float,
                      min_loser_age: int = 2, protect_both_min: int = 10) -> set:
    """For track pairs overlapping at one timestamp with bbox (rotated BEV) IoU ≥ iou_thr,
    mark the track with fewer history (n_frames) as noise for removal. If n_frames ties, keep both (score unused).

    - If either of the overlapping two is static (classified as a static object), keep both.
    - If both tracks have n_frames ≥ protect_both_min, keep both (long tracks are not removed against each other).
    - If the loser's n_frames is below min_loser_age, do not remove.
    Class restriction (e.g. REGULAR_VEHICLE only) is applied by the caller filtering the track candidates.

    tracks : [{ 'key': hashable, 'n': n_frames(int), 'static': bool(optional),
               'frames': {ts(int): (cx, cy, l, w, yaw)} }]
    Returns: set of keys to remove.
    """
    from collections import defaultdict
    nfr = {t["key"]: int(t["n"]) for t in tracks}
    sta = {t["key"]: bool(t.get("static", False)) for t in tracks}
    bucket: dict = defaultdict(list)        # ts → [(key, poly)]
    for t in tracks:
        for ts, (cx, cy, l, w, yaw) in t["frames"].items():
            bucket[int(ts)].append((t["key"], _box_poly(cx, cy, l, w, yaw)))
    removed: set = set()
    for items in bucket.values():
        m = len(items)
        for i in range(m):
            ki, pi = items[i]
            for j in range(i + 1, m):
                kj, pj = items[j]
                if _bev_iou(pi, pj) >= iou_thr:
                    # if either is static (static classification) → keep both.
                    if sta[ki] or sta[kj]:
                        continue
                    # both have history >= protect_both_min → long tracks, keep both.
                    if min(nfr[ki], nfr[kj]) >= protect_both_min:
                        continue
                    # the one with fewer n_frames is the loser. Remove only when loser n_frames >= min_loser_age.
                    if nfr[ki] < nfr[kj]:
                        if nfr[ki] >= min_loser_age:
                            removed.add(ki)
                    elif nfr[kj] < nfr[ki]:
                        if nfr[kj] >= min_loser_age:
                            removed.add(kj)
                    # n_frames tie → keep both
    return removed


def dynamic_velocity_yaw(world_yaws, vx, vy, px, py,
                         tau_dyn: float, v_min: float) -> list | None:
    """For a genuinely moving track (net_disp ≥ tau_dyn), if the IMM yaw is opposite to
    the motion (velocity) direction, flip by +180 to align front/back only. **The yaw
    value is preserved** (not replaced by velocity).

    Gate: net_disp = ‖median(last 3) − median(first 3)‖ ≥ tau_dyn  → dynamic track.
    Apply (per-frame): compared to the velocity direction, if cos(yaw − vdir) < 0 (opposite, >90°), yaw += 180.
      vdir is atan2(vy,vx) of trusted-speed (‖v‖≥v_min) frames; low-speed frames use the
      nearest trusted frame's vdir (heading does not teleport) → also corrects the opposite yaw at acceleration onset.
    Returns None (no change) if net_disp < tau_dyn or there is no trusted-speed frame.

    Gated by net movement rather than instantaneous ‖v‖ → prevents misjudging stationary objects whose ‖v‖ spikes from detection noise.
    """
    n = len(world_yaws)
    if n < 2:
        return None
    qx = np.asarray(px, dtype=np.float64)
    qy = np.asarray(py, dtype=np.float64)
    K = 3
    p0 = (float(np.median(qx[:K])), float(np.median(qy[:K])))
    p1 = (float(np.median(qx[-K:])), float(np.median(qy[-K:])))
    if float(np.hypot(p1[0] - p0[0], p1[1] - p0[1])) < float(tau_dyn):
        return None
    vxa = np.asarray(vx, dtype=np.float64)
    vya = np.asarray(vy, dtype=np.float64)
    spd = np.hypot(vxa, vya)
    rel = np.where(spd >= float(v_min))[0]             # trusted-speed frame indices
    if rel.size == 0:
        return None
    vdir = np.arctan2(vya, vxa)
    out = []
    for i in range(n):
        j = int(rel[int(np.argmin(np.abs(rel - i)))])  # nearest trusted frame
        y = float(world_yaws[i])
        if float(np.cos(y - vdir[j])) < 0.0:           # opposite to velocity (>90°) → flip
            y = y + np.pi
        out.append(_wrap(y))
    return out


def _world_to_ego(wx: float, wy: float,
                  ego_x: float, ego_y: float, ego_yaw: float
                  ) -> tuple[float, float]:
    """world (city) frame (wx, wy) → ego frame (ex, ey).

    ego_obj = R(-ego_yaw) · (world_obj - ego_pos).
    """
    dx = wx - ego_x
    dy = wy - ego_y
    c, s = float(np.cos(ego_yaw)), float(np.sin(ego_yaw))
    ex = c * dx + s * dy
    ey = -s * dx + c * dy
    return ex, ey


def _track_level_lwhs(tr: Track,
                      params: PostProcessParams
                      ) -> tuple[float, float, float, float]:
    """score-weighted (l, w, h, s). If the score sum is 0, simple average.

    Each dimension is min-floored. score is also clipped to [0, 1].
    """
    if not tr.frame_records:
        return params.min_l, params.min_w, params.min_h, 0.0

    L = np.array([r.length_meas for r in tr.frame_records], dtype=np.float64)
    W = np.array([r.width_meas for r in tr.frame_records], dtype=np.float64)
    H = np.array([r.height_meas for r in tr.frame_records], dtype=np.float64)
    S = np.array([r.score_meas for r in tr.frame_records], dtype=np.float64)

    weights = S.copy()
    if weights.sum() <= 0.0:
        weights = np.ones_like(weights)
    w_sum = float(weights.sum())

    l_avg = float((weights * L).sum() / w_sum)
    w_avg = float((weights * W).sum() / w_sum)
    h_avg = float((weights * H).sum() / w_sum)
    s_avg = float((weights * S).sum() / w_sum)

    l_avg = max(l_avg, params.min_l)
    w_avg = max(w_avg, params.min_w)
    h_avg = max(h_avg, params.min_h)
    s_avg = float(np.clip(s_avg, 0.0, 1.0))
    return l_avg, w_avg, h_avg, s_avg


# ── track-level yaw flip correction (motion based) ──────────────────
# Same algorithm as rts_smoothing.yaw_correction._decide_dynamic.
# The detection yaw is body orientation (precise) but has ±180° flip ambiguity.
# Resolve the flip by taking the motion direction from the track's overall bidirectional displacement.
_YAW_TAU_STATIC = 2.0    # m — static if track world bbox diagonal < this value (skip correction)
_YAW_TAU_MOTION = 0.2    # m — motion direction is trusted only with displacement above this


def _resolve_track_yaws(tr, params) -> list[float]:
    """Return the ego-frame output yaw for each of the track's frame_records (flip resolved).

    IMM-active track (imm_yaw_world filled) → base yaw = IMM mixed yaw (precise), but
    correct only the ±180° flip based on the motion direction (KF velocity heading).
    Aligns cases where the measurement yaw flipped on some frames and shook the IMM too
    to the motion direction. The value itself stays IMM (not fully replaced by velocity
    — IMM also expresses reverse/lateral movement).

    CV single track (imm_yaw_world=None) → existing hybrid:
      - frames with ‖v_KF‖ >= dynamic_yaw_threshold → velocity direction (motion heading).
      - low-speed frames → resolve flip by aligning the detection yaw to the nearest reliable frame's motion direction.
      - if no reliable frame → detection yaw as is.
    """
    recs = tr.frame_records
    n = len(recs)
    if n == 0:
        return []
    thr = float(params.dynamic_yaw_threshold)

    # ── IMM-active branch ──────────────────────────────────────────────
    # base yaw = IMM mixed yaw (world). If yaw_corr_enabled, apply the best-scoring logic
    # of rts_smoothing yaw_correction (per-track static decision d_max<tau_static → unify
    # majority direction to ±π, dynamic → resolve motion-direction flip) to the base yaw
    # (value preserved, ±π flip only). If disabled, raw.
    if any(r.imm_yaw_world is not None for r in recs):
        ego_yaw_a = np.array([float(r.ego_yaw) for r in recs], dtype=np.float64)
        base_w = np.array(
            [float(r.imm_yaw_world) if r.imm_yaw_world is not None
             else float(r.yaw_meas) + float(r.ego_yaw) for r in recs],
            dtype=np.float64)
        if not params.yaw_corr_enabled or n < 2:
            return [_wrap(float(base_w[i] - ego_yaw_a[i])) for i in range(n)]
        # reuse rts_smoothing's static/dynamic flip decision functions (guarantees identical logic).
        import sys as _sys
        from .io_utils import PROJECT_ROOT as _PR
        if str(_PR) not in _sys.path:
            _sys.path.insert(0, str(_PR))
        from tools.rts_smoothing.src.yaw_correction import (
            _decide_static_majority_flip, _decide_dynamic,
            YawCorrectionParams as _YC,
        )
        px = np.array([float(r.kf_x) for r in recs], dtype=np.float64)
        py = np.array([float(r.kf_y) for r in recs], dtype=np.float64)
        d_max = float(np.hypot(px.max() - px.min(), py.max() - py.min()))
        if d_max < params.yaw_tau_static:
            flip = _decide_static_majority_flip(base_w)          # static → unify to majority direction
        else:
            res = _decide_dynamic(np.stack([px, py], axis=1), base_w,
                                  _YC(tau_static=params.yaw_tau_static,
                                      tau_motion=params.yaw_tau_motion))
            flip = res[0] if res is not None else np.zeros(n, dtype=bool)
        corr = np.where(flip,
                        np.arctan2(np.sin(base_w + np.pi), np.cos(base_w + np.pi)),
                        base_w)
        return [_wrap(float(corr[i] - ego_yaw_a[i])) for i in range(n)]

    # each frame's world-frame motion direction (velocity based) and whether it is reliable.
    vdir_world = np.zeros(n, dtype=np.float64)   # velocity direction (world frame)
    reliable = np.zeros(n, dtype=bool)
    yaw_meas = np.array([float(r.yaw_meas) for r in recs], dtype=np.float64)
    ego_yaw = np.array([float(r.ego_yaw) for r in recs], dtype=np.float64)
    yaw_world = yaw_meas + ego_yaw   # detection yaw, world frame
    for i, r in enumerate(recs):
        v = float(np.hypot(r.kf_vx, r.kf_vy))
        if v >= thr:
            vdir_world[i] = float(np.arctan2(r.kf_vy, r.kf_vx))
            reliable[i] = True

    out = [0.0] * n
    if not reliable.any():
        # fully static — detection yaw as is (ego frame)
        return [_wrap(float(y)) for y in yaw_meas]

    ridx = np.where(reliable)[0]
    for i in range(n):
        if reliable[i]:
            # velocity direction → ego frame
            out[i] = _wrap(float(vdir_world[i] - ego_yaw[i]))
        else:
            # low speed — resolve flip by aligning detection yaw to the nearest reliable frame's motion direction
            j = int(ridx[np.argmin(np.abs(ridx - i))])
            flip = abs(_wrap(yaw_world[i] - vdir_world[j])) >= (np.pi / 2)
            y = yaw_meas[i] + (np.pi if flip else 0.0)
            out[i] = _wrap(float(y))
    return out


# ── STOP_SIGN yaw → opposite of the lane's heading ──────────────────
def _stop_sign_lane_yaw_world(avm, sx: float, sy: float) -> float | None:
    """At a static stop sign's world position (sx, sy), the world yaw 'opposite' to the nearest lane's heading.

    Select the lane the same way as at_stop_sign_'s stop_sign_lane convention:
      among nearby (10m) lanes, the non-intersection lane whose ls.right_lane_boundary
      endpoint (= near the stop line) is closest to the sign position. If none, the
      nearest including intersections.
    Return the opposite (+π) of the selected lane's centerline heading (centerline[-1]-centerline[0]).
    (A stop sign faces vehicles approaching along the heading, so heading = -lane_dir.)
    Returns None if there is no nearby lane.
    """
    pos = np.array([float(sx), float(sy)], dtype=np.float64)
    try:
        ls_list = list(avm.get_nearby_lane_segments(pos, 10))
    except Exception:
        return None
    if not ls_list:
        return None

    def _end_dist(ls) -> float:
        return float(np.linalg.norm(pos - np.asarray(ls.right_lane_boundary.xyz[-1])[:2]))

    best, best_d = None, np.inf
    for ls in ls_list:
        if getattr(ls, "is_intersection", False):
            continue
        d = _end_dist(ls)
        if d < best_d:
            best_d, best = d, ls
    if best is None:   # if no non-intersection, the nearest including intersections
        for ls in ls_list:
            d = _end_dist(ls)
            if d < best_d:
                best_d, best = d, ls
    if best is None:
        return None

    cl = avm.get_lane_segment_centerline(best.id)
    o = np.asarray(cl[-1], dtype=np.float64) - np.asarray(cl[0], dtype=np.float64)
    if float(np.hypot(o[0], o[1])) < 1e-6:
        return None
    return _wrap(float(np.arctan2(o[1], o[0])) + np.pi)   # opposite of the heading


# ── core ──────────────────────────────────────────────────────────
def build_output_df(tracks: list[Track],
                    params: PostProcessParams,
                    log_timestamps: list[int] | None = None,
                    avm=None,
                    ) -> tuple[pd.DataFrame, dict[int, str]]:
    """Collect only confirmed tracks and build a DataFrame in sm_annotations.feather format.

    Output columns: track_uuid, timestamp_ns, tx_m, ty_m, tz_m,
              qw, qx, qy, qz, length_m, width_m, height_m, score, category

    Returns
    -------
    df : pd.DataFrame
    uuid_map : dict[int, str]
        Track.track_id (int) → newly assigned UUID4 string. Used by summarize_tracks
        when emitting per-track metadata (class_scores, etc.) keyed by uuid.
        Unconfirmed tracks are not included in this map (matches df).
    """
    rows = []
    n_emit_tracks = 0
    uuid_map: dict[int, str] = {}
    # interpolation must fill only 'actual log frame timestamps'.
    # Fake timestamps created by computation cause a KeyError in downstream eval (ego pose/cache lookup).
    _log_ts_sorted = sorted(int(t) for t in log_timestamps) if log_timestamps else None

    # ── [ABLATION lever2] relabel_only — keep raw detection + replace only track_uuid with the consistent ID.
    if params.relabel_only:
        for tr in tracks:
            if not tr.frame_records:
                continue            # regardless of confirmed, output all tracks (preserve DetA)
            if params.relabel_confirmed_only and not tr.is_confirmed:
                continue            # single-frame junk removal mode
            n_emit_tracks += 1
            new_uuid = str(uuid.uuid4())
            uuid_map[int(tr.track_id)] = new_uuid
            recs = sorted(tr.frame_records, key=lambda x: int(x.timestamp_ns))

            def _emit(ts, txw, tyw, z, yaw, l, w, h, sc, cat, ex, ey, eyaw, cxx, cxy, cyy):
                tx_ego, ty_ego = _world_to_ego(txw, tyw, ex, ey, eyaw)
                qw, qx, qy, qz = _yaw_to_quat(_wrap(float(yaw)))
                rows.append({
                    "track_uuid": new_uuid, "timestamp_ns": int(ts),
                    "tx_m": float(tx_ego), "ty_m": float(ty_ego), "tz_m": float(z),
                    "qw": qw, "qx": qx, "qy": qy, "qz": qz,
                    "length_m": float(l), "width_m": float(w), "height_m": float(h),
                    "score": float(sc), "category": str(cat),
                    "cov_xx_world": float(cxx), "cov_xy_world": float(cxy), "cov_yy_world": float(cyy),
                })

            # nominal dt (frame interval) — median of consecutive record time differences.
            difs = [int(recs[i + 1].timestamp_ns) - int(recs[i].timestamp_ns) for i in range(len(recs) - 1)]
            ndt = float(np.median(difs)) if difs else 0.0

            for idx, r in enumerate(recs):
                # IMM-active track (imm_yaw_world present) → output IMM mixed posterior.
                # CV single track → output existing raw detection (relabel_only's original behavior).
                # coords: kf_x/kf_y are world frame (the xy of IMM mixed synced by the tracker).
                #         _world_to_ego inside _emit already handles the world→ego transform.
                # yaw: imm_yaw_world (world frame) → emit after subtracting to ego frame.
                if r.imm_yaw_world is not None:
                    _emit(r.timestamp_ns, r.kf_x, r.kf_y, r.z_meas,
                          r.imm_yaw_world - r.ego_yaw,
                          r.length_meas, r.width_meas, r.height_meas, r.score_meas, r.category_meas,
                          r.ego_x, r.ego_y, r.ego_yaw, r.cov_xx, r.cov_xy, r.cov_yy)
                else:
                    _emit(r.timestamp_ns, r.meas_x_world, r.meas_y_world, r.z_meas, r.yaw_meas,
                          r.length_meas, r.width_meas, r.height_meas, r.score_meas, r.category_meas,
                          r.ego_x, r.ego_y, r.ego_yaw, r.cov_xx, r.cov_xy, r.cov_yy)
                # gap interpolation — fill the 'actual log frames' between two records by interpolation.
                # Must pick from _log_ts_sorted (actual frame timestamps) or eval lookup breaks.
                if (params.interpolate_max_gap > 0 and _log_ts_sorted and idx + 1 < len(recs)):
                    r2 = recs[idx + 1]
                    t0, t1 = int(r.timestamp_ns), int(r2.timestamp_ns)
                    # actual log timestamps existing between (t0, t1) (excluding both ends)
                    lo = bisect.bisect_right(_log_ts_sorted, t0)
                    hi = bisect.bisect_left(_log_ts_sorted, t1)
                    mids = _log_ts_sorted[lo:hi]
                    if 1 <= len(mids) <= params.interpolate_max_gap and t1 > t0:
                        # if the interpolated positions are also IMM-active, use kf_x/y + imm_yaw.
                        use_imm_interp = (r.imm_yaw_world is not None
                                          and r2.imm_yaw_world is not None)
                        for ts_i in mids:
                            a = (ts_i - t0) / (t1 - t0)   # 0..1 interpolation factor (proportional to actual time)
                            lerp = lambda u, v: float(u + a * (v - u))
                            if use_imm_interp:
                                src_x1, src_y1 = r.kf_x, r.kf_y
                                src_x2, src_y2 = r2.kf_x, r2.kf_y
                                src_yaw_ego = r.imm_yaw_world - r.ego_yaw
                            else:
                                src_x1, src_y1 = r.meas_x_world, r.meas_y_world
                                src_x2, src_y2 = r2.meas_x_world, r2.meas_y_world
                                src_yaw_ego = r.yaw_meas
                            _emit(ts_i,
                                  lerp(src_x1, src_x2), lerp(src_y1, src_y2),
                                  lerp(r.z_meas, r2.z_meas), src_yaw_ego,
                                  lerp(r.length_meas, r2.length_meas), lerp(r.width_meas, r2.width_meas),
                                  lerp(r.height_meas, r2.height_meas),
                                  0.9 * min(r.score_meas, r2.score_meas), r.category_meas,
                                  lerp(r.ego_x, r2.ego_x), lerp(r.ego_y, r2.ego_y), lerp(r.ego_yaw, r2.ego_yaw),
                                  -1.0, 0.0, 0.0)   # cov_xx=-1 = interpolated marker (unused by eval, identifies for qualitative validation)
        if not rows:
            df = pd.DataFrame(columns=[
                "track_uuid", "timestamp_ns", "tx_m", "ty_m", "tz_m",
                "qw", "qx", "qy", "qz", "length_m", "width_m", "height_m",
                "score", "category", "cov_xx_world", "cov_xy_world", "cov_yy_world"])
        else:
            df = pd.DataFrame(rows).sort_values(["timestamp_ns", "track_uuid"]).reset_index(drop=True)
        df.attrs["n_emit_tracks"] = n_emit_tracks
        return df, uuid_map

    # ── IoU clean pre-pass — compute removal targets among bbox-overlapping tracks in one frame (shorter one) ──
    # Rotated IoU using world frame box (kf_x/kf_y, track-level l/w, world yaw). Confirmed only.
    # REGULAR_VEHICLE only (large objects like BUS/TRUCK are excluded since their boxes are big). loser age>=2.
    removed_track_ids: set = set()
    if params.iou_clean_enabled:
        _clean_tracks = []
        for tr in tracks:
            if not tr.is_confirmed or not tr.frame_records:
                continue
            if (tr.get_rep_class() or "") != "REGULAR_VEHICLE":
                continue
            _l, _w, _h, _sc = _track_level_lwhs(tr, params)
            _frames = {}
            _cmx, _cmy = [], []
            for r in tr.frame_records:
                _yw = (float(r.imm_yaw_world) if r.imm_yaw_world is not None
                       else float(r.yaw_meas) + float(r.ego_yaw))
                _frames[int(r.timestamp_ns)] = (float(r.kf_x), float(r.kf_y), _l, _w, _yw)
                _cmx.append(float(r.meas_x_world)); _cmy.append(float(r.meas_y_world))
            _cst = classify_static(_cmx, _cmy, params.static_tau_static, params.static_tau_motion)
            _clean_tracks.append({"key": int(tr.track_id),
                                  "n": len(tr.frame_records),
                                  "static": _cst,
                                  "frames": _frames})
        removed_track_ids = iou_clean_removed(
            _clean_tracks, params.iou_clean_thr,
            protect_both_min=params.iou_clean_protect_both)

    for tr in tracks:
        if not tr.is_confirmed:
            continue
        if not tr.frame_records:
            continue
        if int(tr.track_id) in removed_track_ids:
            continue   # IoU clean — remove noise track
        n_emit_tracks += 1

        # track-level constants
        l_avg, w_avg, h_avg, s_avg = _track_level_lwhs(tr, params)
        rep_class = tr.get_rep_class() or "REGULAR_VEHICLE"
        new_uuid = str(uuid.uuid4())
        uuid_map[int(tr.track_id)] = new_uuid

        # track-level motion-based yaw flip correction (resolve ±180° ambiguity of detection yaw).
        resolved_yaws = _resolve_track_yaws(tr, params)

        # static detection (measurement world position based, per-track). The posterior can be
        # artificially fixed by the static v=0 forcing, so use the measurement (min/max diff) that
        # reflects detection noise/actual movement as is. A flag common to all track frames.
        _mx = [float(r.meas_x_world) for r in tr.frame_records]
        _my = [float(r.meas_y_world) for r in tr.frame_records]
        track_is_static = classify_static(
            _mx, _my, params.static_tau_static, params.static_tau_motion)

        # jitter-static yaw freeze — unify the output yaw of a pinwheel track to the dominant direction.
        # IMM-active tracks only. Both gate and RANSAC use MEASUREMENT world yaw — the IMM output
        # tracks flips continuously and smears the axis, but the measurement axis (mod180) is stable,
        # so the dominant direction is accurate.
        _was_frozen = False
        if (params.yaw_freeze_enabled
                and all(r.imm_yaw_world is not None for r in tr.frame_records)):
            _yw = [_wrap(float(r.yaw_meas) + float(r.ego_yaw)) for r in tr.frame_records]
            _theta = jitter_freeze_yaw(_yw, _mx, _my,
                                       params.yaw_freeze_tau_total,
                                       params.yaw_freeze_tau_net)
            if _theta is not None:
                resolved_yaws = [_wrap(_theta - float(r.ego_yaw))
                                 for r in tr.frame_records]
                _was_frozen = True

        # dynamic velocity yaw — align the yaw of a genuinely moving track (net_disp≥tau_dyn) to the velocity direction.
        # Exclude frozen static spinners (_was_frozen). IMM-active tracks only.
        if (params.dyn_yaw_enabled and not _was_frozen
                and all(r.imm_yaw_world is not None for r in tr.frame_records)):
            _recs = tr.frame_records
            _egy = [float(r.ego_yaw) for r in _recs]
            _wy = [_wrap(resolved_yaws[i] + _egy[i]) for i in range(len(_recs))]
            _vx = [float(r.kf_vx) for r in _recs]
            _vy = [float(r.kf_vy) for r in _recs]
            _wyd = dynamic_velocity_yaw(_wy, _vx, _vy, _mx, _my,
                                        params.dyn_yaw_tau_dyn, params.dyn_yaw_vmin)
            if _wyd is not None:
                resolved_yaws = [_wrap(_wyd[i] - _egy[i]) for i in range(len(_recs))]

        # STOP_SIGN yaw → opposite (anti-parallel) of the nearest lane's heading. Replace the static
        # sign's raw detection yaw error with the map centerline (satisfies the at_stop_sign pass condition).
        # The sign is static, so compute the world yaw from a single median of measured world positions, then transform per-frame to ego.
        if (params.stop_sign_yaw_to_lane and avm is not None
                and rep_class == "STOP_SIGN"):
            _sx = float(np.median(_mx))
            _sy = float(np.median(_my))
            _yaw_w = _stop_sign_lane_yaw_world(avm, _sx, _sy)
            if _yaw_w is not None:
                resolved_yaws = [_wrap(_yaw_w - float(r.ego_yaw))
                                 for r in tr.frame_records]

        for ridx_f, r in enumerate(tr.frame_records):
            # world → ego position inverse transform
            tx_ego, ty_ego = _world_to_ego(
                r.kf_x, r.kf_y, r.ego_x, r.ego_y, r.ego_yaw,
            )

            # yaw : detection yaw with flip resolved by motion (ego frame).
            yaw = resolved_yaws[ridx_f]
            qw, qx, qy, qz = _yaw_to_quat(yaw)

            # [ABLATION lever1] per-frame raw score/size (keep track aggregation, gain tuning granularity)
            out_score = float(r.score_meas) if params.per_frame_score else s_avg
            if params.per_frame_size:
                out_l = max(float(r.length_meas), params.min_l)
                out_w = max(float(r.width_meas),  params.min_w)
                out_h = max(float(r.height_meas), params.min_h)
            else:
                out_l, out_w, out_h = l_avg, w_avg, h_avg

            rows.append({
                "track_uuid": new_uuid,
                "timestamp_ns": int(r.timestamp_ns),
                "tx_m": float(tx_ego),
                "ty_m": float(ty_ego),
                "tz_m": float(r.z_meas),
                "qw": qw, "qx": qx, "qy": qy, "qz": qz,
                "length_m": out_l,
                "width_m":  out_w,
                "height_m": out_h,
                "score":    out_score,
                "category": rep_class,
                # KF posterior cov xy block — WORLD frame (rotated by anchor_ego on the server).
                "cov_xx_world": float(r.cov_xx),
                "cov_xy_world": float(r.cov_xy),
                "cov_yy_world": float(r.cov_yy),
                # static detection flag (per-track, measurement position based). For visualization/diagnostics.
                "is_static": int(track_is_static),
                # measured world position/yaw (for server live static decision/yaw freeze). Unused by eval.
                "meas_x_world": float(r.meas_x_world),
                "meas_y_world": float(r.meas_y_world),
                "meas_yaw_world": _wrap(float(r.yaw_meas) + float(r.ego_yaw)),
                # per-frame world velocity (for server dynamic-velocity-yaw). Unused by eval.
                "kf_vx": float(r.kf_vx),
                "kf_vy": float(r.kf_vy),
            })

    if not rows:
        df = pd.DataFrame(columns=[
            "track_uuid", "timestamp_ns",
            "tx_m", "ty_m", "tz_m",
            "qw", "qx", "qy", "qz",
            "length_m", "width_m", "height_m",
            "score", "category",
            "cov_xx_world", "cov_xy_world", "cov_yy_world", "is_static",
            "meas_x_world", "meas_y_world", "meas_yaw_world", "kf_vx", "kf_vy",
        ])
    else:
        df = pd.DataFrame(rows)
        df = df.sort_values(["timestamp_ns", "track_uuid"]).reset_index(drop=True)

    df.attrs["n_emit_tracks"] = n_emit_tracks
    return df, uuid_map


# ── diagnostic statistics ─────────────────────────────────────────
def summarize_tracks(tracks: list[Track],
                     uuid_map: dict[int, str] | None = None,
                     imm_summary_by_debug_uuid: dict[str, dict] | None = None,
                     debug_uuid_for_track_id: dict[int, str] | None = None) -> dict:
    """Diagnostic summary of the tracking result.

    If uuid_map is given, add a per-track section (class_scores, etc.). Emit only
    tracks that are confirmed and present in uuid_map — 1:1 matching with the output
    rows of build_output_df.

    imm_summary_by_debug_uuid: result of tracker.debug_recorder.summarize_per_track() —
    keys are the temporary uuid of the debug stage (str(track_id)). Receives the
    track_id → debug_uuid mapping via debug_uuid_for_track_id and merges into
    per_track[new_uuid].imm.
    """
    n_total = len(tracks)
    n_conf = sum(1 for t in tracks if t.is_confirmed)
    n_init_only = n_total - n_conf
    n_emit_frames = sum(len(t.frame_records) for t in tracks if t.is_confirmed)
    n_drop_frames = sum(len(t.frame_records) for t in tracks if not t.is_confirmed)
    age_hist = sorted([t.age for t in tracks if t.is_confirmed], reverse=True)[:5]
    out: dict = {
        "n_tracks_total": n_total,
        "n_tracks_confirmed": n_conf,
        "n_tracks_unconfirmed_dropped": n_init_only,
        "n_frames_emit": n_emit_frames,
        "n_frames_dropped": n_drop_frames,
        "top5_confirmed_track_age": age_hist,
    }
    if uuid_map is not None:
        per_track: dict[str, dict] = {}
        for tr in tracks:
            if not tr.is_confirmed:
                continue
            new_uuid = uuid_map.get(int(tr.track_id))
            if new_uuid is None:
                # build_output_df emits only with both confirmed + frame_records.
                # A track missing here is the case where frame_records was empty and it was not output to df.
                continue
            per_track[new_uuid] = {
                "class_scores": {k: float(v) for k, v in tr.class_scores.items()},
                "rep_class":    tr.get_rep_class() or "REGULAR_VEHICLE",
                "age":          int(tr.age),
                "n_frames":     int(len(tr.frame_records)),
                "is_confirmed": bool(tr.is_confirmed),
            }
            # merge IMM summary — lookup by debug_uuid. CV-only tracks may also have keys
            # (DebugRecorder accumulates the summary in record_cv_4d too).
            if imm_summary_by_debug_uuid is not None and debug_uuid_for_track_id is not None:
                debug_uuid = debug_uuid_for_track_id.get(int(tr.track_id))
                if debug_uuid is not None:
                    imm_sec = imm_summary_by_debug_uuid.get(debug_uuid)
                    if imm_sec is not None:
                        per_track[new_uuid]["imm"] = imm_sec
        out["per_track"] = per_track
    return out
