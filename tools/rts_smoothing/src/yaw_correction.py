"""Stage-1 yaw correction — dynamic-only.

Does not correct the yaw of stationary objects. This stage decides only the
per-frame ±π flip for tracks that have sufficient motion signal.

Pipeline
--------
[1] skip check
    · category ∈ {EGO_VEHICLE, PEDESTRIAN}        → skip_<category>
    · track length N < n_min                       → skip_short
    · world displacement d_max < τ_static (noise floor)  → skip_static
[2] _decide_dynamic — bidirectional, nearest-with-threshold
    · for each frame i, pick the frame j with minimum |i − j| among those with
      |p[j] − p[i]| ≥ τ_motion as the motion oracle (search forward·backward simultaneously).
    · the displacement vector is always oriented forward in time to match the
      motion travel direction.
    · reliable frame: |wrap(yaw_world[i] − yaw_base)| ≥ π/2 → flip.
    · other frames: anchor-snap to the corrected yaw of the nearest reliable frame.
    · very rare case where not a single reliable frame is found → skip_no_motion.

Constraints
-----------
Every decision is only a per-frame boolean flip ∈ {0, 1}; the application is exactly +0 or +π.
The yaw value is never changed to anything else → bbox body axis (major axis) absolutely preserved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

from .track_state import TrackState, Tracks


@dataclass
class YawCorrectionParams:
    tau_static: float = 2.0       # m  — if d_max < τ_static, judged static → skip_static
    tau_motion: float = 0.2       # m  — bidirectional Δp* threshold (reliable decision)
    n_min: int = 10               # frame count < n_min → skip_short
    skip_categories: tuple = ("EGO_VEHICLE", "PEDESTRIAN")
    # ── static object yaw stabilization (optional). If True, unify the yaw of a track with
    # d_max < τ_static toward the majority direction (+π flip only on minority frames). The yaw
    # value itself / bbox position / size / score are never touched. If False (default), pass through raw as skip_static.
    static_yaw_stabilize: bool = False


@dataclass
class TrackCorrectionInfo:
    uuid: str
    category: str
    # corrected_dynamic | corrected_dynamic_anchor
    # | skip_static | skip_no_motion | skip_short | skip_<category>
    status: str
    n_frames: int
    n_flipped: int
    n_reliable: int = 0               # number of reliable frames (decided directly)
    n_lowspeed: int = 0               # number of anchor-snapped frames
    flip_transitions: int = 0
    d_max: Optional[float] = None     # world displacement bbox diagonal (for diagnostics)
    is_dynamic: Optional[bool] = None  # d_max ≥ τ_static


# ── helpers ─────────────────────────────────────────────────────────────────
def _wrap(x):
    return np.arctan2(np.sin(x), np.cos(x))


def _ego_pose_lookup(tracks: Tracks):
    ts_arr = tracks.ego_poses_ts
    xyz_arr = tracks.ego_poses_xyz
    yaw_arr = tracks.ego_poses_yaw

    if len(ts_arr) == 0:
        def _f(_ts):
            return np.zeros(3, dtype=np.float64), 0.0
        return _f

    def _f(ts: int):
        idx = int(np.searchsorted(ts_arr, ts))
        if idx < len(ts_arr) and int(ts_arr[idx]) == int(ts):
            return xyz_arr[idx], float(yaw_arr[idx])
        if idx == 0:
            return xyz_arr[0], float(yaw_arr[0])
        if idx >= len(ts_arr):
            return xyz_arr[-1], float(yaw_arr[-1])
        before, after = int(ts_arr[idx - 1]), int(ts_arr[idx])
        if (ts - before) <= (after - ts):
            return xyz_arr[idx - 1], float(yaw_arr[idx - 1])
        return xyz_arr[idx], float(yaw_arr[idx])

    return _f


def _to_world(track: TrackState, ego_at) -> tuple[np.ndarray, np.ndarray]:
    """ego frame box → world (city) frame. Returns: (p_world (N,2), yaw_world (N,))."""
    N = len(track)
    p = np.zeros((N, 2), dtype=np.float64)
    y = np.zeros(N, dtype=np.float64)
    for i in range(N):
        ts = int(track.timestamps_ns[i])
        cxyz, cyaw = ego_at(ts)
        c, s = math.cos(cyaw), math.sin(cyaw)
        tx, ty = float(track.translations_m[i, 0]), float(track.translations_m[i, 1])
        p[i, 0] = cxyz[0] + c * tx - s * ty
        p[i, 1] = cxyz[1] + s * tx + c * ty
        y[i] = float(track.yaws_rad[i]) + cyaw
    return p, y


def _flip_to_yaws(yaws: np.ndarray, flip: np.ndarray) -> np.ndarray:
    out = yaws.astype(np.float64).copy()
    out[flip] = _wrap(out[flip] + math.pi)
    return out


def _flip_to_quats(quats: Optional[np.ndarray], flip: np.ndarray) -> Optional[np.ndarray]:
    """q_z(π) ⊗ q_old. Since q_z(π) = (qw=0, qx=0, qy=0, qz=1),
        (qw, qx, qy, qz) → (-qz, -qy, qx, qw).

    Canonicalization: since q and −q represent the same rotation, if the resulting
    quaternion's qw becomes negative, flip the entire sign to normalize to the
    canonical form with qw ≥ 0. This way, extracting yaw downstream via
    `2·atan2(qz, qw)` always falls in the [−π, π] range.
    """
    if quats is None:
        return None
    out = quats.astype(np.float64).copy()
    if not flip.any():
        return out
    qw = out[flip, 0].copy()
    qx = out[flip, 1].copy()
    qy = out[flip, 2].copy()
    qz = out[flip, 3].copy()
    out[flip, 0] = -qz
    out[flip, 1] = -qy
    out[flip, 2] = qx
    out[flip, 3] = qw
    # canonicalize: so that qw ≥ 0 (q and −q are the same rotation).
    neg = out[:, 0] < 0
    if neg.any():
        out[neg] *= -1.0
    return out


# ── core decisions ──────────────────────────────────────────────────────────
def _decide_static_majority_flip(yaws: np.ndarray) -> np.ndarray:
    """Return a flip mask to unify a static track's yaw toward the majority direction.

    Does not change the yaw value itself, only decides the ±π flip — minority
    frames (opposite direction to the majority) are True, majority frames are
    False. When the caller applies it via _flip_to_yaws / _flip_to_quats, all
    frames point the same direction. bbox position / size / score are preserved.

    Algorithm — "2·yaw" trick
      1. circular mean of 2·yaw = atan2(Σ sin(2θ), Σ cos(2θ)) → "line direction"
         (estimate the majority's common line direction while ignoring ±π flips).
      2. classify whether each frame's yaw is along the line direction (+) or opposite (+π):
         |wrap(yaw − line_dir)| ≤ π/2 → forward, otherwise → backward.
      3. decide the larger group as the majority — flip=True only for the minority.

    On a tie (forward == backward), forward is treated as the majority so only backward is flipped.
    """
    yaws = np.asarray(yaws, dtype=np.float64)
    N = len(yaws)
    if N < 2:
        return np.zeros(N, dtype=bool)
    line_dir = 0.5 * math.atan2(
        float(np.sum(np.sin(2.0 * yaws))),
        float(np.sum(np.cos(2.0 * yaws))),
    )
    diff = _wrap(yaws - line_dir)
    forward = np.abs(diff) <= math.pi / 2
    if forward.sum() >= N - forward.sum():
        return ~forward            # forward is majority → flip only backward
    return forward                  # backward is majority → flip only forward


def _decide_dynamic(p_world: np.ndarray, yaw_world: np.ndarray,
                    params: YawCorrectionParams):
    """Bidirectional, threshold-anchored motion direction.

    For each frame i, use the frame j with smallest |i − j| among those with
    |p[j] − p[i]| ≥ τ_motion as the motion oracle (the temporally nearest
    meaningful displacement). Because forward·backward are searched simultaneously,
    a stop frame just before departure, a stop frame just after arrival, and
    low-speed crawling can all be promoted to reliable.

    If both exist at the same |i − j|, prefer the one with larger displacement
    (= higher signal-to-noise). yaw_base is always computed from the forward-in-time displacement.

    Returns
    -------
    None
        when the track has not a single |Δp| ≥ τ_motion pair (caller uses skip_no_motion).
    (flip, n_reliable, n_lowspeed)
        otherwise. If n_reliable == N, all frames were decided directly;
        if n_reliable < N, the shortfall is filled by anchor snap.
    """
    N = len(yaw_world)
    flip = np.zeros(N, dtype=bool)
    reliable = np.zeros(N, dtype=bool)

    for i in range(N):
        diffs = p_world - p_world[i]
        dists = np.linalg.norm(diffs, axis=1)
        mask = dists >= params.tau_motion
        mask[i] = False
        if not mask.any():
            continue
        valid_idx = np.where(mask)[0]
        order = np.lexsort((-dists[valid_idx], np.abs(valid_idx - i)))
        j = int(valid_idx[order[0]])
        # forward-in-time displacement (if j > i, p[j]−p[i]; if j < i, p[i]−p[j])
        if j > i:
            dp = p_world[j] - p_world[i]
        else:
            dp = p_world[i] - p_world[j]
        yaw_base = math.atan2(dp[1], dp[0])
        reliable[i] = True
        flip[i] = abs(float(_wrap(yaw_world[i] - yaw_base))) >= math.pi / 2

    n_reliable = int(reliable.sum())
    if n_reliable == 0:
        return None

    # anchor snap — non-reliable frames branch at ±π/2 from the nearest reliable's corrected yaw
    reliable_idx = np.where(reliable)[0]
    for i in range(N):
        if reliable[i]:
            continue
        j = int(reliable_idx[np.argmin(np.abs(reliable_idx - i))])
        yaw_anchor = yaw_world[j] + (math.pi if flip[j] else 0.0)
        flip[i] = abs(float(_wrap(yaw_world[i] - yaw_anchor))) >= math.pi / 2

    n_lowspeed = N - n_reliable
    return flip, n_reliable, n_lowspeed


def correct_track(track: TrackState, ego_at,
                  params: YawCorrectionParams) -> tuple[TrackCorrectionInfo, TrackState]:
    """Correct a single track. Returns (info, new_track).

    Flow
    ----
    1. skip check (category / length / d_max < τ_static).
    2. ego→world conversion.
    3. _decide_dynamic — if reliable frames are found, use that decision + anchor-snap the non-reliable ones.

    Static objects (d_max < τ_static) are excluded from correction — to prevent
    the regression of mistaking yaw noise for motion, and based on the analysis
    that scenario mining prompts have low dependence on stationary-object-heading.
    """
    N = len(track)
    cat = track.category

    if cat in params.skip_categories:
        info = TrackCorrectionInfo(uuid=track.uuid, category=cat, n_frames=N,
                                   status=f"skip_{cat.lower()}", n_flipped=0)
        return info, track

    if N < params.n_min:
        info = TrackCorrectionInfo(uuid=track.uuid, category=cat, n_frames=N,
                                   status="skip_short", n_flipped=0)
        return info, track

    p_world, yaw_world = _to_world(track, ego_at)
    dx = float(p_world[:, 0].max() - p_world[:, 0].min())
    dy = float(p_world[:, 1].max() - p_world[:, 1].min())
    d_max = math.sqrt(dx * dx + dy * dy)
    is_dynamic = d_max >= params.tau_static

    if not is_dynamic:
        # static track — if the option is on, unify yaw toward the majority direction (±π flip only, bbox preserved).
        if params.static_yaw_stabilize:
            flip = _decide_static_majority_flip(np.asarray(track.yaws_rad, dtype=np.float64))
            n_flipped = int(flip.sum())
            transitions = int(np.sum(flip[1:] != flip[:-1])) if N > 1 else 0
            new_track = TrackState(
                uuid=track.uuid,
                category=track.category,
                timestamps_ns=track.timestamps_ns,
                translations_m=track.translations_m,    # ← preserve bbox position
                yaws_rad=_flip_to_yaws(track.yaws_rad, flip),
                sizes_m=track.sizes_m,                  # ← preserve bbox size
                scores=track.scores,                    # ← preserve score
                quaternions=_flip_to_quats(track.quaternions, flip),
            )
            info = TrackCorrectionInfo(
                uuid=track.uuid, category=cat, n_frames=N,
                status="corrected_static_stabilized",
                n_flipped=n_flipped, flip_transitions=transitions,
                d_max=d_max, is_dynamic=False,
            )
            return info, new_track
        info = TrackCorrectionInfo(uuid=track.uuid, category=cat, n_frames=N,
                                   status="skip_static", n_flipped=0,
                                   d_max=d_max, is_dynamic=False)
        return info, track

    res = _decide_dynamic(p_world, yaw_world, params)
    if res is None:
        # d_max is sufficient but reliable is 0 — a nearly-never case (e.g., an
        # abnormal track where almost all displacement is concentrated only in
        # pairs with far-apart timestamps). Safe hold.
        info = TrackCorrectionInfo(uuid=track.uuid, category=cat, n_frames=N,
                                   status="skip_no_motion", n_flipped=0,
                                   d_max=d_max, is_dynamic=True)
        return info, track

    flip, n_reliable, n_lowspeed = res
    status = "corrected_dynamic" if n_reliable == N else "corrected_dynamic_anchor"
    n_flipped = int(flip.sum())
    transitions = int(np.sum(flip[1:] != flip[:-1])) if N > 1 else 0

    new_track = TrackState(
        uuid=track.uuid,
        category=track.category,
        timestamps_ns=track.timestamps_ns,
        translations_m=track.translations_m,
        yaws_rad=_flip_to_yaws(track.yaws_rad, flip),
        sizes_m=track.sizes_m,
        scores=track.scores,
        quaternions=_flip_to_quats(track.quaternions, flip),
    )

    info = TrackCorrectionInfo(
        uuid=track.uuid, category=cat, n_frames=N, status=status,
        n_flipped=n_flipped, n_reliable=n_reliable, n_lowspeed=n_lowspeed,
        flip_transitions=transitions, d_max=d_max, is_dynamic=True,
    )
    return info, new_track


def correct_yaws_stage1(tracks: Tracks,
                        params: Optional[YawCorrectionParams] = None
                        ) -> tuple[Tracks, dict]:
    """Apply stage-1 yaw correction to all tracks. Returns (new Tracks, summary).

    summary structure:
        {
          "total":     int,
          "by_status": {status: count},
          "by_status_flipped": {status: total flipped frames},
          "tracks":    [TrackCorrectionInfo dict, ...],
          "params":    YawCorrectionParams asdict,
        }
    """
    if params is None:
        params = YawCorrectionParams()
    ego_at = _ego_pose_lookup(tracks)

    new_tracks: dict[str, TrackState] = {}
    infos: list[TrackCorrectionInfo] = []
    by_status: dict[str, int] = {}
    by_status_flipped: dict[str, int] = {}

    for uuid, tr in tracks.tracks.items():
        info, new_tr = correct_track(tr, ego_at, params)
        new_tracks[uuid] = new_tr
        infos.append(info)
        by_status[info.status] = by_status.get(info.status, 0) + 1
        by_status_flipped[info.status] = by_status_flipped.get(info.status, 0) + info.n_flipped

    new = Tracks(
        log_id=tracks.log_id,
        split=tracks.split,
        tracks=new_tracks,
        ego_poses_ts=tracks.ego_poses_ts,
        ego_poses_xyz=tracks.ego_poses_xyz,
        ego_poses_yaw=tracks.ego_poses_yaw,
    )
    summary = {
        "total": len(infos),
        "by_status": by_status,
        "by_status_flipped": by_status_flipped,
        "tracks": [asdict(i) for i in infos],
        "params": asdict(params),
    }
    return new, summary
