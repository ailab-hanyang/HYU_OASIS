"""Cost matrix + gating + matching (Hungarian / Greedy branch).

Flow
----
1. Compute the n_meas × n_track cost matrix (L2 or Mahalanobis — config branch)
2. Gating — gate-violating cells are filled with INF (infinity)
   · L2 distance > max_association_dist_m (PEDESTRIAN/STATIC use a separate narrow gate)
   · class isolation (PEDESTRIAN ↔ non-PEDESTRIAN)
3. Matching algorithm branch (config: tracking.match_algorithm)
   · 'greedy'    : lock the (i, j) pairs from smallest cost first. A locked row/col is not reused.
                    Same as C++ multi_class_object_tracking.
   · 'hungarian' : scipy.optimize.linear_sum_assignment — minimizes the total cost sum.
                    May cause ID swaps in densely-populated areas (see greedy_vs_hungarian.html).
4. Discard INF / gate-violating pairs from the matching result
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .track import Track, Measurement
from .kalman import H_MATRIX, build_R, KFParams


INF = 1e9


# ── Static (= non-moving) categories — per Argoverse 2 classification
# Measurements of these categories are gated by static_max_association_dist_m (usually very narrow).
STATIC_CATEGORIES = frozenset({
    "BOLLARD",
    "STOP_SIGN",
    "SIGN",
    "MOBILE_PEDESTRIAN_CROSSING_SIGN",
    "CONSTRUCTION_CONE",
    "CONSTRUCTION_BARREL",
})


# ── Class groups — for class_isolation_grouped.
# Class flicker within the same group (e.g. REGULAR_VEHICLE↔BOX_TRUCK, BICYCLE↔BICYCLIST) is allowed
# for association, while cross-group (BICYCLE↔VEHICLE, VEHICLE↔BOLLARD etc. — over-merge causing FP) is blocked.
_CLASS_GROUPS = {
    "vehicle": {"REGULAR_VEHICLE", "LARGE_VEHICLE", "BOX_TRUCK", "TRUCK", "TRUCK_CAB",
                "VEHICULAR_TRAILER", "BUS", "ARTICULATED_BUS", "SCHOOL_BUS",
                "RAILED_VEHICLE", "MESSAGE_BOARD_TRAILER", "TRAFFIC_LIGHT_TRAILER"},
    "two_wheeler": {"BICYCLE", "BICYCLIST", "MOTORCYCLE", "MOTORCYCLIST",
                    "WHEELED_DEVICE", "WHEELED_RIDER"},
    "pedestrian": {"PEDESTRIAN", "STROLLER", "DOG", "OFFICIAL_SIGNALER"},
    "static": {"BOLLARD", "STOP_SIGN", "SIGN", "MOBILE_PEDESTRIAN_CROSSING_SIGN",
               "CONSTRUCTION_CONE", "CONSTRUCTION_BARREL"},
}
_CAT_TO_GROUP = {c: g for g, cs in _CLASS_GROUPS.items() for c in cs}


def _class_group(cat: str) -> str:
    # an unregistered class is its own group (unique isolation)
    return _CAT_TO_GROUP.get(cat, cat)


@dataclass
class AssociationParams:
    cost_metric: str                     # 'l2' or 'maha'
    max_association_dist_m: float
    ped_max_association_dist_m: float    # PEDESTRIAN measurements only (usually smaller)
    static_max_association_dist_m: float # STATIC category measurements only (narrowest)
    pedestrian_class_isolation: bool
    class_isolation: bool                 # True blocks when measurement class != track rep class (exactly) (strict)
    class_isolation_grouped: bool         # True blocks only cross-group — allows same-group flicker (Vehicle↔Truck)
    class_mismatch_cost_factor: float     # >1.0 applies a multiplicative cost penalty on class mismatch instead of hard-block (soft)
    class_mismatch_exact: bool            # apply the soft penalty on an exact-class (True) / cross-group (False) basis
    class_isolation_min_score: float      # apply class isolation only to measurements with score >= this value (exempt low-confidence flicker)
    class_iso_bypass_dist: float          # if l2 <= this value, exempt from class isolation (allow nearby cross-class = flicker)
    confirmed_max_association_dist_m: float  # >0 makes confirmed tracks use this narrow gate (blocks lead-vehicle over-merge)
    match_algorithm: str                 # 'greedy' or 'hungarian'
    # ── yaw_change_gate — when a track is moving, the displacement direction must match the track heading.
    # Blocks the case where a vehicle crosses lanes and ID-swaps onto another vehicle in the adjacent lane.
    # Active condition: track.age >= min_age AND ||v|| >= min_speed.
    # angle = angle(displacement_vector, track_velocity_vector). If this exceeds max_deg, it is a gate violation.
    yaw_gate_enabled: bool
    yaw_gate_max_deg: float
    yaw_gate_min_speed: float
    yaw_gate_min_age: int
    # ── lateral (perpendicular to heading) association gate — blocks sideways-jump mismatches ──
    # Decompose the residual into longitudinal/lateral relative to the track heading → reject if lateral
    # component > lat_max. A car almost never moves sideways. Only when ‖v‖ ≥ lat_min_speed (direction trusted).
    # lat_max=0 disables it.
    adaptive_gate_lat_max: float = 0.0     # m — lateral residual upper bound (0=disabled)
    adaptive_gate_lat_min_speed: float = 1.0  # m/s — apply the lateral gate only above this speed

    @classmethod
    def from_config(cls, t_cfg: dict) -> "AssociationParams":
        return cls(
            cost_metric=str(t_cfg.get("cost_metric", "maha")).lower(),
            max_association_dist_m=float(t_cfg.get("max_association_dist_m", 3.0)),
            ped_max_association_dist_m=float(
                t_cfg.get("pedestrian_max_association_dist_m",
                          t_cfg.get("max_association_dist_m", 3.0))
            ),
            static_max_association_dist_m=float(
                t_cfg.get("static_max_association_dist_m",
                          t_cfg.get("max_association_dist_m", 3.0))
            ),
            pedestrian_class_isolation=bool(t_cfg.get("pedestrian_class_isolation", True)),
            class_isolation=bool(t_cfg.get("class_isolation", False)),
            class_isolation_grouped=bool(t_cfg.get("class_isolation_grouped", False)),
            class_mismatch_cost_factor=float(t_cfg.get("class_mismatch_cost_factor", 1.0)),
            class_mismatch_exact=bool(t_cfg.get("class_mismatch_exact", False)),
            class_isolation_min_score=float(t_cfg.get("class_isolation_min_score", 0.0)),
            class_iso_bypass_dist=float(t_cfg.get("class_iso_bypass_dist", 0.0)),
            confirmed_max_association_dist_m=float(t_cfg.get("confirmed_max_association_dist_m", 0.0)),
            match_algorithm=str(t_cfg.get("match_algorithm", "greedy")).lower(),
            yaw_gate_enabled=bool((t_cfg.get("yaw_change_gate") or {}).get("enabled", False)),
            yaw_gate_max_deg=float((t_cfg.get("yaw_change_gate") or {}).get("max_deg", 90.0)),
            yaw_gate_min_speed=float((t_cfg.get("yaw_change_gate") or {}).get("min_speed", 1.0)),
            yaw_gate_min_age=int((t_cfg.get("yaw_change_gate") or {}).get("min_age", 3)),
            adaptive_gate_lat_max=float((t_cfg.get("adaptive_gate") or {}).get("lat_max_m", 0.0)),
            adaptive_gate_lat_min_speed=float((t_cfg.get("adaptive_gate") or {}).get("lat_min_speed", 1.0)),
        )


# ── single-pair cost ───────────────────────────────────────────────
def _l2(meas: Measurement, track: Track) -> float:
    return float(np.hypot(meas.tx - track.state[0], meas.ty - track.state[1]))


def _mahalanobis(meas: Measurement, track: Track,
                 kf_params: KFParams) -> float:
    """Mahalanobis on xy with bounds (same as C++).

    inv_cov is the inverse of (track.cov[:2,:2] + R) — the innovation covariance.
    sqrt of `(z − μ)^T S^{-1} (z − μ)`.
    Upper/lower bounds: l2 / 5  ≤  maha  ≤  l2 · 3
    """
    diff = np.array([meas.tx - track.state[0], meas.ty - track.state[1]],
                    dtype=np.float64)
    P_xy = track.cov[:2, :2]
    # R is also anisotropic along the measurement's yaw_world direction (optional)
    R = build_R(kf_params, meas.score, yaw_hint=float(meas.yaw_world))
    S = P_xy + R
    try:
        Sinv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        return _l2(meas, track) * 3.0
    md = float(np.sqrt(max(0.0, diff @ Sinv @ diff)))
    l2 = float(np.linalg.norm(diff))
    # C++ clamp
    if md > l2 * 3.0:
        md = l2 * 3.0
    if md < l2 / 5.0:
        md = l2 / 5.0
    return md


# ── cost matrix ─────────────────────────────────────────────────────
def build_cost_matrix(measurements: list[Measurement],
                      tracks: list[Track],
                      assoc_params: AssociationParams,
                      kf_params: KFParams) -> tuple[np.ndarray, np.ndarray]:
    """Returns the cost matrix + L2 distances (kept separately for gating).

    cost: shape (n_meas, n_tracks). Gate-violating cells are INF.
    l2_arr: shape (n_meas, n_tracks) — always L2 distance (re-checked in post-filter).
    """
    n_meas = len(measurements)
    n_tracks = len(tracks)
    cost = np.full((n_meas, n_tracks), INF, dtype=np.float64)
    l2_arr = np.full((n_meas, n_tracks), INF, dtype=np.float64)

    use_maha = assoc_params.cost_metric == "maha"
    max_d = assoc_params.max_association_dist_m
    ped_max_d = assoc_params.ped_max_association_dist_m
    static_max_d = assoc_params.static_max_association_dist_m
    iso = assoc_params.pedestrian_class_isolation

    for i, m in enumerate(measurements):
        m_is_ped = (m.category == "PEDESTRIAN")
        m_is_static = (m.category in STATIC_CATEGORIES)
        # gate branch based on measurement category (PED → ped, STATIC → static, otherwise → general)
        if m_is_ped:
            eff_max_d = ped_max_d
        elif m_is_static:
            eff_max_d = static_max_d
        else:
            eff_max_d = max_d

        for j, tr in enumerate(tracks):
            if not tr.is_init:
                continue

            l2 = _l2(m, tr)
            l2_arr[i, j] = l2
            # maturity-adaptive gate: a confirmed track trusts the KF prediction → narrow gate (blocks lead-vehicle over-merge).
            # a new track uses a loose gate (bootstrap). Active only when confirmed_max_association_dist_m > 0.
            eff_d = eff_max_d
            if assoc_params.confirmed_max_association_dist_m > 0.0 and tr.is_confirmed:
                eff_d = min(eff_max_d, assoc_params.confirmed_max_association_dist_m)
            if l2 > eff_d:
                continue   # gate violation — leave as INF

            # lateral (perpendicular to heading) gate — blocks sideways-jump mismatches. Decompose the residual
            # into longitudinal/lateral → reject if lateral component > lat_max. Direction from velocity (only when ‖v‖≥lat_min_speed; low speed distrusted).
            if assoc_params.adaptive_gate_lat_max > 0.0:
                _tvx = float(tr.state[2]); _tvy = float(tr.state[3])
                _spd = float(np.hypot(_tvx, _tvy))
                if _spd >= assoc_params.adaptive_gate_lat_min_speed and _spd > 1e-6:
                    _dx = m.tx - float(tr.state[0]); _dy = m.ty - float(tr.state[1])
                    _ct = _tvx / _spd; _st = _tvy / _spd
                    _lat = abs(-_st * _dx + _ct * _dy)   # component perpendicular to the heading
                    if _lat > assoc_params.adaptive_gate_lat_max:
                        continue   # lateral gate violation

            # Class isolation
            if iso:
                rep = tr.get_rep_class()
                t_is_ped = (rep == "PEDESTRIAN")
                if m_is_ped != t_is_ped:
                    continue   # block PED ↔ non-PED
            # class isolation/penalty — prevents multi-class tracks from over-merge (BICYCLE+VEHICLE etc.).
            # However, detector class flicker (Vehicle→Truck→Vehicle, misclassification of the same object) must be preserved:
            #   - strict   : block on any exact class difference (breaks flicker too — over-fragmentation risk)
            #   - grouped  : block only cross-group, allow same-group flicker (recommended)
            #   - soft     : cost penalty on class mismatch instead of blocking (handles flicker most smoothly)
            _rep = tr.get_rep_class() or m.category
            _class_mismatch_strict = (m.category != _rep)
            _class_mismatch_group = (_class_group(m.category) != _class_group(_rep))
            # isolation-exemption (flicker-preservation) conditions:
            #  - score-gate: score < min_score (low confidence = higher misdetection probability). [experiment: most are low-score so counterproductive]
            #  - dist-gate (recommended): l2 <= bypass_dist (very close to the track's predicted position = same object's class wobble).
            #    Far cross-class (over-merge) is still blocked. Selectively recovers only flicker fragmentation.
            _iso_active = (m.score >= assoc_params.class_isolation_min_score) \
                and (l2 > assoc_params.class_iso_bypass_dist)
            if assoc_params.class_isolation and _class_mismatch_strict and _iso_active:
                continue
            if assoc_params.class_isolation_grouped and _class_mismatch_group and _iso_active:
                continue

            # yaw_change_gate — displacement direction vs track heading.
            # Track heading = track.last_yaw (the yaw_world of the most recently matched measurement).
            # KF velocity is not used because it is noisy in the first 1-2 frames.
            # Blocks the case where a vehicle crosses lanes and ID-swaps onto another vehicle in the adjacent lane:
            #   - the track is stable (age >= min_age) and non-static (velocity >= min_speed)
            #   - the displacement vector reaches far enough to the measurement position (>=0.3 m)
            #   - reject if the angle between displacement and last_yaw exceeds max_deg
            if (assoc_params.yaw_gate_enabled
                    and tr.age >= assoc_params.yaw_gate_min_age
                    and tr.last_yaw is not None):
                tvx = float(tr.state[2]); tvy = float(tr.state[3])
                track_speed = float(np.hypot(tvx, tvy))
                if track_speed >= assoc_params.yaw_gate_min_speed:
                    dx = m.tx - float(tr.state[0])
                    dy = m.ty - float(tr.state[1])
                    disp_mag = float(np.hypot(dx, dy))
                    if disp_mag >= 0.3:   # small displacement has unstable direction → skip
                        track_dir = float(tr.last_yaw)   # world frame yaw
                        disp_dir = float(np.arctan2(dy, dx))
                        a = disp_dir - track_dir
                        diff_deg = abs(np.degrees(np.arctan2(np.sin(a), np.cos(a))))
                        # both forward / backward are OK — since yaw is the body direction, ±180° is the same lane.
                        # So compare only the distance to the nearest axis via min(diff_deg, 180 - diff_deg).
                        diff_axis = min(diff_deg, 180.0 - diff_deg)
                        if diff_axis > assoc_params.yaw_gate_max_deg:
                            continue   # yaw gate violation

            if use_maha:
                c = _mahalanobis(m, tr, kf_params)
            else:
                c = l2
            # soft class penalty — amplify cost on class mismatch (not blocking). Nearby flicker still matches
            # after the penalty, while a far lead vehicle (over-merge) is dropped by it. class_mismatch_exact=True
            # uses the exact-class basis (penalizes within-family too → combines strict's over-merge blocking with
            # flicker preservation); False uses the cross-group basis.
            if assoc_params.class_mismatch_cost_factor > 1.0:
                _mm = _class_mismatch_strict if assoc_params.class_mismatch_exact else _class_mismatch_group
                if _mm:
                    c *= assoc_params.class_mismatch_cost_factor
            cost[i, j] = c

    return cost, l2_arr


# ── Hungarian matching + post-filter ────────────────────────────────
def hungarian_match(cost: np.ndarray, l2_arr: np.ndarray,
                    max_dist: float) -> list[tuple[int, int]]:
    """linear_sum_assignment + removal of gate violations.

    Hungarian minimizes the total cost sum. Track swaps are possible.

    Returns
    -------
    list of (meas_idx, track_idx) — valid matches only.
    """
    n_meas, n_tracks = cost.shape
    if n_meas == 0 or n_tracks == 0:
        return []

    # scipy allows INF, but a large finite value is safer
    cost_finite = np.where(np.isfinite(cost), cost, INF)
    row_ind, col_ind = linear_sum_assignment(cost_finite)

    matches = []
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] >= INF:
            continue
        if l2_arr[r, c] > max_dist:
            continue
        matches.append((int(r), int(c)))
    return matches


# ── Greedy matching ──────────────────────────────────────────────────
def greedy_match(cost: np.ndarray, l2_arr: np.ndarray,
                 max_dist: float) -> list[tuple[int, int]]:
    """Same as C++ MatchPairs — lock the (i, j) pairs from smallest cost first.

    A locked row/col is not reused. Obvious matches are never broken, so track swaps
    do not occur (stable in dense areas).

    Returns
    -------
    list of (meas_idx, track_idx) — valid matches only.
    """
    n_meas, n_tracks = cost.shape
    if n_meas == 0 or n_tracks == 0:
        return []

    # candidates are gate-passing + finite pairs only. (i, j, cost) tuples.
    rows, cols = np.where(np.isfinite(cost) & (l2_arr <= max_dist))
    if len(rows) == 0:
        return []
    pair_costs = cost[rows, cols]
    order = np.argsort(pair_costs, kind="stable")

    used_row = np.zeros(n_meas, dtype=bool)
    used_col = np.zeros(n_tracks, dtype=bool)
    matches: list[tuple[int, int]] = []
    for k in order:
        r, c = int(rows[k]), int(cols[k])
        if used_row[r] or used_col[c]:
            continue
        if cost[r, c] >= INF:
            continue
        matches.append((r, c))
        used_row[r] = True
        used_col[c] = True
    return matches


# ── dispatcher ──────────────────────────────────────────────────────
def match_pairs(cost: np.ndarray, l2_arr: np.ndarray,
                max_dist: float, algorithm: str = "greedy"
                ) -> list[tuple[int, int]]:
    """Matching algorithm dispatcher. 'greedy' (default) or 'hungarian'."""
    algo = algorithm.lower()
    if algo == "greedy":
        return greedy_match(cost, l2_arr, max_dist)
    if algo == "hungarian":
        return hungarian_match(cost, l2_arr, max_dist)
    raise ValueError(f"unknown match_algorithm: {algorithm!r} "
                     f"(expected 'greedy' or 'hungarian')")
