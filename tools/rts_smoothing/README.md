# rts_smoothing — Track post-processing smoothing (RTS / IMM smoother)

Takes the tracker output (`sm_annotations.feather`) produced by `multi_class_tracking` and applies
**backward smoothing** to refine the position/yaw trajectory. It does not change detections or track
IDs (association); it only smooths the **state (x, y, yaw, …)**, so it mainly affects DetA/LocA and
atomic-function decisions (velocity/yaw dependent).

```
tracker_predictions/<src_tracker>/<split>/<log>/sm_annotations.feather   (input)
        │   (+ imm_smooth_inputs.feather  ← sidecar when smoother_mode=imm)
        ▼
   [smoother]  rts (11-D single CV-RTS)  |  imm (per-model RTS based on sidecar)
        ▼
   [yaw_postproc]  reapply post-smoothing yaw post-processing (unified with tracking)
        ▼
tracker_predictions/<dst_tracker>/<split>/<log>/sm_annotations.feather   (output)
```

---

## 0. Prerequisite — you must run `multi_class_tracking` first ⚠️

> **This package contains both the "pre" re-tracking preprocessing and the "post" smoothing.**
> Full pipeline (details: [multi_class_tracking/README.md](../multi_class_tracking/README.md)):
> ```
> (1) ego offset   derive_ego_from_base   Le3DE2E_Tracking → _ego          (before re-tracking)
> (2) yaw-fix       apply_yaw_correction   _ego → _ego_yawfix               (before re-tracking)
> (3) re-tracking   multi_class_tracking   _ego_yawfix → IMM dst(+sidecar)
> (4) smoothing     apply_rts_smoothing    IMM dst → smoothed              (← main focus of this doc)
> ```

`apply_rts_smoothing` (step 4, smoothing) **does not run standalone.** Its input (`src_tracker`) must
be the **output (step 3)** of [`tools/multi_class_tracking`](../multi_class_tracking/README.md).

- **All modes**: the `src_tracker` dir must contain `sm_annotations.feather`
  (= the result of `multi_class_tracking.apply_tracking`).
- **`smoother_mode: imm` (IMM-S)**: additionally the **`imm_smooth_inputs.feather` sidecar is required**.
  This sidecar is generated **only when `multi_class_tracking` is run with `motion_model: imm` (IMM
  filter)** (`tracking.save_smooth_inputs` defaults to true → generated automatically when IMM tracks
  exist; `imm.save_debug` is for the separate `imm_debug.feather` file and is unrelated to the sidecar).
  So you **must first run IMM-based re-tracking** and then point `src_tracker` at that dst for the IMM
  smoother to work. (CV-only tracking results have no IMM tracks, so the sidecar is empty and imm mode is impossible.)

### Correct execution order
```bash
cd /path/to/HYU_OASIS

# 1) (pre) multi_class_tracking — IMM-filter-based re-tracking
#    config: tracking.motion_model=imm  (sidecar auto-generated via save_smooth_inputs default true)
#    → produces sm_annotations.feather + imm_smooth_inputs.feather in the dst dir
python -m tools.multi_class_tracking.apply_tracking --all --workers 4 --force
#    (dst e.g.: Le3DE2E_Tracking_ego_yawfix2_track6)

# 2) (post) rts_smoothing — smooth, pointing src_tracker at the dst above
#    config: src_tracker=<dst above>, smoother_mode=imm
python -m tools.rts_smoothing.apply_rts_smoothing --smoother_mode imm --force
```

> If the sidecar is missing, imm mode skips that log as `missing_sidecar`. Make sure step 1) (IMM
> tracking) finished first.

### One-command run — `run_imm_smoothing.sh`
Run the whole pipeline (1–4: ego offset → yaw-fix → IMM re-tracking → IMM smoothing) in one command:
```bash
bash tools/scripts/run_imm_smoothing.sh                          # whole split
LOGS="3de5b5d6-68c4-3c95-84ed-be7c83d829f8" \
    bash tools/scripts/run_imm_smoothing.sh                      # specific log only
SKIP_EGO=1 SKIP_YAWFIX=1 bash tools/scripts/run_imm_smoothing.sh # skip preprocessing (ego/yawfix)
```
- Env overrides: `SPLIT WORKERS BASE EGO YAWFIX TRACK SMOOTH LOGS SKIP_EGO SKIP_YAWFIX`.
- Script location: [tools/scripts/run_imm_smoothing.sh](../scripts/run_imm_smoothing.sh). Prerequisite: `tracking.motion_model: imm`.

---

## 1. The two smoothers (`smoother_mode`)

| mode | code | input | description |
|---|---|---|---|
| **`rts`** | [src/rts_smoother.py](src/rts_smoother.py) | `sm_annotations.feather` | 11-D single-model RTS. State `[x,y,z,θ,l,w,h,s,vx,vy,vz]`, forward Kalman + backward RTS. Smooths all tracks (except skipped), including static categories. |
| **`imm`** | [src/imm_smoother.py](src/imm_smoother.py) | `imm_smooth_inputs.feather` (sidecar) | Reads the forward IMM per-model prior/posterior·F·μ from the sidecar and does **per-model RTS backward + forward μ combination**. IMM tracks only (vehicle/two_wheeler/pedestrian). Static single-CV tracks have no sidecar → pass through original. |

> In `imm` mode, `src_tracker` must be the **IMM dst** of `multi_class_tracking` (i.e. a result holding
> `imm_smooth_inputs.feather`).

---

## 2. File structure

```
tools/rts_smoothing/
├── apply_rts_smoothing.py     # main CLI — apply rts/imm smoothing
├── apply_yaw_correction.py    # yaw-fix (stage-1 flip correction) — produces *_yawfix "before" re-tracking
│                              #   (input to multi_class_tracking. A separate stage from apply_rts_smoothing)
├── derive_ego_from_base.py    # ego offset — correct EGO box rear-axle→body-center "before" re-tracking
│                              #   (base Le3DE2E_Tracking → _ego. before yaw-fix)
├── config/config.yaml         # all parameters
└── src/
    ├── load_tracks.py         # sm_annotations + ego pose → Tracks (collection of TrackState)
    ├── track_state.py         # TrackState / Tracks data structures (ego frame)
    ├── kalman_predict.py      # 11-D predict (build F)
    ├── kalman_update.py       # 11-D update (build H)
    ├── rts_smoother.py        # smoother_mode=rts — forward/backward RTS
    ├── imm_smoother.py        # smoother_mode=imm — per-model RTS + μ combination
    ├── yaw_correction.py      # stage-1 yaw flip correction (static majority unify / dynamic motion flip)
    └── yaw_postproc.py        # reapply post-smoothing yaw post-processing (same functions as tracking)
```

---

## 3. Running

Uses `src_tracker`/`dst_tracker`/`split`/`smoother_mode` from config (`config/config.yaml`) as-is;
CLI args take precedence if given (**priority: CLI > config > default**).

```bash
cd /path/to/HYU_OASIS

# config as-is (whole split)
python -m tools.rts_smoothing.apply_rts_smoothing --force

# explicit imm mode + single log
python -m tools.rts_smoothing.apply_rts_smoothing --smoother_mode imm \
    --src_tracker Le3DE2E_Tracking_ego_yawfix2_track6 \
    --logs 3de5b5d6-68c4-3c95-84ed-be7c83d829f8 --workers 1 --force

# override dst name
python -m tools.rts_smoothing.apply_rts_smoothing --dst_tracker MyName --force
```

| arg | default | description |
|---|---|---|
| `--src_tracker` | config `src_tracker` | input tracker dir |
| `--dst_tracker` | config `dst_tracker` (null→auto `<src>_<imm_smooth\|rts>`) | output tracker dir |
| `--split` | config `split` | val / test |
| `--smoother_mode` | config `smoother_mode` | `rts` \| `imm` |
| `--logs` | (none → whole split) | list of log_id to process |
| `--workers` | 4 | parallel processes |
| `--force` | off | overwrite if dst exists (otherwise skip) |

Output: `tracker_predictions/<dst_tracker>/<split>/<log>/sm_annotations.feather` (+ per-log summary json).
Detections/score/track_uuid are preserved; only `tx_m/ty_m/qw..qz` (position·yaw), etc. are replaced with smoothed values.

---

## 4. config parameters (key)

### Common
- `split`, `log_id`, `src_tracker`, `dst_tracker`, `smoother_mode`

### `imm_smoothing` (smoother_mode=imm)
- `n_min` : if frame count < n_min, skip (pass through original)
- `freeze_initial` : do not backward-smooth the first frame (k=0) — prevents the initial box from being dragged toward the next measurement by the large initial P0
- `static_cov_inflate` : inflate a static frame's T+1 prior cov (the information matrix of the backward gain) by `multiplier` to weaken the gain → trust the measurement (R) more

### `rts_smoothing` (smoother_mode=rts)
- `Q` (11-D) / `R` (8-D) / `P0` (11-D) : process/measurement/initial cov diagonals
- `n_min`, `freeze_initial`, `static_cov_inflate` : (same meaning as imm, **separate values for rts mode**)
- `Q_aligned` : direction/class-aware process noise (optional)

### `yaw_postproc` (common to rts·imm, post-smoothing yaw post-processing)
Reapplies tracking's yaw post-processing — lost when the smoother overwrites yaw — with the **same
functions**, to unify them. Flags/thresholds reuse the `post_processing` values from the
`multi_class_tracking` config (stop_sign / jitter_freeze / dynamic_velocity_yaw). Plus two
smoothing-only stages:
- `static_yaw_jump_hold` : on a static frame, if yaw jumps by `thr_deg` or more from the previous frame, keep the previous yaw
- `yaw_lock_to_track` : if the smoothed yaw diverges by `thr_deg` or more from the tracking-output yaw
  (already flip-resolved), replace it with the tracking yaw (speed-independent — fixes the smoother's ±180° flip smear)

### `yaw_correction` (stage-1, optional / server stage 1·2)
Static majority yaw unification + dynamic motion flip resolution. Used by `apply_yaw_correction.py`
standalone or by the server viewer's 1-stage/2-stage.

---

## 5. server.py / viewer integration

The viewer's **Smoothing** toggle computes live via `/api/smoothing/tracks?stage=...`.
- stage 1 = yaw_correction, stage 2 = +RTS, **stage 3 = IMM smoother (IMM-S)**.
- IMM-S reads the config above (`imm_smoothing`/`yaw_postproc`) per request via `from_config`
  (config value changes → reflected live after a server restart; the response cache key includes the key parameters).
- The viewer's **IMM-Tune** panel (stopSign/jFreeze/dynYaw/statInfl/statYawHold/yawLock) lets you toggle
  each post-processing stage on/off for qualitative comparison.

> If you change config parameters, **restart** server.py so the module code is refreshed (values are re-read per request).

---

## 6. Coordinate-frame notes

- `TrackState` is in the **ego frame** (same as sm_annotations; ego differs per frame). Use
  `Tracks.ego_poses_*` for ego↔world conversion.
- The smoother's internal yaw residual/state yaw is wrapped to [-π, π]. `yaw_postproc` runs in the
  world frame and then converts back to the ego frame (quat), updating only `qw..qz`.
```
