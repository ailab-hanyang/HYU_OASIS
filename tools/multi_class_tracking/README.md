# multi_class_tracking — Rule-based re-tracking (CV / IMM)

Takes Le3DE2E detection results (`sm_annotations.feather`) as input, **generates a new track ID
sequence**, and refines position/yaw with the KF/IMM posterior. The detections themselves
(class, score, box size) are preserved; only the **`track_uuid` and the post-processed (tx, ty, yaw)**
are replaced → improved ID consistency (AssA) + trajectory quality.

```
sm_annotations.feather (Le3DE2E detection, incl. EGO)
        │  separate EGO (pass-through) + score pre-filter
        ▼
   [frame loop]  predict → cost/gating/match(Hungarian|greedy) → update → init/age
        │        motion_model = cv (4D) | imm (4 sub-filter)
        ▼
   [post_process]  world→ego transform + yaw flip resolution + static/jitter/IoU/stop_sign post-proc
        ▼
tracker_predictions/<dst_tracker>/<split>/<log>/
   sm_annotations.feather      # new uuid + post-processed state (downstream eval/smoothing input)
   imm_debug.feather           # (imm) per-sub-filter debug
   imm_smooth_inputs.feather   # (imm) RTS smoother sidecar — input to rts_smoothing(imm)
   tracking_summary.json       # diagnostic summary
```

---

## 0. Prerequisite — produce yaw-fixed detections first ⚠️

This package's input (`src_tracker`) must be **detections with ego offset + yaw correction applied**.
Both preprocessing steps are handled by the [`tools/rts_smoothing`](../rts_smoothing/README.md) package:
- **ego offset** — `derive_ego_from_base` : correct the EGO box from rear-axle → body-center offset
  (`Le3DE2E_Tracking` → `Le3DE2E_Tracking_ego`).
- **yaw-fix** — `apply_yaw_correction` : stage-1 yaw flip correction (`*_ego` → `*_ego_yawfix`).

So **before re-tracking, run ego offset → yaw-fix in that order** to produce a `*_yawfix` tracker,
then point `src_tracker` at it. (The config's `src_tracker: Le3DE2E_Tracking_ego_yawfix_...` is that result.)

### Full pipeline order
```bash
cd /home/ailab/AILabDataset/03_Shared_Repository/jeongwoo/HYU_RefAV

# 1) (pre) ego offset — rts_smoothing package. EGO box rear-axle→body-center offset
#    Le3DE2E_Tracking → Le3DE2E_Tracking_ego
python -m tools.rts_smoothing.derive_ego_from_base \
    --src_tracker Le3DE2E_Tracking \
    --dst_tracker Le3DE2E_Tracking_ego \
    --split val --workers 8

# 2) (pre) yaw-fix — rts_smoothing package. stage-1 yaw flip correction
#    Le3DE2E_Tracking_ego → Le3DE2E_Tracking_ego_yawfix
python -m tools.rts_smoothing.apply_yaw_correction \
    --src_tracker Le3DE2E_Tracking_ego \
    --dst_tracker Le3DE2E_Tracking_ego_yawfix \
    --split val --workers 8

# 3) (this package) IMM re-tracking — using the yawfix result above as src_tracker
python -m tools.multi_class_tracking.apply_tracking --all --workers 4 --force

# 4) (post) backward smoothing — rts_smoothing
python -m tools.rts_smoothing.apply_rts_smoothing --smoother_mode imm --force
```

> Skipping ego offset / yaw-fix and re-tracking directly on the raw detection (`Le3DE2E_Tracking`)
> lets EGO position error + detection yaw ±180° flips leak in, degrading position/heading quality.

### One-command run — `run_imm_smoothing.sh`
Run the whole pipeline (1–4 above) in a single command (details: [tools/scripts/run_imm_smoothing.sh](../scripts/run_imm_smoothing.sh)):
```bash
bash tools/scripts/run_imm_smoothing.sh                          # whole split
LOGS="3de5b5d6-68c4-3c95-84ed-be7c83d829f8" \
    bash tools/scripts/run_imm_smoothing.sh                      # specific log only
SKIP_EGO=1 SKIP_YAWFIX=1 bash tools/scripts/run_imm_smoothing.sh # skip preprocessing stages
```
- Env overrides: `SPLIT WORKERS BASE EGO YAWFIX TRACK SMOOTH LOGS SKIP_EGO SKIP_YAWFIX`.
- Prerequisite: `tracking.motion_model: imm` (sidecar generation → required for IMM smoothing). The script checks this automatically.

---

## 1. Motion model (`tracking.motion_model`)

| mode | code | state | applies to |
|---|---|---|---|
| **`cv`** | [src/kalman.py](src/kalman.py) | 4D `[x,y,vx,vy]` | all categories |
| **`imm`** | [src/imm_filter.py](src/imm_filter.py) | 4 sub-filters | vehicle/two_wheeler/pedestrian. **static categories (SIGN/BOLLARD/CONE, etc.) always use 4D CV** |

IMM sub-filters (mu/M order) = **[CV, CA, CTRV, CTRA]**, measurement 3D `[x, y, yaw]`.
- Group (vehicle/two_wheeler/pedestrian) differences come only from `mu_by_group`/`M_by_group`; Q/R/P0 are shared per model.
- EGO_VEHICLE is excluded from input frames (not tracked) and concatenated to the output as-is (pass-through).

---

## 2. File structure

```
tools/multi_class_tracking/
├── apply_tracking.py      # main CLI — re-tracking (per log)
├── apply_stitch.py        # (optional) apply stitching of broken tracks
├── config/config.yaml     # all parameters
└── src/
    ├── track.py           # Track / Measurement / Frame / FrameRecord
    ├── io_utils.py        # load_frames (separates EGO pass-through), save_output_feather
    ├── kalman.py          # 4D CV KF (predict/update, anisotropic Q/R)
    ├── association.py     # cost matrix + gating + Hungarian/greedy matching
    ├── motion_models.py   # IMM sub-filters (CV/CA/CTRV/CTRA, native↔standard conversion)
    ├── imm_filter.py      # IMM (mixing/predict/update/combination)
    ├── imm_config.py      # config → IMMParams
    ├── imm_debug.py       # DebugRecorder + SmoothInputRecorder (sidecar)
    ├── tracker.py         # MultiClassTracker — frame loop + lifecycle
    ├── post_process.py    # world→ego, yaw flip resolution, static/jitter/IoU/stop_sign post-proc
    └── stitch.py          # track stitching logic
```

---

## 3. Running

```bash
cd /home/ailab/AILabDataset/03_Shared_Repository/jeongwoo/HYU_RefAV

# config single log
python -m tools.multi_class_tracking.apply_tracking

# whole split
python -m tools.multi_class_tracking.apply_tracking --all --workers 4 --force

# specific log + explicit dst
python -m tools.multi_class_tracking.apply_tracking \
    --logs 3de5b5d6-68c4-3c95-84ed-be7c83d829f8 \
    --dst_tracker Le3DE2E_Tracking_ego_yawfix2_track6 --force --workers 1
```

| arg | default | description |
|---|---|---|
| `--src_tracker` | config `src_tracker` | input detection tracker dir |
| `--dst_tracker` | config `dst_tracker` (null→`<src>_retrack`) | output dir |
| `--split` | config `split` | val / test |
| `--logs` | config `log_id` (single) | list of log_id to process |
| `--all` | off | process the whole split |
| `--workers` | 1 | parallel processes |
| `--force` | off | overwrite if dst exists |

---

## 4. config parameters (key)

### `tracking` — forward tracking
- **`motion_model`** : `cv` | `imm`
- **4D CV KF** : `Q`/`R`/`P0` (static categories + static objects in imm mode), `direction_skew` (heading-axis anisotropic Q), `init_cov_yaw_aligned` (P0 anisotropic from detection yaw)
- **`imm`** : `save_debug`, `mu_by_group`, `M_by_group`, per-model `CV/CA/CTRV/CTRA` (q_xy/q_v/q_a/q_yaw/q_yaw_rate/r_xy/r_yaw/p0_*). CTRV/CTRA override with `static_q_yaw/q_yaw_rate/r_yaw` when stationary
- **Detection filter** : `detection_score_threshold`, `conf_low_threshold`/`conf_low_R_scale`
- **Association** : `cost_metric` (l2|maha), `match_algorithm` (greedy|hungarian), `max_association_dist_m`, `pedestrian_/static_max_association_dist_m`, `pedestrian_class_isolation`, `adaptive_gate` (lateral gate), `yaw_change_gate`
- **Lifecycle** : `max_history`, `max_history_for_outdated`, `static_max_history_for_outdated`, `confirmed_age_min`, `confirmed_detect_min`, `class_alpha`

### `post_processing` — after forward tracking
| key | effect |
|---|---|
| `relabel_only` | **(unused — keep `false`.)** ablation lever (true = preserve raw boxes + replace only uuid). The current pipeline must emit the KF·IMM posterior, so keep false. |
| `dynamic_yaw_threshold` | speed threshold for dynamic-frame yaw estimation |
| `min_lwh` | l/w/h min floor |
| `static_detection` | is_static flag (viewer "Static" toggle, tau_static/tau_motion) |
| `yaw_jitter_freeze` | freeze the yaw of in-place-rotating (pinwheel) static tracks to a RANSAC dominant axis |
| `iou_clean` | remove the shorter noise track among overlapping REGULAR_VEHICLEs (protect_both_min guard) |
| `dynamic_velocity_yaw` | align dynamic-track yaw to the velocity direction via ±180 flip |
| `stop_sign_yaw_to_lane` | fix STOP_SIGN yaw to the opposite of the nearest lane direction (at_stop_sign FN↓, requires AV2 map) |

> Most post_processing toggles/thresholds can be validated live in the **Tracking Tuning** panel of server.py.

---

## 5. Output → downstream

- `sm_annotations.feather` : input for downstream eval (refAV) / `rts_smoothing`.
- `imm_smooth_inputs.feather` : **sidecar for the imm smoother (IMM-S) of `rts_smoothing`**. With
  `motion_model: imm` + `tracking.save_smooth_inputs` (default true), it is generated automatically
  when IMM tracks exist (per-model prior/posterior native state·cov·F·μ). `imm.save_debug` is for the
  separate `imm_debug.feather` file.
- Coordinates: KF/IMM tracks in the world (city) frame → on output, **world→ego inverse transform** with the per-frame ego pose.

---

## 6. Related tools

- **`tools/rts_smoothing`** : applies backward smoothing (RTS/IMM) to this output ([../rts_smoothing/README.md](../rts_smoothing/README.md)).
- **`tools/output_verification/server.py`** + `visualization/3d_perception/viewer.html` : BEV viewer +
  live Tracking/Smoothing tuning.
