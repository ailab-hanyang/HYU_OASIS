# layer1_context

An L1 pipeline that attaches context labels to AV2 ring-camera images with a vLLM.
The input is ring-camera JPEGs; the output is per-timestamp JSON (per-camera booleans
+ ego-centric booleans). RefAV atomic functions (e.g. `near_infrastructure`) read this
JSON directly.

## At a glance

```
raw sensor frames + tracker feather 2Hz timestamps
            │
            ▼
   [annotate.runner]  ─── 7 per-camera calls + 1 ego 5-view call
   [annotate.engine]      (vLLM offline LLM.chat)
            │
            ▼
   raw JSON  output/layer1_context/<split>/<log_id>/<ts>.json
            │
            ▼
   [postprocess.runner]  ─── majority-vote smoothing + confirmed-run dilation
            │
            ▼
   processed JSON  output/layer1_context/<split>_processed/<log_id>/<ts>.json
            │
            ▼
   refAV/utils.py:get_context_annotations  →  refAV/atomic_functions.py
```

## Folder layout

| Folder | Role |
|---|---|
| [config/](config/) | `settings.yaml` (model path / vLLM options / input dataset) + `loader.py` (yaml → dict) |
| [prompts/](prompts/) | `schema.py` — `CONTEXT_SCHEMA` (4 categories, 28 items), `CAMERA_NAMES`, `EGO_CAMERA_NAMES`, and the SYSTEM/USER prompt strings |
| [annotate/](annotate/) | `engine.py` (vLLM wrapper, `VLLMAnnotator`) + `runner.py` (log iteration, batch splitting, JSON saving) |
| [postprocess/](postprocess/) | `runner.py` (smoothing+dilation pipeline) + `smoothing.py` (1D bool time-series primitives) |
| [scripts/](scripts/) | CLI entry points + smoke-test / full-inference `.sh` |
| [docker/](docker/) | `Dockerfile` + `run_container.sh` (vLLM runtime for the H100 server) |

## Inference calls (per timestamp)

- per-camera: 7 cameras × single-image → `infra` (7) / `weather` (4) / `time_of_day` (2)
- ego: front_center + front_left + front_right + rear_left + rear_right (5 images) as one call → `ego` (15)

8 calls total. Splitting the two batches lets the vLLM prefix cache hit well.

## Output JSON schema

```json
{
  "log_id": "...",
  "timestamp_ns": 315969904359876000,
  "per_camera": {
    "ring_front_center": {
      "image_path": "sensors/cameras/ring_front_center/<cam_ts>.jpg",
      "camera_timestamp_ns": ...,
      "infra":       { "bus_stop": false, ... 7 keys },
      "weather":     { "rain": false, ... 4 keys },
      "time_of_day": { "dusk_dawn": false, "daylight": true }
    },
    ... 6 more cameras
  },
  "ego": {
    "bridge": false, "brick_street": false, "pothole": false,
    "filled_pothole": false, "storm_grate": false, "road_damage": false,
    "streetcar_tracks": false, "roundabout": false, "school_zone": false,
    "construction_zone": false, "speed_limit_zone": false,
    "shadow_of_building": false, "green_light": false, "yellow_light": false,
    "broken_traffic_light": false
  }
}
```

Total categories/items: `infra` 7 / `ego` 15 / `weather` 4 / `time_of_day` 2 = **28**.
`construction_zone` deliberately appears in both `infra` and `ego` (object matching vs. ego-inside judgment).

## Usage

### 0. Enter the container (on the host)

```bash
bash tools/layer1_context/docker/run_container.sh
```

All subsequent commands run inside the container, from the repo root (`HYU_OASIS/`).

### 1. Smoke test — infer + postprocess + validate one log

```bash
bash tools/layer1_context/scripts/test.sh                       # default log_id, val
bash tools/layer1_context/scripts/test.sh <LOG_ID>
bash tools/layer1_context/scripts/test.sh <LOG_ID> test
```

Both raw JSON and processed JSON are validated against the schema automatically.

### 2. Full-split inference + postprocess

```bash
bash tools/layer1_context/scripts/annotate.sh                   # default: val
bash tools/layer1_context/scripts/annotate.sh val 8             # postprocess with 8 workers
```

This is a long job, so `tmux`/`nohup` is recommended. On restart after an interruption it
resumes via the `existing` skip in [annotate/runner.py:91-92](annotate/runner.py#L91-L92).

### 3. Individual steps (as needed)

```bash
PYTHONPATH=. python -m tools.layer1_context.scripts.annotate \
    --log-ids <ID1> <ID2> --split val
PYTHONPATH=. python -m tools.layer1_context.scripts.annotate --dry-run

PYTHONPATH=. python -m tools.layer1_context.scripts.postprocess \
    --split val --log-ids <ID> --workers 8

PYTHONPATH=. python -m tools.layer1_context.scripts.validate \
    output/layer1_context/val_processed/<ID>
```

`--log-ids` / `--split` override `dataset.log_ids` / `dataset.splits` in `settings.yaml`
(no need to edit the yaml directly).

## Key settings.yaml fields

| Key | Meaning | Default |
|---|---|---|
| `paths.output_dir` | raw JSON output root (postprocess derives `_processed` from it) | `output/layer1_context` |
| `vlm.model_path` | checkpoint the vLLM loads (HF-style id or local path) | (set to your checkpoint) |
| `vlm.tensor_parallel_size` | number of GPUs for TP sharding | 4 |
| `vlm.max_model_len` | context length. ego 5-image + EGO_USER_PROMPT | 24576 |
| `vlm.limit_mm_per_prompt` | forced to `{"image": 5}` in code (required for multi-view ego) | (code) |
| `inference.temperature` | 0.0 (deterministic, for JSON output) | 0.0 |
| `inference.enable_thinking` | Qwen3 reasoning mode. Off for JSON-only output | false |
| `dataset.tracker_dir` | parent dir of the tracker feather holding the 2Hz timestamps | `output/tracker_predictions/Le3DE2E_Tracking` |

## RefAV-side integration

- [refAV/paths.py](../../refAV/paths.py) `CONTEXT_ANNOTATIONS_DIR` must match the output root.
- [refAV/utils.py](../../refAV/utils.py) `get_context_annotations` auto-branches between the
  new schema (`per_camera`) and the old schema (`cameras`).
- [refAV/atomic_functions.py](../../refAV/atomic_functions.py) `near_infrastructure`'s
  infra `Literal` must match `CONTEXT_SCHEMA["infra"]`.

## Troubleshooting

- `ModuleNotFoundError: No module named 'tools.layer1_context'` → `PYTHONPATH=.` missing. Run from the repo root.
- `ModuleNotFoundError: No module named 'vllm'` → running on the host. Enter the container and retry.
- OOM after vLLM engine init → lower `vlm.gpu_memory_utilization` (e.g. 0.92 → 0.85).
