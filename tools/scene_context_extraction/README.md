# Scene Context Extraction

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
   raw JSON  output/scene_context/<split>/<log_id>/<ts>.json
            │
            ▼
   [postprocess.runner]  ─── majority-vote smoothing + confirmed-run dilation
            │
            ▼
   processed JSON  output/scene_context/<split>_processed/<log_id>/<ts>.json
            │
            ▼
   refAV/utils.py:get_context_annotations  →  refAV/atomic_functions.py
```

## Folder layout

| Path | Role |
|---|---|
| [annotate.py](annotate.py) · [postprocess.py](postprocess.py) · [validate.py](validate.py) | CLI entry points (`python -m tools.scene_context_extraction.<name>`) |
| [config/](config/) | `settings.yaml` (model path / vLLM options / input dataset) |
| [src/](src/) | `schema.py` (`CONTEXT_SCHEMA`, `CAMERA_NAMES`, prompt strings) · `engine.py` (`VLLMAnnotator`) · `annotate_runner.py` (log iteration, batching, JSON saving) · `postprocess_runner.py` (smoothing+dilation) · `smoothing.py` (1D bool primitives) · `loader.py` (yaml → dict) |
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
bash tools/scene_context_extraction/docker/run_container.sh
```

All subsequent commands run inside the container, from the repo root (`HYU_OASIS/`).

The whole pipeline (annotate → postprocess → validate) runs via
[`tools/scripts/run_scene_context_extraction.sh`](../scripts/run_scene_context_extraction.sh).
Overrides: `SPLIT`, `WORKERS`, `LOGS` (subset of logs), `CONFIG` (alt settings.yaml),
`SKIP_VALIDATE`.

### 1. Smoke test — one log

```bash
LOGS="02678d04-cc9f-3148-9f95-1ba66347dff9" bash tools/scripts/run_scene_context_extraction.sh
```

### 2. Full-split run

```bash
bash tools/scripts/run_scene_context_extraction.sh                  # whole val split
WORKERS=8 bash tools/scripts/run_scene_context_extraction.sh        # more postprocess workers
SKIP_VALIDATE=1 bash tools/scripts/run_scene_context_extraction.sh  # skip the validate stage
```

This is a long job, so `tmux`/`nohup` is recommended. On restart after an interruption it
resumes via the `existing` skip in [src/annotate_runner.py:91-92](src/annotate_runner.py#L91-L92).

### 3. Individual steps (as needed)

```bash
PYTHONPATH=. python -m tools.scene_context_extraction.annotate \
    --log-ids <ID1> <ID2> --split val
PYTHONPATH=. python -m tools.scene_context_extraction.annotate --dry-run

PYTHONPATH=. python -m tools.scene_context_extraction.postprocess \
    --split val --log-ids <ID> --workers 8

PYTHONPATH=. python -m tools.scene_context_extraction.validate \
    output/scene_context/val_processed/<ID>
```

`--log-ids` / `--split` override `dataset.log_ids` / `dataset.splits` in `settings.yaml`
(no need to edit the yaml directly).

## Key settings.yaml fields

| Key | Meaning | Default |
|---|---|---|
| `paths.output_dir` | raw JSON output root (postprocess derives `_processed` from it) | `output/scene_context` |
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

- `ModuleNotFoundError: No module named 'tools.scene_context_extraction'` → `PYTHONPATH=.` missing. Run from the repo root.
- `ModuleNotFoundError: No module named 'vllm'` → running on the host. Enter the container and retry.
- OOM after vLLM engine init → lower `vlm.gpu_memory_utilization` (e.g. 0.92 → 0.85).
