# layer1_context

AV2 ring camera 이미지에 vLLM 으로 컨텍스트 라벨을 붙이는 L1 파이프라인.
입력은 ring camera JPEG, 출력은 timestamp 단위 JSON (camera 별 boolean
+ ego-centric boolean) 입니다. RefAV atomic function (`near_infrastructure`
등) 이 이 JSON 을 직접 읽습니다.

## 한눈에

```
원본 sensor frames + tracker feather 2Hz timestamps
            │
            ▼
   [annotate.runner]  ─── 7회 per-camera 호출 + 1회 ego 3-view 호출
   [annotate.engine]      (vLLM offline LLM.chat)
            │
            ▼
   raw JSON  output/context_annotations_v4/<split>/<log_id>/<ts>.json
            │
            ▼
   [postprocess.runner]  ─── majority-vote smoothing + confirmed-run dilation
            │
            ▼
   processed JSON  output/context_annotations_v4/<split>_processed/<log_id>/<ts>.json
            │
            ▼
   refAV/utils.py:get_context_annotations  →  refAV/atomic_functions.py
```

## 폴더 구조

| 폴더 | 역할 |
|---|---|
| [config/](config/) | `settings.yaml` (모델 경로 / vLLM 옵션 / 입력 dataset) + `loader.py` (yaml → dict) |
| [prompts/](prompts/) | `schema.py` — `CONTEXT_SCHEMA` (4 카테고리 35 항목), `CAMERA_NAMES`, `EGO_CAMERA_NAMES`, 그리고 SYSTEM/USER prompt 문자열 |
| [annotate/](annotate/) | `engine.py` (vLLM wrapper, `VLLMAnnotator`) + `runner.py` (log 순회, batch 분리, JSON 저장) |
| [postprocess/](postprocess/) | `runner.py` (smoothing+dilation 파이프라인) + `smoothing.py` (1D bool 시계열 primitives) |
| [tools/](tools/) | CLI entry points + smoke test / 전체 추론 .sh |
| [docker/](docker/) | `Dockerfile` + `run_container.sh` (H100 서버용 vLLM 런타임) |

## 추론 호출 단위 (timestamp 1개당)

- per-camera: 7 카메라 × single-image → `infra` (12) / `weather` (4) / `time_of_day` (2)
- ego: front_center + front_left + front_right + rear_left + rear_right 5장 묶어 1회 → `ego` (17)

총 8회 호출. 두 batch 가 분리되어 vLLM prefix-cache 가 잘 적중합니다.

## 출력 JSON 스키마

```json
{
  "log_id": "...",
  "timestamp_ns": 315969904359876000,
  "per_camera": {
    "ring_front_center": {
      "image_path": "sensors/cameras/ring_front_center/<cam_ts>.jpg",
      "camera_timestamp_ns": ...,
      "infra":       { "bus_stop": false, ... 12 keys },
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
    "one_way_road": false, "turn_lane": false, "shadow_of_building": false,
    "green_light": false, "yellow_light": false, "broken_traffic_light": false
  }
}
```

총 카테고리/항목 수: `infra` 12 / `ego` 17 / `weather` 4 / `time_of_day` 2 = **35**.
`construction_zone` 만 `infra` 와 `ego` 에 의도적으로 병존합니다 (다른 객체 매칭용 vs ego-inside 판정).

## 사용법

### 0. 컨테이너 진입 (호스트)

```bash
bash layer1_context/docker/run_container.sh
```

이후 모든 명령은 컨테이너 안 + 레포 루트 (`HYU_RefAV/`) 에서 실행합니다.

### 1. Smoke test — 1개 log 추론 + postprocess + validate

```bash
bash layer1_context/tools/test.sh                       # default log_id, val
bash layer1_context/tools/test.sh <LOG_ID>
bash layer1_context/tools/test.sh <LOG_ID> test
```

raw JSON / processed JSON 모두 schema 검증까지 자동 수행.

### 2. 전체 split 추론 + postprocess

```bash
bash layer1_context/tools/annotate.sh                   # default: val
bash layer1_context/tools/annotate.sh val 8             # postprocess 8 workers
```

긴 작업이라 `tmux`/`nohup` 권장. 중단 후 재실행 시 [annotate/runner.py:91-92](annotate/runner.py#L91-L92)
의 `existing` 스킵으로 이어 진행합니다.

### 3. 개별 단계 (필요 시)

```bash
PYTHONPATH=. python -m layer1_context.tools.annotate \
    --log-ids <ID1> <ID2> --split val
PYTHONPATH=. python -m layer1_context.tools.annotate --dry-run

PYTHONPATH=. python -m layer1_context.tools.postprocess \
    --split val --log-ids <ID> --workers 8

PYTHONPATH=. python -m layer1_context.tools.validate \
    output/context_annotations_v4/val_processed/<ID>
```

`--log-ids` / `--split` 는 `settings.yaml` 의 `dataset.log_ids` / `dataset.splits`
를 override 합니다 (yaml 직접 편집 불필요).

## settings.yaml 핵심 항목

| 키 | 의미 | 기본값 |
|---|---|---|
| `paths.output_dir` | raw JSON 출력 root (postprocess 의 default 도 여기 기준 `_processed`) | `output/context_annotations_v4` |
| `vlm.model_path` | vLLM 이 로드할 체크포인트 경로 (HF style) | Qwen3.5-122B-A10B-FP8 |
| `vlm.tensor_parallel_size` | TP 분산 GPU 수 | 4 |
| `vlm.max_model_len` | 컨텍스트 길이. ego 3-image + EGO_USER_PROMPT 가 ~3.7K | 8192 |
| `vlm.limit_mm_per_prompt` | 코드에서 `{"image": 3}` 강제 (multi-view ego 위해 필수) | (코드) |
| `inference.temperature` | 0.0 (결정론적, JSON 출력용) | 0.0 |
| `inference.enable_thinking` | Qwen3 reasoning mode. JSON-only 위해 off | false |
| `dataset.tracker_dir` | 2Hz timestamp 가 들어있는 tracker feather 의 부모 dir | `output/tracker_predictions/Le3DE2E_Tracking` |

## RefAV 측 연동

- [refAV/paths.py](../refAV/paths.py) `CONTEXT_ANNOTATIONS_DIR` 가 출력 root 와 일치해야 합니다.
- [refAV/utils.py](../refAV/utils.py) `get_context_annotations` 는 신 스키마 (`per_camera`)
  와 구 스키마 (`cameras`) 를 자동 분기합니다.
- [refAV/atomic_functions.py](../refAV/atomic_functions.py) `near_infrastructure` 의
  infra `Literal` 은 `CONTEXT_SCHEMA["infra"]` 와 동일해야 합니다.

## 트러블슈팅

- `ModuleNotFoundError: No module named 'layer1_context'` → `PYTHONPATH=.` 누락. 레포 루트에서 실행.
- `ModuleNotFoundError: No module named 'vllm'` → 호스트에서 실행 중. 컨테이너 진입 후 재시도.
- vLLM engine init 후 OOM → `vlm.gpu_memory_utilization` 낮추기 (0.92 → 0.85 등).
