# VLM server (for `get_visual_actor` / `get_visual_behavior`)

The visual atoms `get_visual_actor` / `get_visual_behavior` classify each track's best crop
with a **vLLM**-served vision-language model (OpenAI-compatible API): one image per request,
a compact `{"match": true|false}` reply.

> **Our reference setup:** `Qwen3.6-35B-A3B` (served name `qwen3.6-35b`) on **4× H100 NVL** —
> a separate, smaller model from the `scene_context_extraction` annotator. It is not a public
> checkpoint, so point `CKPT` at your own vision-language model and set `REFAV_VLM_MODEL` to match.

## Launch

```bash
CKPT=/path/to/weights bash tools/scripts/run_vlm_server.sh    # start all replicas + health-check
```

One replica per GPU (ports 8000.., the `REFAV_VLM_ENDPOINTS` set); waits for readiness, runs the
check. Other actions (first arg): `stop`, `restart`, `status`, `check`.

- **`MODE=docker`** (default): prebuilt `vllm/vllm-openai:latest`, auto-pulled, no Dockerfile
- **`MODE=native`**: runs `vllm serve` on the host
- **`CKPT`**: local path or HF hub id
- override `GPUS` (e.g. `"0 1"` for 2 GPUs), `MODEL_NAME`, `BASE_PORT`, `TP`, `MAX_MODEL_LEN`, … via env

> **<4 GPUs:** the endpoint set shrinks with `GPUS` — export the `REFAV_VLM_ENDPOINTS` the
> launcher prints, else the check flags the never-started ports as down.

<details><summary>Manual commands (what the launcher runs per GPU)</summary>

```bash
# docker (default); repeat with CUDA_VISIBLE_DEVICES=1 --port 8001, … per GPU
docker run -d --name vlm_8000 --gpus all -e CUDA_VISIBLE_DEVICES=0 --shm-size=16g \
  --ipc=host -p 8000:8000 -v "$CKPT:$CKPT:ro" vllm/vllm-openai:latest \
  --model "$CKPT" --served-model-name qwen3.6-35b --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.85 --max-model-len 8192 --trust-remote-code --port 8000

# native
CUDA_VISIBLE_DEVICES=0 vllm serve "$CKPT" --served-model-name qwen3.6-35b \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.85 --max-model-len 8192 \
  --trust-remote-code --port 8000
```
</details>

## Verify (before every run)

```bash
python tools/vlm_server/check_connection.py    # defaults to localhost:8000-3
```

Stdlib-only, dataset-free:

- health-checks each endpoint, then round-trips one real completion
- exit `0` OK · `1` nothing reachable · `2` reachable but completion/parse failed
- env vars (shared with the atoms): `REFAV_VLM_ENDPOINTS` (default `localhost:8000-3`), `REFAV_VLM_MODEL` (default `qwen3.6-35b`)
- **remote host:** point `REFAV_VLM_ENDPOINTS` at it (or SSH-forward ports 8000-3 to localhost)

> **Runtime:** `_vlm_call` fails over across endpoints; it raises `VlmServerError` (aborting the
> run) only when *every* endpoint is down, or immediately on a reply with no parseable `{"match": …}`.
