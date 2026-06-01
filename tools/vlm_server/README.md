# VLM server (for `get_visual_actor` / `get_visual_behavior`)

The atomic functions **`get_visual_actor`** and **`get_visual_behavior`**
(`refAV/atomic_functions.py` → `_visual_filter` in `refAV/utils.py`) classify each
track's best camera crop with a vision-language model served by **vLLM**
(OpenAI-compatible API). Each request sends **one image** (the object's tight crop)
plus a short text question and expects a compact `{"match": true|false}` reply.

> This is a **different** model/server from the `layer1_context` scene-annotation
> pipeline. Here we serve a smaller per-object classifier (default **Qwen3.6-35B-A3B**,
> served-model-name `qwen3.6-35b`); `tools/layer1_context` uses its own larger model.

> ⚠️ **If this server is not running, the visual atoms fail silently.** On any
> connection error/timeout `_vlm_call` returns `None`, which `_visual_filter` records
> as "no match" — so `get_visual_actor` / `get_visual_behavior` just return an empty
> set with **no error**. Always run the health check below before a mining/eval run.
> The same applies per-endpoint: if one replica is down, the requests routed to it
> become "no match", so keep the whole fleet healthy.

---

## What the atoms expect (deployment contract)

- One or more **vLLM replicas** exposing the OpenAI-compatible API, by default on
  ports **8000–8003** (one per GPU; the atoms round-robin across them).
- **served-model-name `qwen3.6-35b`** (weights: `Qwen3.6-35B-A3B`, or any compatible
  vision-language model — set the name to match).
- The atoms read two env vars (`refAV/utils.py::_vlm_endpoints` / `_vlm_call`):

  | env var | default | meaning |
  |---|---|---|
  | `REFAV_VLM_ENDPOINTS` | `http://localhost:8000,http://localhost:8001,http://localhost:8002,http://localhost:8003` | comma-separated base URLs |
  | `REFAV_VLM_MODEL` | `qwen3.6-35b` | served-model-name sent in each request |

`check_vllm.py` reads the **same** env vars (and `--endpoints` / `--model` flags),
so a green run there means the atoms will connect too.

---

## 1. Launch the server

Set `CKPT` to your model weights (a local path or a HuggingFace hub id). The atoms
send **one image per request**, so the default vLLM image limit (1) is sufficient —
you do **not** need `--limit-mm-per-prompt`.

**Option A — `vllm serve` directly** (one replica per GPU; repeat for each GPU):
```bash
CKPT=/path/to/Qwen3.6-35B-A3B   # local dir or HF id

# GPU 0 -> port 8000  (repeat with CUDA_VISIBLE_DEVICES=1 --port 8001, etc.)
CUDA_VISIBLE_DEVICES=0 vllm serve "$CKPT" \
  --served-model-name qwen3.6-35b \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --trust-remote-code \
  --port 8000
```

**Option B — Docker** (official vLLM image; one container per GPU):
```bash
CKPT=/path/to/Qwen3.6-35B-A3B

for N in 0 1 2 3; do
  docker run -d --name vlm_$N --gpus all -e CUDA_VISIBLE_DEVICES=$N \
    --shm-size=16g --ipc=host -p $((8000+N)):8000 \
    -v "$CKPT:$CKPT:ro" \
    vllm/vllm-openai:latest \
    --model "$CKPT" --served-model-name qwen3.6-35b \
    --tensor-parallel-size 1 --gpu-memory-utilization 0.85 \
    --max-model-len 8192 --trust-remote-code --port 8000
done
```

A single replica is fine for small runs — set `REFAV_VLM_ENDPOINTS=http://localhost:8000`.
More replicas just add throughput (the atoms round-robin across whatever you list).

---

## 2. Verify connectivity (do this before every run)

`check_vllm.py` is **self-contained and dataset-free**: it embeds a tiny test image
and uses only the Python standard library, exercising the exact request path the real
atoms use (single image + text, `temperature=0`, `max_tokens=20`, thinking disabled).

```bash
python tools/vlm_server/check_vllm.py            # uses localhost:8000-3
```

It (1) health-checks every endpoint (`GET /v1/models`) and (2) round-trips one real
multimodal chat completion on the first healthy endpoint and parses the `{"match": …}`
JSON. Exit code `0` = all good, `1` = nothing reachable, `2` = reachable but the
completion/JSON-parse failed.

A healthy run:
```
== 1. health check (4 endpoint(s)) ==
  [OK]   http://localhost:8000/v1/models  models=['qwen3.6-35b']
  ... (x4)
== 2. single-image multimodal round-trip on http://localhost:8000 ==
  raw content : '{"match": true}'   (0.10s)
  parsed JSON : {'match': True}
ALL GOOD: 4/4 endpoint(s) healthy + round-trip + JSON parse OK.
```

**Is anything up at all?**
```bash
for p in 8000 8001 8002 8003; do echo -n "$p: "; curl -s localhost:$p/v1/models | head -c 80; echo; done
```

---

## 3. Running mining from a different host

If the vLLM ports live on a remote GPU host and you run mining elsewhere, either:

**A) SSH tunnel** (recommended when the ports are bound to the host's localhost):
```bash
ssh -N \
  -L 8000:localhost:8000 -L 8001:localhost:8001 \
  -L 8002:localhost:8002 -L 8003:localhost:8003 \
  <user>@<GPU_HOST>
# in another shell, defaults (localhost:8000-3) now reach the remote server:
python tools/vlm_server/check_vllm.py
```

**B) Direct** (only if the ports are reachable on the network):
```bash
REFAV_VLM_ENDPOINTS=http://<GPU_HOST>:8000,http://<GPU_HOST>:8001,http://<GPU_HOST>:8002,http://<GPU_HOST>:8003 \
  python tools/vlm_server/check_vllm.py
```

Whatever makes the check pass, export the **same** `REFAV_VLM_ENDPOINTS` for the
actual mining run so the atoms hit the identical endpoints.

---

## Troubleshooting

| Symptom (from `check_vllm.py`) | Likely cause | Fix |
|---|---|---|
| `[FAIL] … URLError … Connection refused` on every endpoint | vLLM not running, or ports not reachable from here | check the server with `docker ps` / `curl localhost:8000/v1/models`; if running but unreachable, use the **SSH tunnel** (section 3) |
| Some endpoints `[OK]`, some `[FAIL]` | a replica/GPU is down | restart the missing replica — the atoms route some tracks to it, so a partial fleet silently drops those to "no match" |
| `[FAIL] HTTP 404` on `/v1/models` | wrong base URL / path | endpoints must be the **base** (`http://host:8000`); the code appends `/v1/...` |
| `[FAIL] HTTP 400/404` on round-trip, body mentions the model | `REFAV_VLM_MODEL` ≠ served-model-name | set `--model` / `REFAV_VLM_MODEL` to the `--served-model-name` the server was launched with |
| round-trip OK but `no parseable {"match": …} JSON` | thinking mode not disabled → the model reasons until `max_tokens` and never emits the JSON | the server must honor `chat_template_kwargs={"enable_thinking": false}`; if a build ignores it, raise `max_tokens` or disable thinking in the chat template |
| smoke test passes, but real mining returns 0 matches | not a connectivity issue | check crops exist for the log (`get_best_crop`), and gather the candidate set BROADLY (see the `get_visual_actor` docstring) |

> Note: real AV2 crops are occasionally truncated JPEGs; the pipeline sets
> `PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True` in `refAV/utils.py`. The smoke test's
> embedded image is clean, so it only validates connectivity.
