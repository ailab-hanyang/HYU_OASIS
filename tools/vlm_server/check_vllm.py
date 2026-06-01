#!/usr/bin/env python3
"""
Standalone smoke test for the VLM (Qwen) vLLM server connection.

It verifies that the OpenAI-compatible vLLM endpoints used by the atomic functions
`get_visual_actor` / `get_visual_behavior` (refAV/atomic_functions.py -> `_visual_filter`
in refAV/utils.py) are reachable and answer a real multimodal chat completion.

No project / dataset / heavy deps are needed: Python 3 standard library ONLY. A tiny
JPEG is embedded below, so the full path (HTTP -> multimodal chat completion -> JSON
parse) is exercised exactly like the real atomic-function call, which sends ONE image
per request.

Usage:
    python tools/vlm_server/check_vllm.py
    REFAV_VLM_ENDPOINTS=http://localhost:8000,http://localhost:8001 \
        python tools/vlm_server/check_vllm.py
    python tools/vlm_server/check_vllm.py --endpoints http://my-host:8000 --model qwen3.6-35b

Exit codes:
    0 = all endpoints healthy AND round-trip + JSON parse OK
    1 = no endpoint reachable
    2 = an endpoint answered but the round-trip / JSON parse failed
"""
import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error

# A 64x64 solid-red JPEG (base64). Embedded so the test needs neither PIL nor any
# dataset image -- it exercises the same multimodal request shape as the real call.
TEST_IMAGE_B64 = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAUDBAQEAwUEBAQFBQUGBwwIBwcHBw8LCwkMEQ8SEhEPERET"
    "FhwXExQaFRERGCEYGh0dHx8fExciJCIeJBweHx7/2wBDAQUFBQcGBw4ICA4eFBEUHh4eHh4eHh4eHh4e"
    "Hh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh4eHh7/wAARCABAAEADASIAAhEBAxEB/8QA"
    "HwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIh"
    "MUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVW"
    "V1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXG"
    "x8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQF"
    "BgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAV"
    "YnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOE"
    "hYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq"
    "8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDlaKKK+ZP3AKKKKACiiigAooooAKKKKACiiigAooooAKKKKACi"
    "iigAooooAKKKKACiiigAooooAKKKKACiiigAooooA//Z"
)

DEFAULT_ENDPOINTS = (
    "http://localhost:8000,http://localhost:8001,"
    "http://localhost:8002,http://localhost:8003"
)


def parse_endpoints(raw):
    return [u.strip().rstrip("/") for u in raw.split(",") if u.strip()]


def check_health(base, timeout):
    try:
        with urllib.request.urlopen(base + "/v1/models", timeout=timeout) as r:
            data = json.loads(r.read())
        return True, [m.get("id") for m in data.get("data", [])]
    except Exception as e:  # noqa: BLE001 - report any failure verbatim
        return False, repr(e)


def round_trip(base, model, timeout):
    """Mirror refAV/utils.py::_vlm_call exactly: ONE image per request, thinking off.

    The real call sends a single tight crop per track, so this sends one image too
    (temperature=0, max_tokens=20, chat_template_kwargs enable_thinking=False).
    """
    payload = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": 20,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {"role": "system",
             "content": "You are a precise visual classifier. Output only a compact JSON object."},
            {"role": "user", "content": [
                {"type": "text",
                 "text": "Does the centered object in this crop clearly look predominantly red? "
                         'Respond with ONLY a JSON object: {"match": true} or {"match": false}.'},
                {"type": "image_url",
                 "image_url": {"url": "data:image/jpeg;base64," + TEST_IMAGE_B64}},
            ]},
        ],
    }
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read())
    dt = time.time() - t0
    content = body["choices"][0]["message"]["content"]
    a, b = content.find("{"), content.rfind("}")
    parsed = None
    if a != -1 and b > a:
        try:
            parsed = json.loads(content[a:b + 1])
        except Exception:
            parsed = None
    return dt, content, parsed


def main():
    ap = argparse.ArgumentParser(description="VLM vLLM connection smoke test (stdlib only).")
    ap.add_argument("--endpoints",
                    default=os.environ.get("REFAV_VLM_ENDPOINTS", DEFAULT_ENDPOINTS),
                    help="comma-separated base URLs (default: env REFAV_VLM_ENDPOINTS or localhost:8000-3)")
    ap.add_argument("--model",
                    default=os.environ.get("REFAV_VLM_MODEL", "qwen3.6-35b"),
                    help="served-model-name (default: env REFAV_VLM_MODEL or qwen3.6-35b)")
    ap.add_argument("--timeout", type=float, default=20.0, help="per-request timeout seconds")
    args = ap.parse_args()

    eps = parse_endpoints(args.endpoints)
    print(f"REFAV_VLM_MODEL     = {args.model}")
    print(f"REFAV_VLM_ENDPOINTS = {args.endpoints}")

    print(f"\n== 1. health check ({len(eps)} endpoint(s)) ==")
    healthy = []
    for base in eps:
        ok, info = check_health(base, args.timeout)
        if ok:
            print(f"  [OK]   {base}/v1/models  models={info}")
            healthy.append(base)
        else:
            print(f"  [FAIL] {base}/v1/models  {info}")
    if not healthy:
        print("\nNo endpoint reachable -> see README 'Troubleshooting': vLLM not running, "
              "wrong host/port, or ports not reachable from here (use an SSH tunnel).")
        return 1

    print(f"\n== 2. single-image multimodal round-trip on {healthy[0]} ==")
    try:
        dt, content, parsed = round_trip(healthy[0], args.model, args.timeout)
    except urllib.error.HTTPError as e:
        detail = e.read()[:300]
        print(f"  [FAIL] HTTP {e.code}: {detail!r}")
        print("  -> model-name mismatch? set --model / REFAV_VLM_MODEL to the served-model-name "
              "the server was launched with.")
        return 2
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] {e!r}")
        return 2

    print(f"  raw content : {content!r}   ({dt:.2f}s)")
    if parsed is not None and "match" in parsed:
        print(f"  parsed JSON : {parsed}")
        print(f"\nALL GOOD: {len(healthy)}/{len(eps)} endpoint(s) healthy + round-trip + JSON parse OK.")
        print("get_visual_actor / get_visual_behavior will be able to reach this server.")
        return 0

    print('  [FAIL] response carried no parseable {"match": ...} JSON.')
    print("  -> is thinking mode disabled (chat_template_kwargs enable_thinking=false)? "
          "is max_tokens large enough? See README 'Troubleshooting'.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
