"""VLLMAnnotator — batch annotate AV2 camera images via vLLM offline API.

Loads a single vLLM engine across all configured GPUs (tensor parallel),
accepts a list of image paths, returns a list of validated context dicts.
"""

import base64
import json
import logging
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image
from vllm import LLM, SamplingParams

from tools.layer1_context.src.schema import (
    CONTEXT_SCHEMA,
    EGO_SYSTEM_PROMPT,
    EGO_USER_PROMPT,
    SYSTEM_PROMPT,
    USER_PROMPT,
)

logger = logging.getLogger(__name__)


def _pil_to_data_url(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=90)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


_PER_CAMERA_CATEGORIES = ("infra", "weather", "time_of_day")


def _empty_per_camera_result() -> Dict[str, Dict[str, bool]]:
    return {cat: {k: False for k in CONTEXT_SCHEMA[cat]} for cat in _PER_CAMERA_CATEGORIES}


def _empty_ego_result() -> Dict[str, bool]:
    return {k: False for k in CONTEXT_SCHEMA["ego"]}


def _coerce_to_schema(parsed: Dict[str, Any]) -> Dict[str, Dict[str, bool]]:
    """Per-camera output: keep only infra/weather/time_of_day with their fixed keys."""
    result = {}
    for category in _PER_CAMERA_CATEGORIES:
        cat_data = parsed.get(category) or {}
        result[category] = {k: bool(cat_data.get(k, False)) for k in CONTEXT_SCHEMA[category]}
    return result


def _coerce_ego_schema(parsed: Dict[str, Any]) -> Dict[str, bool]:
    """Ego output: flat 15-key bool dict."""
    return {k: bool(parsed.get(k, False)) for k in CONTEXT_SCHEMA["ego"]}


def _build_messages(image: Image.Image) -> List[Dict[str, Any]]:
    # Image last so the long shared prefix (system + USER_PROMPT) hits the
    # vLLM prefix cache. Per-request prefill drops from ~5K to ~1.5K tokens.
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": USER_PROMPT},
                {"type": "image_url", "image_url": {"url": _pil_to_data_url(image)}},
            ],
        },
    ]


_EGO_IMAGE_ORDER_NOTE = (
    "You are given five images from the same timestamp, in this order:\n"
    "  Image 1: ring_front_center (PRIMARY ANCHOR — forward path; ONLY view for traffic signals)\n"
    "  Image 2: ring_front_left  (left side at the current position — what is flanking the ego now)\n"
    "  Image 3: ring_front_right (right side at the current position — what is flanking the ego now)\n"
    "  Image 4: ring_rear_left   (left rearward — peripheral / post-passage cues)\n"
    "  Image 5: ring_rear_right  (right rearward — peripheral / post-passage cues)\n\n"
)


def _build_ego_messages(images: List[Image.Image]) -> List[Dict[str, Any]]:
    # Text first (long shared prefix → prefix cache hit), images last in fixed
    # order matching _EGO_IMAGE_ORDER_NOTE.
    user_content: List[Dict[str, Any]] = [
        {"type": "text", "text": _EGO_IMAGE_ORDER_NOTE + EGO_USER_PROMPT},
    ]
    for img in images:
        user_content.append(
            {"type": "image_url", "image_url": {"url": _pil_to_data_url(img)}}
        )
    return [
        {"role": "system", "content": EGO_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


class VLLMAnnotator:
    def __init__(self, config: Dict[str, Any]):
        vlm = config["vlm"]
        inf = config["inference"]

        llm_kwargs = dict(
            model=vlm["model_path"],
            tensor_parallel_size=vlm.get("tensor_parallel_size", 4),
            gpu_memory_utilization=vlm.get("gpu_memory_utilization", 0.92),
            max_model_len=vlm.get("max_model_len", 8192),
            max_num_seqs=vlm.get("max_num_seqs", 256),
            enable_prefix_caching=vlm.get("enable_prefix_caching", True),
            trust_remote_code=vlm.get("trust_remote_code", True),
            limit_mm_per_prompt={"image": 5},
        )
        if vlm.get("mm_processor_kwargs"):
            llm_kwargs["mm_processor_kwargs"] = vlm["mm_processor_kwargs"]

        logger.info("Initializing vLLM engine: %s", vlm["model_path"])
        self.llm = LLM(**llm_kwargs)

        self.sampling = SamplingParams(
            temperature=inf.get("temperature", 0.0),
            max_tokens=inf.get("max_tokens", 512),
        )
        self.chat_template_kwargs = {
            "enable_thinking": inf.get("enable_thinking", False),
        }

    def annotate_batch(self, image_paths: List[Path]) -> List[Dict[str, Any]]:
        if not image_paths:
            return []

        conversations = []
        for path in image_paths:
            image = Image.open(path).convert("RGB")
            conversations.append(_build_messages(image))

        outputs = self.llm.chat(
            messages=conversations,
            sampling_params=self.sampling,
            chat_template_kwargs=self.chat_template_kwargs,
            use_tqdm=False,
        )

        results = []
        for path, output in zip(image_paths, outputs):
            text = output.outputs[0].text
            try:
                parsed = json.loads(text)
                results.append(_coerce_to_schema(parsed))
            except json.JSONDecodeError as e:
                logger.warning("JSON parse failed for %s: %s | raw=%r", path, e, text[:200])
                results.append(_empty_per_camera_result())
        return results

    def annotate_ego_batch(self, image_groups: List[List[Path]]) -> List[Dict[str, bool]]:
        """Annotate ego (multi-view) for a batch of timestamps.

        Each group is the per-timestamp image paths in EGO_CAMERA_NAMES order.
        Returns a list of 15-key bool dicts aligned with the input.
        """
        if not image_groups:
            return []

        from tools.layer1_context.src.schema import EGO_CAMERA_NAMES
        expected = len(EGO_CAMERA_NAMES)
        conversations = []
        for group in image_groups:
            assert len(group) == expected, f"ego group must have {expected} images, got {len(group)}"
            images = [Image.open(p).convert("RGB") for p in group]
            conversations.append(_build_ego_messages(images))

        outputs = self.llm.chat(
            messages=conversations,
            sampling_params=self.sampling,
            chat_template_kwargs=self.chat_template_kwargs,
            use_tqdm=False,
        )

        results = []
        for group, output in zip(image_groups, outputs):
            text = output.outputs[0].text
            try:
                parsed = json.loads(text)
                results.append(_coerce_ego_schema(parsed))
            except json.JSONDecodeError as e:
                logger.warning("Ego JSON parse failed for %s: %s | raw=%r", group[0], e, text[:200])
                results.append(_empty_ego_result())
        return results
