"""
Multi-agent (Function Selector + Code Writer) pipeline for HYU_RefAV.

Two-stage prompting:
  Stage 1 - Selector: prompt + auto INDEX + few-shot -> JSON with selected_funcs.
  Stage 2 - Coder:    prompt + selected_funcs + filtered docs/examples -> Python code.

The pipeline is designed so that:
  - atomic_functions.py algorithm-only edits do NOT invalidate any LLM cache
  - atomic_functions.txt docstring edits invalidate ONLY prompts that selected that function
  - examples.txt edits invalidate ONLY prompts that selected the affected function
  - Selector failure falls back to single-agent code generation (caller responsibility)

The actual LLM calls are made via predict_scenario_anthropic (same as single-agent path).
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Optional

from refAV import paths

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INVARIANT_FUNCS = {
    "get_objects_of_category",
    "scenario_and",
    "scenario_or",
    "scenario_not",
    "output_scenario",
    "reverse_relationship",
}

# Layer assignment — single source of truth for the 4-layer cascade
# (Context -> Actor -> Behavior -> Relation). See
# docs/four_layer_cascade_design.html for the design.
LAYER_NAMES = ("layer1_context", "layer2_actor", "layer3_behavior", "layer4_relation")

LAYER1_CONTEXT = [
    "is_weather", "is_time_of_day", "near_infrastructure", "within_camera_view",
    "on_lane_type", "near_intersection", "on_intersection", "at_pedestrian_crossing",
    "at_stop_sign", "in_drivable_area", "on_road", "in_turn_lane",
    "in_parallel_parking", "near_construction_objects",
    "on_median", "on_road_with_n_lanes",
]
LAYER2_ACTOR = ["is_category", "active", "stationary",
                "get_visual_actor"]
LAYER3_BEHAVIOR = [
    "turning", "changing_lanes", "has_velocity", "accelerating",
    "has_lateral_acceleration", "braking", "braking_hard", "reversing",
    "waiting_to_turn", "get_visual_behavior",
]
LAYER4_RELATION = [
    "has_objects_in_relative_direction", "get_objects_in_relative_direction",
    "near_objects", "following", "being_crossed_by", "being_overtaken",
    "cut_in_front_of", "in_same_lane", "on_relative_side_of_road",
    "between_two_objects", "group_of",
    "heading_in_relative_direction_to", "facing_toward", "heading_toward",
    "nth_object_in_direction",
]
LAYER_OF: dict[str, str] = {
    **{f: "layer1_context"  for f in LAYER1_CONTEXT},
    **{f: "layer2_actor"    for f in LAYER2_ACTOR},
    **{f: "layer3_behavior" for f in LAYER3_BEHAVIOR},
    **{f: "layer4_relation" for f in LAYER4_RELATION},
}

# Helpers auto-added when Selector output is narrow / low-confidence. One bucket
# per layer so the augmented atom lands in the correct cascade block.
COMMON_HELPERS_BY_LAYER: dict[str, list[str]] = {
    "layer1_context":  ["on_intersection"],
    "layer3_behavior": ["has_velocity"],
    "layer4_relation": ["has_objects_in_relative_direction",
                        "get_objects_in_relative_direction", "near_objects"],
}

PROMPT_DIR = paths.PROMPT_DIR / "RefAV"
SHARED_PROMPT_DIR = PROMPT_DIR / "shared"   # atomic_functions.txt, categories.txt, examples.txt
MULTI_PROMPT_DIR  = PROMPT_DIR / "multi"    # deconstructor/selector/coder prompts + shared_guide.txt
ATOMIC_PY = paths.REFAV_PATH / "refAV" / "atomic_functions.py" if hasattr(paths, "REFAV_PATH") else Path(__file__).parent / "atomic_functions.py"


# ---------------------------------------------------------------------------
# Utility: function index / docs / examples extraction
# ---------------------------------------------------------------------------

def _all_function_names(atomic_py: Path = ATOMIC_PY) -> list[str]:
    """All top-level def names in atomic_functions.py (no leading underscore)."""
    tree = ast.parse(atomic_py.read_text())
    names = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            names.append(node.name)
    return names


def valid_selectable_funcs(atomic_py: Path = ATOMIC_PY) -> set[str]:
    """Set of selectable function names (all top-level defs minus INVARIANT)."""
    return set(_all_function_names(atomic_py)) - INVARIANT_FUNCS


def _validate_layer_assignment(atomic_py: Path = ATOMIC_PY) -> None:
    """Module-load self-check: every selectable atomic function is assigned to
    exactly one layer, no extras, no intra-layer duplicates. Catches drift
    between atomic_functions.py and the LAYER_* constants at import time."""
    valid = valid_selectable_funcs(atomic_py)
    assigned = set(LAYER_OF)
    missing = valid - assigned
    extra = assigned - valid
    dups = [lst for lst in (LAYER1_CONTEXT, LAYER2_ACTOR, LAYER3_BEHAVIOR, LAYER4_RELATION)
            if len(lst) != len(set(lst))]
    assert not missing, f"Layer-unassigned funcs: {sorted(missing)}"
    assert not extra, f"Layer-assigned but invalid funcs: {sorted(extra)}"
    assert not dups, f"Duplicates inside a layer: {dups}"


_validate_layer_assignment()


def extract_function_index(atomic_py: Path = ATOMIC_PY) -> str:
    """
    Auto-generated FUNCTION INDEX (English): `- name : first-line of docstring`.
    INVARIANT functions are excluded — they live in selector_system_prompt's
    [INVARIANT FUNCTIONS] section.
    """
    tree = ast.parse(atomic_py.read_text())
    lines = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith("_"):
            if node.name in INVARIANT_FUNCS:
                continue
            doc = ast.get_docstring(node) or ""
            first = doc.strip().split("\n")[0].strip() if doc else "(no docstring)"
            if len(first) > 110:
                first = first[:107] + "..."
            lines.append(f"- {node.name:38s} : {first}")
    return "\n".join(lines)


def extract_layered_function_index(atomic_py: Path = ATOMIC_PY) -> dict[str, str]:
    """Layered variant of extract_function_index. Returns {layer_name:
    rendered_index_block}, used both to populate the Selector prompt slots and
    as an in-memory fallback when per-layer .txt files are missing on disk."""
    tree = ast.parse(atomic_py.read_text())
    by_layer: dict[str, list[str]] = {ln: [] for ln in LAYER_NAMES}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("_") or node.name in INVARIANT_FUNCS:
            continue
        layer = LAYER_OF.get(node.name)
        if not layer:
            continue
        doc = ast.get_docstring(node) or ""
        first = doc.strip().split("\n")[0].strip() if doc else "(no docstring)"
        if len(first) > 110:
            first = first[:107] + "..."
        by_layer[layer].append(f"- {node.name:38s} : {first}")
    return {ln: "\n".join(lines) for ln, lines in by_layer.items()}


_DEF_RE = re.compile(r"^def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", re.MULTILINE)


def _split_atomic_txt_blocks(atomic_txt: Path) -> dict[str, str]:
    """Parse atomic_functions.txt into {func_name: full def block} entries."""
    text = atomic_txt.read_text()
    blocks: dict[str, str] = {}
    matches = list(_DEF_RE.finditer(text))
    for i, m in enumerate(matches):
        name = m.group(1)
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        blocks[name] = text[start:end].rstrip() + "\n"
    return blocks


def extract_function_docs(funcs: list[str], atomic_txt: Optional[Path] = None) -> str:
    """Concatenated def + docstring blocks from atomic_functions.txt for the
    given function names, in atomic_functions.txt order."""
    if atomic_txt is None:
        atomic_txt = SHARED_PROMPT_DIR / "atomic_functions.txt"
    blocks = _split_atomic_txt_blocks(atomic_txt)
    out = []
    # Preserve original order
    for name in blocks:
        if name in funcs:
            out.append(blocks[name])
    return "\n".join(out)


def extract_layered_function_docs(
    selected_by_layer: dict[str, list[str]],
    atomic_txt: Optional[Path] = None,
) -> dict[str, str]:
    """Layered variant of extract_function_docs. Returns {layer_name:
    concatenated_def_blocks} for the per-layer selected functions, preserving
    atomic_functions.txt order within each layer."""
    if atomic_txt is None:
        atomic_txt = SHARED_PROMPT_DIR / "atomic_functions.txt"
    blocks = _split_atomic_txt_blocks(atomic_txt)
    out: dict[str, str] = {}
    for ln in LAYER_NAMES:
        funcs_set = set(selected_by_layer.get(ln, []))
        out[ln] = "\n".join(blocks[n] for n in blocks if n in funcs_set)
    return out


def extract_invariant_docs(atomic_txt: Optional[Path] = None) -> str:
    """Docs for INVARIANT functions (always prepended)."""
    if atomic_txt is None:
        atomic_txt = SHARED_PROMPT_DIR / "atomic_functions.txt"
    blocks = _split_atomic_txt_blocks(atomic_txt)
    out = []
    for name in blocks:
        if name in INVARIANT_FUNCS:
            out.append(blocks[name])
    return "\n".join(out)


def filter_examples_for_funcs(
    funcs: list[str],
    examples_txt: Optional[Path] = None,
    basic_count: int = 5,
) -> str:
    """
    From examples.txt, return blocks that demonstrate the selected functions
    plus a fixed prefix of basic patterns.

    Layout:
      1. First `basic_count` ```python``` blocks of examples.txt are ALWAYS
         included. These cover the most common geometric / kinematic primitives
         (has_velocity, has_obj_in_dir, near_objects, etc.) that nearly every
         Coder needs even when not explicitly in selected_funcs. This addresses
         the prior failure mode where the filter removed all has_velocity
         examples and Coder defaulted to `stationary`.
      2. After the basic prefix, only blocks that call at least one function
         in `funcs` are appended. ANTI-PATTERN / SPECIAL-CASE blocks that
         mention any selected function are kept regardless (preserved guidance).
    """
    if examples_txt is None:
        examples_txt = SHARED_PROMPT_DIR / "examples.txt"
    text = examples_txt.read_text()
    blocks = re.findall(r"```python\n(.*?)```", text, re.DOTALL)

    keep: list[str] = []
    # 1) basic prefix — always included
    basic_blocks = blocks[:basic_count]
    for block in basic_blocks:
        keep.append("```python\n" + block + "```")

    # 2) function-matched blocks after the prefix
    for block in blocks[basic_count:]:
        calls = set(re.findall(r"\b([a-z_][a-z0-9_]*)\s*\(", block))
        is_special = "ANTI-PATTERN" in block or "SPECIAL-CASE" in block
        if calls & set(funcs):
            keep.append("```python\n" + block + "```")
        elif is_special and any(f in block for f in funcs):
            keep.append("```python\n" + block + "```")
    return "\n".join(keep)


def filter_layered_examples_for_funcs(
    selected_by_layer: dict[str, list[str]],
    examples_txt: Optional[Path] = None,
    basic_count: int = 5,
) -> dict[str, str]:
    """Layered variant of filter_examples_for_funcs. Returns
      {"basic": <prefix>, "layer1_context": ..., "layer2_actor": ...,
       "layer3_behavior": ..., "layer4_relation": ...}
    The first `basic_count` python blocks form the basic prefix (always shown
    regardless of layer). Each remaining block is assigned to the first layer
    (in cascade order L1 -> L2 -> L3 -> L4) whose selected functions appear in
    the block's call set — one block per layer slot."""
    if examples_txt is None:
        examples_txt = SHARED_PROMPT_DIR / "examples.txt"
    text = examples_txt.read_text()
    blocks = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
    basic = "\n".join("```python\n" + b + "```" for b in blocks[:basic_count])
    out_lists: dict[str, list[str]] = {ln: [] for ln in LAYER_NAMES}
    for block in blocks[basic_count:]:
        calls = set(re.findall(r"\b([a-z_][a-z0-9_]*)\s*\(", block))
        for ln in LAYER_NAMES:
            funcs_ln = set(selected_by_layer.get(ln, []))
            if calls & funcs_ln:
                out_lists[ln].append("```python\n" + block + "```")
                break
    return {"basic": basic,
            **{ln: "\n".join(out_lists[ln]) for ln in LAYER_NAMES}}


# ---------------------------------------------------------------------------
# Hash + atomic write
# ---------------------------------------------------------------------------

def _md5(*parts: str) -> str:
    h = hashlib.md5()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def compute_deconstructor_hash(
    description: str,
    deconstructor_sys: str,
    few_shot: str,
) -> str:
    return _md5(description, deconstructor_sys, few_shot)


def compute_selector_hash(
    description: str,
    selector_sys: str,
    function_indexes: dict[str, str],
    few_shot: str,
    shared_guide: str = "",
    deconstructed: str = "",
) -> str:
    """Layer-aware hash. function_indexes is {layer_name: rendered_index} —
    per-layer sub-hashes are folded into the combined hash so that editing one
    layer's index does not silently look identical to another layer's edit."""
    layer_hashes = [_md5(function_indexes.get(ln, "")) for ln in LAYER_NAMES]
    return _md5(description, selector_sys, *layer_hashes,
                few_shot, shared_guide, deconstructed)


def compute_coder_hash(
    description: str,
    selected_by_layer: dict[str, list[str]],
    selected_docs: dict[str, str],
    invariant_docs: str,
    filtered_examples: dict[str, str],
    categories: str,
    coder_sys: str,
    shared_guide: str = "",
    deconstructed: str = "",
) -> str:
    """Layer-aware hash. Dict inputs are serialized deterministically so that
    changing the order of layers (or the order of funcs within a layer) does
    not affect the hash."""
    sel_repr = json.dumps({ln: sorted(selected_by_layer.get(ln, []))
                           for ln in LAYER_NAMES}, sort_keys=True)
    docs_repr = json.dumps({ln: selected_docs.get(ln, "") for ln in LAYER_NAMES},
                           sort_keys=True)
    ex_repr = json.dumps({k: filtered_examples.get(k, "")
                          for k in ("basic", *LAYER_NAMES)}, sort_keys=True)
    return _md5(
        description,
        sel_repr,
        docs_repr,
        invariant_docs,
        ex_repr,
        categories,
        coder_sys,
        shared_guide,
        deconstructed,
    )


def atomic_write_text(path: Path, text: str) -> None:
    """POSIX atomic rename — safe against parallel subprocess writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2))


# ---------------------------------------------------------------------------
# Selector output validation
# ---------------------------------------------------------------------------

def parse_deconstructor_json(text: str) -> Optional[dict]:
    """
    Parse Deconstructor raw output. Returns dict
      {subject, essential_filters, descriptive_context, confidence}
    or None on failure (caller should fall back to using the raw description).

    Validation:
      - subject: non-empty string
      - essential_filters: dict with exactly the 4 LAYER_NAMES keys, each a
        list of strings (any/all may be empty — that is a valid
        "bare-category fallback" signal)
      - descriptive_context: list of strings (may be empty)
      - confidence: high|medium|low (default medium)
    """
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    raw = m.group(1) if m else text
    s = raw.find("{")
    e = raw.rfind("}")
    if s < 0 or e <= s:
        return None
    try:
        obj = json.loads(raw[s : e + 1])
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    subject = obj.get("subject")
    if not isinstance(subject, str) or not subject.strip():
        return None
    essential = obj.get("essential_filters", {})
    if not isinstance(essential, dict):
        return None
    norm_ess: dict[str, list[str]] = {}
    for layer in LAYER_NAMES:
        v = essential.get(layer, [])
        if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
            return None
        norm_ess[layer] = v
    context = obj.get("descriptive_context", [])
    if not isinstance(context, list) or not all(isinstance(x, str) for x in context):
        return None
    confidence = obj.get("confidence", "medium")
    if confidence not in {"high", "medium", "low"}:
        confidence = "medium"
    return {
        "subject": subject.strip(),
        "essential_filters": norm_ess,
        "descriptive_context": context,
        "confidence": confidence,
    }


def render_deconstructed_block(deconstructed: Optional[dict]) -> str:
    """Render the deconstructed dict into a human-readable block that is
    injected into the Selector and Coder prompts. Returns "" if None — that
    is treated as "no deconstruction available" and downstream prompts will
    use the raw description.
    """
    if not deconstructed:
        return ""
    subj = deconstructed.get("subject", "")
    ess = deconstructed.get("essential_filters", {}) or {}
    ctx = deconstructed.get("descriptive_context", [])
    conf = deconstructed.get("confidence", "medium")
    lines = [
        f"Subject (the referred object): {subj}",
        "Essential filters (encode each as one atom in the matching layer):",
        f"  L1 (context):  {json.dumps(ess.get('layer1_context', []),  ensure_ascii=False)}",
        f"  L2 (actor):    {json.dumps(ess.get('layer2_actor', []),    ensure_ascii=False)}",
        f"  L3 (behavior): {json.dumps(ess.get('layer3_behavior', []), ensure_ascii=False)}",
        f"  L4 (relation): {json.dumps(ess.get('layer4_relation', []), ensure_ascii=False)}",
        f"Descriptive context (DO NOT add atoms for these): {json.dumps(ctx, ensure_ascii=False)}",
        f"Deconstructor confidence: {conf}",
    ]
    return "\n".join(lines)


def parse_selector_json(text: str, atomic_py: Path = ATOMIC_PY) -> Optional[dict]:
    """
    Parse Selector raw output. Returns dict
      {selected_funcs_by_layer, selected_funcs, rationale, confidence,
       valid_funcs_intersected, helpers_added}
    or None on failure (caller should fall back).

    Pipeline:
      1) Strip ```json``` fences, find first { ... last }.
      2) json.loads -> dict with `selected_funcs_by_layer` (4-key dict).
      3) Per-layer validation: each value is list[str].
      4) Drop INVARIANT, intersect with valid_selectable_funcs, dedupe across
         all layers (a function chosen in two layers is kept only once).
      5) Route every kept function to its true layer per LAYER_OF (silent
         re-route for Selector mis-placement — keeps the pipeline robust).
      6) If 0 functions remain across all layers -> return None.
      7) Per-layer COMMON_HELPERS_BY_LAYER augment when total < 3 or
         confidence in {low, medium}.
      8) Per-layer cap of 8 (slice + downgrade confidence to low).
    """
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    raw = m.group(1) if m else text
    s = raw.find("{")
    e = raw.rfind("}")
    if s < 0 or e <= s:
        return None
    try:
        obj = json.loads(raw[s : e + 1])
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    by_layer = obj.get("selected_funcs_by_layer")
    if not isinstance(by_layer, dict):
        return None

    rationale = obj.get("rationale", "")
    confidence = obj.get("confidence", "medium")
    if confidence not in {"high", "medium", "low"}:
        confidence = "medium"

    valid = valid_selectable_funcs(atomic_py)
    cleaned: dict[str, list[str]] = {ln: [] for ln in LAYER_NAMES}
    seen: set[str] = set()
    for layer in LAYER_NAMES:
        raw_list = by_layer.get(layer, [])
        if not isinstance(raw_list, list):
            return None
        for f in raw_list:
            if not isinstance(f, str):
                continue
            if f in INVARIANT_FUNCS:
                continue
            if f not in valid:
                continue
            if f in seen:
                continue
            seen.add(f)
            target = LAYER_OF.get(f, layer)        # silent re-route on mis-placement
            cleaned[target].append(f)

    total = sum(len(v) for v in cleaned.values())
    if total == 0:
        return None

    augmented = {ln: list(v) for ln, v in cleaned.items()}
    # Helper-augmentation policy (v3 tightening): v2 eval showed
    # count=5 (n=21) HOTA 0.480 best, count=6 (n=149) 0.462, count=7 (n=163)
    # **0.339** worst. Move threshold from <6 to <5 so the augmentation stops
    # earlier and fewer cases bleed into the count=7+ degradation bucket.
    if confidence in ("low", "medium") or total < 5:
        for layer, helpers in COMMON_HELPERS_BY_LAYER.items():
            for h in helpers:
                if h in valid and h not in augmented[layer]:
                    augmented[layer].append(h)

    for ln in LAYER_NAMES:
        if len(augmented[ln]) > 6:
            augmented[ln] = augmented[ln][:6]
            confidence = "low"

    flat = [f for ln in LAYER_NAMES for f in augmented[ln]]
    return {
        "selected_funcs_by_layer": augmented,
        "selected_funcs": flat,                    # back-compat flat list
        "rationale": rationale,
        "confidence": confidence,
        "valid_funcs_intersected": cleaned,
        "helpers_added": {ln: [h for h in augmented[ln] if h not in cleaned[ln]]
                          for ln in LAYER_NAMES},
    }


# ---------------------------------------------------------------------------
# Context builders
# ---------------------------------------------------------------------------

def _safe_inject(template: str, **slots: str) -> str:
    """str.format() alternative that does not choke on literal { } in the template.

    Each slot is filled via plain .replace() of {slot_name}. Anything else
    (e.g. JSON examples with literal { }) is left untouched.
    """
    out = template
    for k, v in slots.items():
        out = out.replace("{" + k + "}", v)
    return out


def _load_shared_guide(prompt_dir: Path) -> str:
    """Optional domain guide loaded by both Selector and Coder. Empty string
    if `shared_guide.txt` is absent or empty — so callers can no-op it by
    blanking the file."""
    path = prompt_dir / "shared_guide.txt"
    if not path.exists():
        return ""
    return path.read_text()


def _load_few_shot(prompt_dir: Path, split: str = "val") -> str:
    """Load the selector few-shot file matching the eval split. Returns "" if
    the file is missing — caller's build_selector_context will inject an
    empty few-shot block."""
    candidate = prompt_dir / f"selector_few_shot_{split}.txt"
    return candidate.read_text() if candidate.exists() else ""


def _load_deconstructor_few_shot(prompt_dir: Path) -> str:
    """Single shared file across splits — descriptions overlap heavily and
    the deconstructor's decisions are about phrase types, not split-specific
    GT distributions."""
    path = prompt_dir / "deconstructor_few_shot.txt"
    if not path.exists():
        return ""
    return path.read_text()


def build_deconstructor_context(
    description: str,
    prompt_dir: Path = MULTI_PROMPT_DIR,
) -> str:
    sys = (prompt_dir / "deconstructor_system_prompt.txt").read_text()
    few_shot = _load_deconstructor_few_shot(prompt_dir)
    return _safe_inject(
        sys,
        deconstructor_few_shot=few_shot,
        natural_language_description=description,
    )


def _load_layered_function_indexes(prompt_dir: Path) -> dict[str, str]:
    """Load 4 per-layer function index files. If none exist on disk, fall back
    to extracting them in-memory from atomic_functions.py."""
    idx: dict[str, str] = {}
    for i, ln in enumerate(LAYER_NAMES, start=1):
        p = prompt_dir / f"selector_function_index_layer{i}.txt"
        idx[ln] = p.read_text() if p.exists() else ""
    if not any(idx.values()):
        idx = extract_layered_function_index()
    return idx


def build_selector_context(
    description: str,
    prompt_dir: Path = MULTI_PROMPT_DIR,
    split: str = "val",
    deconstructed: Optional[dict] = None,
) -> str:
    sys = (prompt_dir / "selector_system_prompt.txt").read_text()
    idx = _load_layered_function_indexes(prompt_dir)
    few_shot = _load_few_shot(prompt_dir, split)
    shared = _load_shared_guide(prompt_dir)
    return _safe_inject(
        sys,
        shared_guide=shared,
        function_index_layer1=idx["layer1_context"],
        function_index_layer2=idx["layer2_actor"],
        function_index_layer3=idx["layer3_behavior"],
        function_index_layer4=idx["layer4_relation"],
        few_shot_examples=few_shot,
        natural_language_description=description,
        deconstructed=render_deconstructed_block(deconstructed),
    )


def build_coder_context(
    description: str,
    selected_funcs_by_layer: dict[str, list[str]],
    prompt_dir: Path = MULTI_PROMPT_DIR,
    selector_confidence: str = "medium",
    selector_rationale: str = "",
    deconstructed: Optional[dict] = None,
) -> str:
    coder_sys = (prompt_dir / "coder_system_prompt.txt").read_text()
    # Shared assets live in SHARED_PROMPT_DIR (independent of multi/single split).
    categories = (SHARED_PROMPT_DIR / "categories.txt").read_text()
    invariant_docs = extract_invariant_docs(SHARED_PROMPT_DIR / "atomic_functions.txt")
    selected_docs = extract_layered_function_docs(
        selected_funcs_by_layer, SHARED_PROMPT_DIR / "atomic_functions.txt")
    filtered_ex = filter_layered_examples_for_funcs(
        selected_funcs_by_layer, SHARED_PROMPT_DIR / "examples.txt")
    shared = _load_shared_guide(prompt_dir)
    return _safe_inject(
        coder_sys,
        shared_guide=shared,
        invariant_docs=invariant_docs,
        selected_funcs_docs_layer1=selected_docs["layer1_context"],
        selected_funcs_docs_layer2=selected_docs["layer2_actor"],
        selected_funcs_docs_layer3=selected_docs["layer3_behavior"],
        selected_funcs_docs_layer4=selected_docs["layer4_relation"],
        filtered_examples_basic=filtered_ex["basic"],
        filtered_examples_layer1=filtered_ex["layer1_context"],
        filtered_examples_layer2=filtered_ex["layer2_actor"],
        filtered_examples_layer3=filtered_ex["layer3_behavior"],
        filtered_examples_layer4=filtered_ex["layer4_relation"],
        categories=categories,
        natural_language_description=description,
        selector_confidence=selector_confidence,
        selector_rationale=selector_rationale,
        deconstructed=render_deconstructed_block(deconstructed),
    )


# ---------------------------------------------------------------------------
# End-to-end driver (LLM call abstracted)
# ---------------------------------------------------------------------------

def predict_scenario_multi_agent(
    description: str,
    output_dir: Path,
    model_name: str,
    llm_call,
    split: str = "val",
) -> Path:
    """
    Two-stage pipeline. `llm_call(prompt: str, model_name: str) -> str` is injected
    so callers can plug in predict_scenario_anthropic (production) or a sub-agent
    stub (simulation / tests).

    Files written into output_dir / model_name / multi_agent / :
      - {description}_selector.json
      - {description}.txt   (Python code)
      - {description}_meta.json

    Returns the path to the saved code .txt.

    On Selector failure (None from parse_selector_json), the caller is expected
    to invoke the single-agent fallback. We surface this by returning None.
    """
    from refAV.code_generation import extract_and_save_code_blocks  # local import to avoid cycle

    # Layout: code lives directly in output_dir/model_name/ so the existing
    # eval pipeline keeps working unchanged. JSON sidecars (selector/meta) go
    # into output_dir/model_name/meta/ so the root stays clean and eval's
    # glob on *.txt never sees the JSON files.
    code_root = output_dir / model_name
    meta_root = code_root / "meta"
    code_root.mkdir(parents=True, exist_ok=True)
    meta_root.mkdir(parents=True, exist_ok=True)
    code_path = code_root / f"{description}.txt"
    decon_path = meta_root / f"{description}_deconstructor.json"
    sel_path = meta_root / f"{description}_selector.json"
    meta_path = meta_root / f"{description}_meta.json"

    # ---- Stage 1: Deconstructor ---------------------------------------
    decon_sys = (MULTI_PROMPT_DIR / "deconstructor_system_prompt.txt").read_text() \
        if (MULTI_PROMPT_DIR / "deconstructor_system_prompt.txt").exists() else ""
    decon_few_shot = _load_deconstructor_few_shot(MULTI_PROMPT_DIR)
    decon_hash = compute_deconstructor_hash(description, decon_sys, decon_few_shot)

    deconstructed: Optional[dict] = None
    if decon_path.exists():
        try:
            cached = json.loads(decon_path.read_text())
            if cached.get("deconstructor_input_hash") == decon_hash:
                deconstructed = {
                    "subject": cached.get("subject", ""),
                    "essential_filters": cached.get("essential_filters", []),
                    "descriptive_context": cached.get("descriptive_context", []),
                    "confidence": cached.get("confidence", "medium"),
                }
        except Exception:
            deconstructed = None

    if deconstructed is None and decon_sys:
        decon_prompt = build_deconstructor_context(description)
        decon_raw = llm_call(decon_prompt, model_name)
        deconstructed = parse_deconstructor_json(decon_raw)
        if deconstructed is not None:
            atomic_write_json(decon_path, {
                **deconstructed,
                "model_name": model_name,
                "deconstructor_input_hash": decon_hash,
            })
        # If deconstructor fails we proceed without it (selector/coder behave
        # as in the 2-stage pipeline).

    decon_serialized = json.dumps(deconstructed, ensure_ascii=False, sort_keys=True) \
        if deconstructed else ""

    # ---- Stage 2: Selector --------------------------------------------
    selector_sys = (MULTI_PROMPT_DIR / "selector_system_prompt.txt").read_text()
    function_indexes = _load_layered_function_indexes(MULTI_PROMPT_DIR)
    few_shot = _load_few_shot(MULTI_PROMPT_DIR, split)
    shared = _load_shared_guide(MULTI_PROMPT_DIR)
    selector_hash = compute_selector_hash(
        description, selector_sys, function_indexes, few_shot, shared, decon_serialized,
    )

    sel: Optional[dict] = None
    if sel_path.exists():
        try:
            cached = json.loads(sel_path.read_text())
            if cached.get("selector_input_hash") == selector_hash:
                sel = cached
        except Exception:
            sel = None

    if sel is None:
        sel_prompt = build_selector_context(description, split=split, deconstructed=deconstructed)
        sel_raw = llm_call(sel_prompt, model_name)
        sel = parse_selector_json(sel_raw)
        if sel is None:
            # Caller is responsible for invoking the single-agent fallback.
            atomic_write_json(meta_path, {
                "fallback_triggered": True,
                "fallback_reason": "selector_json_parse_or_validation_fail",
                "deconstructor_input_hash": decon_hash,
                "selector_input_hash": selector_hash,
            })
            return None
        sel["model_name"] = model_name
        sel["selector_input_hash"] = selector_hash
        atomic_write_json(sel_path, sel)

    selected_by_layer = sel["selected_funcs_by_layer"]

    # ---- Stage 3: Coder ------------------------------------------------
    coder_sys = (MULTI_PROMPT_DIR / "coder_system_prompt.txt").read_text()
    categories = (SHARED_PROMPT_DIR / "categories.txt").read_text()
    invariant_docs = extract_invariant_docs()
    selected_docs = extract_layered_function_docs(selected_by_layer)
    filtered_examples = filter_layered_examples_for_funcs(selected_by_layer)
    coder_hash = compute_coder_hash(
        description, selected_by_layer, selected_docs, invariant_docs,
        filtered_examples, categories, coder_sys, shared, decon_serialized,
    )

    meta_existing = {}
    if meta_path.exists():
        try:
            meta_existing = json.loads(meta_path.read_text())
        except Exception:
            meta_existing = {}
    if code_path.exists() and meta_existing.get("coder_input_hash") == coder_hash:
        return code_path

    coder_prompt = build_coder_context(
        description,
        selected_by_layer,
        selector_confidence=sel.get("confidence", "medium"),
        selector_rationale=sel.get("rationale", ""),
        deconstructed=deconstructed,
    )
    coder_raw = llm_call(coder_prompt, model_name)
    saved_paths = extract_and_save_code_blocks(coder_raw, output_dir=code_root, description=description)
    out_path = Path(saved_paths[-1]) if saved_paths else code_path

    # Meta
    layer_sizes = {ln: len(selected_by_layer.get(ln, [])) for ln in LAYER_NAMES}
    atomic_write_json(meta_path, {
        "fallback_triggered": False,
        "deconstructor_input_hash": decon_hash,
        "deconstructed": deconstructed,
        "selector_input_hash": selector_hash,
        "coder_input_hash": coder_hash,
        "selected_funcs_by_layer": selected_by_layer,
        "helpers_added": sel.get("helpers_added", {}),
        "layer_sizes": layer_sizes,
        "selected_funcs_count": sum(layer_sizes.values()),
        "confidence": sel.get("confidence"),
        "model_name": model_name,
    })
    return out_path
