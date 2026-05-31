"""
Integrated scenario mining + visualization verification pipeline.

Flow:
    1. Load [log_id, prompts] pairs from log_prompt_pairs_val.json
    2. For each (log_id, prompt):
        a. Generate/execute scenario mining code via refAV.code_generation
        b. Produce prediction pkl via output_scenario()
    3. After mining, launch visualization prepare_log.py to verify results

Usage:
    conda run -n refAV python run/verification_pipeline.py --exp_name exp74
    conda run -n refAV python run/verification_pipeline.py --exp_name exp74 --limit 3
    conda run -n refAV python run/verification_pipeline.py --exp_name exp74 --skip_viz
"""

import argparse
import json
import subprocess
import sys
import yaml
from pathlib import Path
from tqdm import tqdm

from refAV.eval import create_refprog_prediction
from refAV.code_generation import build_context, load_qwen
from refAV.utils import cache_manager, construct_caches
from refAV.dataset_conversion import pickle_to_feather
import refAV.paths as paths


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIZ_SCRIPT = PROJECT_ROOT / "visualization" / "3d_perception" / "scripts" / "prepare_log.py"


def load_log_prompt_pairs(json_path: Path) -> dict:
    """Read [log_id → prompts] mapping from the scenario mining json."""
    with open(json_path, "rb") as f:
        pairs = json.load(f)
    return pairs


def load_experiment_config(exp_name: str) -> dict:
    """Resolve experiment settings (tracker, LLM, split, context) from experiments.yml."""
    with open(paths.EXPERIMENTS, "rb") as f:
        cfg = yaml.safe_load(f)
    if exp_name not in cfg:
        raise KeyError(f"Experiment '{exp_name}' not found in {paths.EXPERIMENTS}")
    entry = cfg[exp_name]
    return {
        "name": entry["name"],
        "tracker": entry["tracker"],
        "llm": entry["LLM"],
        "split": entry["split"],
        "context": entry.get("context", "RefAV"),
    }


def prepare_tracker_feathers(exp_cfg: dict) -> None:
    """Convert tracker pkl to per-log feather files if not already done."""
    split = exp_cfg["split"]
    tracker_pkl = paths.TRACKER_PRED_DIR / f"{exp_cfg['tracker']}_{split}.pkl"
    if not tracker_pkl.exists():
        print(f"[Tracker] pkl not found at {tracker_pkl} — skip feather conversion.")
        return

    dataset_dir = paths.AV2_DATA_DIR
    out_dir = paths.TRACKER_PRED_DIR / exp_cfg["tracker"]
    print(f"[Tracker] Converting {tracker_pkl.name} → per-log feathers in {out_dir}")
    pickle_to_feather(dataset_dir, tracker_pkl, base_output_dir=out_dir)


def mine_scenarios(
    exp_cfg: dict,
    log_prompt_pairs: dict,
    limit: int | None = None,
) -> list[tuple[str, str]]:
    """Run scenario mining for each (log_id, prompt). Returns processed pairs."""
    llm_name = exp_cfg["llm"]
    tracker_name = exp_cfg["tracker"]
    exp_name = exp_cfg["name"]
    context_name = exp_cfg["context"]

    # Build LLM context once and pre-load local model if needed.
    custom_context = build_context(context_path=paths.PROMPT_DIR / context_name)
    scenario_def_dir = paths.LLM_PRED_DIR / context_name

    local_model, local_tokenizer = None, None
    if "qwen" in llm_name.lower():
        print(f"[LLM] Pre-loading Qwen model: {llm_name}")
        local_model, local_tokenizer = load_qwen(llm_name)

    processed: list[tuple[str, str]] = []
    items = list(log_prompt_pairs.items())
    if limit:
        items = items[:limit]

    total = sum(len(prompts) for _, prompts in items)
    pbar = tqdm(total=total, desc="Mining scenarios")

    for log_id, prompts in items:
        cache_manager.clear_all()
        log_dir = paths.TRACKER_PRED_DIR / tracker_name / exp_cfg["split"] / log_id

        # Build semantic_lane + road_side + color caches for this log.
        try:
            construct_caches(log_dir, prompts)
        except Exception as e:
            print(f"[Cache] Failed for {log_id}: {e}")

        cache_manager.load_custom_caches(log_dir)

        for prompt in prompts:
            try:
                create_refprog_prediction(
                    description=prompt,
                    log_id=log_id,
                    llm_name=llm_name,
                    tracker_name=tracker_name,
                    experiment_name=exp_name,
                    custom_context=custom_context,
                    scenario_def_output_dir=scenario_def_dir,
                    local_model=local_model,
                    local_tokenizer=local_tokenizer,
                )
                processed.append((log_id, prompt))
            except Exception as e:
                print(f"[Mining] {log_id} / '{prompt}' failed: {e}")
            pbar.update(1)

    pbar.close()
    return processed


def visualize_logs(log_ids: list[str], force: bool = False) -> None:
    """Invoke prepare_log.py for each log_id to build the viewer payload."""
    if not VIZ_SCRIPT.exists():
        print(f"[Viz] prepare_log.py not found at {VIZ_SCRIPT}")
        return

    for log_id in log_ids:
        cmd = [sys.executable, str(VIZ_SCRIPT), "--log_id", log_id]
        if force:
            cmd.append("--force")
        print(f"[Viz] {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=PROJECT_ROOT)
        if result.returncode != 0:
            print(f"[Viz] prepare_log.py failed for {log_id} (exit {result.returncode})")


def run_verification_pipeline(
    exp_name: str,
    log_prompt_json: Path,
    limit: int | None = None,
    skip_viz: bool = False,
    force_viz: bool = False,
) -> None:
    """End-to-end orchestration: load pairs → mine → visualize."""
    print(f"=== Verification pipeline [{exp_name}] ===")
    exp_cfg = load_experiment_config(exp_name)
    print(f"[Config] tracker={exp_cfg['tracker']} llm={exp_cfg['llm']} "
          f"split={exp_cfg['split']} context={exp_cfg['context']}")

    log_prompt_pairs = load_log_prompt_pairs(log_prompt_json)
    print(f"[Data] {len(log_prompt_pairs)} logs loaded from {log_prompt_json.name}")

    prepare_tracker_feathers(exp_cfg)

    processed = mine_scenarios(exp_cfg, log_prompt_pairs, limit=limit)
    print(f"[Mining] Finished: {len(processed)} (log, prompt) pairs processed")

    if not skip_viz:
        unique_logs = sorted({log_id for log_id, _ in processed})
        print(f"[Viz] Rendering payload for {len(unique_logs)} logs")
        visualize_logs(unique_logs, force=force_viz)

    print("=== Pipeline complete ===")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True,
                        help="Experiment name in run/experiment_configs/experiments.yml")
    parser.add_argument("--log_prompt_json", type=Path,
                        default=PROJECT_ROOT / "scenario_mining_downloads" / "log_prompt_pairs_val.json")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N logs (for quick debugging)")
    parser.add_argument("--skip_viz", action="store_true", help="Run mining only")
    parser.add_argument("--force_viz", action="store_true", help="Re-run prepare_log.py even if cached")
    args = parser.parse_args()

    run_verification_pipeline(
        exp_name=args.exp_name,
        log_prompt_json=args.log_prompt_json,
        limit=args.limit,
        skip_viz=args.skip_viz,
        force_viz=args.force_viz,
    )


if __name__ == "__main__":
    main()
