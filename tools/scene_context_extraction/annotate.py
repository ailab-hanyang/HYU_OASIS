"""Scene Context vLLM Annotation — entry point.

Usage (inside container):
    PYTHONPATH=. python -m tools.scene_context_extraction.annotate
    PYTHONPATH=. python -m tools.scene_context_extraction.annotate --config path/to/settings.yaml
    PYTHONPATH=. python -m tools.scene_context_extraction.annotate --log-ids <id1> <id2> --split val
    PYTHONPATH=. python -m tools.scene_context_extraction.annotate --dry-run
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.scene_context_extraction.src.loader import load_config
from tools.scene_context_extraction.src.annotate_runner import build_annotations


def main() -> None:
    parser = argparse.ArgumentParser(description="Scene Context vLLM Annotation Pipeline")
    parser.add_argument(
        "--config", default=None,
        help="Path to settings.yaml (default: tools/scene_context_extraction/config/settings.yaml)",
    )
    parser.add_argument(
        "--log-ids", nargs="*", default=None,
        help="Override dataset.log_ids in settings (space-separated).",
    )
    parser.add_argument(
        "--split", default=None,
        help="Override dataset.splits in settings (single split name).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Override dry_run=true in settings (list logs, no inference).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.log_ids is not None:
        config["dataset"]["log_ids"] = args.log_ids
    if args.split is not None:
        config["dataset"]["splits"] = [args.split]
    if args.dry_run:
        config["dry_run"] = True

    logging.basicConfig(
        level=logging.DEBUG if config.get("verbose", False) else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    build_annotations(config)


if __name__ == "__main__":
    main()
