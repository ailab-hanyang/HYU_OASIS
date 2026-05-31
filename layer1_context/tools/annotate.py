"""Context Layer vLLM Annotation — entry point.

Usage (inside container):
    PYTHONPATH=. python -m layer1_context.tools.annotate
    PYTHONPATH=. python -m layer1_context.tools.annotate --config path/to/settings.yaml
    PYTHONPATH=. python -m layer1_context.tools.annotate --log-ids <id1> <id2> --split val
    PYTHONPATH=. python -m layer1_context.tools.annotate --dry-run
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from layer1_context.config.loader import load_config
from layer1_context.annotate.runner import build_annotations


def main() -> None:
    parser = argparse.ArgumentParser(description="Context Layer vLLM Annotation Pipeline")
    parser.add_argument(
        "--config", default=None,
        help="Path to settings.yaml (default: layer1_context/config/settings.yaml)",
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
