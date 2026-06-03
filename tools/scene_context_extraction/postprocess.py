"""CLI entry point for post-processing (smoothing + confirmed-run dilation, 2Hz).

Examples:
    # default: output/scene_context/val → .../val_processed
    PYTHONPATH=. python -m tools.scene_context_extraction.postprocess --split val

    # custom paths / smoothing + dilation knobs
    PYTHONPATH=. python -m tools.scene_context_extraction.postprocess \\
        --input-dir output/scene_context/val \\
        --output-dir output/scene_context/val_processed \\
        --mv-window-size 3 --mv-threshold 0.5 \\
        --dilation-min-run 3 --dilation-step 1

    # specific logs, parallel workers
    PYTHONPATH=. python -m tools.scene_context_extraction.postprocess \\
        --split val --workers 8 --log-ids <log_id1> <log_id2>
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.scene_context_extraction.src.postprocess_runner import postprocess_split

DEFAULT_ROOT = Path("output/scene_context")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scene Context post-processing (smoothing + confirmed-run dilation)"
    )
    parser.add_argument(
        "--split", type=str, default="val",
        help="Split name (val / test). Used for default input/output paths.",
    )
    parser.add_argument(
        "--input-dir", type=str, default=None,
        help=f"Input split dir (default: {DEFAULT_ROOT}/<split>).",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help=f"Output split dir (default: {DEFAULT_ROOT}/<split>_processed).",
    )
    parser.add_argument(
        "--mv-window-size", type=int, default=3,
        help="Symmetric window size in frames (2Hz: 3 ≈ 1.5s).",
    )
    parser.add_argument(
        "--mv-threshold", type=float, default=0.5,
        help="Minimum True ratio to output True (default 0.5, strict majority).",
    )
    parser.add_argument(
        "--dilation-min-run", type=int, default=3,
        help="Minimum True-run length (after smoothing) that qualifies for dilation.",
    )
    parser.add_argument(
        "--dilation-step", type=int, default=1,
        help="Symmetric dilation in frames per qualifying run (0 disables).",
    )
    parser.add_argument(
        "--log-ids", type=str, nargs="*", default=None,
        help="Process only these logs (default: all under the split dir).",
    )
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    input_split_dir = (
        Path(args.input_dir) if args.input_dir else DEFAULT_ROOT / args.split
    )
    output_split_dir = (
        Path(args.output_dir)
        if args.output_dir
        else DEFAULT_ROOT / f"{args.split}_processed"
    )

    if not input_split_dir.exists():
        print(f"[ERROR] Input directory does not exist: {input_split_dir}")
        sys.exit(1)

    postprocess_split(
        input_split_dir=input_split_dir,
        output_split_dir=output_split_dir,
        mv_window_size=args.mv_window_size,
        mv_threshold=args.mv_threshold,
        dilation_min_run=args.dilation_min_run,
        dilation_step=args.dilation_step,
        log_ids=args.log_ids,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()
