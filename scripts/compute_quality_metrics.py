#!/usr/bin/env python3
"""Compute real image-quality metrics and QC figures from metadata/images.csv."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from schisto_mobile_ai.data.quality_metrics import format_quality_summary, run_quality_metrics
from schisto_mobile_ai.utils.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for quality-metric generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images-csv",
        type=Path,
        default=REPO_ROOT / "metadata" / "images.csv",
        help="Path to metadata/images.csv.",
    )
    parser.add_argument(
        "--pairs-csv",
        type=Path,
        default=REPO_ROOT / "metadata" / "pairs.csv",
        help="Path to metadata/pairs.csv for BF/DF example figures.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=REPO_ROOT / "data" / "raw",
        help="Root directory used with images.csv relative_path values.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=REPO_ROOT / "metadata" / "image_quality.csv",
        help="Where to save the per-image quality metrics CSV.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=REPO_ROOT / "metadata" / "image_quality_summary.json",
        help="Where to save the JSON summary report.",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=REPO_ROOT / "figures" / "qc",
        help="Directory where QC plots and panels will be written.",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=None,
        help="Optionally limit the number of images processed.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a small quality-metric subset for a quick check.",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=768,
        help="Largest image side used for metric computation after resizing.",
    )
    parser.add_argument(
        "--pair-samples",
        type=int,
        default=6,
        help="How many real BF/DF pairs to show in the sample-pair figure.",
    )
    parser.add_argument(
        "--sharp-samples",
        type=int,
        default=4,
        help="How many blurry and sharp examples to show in the blur panel.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing CSV, JSON, or figure outputs.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce log output.",
    )
    return parser


def main() -> int:
    """Run the real quality-metric pipeline."""
    parser = build_parser()
    args = parser.parse_args()
    logger = configure_logging(quiet=args.quiet)

    result = run_quality_metrics(
        images_csv=args.images_csv,
        pairs_csv=args.pairs_csv,
        raw_dir=args.raw_dir,
        output_csv=args.output_csv,
        summary_json=args.summary_json,
        figures_dir=args.figures_dir,
        subset_size=args.subset_size,
        smoke_test=args.smoke_test,
        max_side=args.max_side,
        pair_samples=args.pair_samples,
        sharp_samples=args.sharp_samples,
        overwrite=args.overwrite,
    )

    print(format_quality_summary(result.summary))
    logger.info("Saved image-quality CSV to %s", result.csv_path)
    logger.info("Saved QC summary JSON to %s", result.summary_path)
    logger.info("Saved QC figures to %s", result.figures_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
