#!/usr/bin/env python3
"""Validate generated metadata tables for patient-level consistency and contrast integrity."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from schisto_mobile_ai.data.validation import (
    build_validation_report,
    format_validation_summary,
    save_validation_report,
)
from schisto_mobile_ai.utils.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for metadata validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images-csv",
        type=Path,
        default=REPO_ROOT / "metadata" / "images.csv",
        help="Path to metadata/images.csv.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT / "metadata" / "validation_report.json",
        help="Where to save the JSON validation report.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce log output.",
    )
    return parser


def main() -> int:
    """Run validation and save the JSON report."""
    parser = build_parser()
    args = parser.parse_args()
    logger = configure_logging(quiet=args.quiet)

    report = build_validation_report(args.images_csv)
    saved_path = save_validation_report(report, args.output_json)
    print(format_validation_summary(report, output_path=saved_path))

    logger.info("Saved metadata validation report to %s", saved_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
