#!/usr/bin/env python3
"""Build reusable metadata tables from the dataset under data/raw/."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from schisto_mobile_ai.data.metadata_builder import (
    analyze_dataset,
    format_audit_summary,
    format_pair_examples,
    write_metadata_outputs,
)
from schisto_mobile_ai.utils.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for metadata export."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=REPO_ROOT / "data" / "raw",
        help="Root directory that contains the downloaded dataset.",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=REPO_ROOT / "metadata",
        help="Directory where images.csv, patients.csv, and pairs.csv will be written.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use a tiny subset for a quick end-to-end check.",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=None,
        help="Limit the number of image files indexed while building metadata.",
    )
    parser.add_argument(
        "--metadata-row-limit",
        type=int,
        default=None,
        help="Optionally limit how many rows are read from each metadata file.",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=5,
        help="Number of complete BF/DF examples to print after writing metadata.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing non-empty metadata directory.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce log output.",
    )
    return parser


def main() -> int:
    """Build the metadata index and save it to the metadata directory."""
    parser = build_parser()
    args = parser.parse_args()
    logger = configure_logging(quiet=args.quiet)

    args.metadata_dir.mkdir(parents=True, exist_ok=True)
    if any(args.metadata_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Metadata directory already contains files: {args.metadata_dir}. "
            "Pass --overwrite to replace its contents."
        )

    result = analyze_dataset(
        args.raw_dir,
        subset_size=args.subset_size,
        smoke_test=args.smoke_test,
        metadata_row_limit=args.metadata_row_limit,
    )
    write_metadata_outputs(result, args.metadata_dir)

    print(format_audit_summary(result))
    print(f"\nMetadata tables written to: {args.metadata_dir}")
    if args.examples > 0:
        print()
        print(format_pair_examples(result.pairs, limit=args.examples))

    logger.info("Saved metadata tables to %s", args.metadata_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
