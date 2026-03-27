#!/usr/bin/env python3
"""Print a small sample of inferred brightfield/darkfield pairs for manual inspection."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from schisto_mobile_ai.data.metadata_builder import format_pair_examples


def build_parser() -> argparse.ArgumentParser:
    """Build the parser for pair-example inspection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs-csv",
        type=Path,
        default=REPO_ROOT / "metadata" / "pairs.csv",
        help="Path to the pairs.csv file produced by build_metadata.py.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="How many complete pairs to print.",
    )
    return parser


def main() -> int:
    """Load pairs.csv and print a small sample for manual review."""
    parser = build_parser()
    args = parser.parse_args()

    if not args.pairs_csv.exists():
        raise FileNotFoundError(
            f"Pairs file does not exist: {args.pairs_csv}. "
            "Run scripts/build_metadata.py first."
        )

    pairs = pd.read_csv(args.pairs_csv)
    print(format_pair_examples(pairs, limit=args.limit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
