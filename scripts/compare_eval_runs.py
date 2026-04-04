#!/usr/bin/env python3
"""Compare multiple patient-level evaluation metrics.json files side by side."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from schisto_mobile_ai.eval.patient_level import compare_metrics_payloads


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for side-by-side metrics comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics-json",
        type=Path,
        nargs="+",
        required=True,
        help="One or more metrics.json files produced by scripts/eval_patient_level.py.",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="roc_auc",
        choices=("accuracy", "sensitivity", "specificity", "precision", "f1", "roc_auc", "n_patients"),
        help="Metric column used to sort the comparison table.",
    )
    return parser


def main() -> int:
    """Load metrics.json files and print a concise comparison table."""
    parser = build_parser()
    args = parser.parse_args()

    payloads = []
    for path in args.metrics_json:
        if not path.exists():
            raise FileNotFoundError(f"metrics.json does not exist: {path}")
        with path.open("r", encoding="utf-8") as handle:
            payloads.append(json.load(handle))

    comparison = compare_metrics_payloads(payloads)
    if comparison.empty:
        print("No metrics payloads were loaded.")
        return 0

    numeric_sort = pd.to_numeric(comparison[args.sort_by], errors="coerce")
    comparison = comparison.assign(_sort_value=numeric_sort).sort_values(
        "_sort_value",
        ascending=False,
        na_position="last",
    ).drop(columns="_sort_value")
    print(comparison.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
