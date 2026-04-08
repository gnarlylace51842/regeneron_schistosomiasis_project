#!/usr/bin/env python3
"""Diagnose whether prediction probabilities are informative or near-collapsed."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from schisto_mobile_ai.eval.diagnostics import (
    load_prediction_frame,
    plot_probability_histograms,
    summarize_probability_frame,
)
from schisto_mobile_ai.utils.io import ensure_dir, write_json
from schisto_mobile_ai.utils.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for prediction diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        required=True,
        help="Path to a validation predictions CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where summary JSON, CSV, and plots will be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing outputs.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce log output.",
    )
    return parser


def _default_output_dir(predictions_csv: Path) -> Path:
    return REPO_ROOT / "results" / "diagnostics" / f"{predictions_csv.parent.name}_prediction_diagnostics"


def _guard_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    ensure_dir(output_dir)
    blocking = [
        output_dir / "probability_summary.json",
        output_dir / "class_probability_stats.csv",
        output_dir / "probability_histogram_by_class.png",
        output_dir / "probability_density_by_class.png",
    ]
    existing = [path for path in blocking if path.exists()]
    if existing and not overwrite:
        existing_text = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            "Diagnostic outputs already exist. Pass --overwrite to replace them: "
            f"{existing_text}"
        )


def _class_stats_frame(summary: dict) -> pd.DataFrame:
    rows = []
    for class_name, stats in summary["by_class"].items():
        rows.append({"class": class_name, **stats})
    return pd.DataFrame(rows).sort_values("class").reset_index(drop=True)


def main() -> int:
    """Run compact prediction diagnostics."""
    parser = build_parser()
    args = parser.parse_args()
    logger = configure_logging(quiet=args.quiet)

    output_dir = args.output_dir or _default_output_dir(args.predictions_csv)
    _guard_output_dir(output_dir, overwrite=args.overwrite)

    predictions = load_prediction_frame(args.predictions_csv)
    summary = summarize_probability_frame(predictions)
    summary["inputs"] = {
        "predictions_csv": str(args.predictions_csv),
    }
    summary["outputs"] = {
        "output_dir": str(output_dir),
        "summary_json": str(output_dir / "probability_summary.json"),
        "class_stats_csv": str(output_dir / "class_probability_stats.csv"),
        "histogram_png": str(output_dir / "probability_histogram_by_class.png"),
        "density_png": str(output_dir / "probability_density_by_class.png"),
    }

    write_json(output_dir / "probability_summary.json", summary)
    _class_stats_frame(summary).to_csv(output_dir / "class_probability_stats.csv", index=False)
    plot_probability_histograms(
        predictions,
        histogram_path=output_dir / "probability_histogram_by_class.png",
        density_path=output_dir / "probability_density_by_class.png",
    )

    overall = summary["overall"]
    constant_check = summary["constant_check"]
    print("Prediction Diagnostic Summary")
    print(f"  n_predictions: {overall['n']}")
    print(f"  mean_probability: {overall['mean']:.6f}")
    print(f"  std_probability: {overall['std']:.6f}")
    print(f"  min_probability: {overall['min']:.6f}")
    print(f"  max_probability: {overall['max']:.6f}")
    print(f"  range_probability: {overall['value_range']:.6f}")
    for band_key in [
        "fraction_within_0p001_of_0p5",
        "fraction_within_0p005_of_0p5",
        "fraction_within_0p01_of_0p5",
        "fraction_within_0p02_of_0p5",
        "fraction_within_0p05_of_0p5",
    ]:
        print(f"  {band_key}: {overall[band_key]:.4f}")
    print(f"  effectively_constant: {constant_check['is_effectively_constant']}")
    if constant_check["reasons"]:
        print(f"  constant_reasons: {', '.join(constant_check['reasons'])}")

    for class_name, stats in summary["by_class"].items():
        print(
            "  class="
            f"{class_name}: n={stats['n']}, mean={stats['mean']:.6f}, std={stats['std']:.6f}, "
            f"min={stats['min']:.6f}, max={stats['max']:.6f}"
        )

    logger.info("Saved prediction diagnostics to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
