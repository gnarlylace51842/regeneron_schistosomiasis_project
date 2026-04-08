#!/usr/bin/env python3
"""Sweep decision thresholds for patient-level predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from schisto_mobile_ai.eval.diagnostics import build_threshold_sweep, plot_threshold_sweep, select_best_threshold
from schisto_mobile_ai.utils.io import ensure_dir, write_json
from schisto_mobile_ai.utils.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for threshold sweeps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--patient-predictions-csv",
        type=Path,
        required=True,
        help="Path to patient_predictions.csv from scripts/eval_patient_level.py.",
    )
    parser.add_argument(
        "--num-thresholds",
        type=int,
        default=201,
        help="Number of thresholds between 0 and 1 to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where sweep CSV, plot, and summary JSON will be written.",
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


def _default_output_dir(path: Path) -> Path:
    return REPO_ROOT / "results" / "diagnostics" / f"{path.parent.name}_threshold_sweep"


def _load_patient_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"target", "patient_probability"}
    missing = required - set(frame.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"patient_predictions.csv is missing required columns: {missing_text}")
    frame["target"] = pd.to_numeric(frame["target"], errors="coerce")
    frame["patient_probability"] = pd.to_numeric(frame["patient_probability"], errors="coerce")
    frame = frame.dropna(subset=["target", "patient_probability"]).reset_index(drop=True)
    if frame.empty:
        raise ValueError("patient_predictions.csv does not contain any valid numeric target/probability rows.")
    return frame


def main() -> int:
    """Run a threshold sweep and save compact outputs."""
    parser = build_parser()
    args = parser.parse_args()
    logger = configure_logging(quiet=args.quiet)

    output_dir = args.output_dir or _default_output_dir(args.patient_predictions_csv)
    ensure_dir(output_dir)
    blocking = [
        output_dir / "threshold_sweep.csv",
        output_dir / "threshold_sweep.png",
        output_dir / "threshold_summary.json",
    ]
    existing = [path for path in blocking if path.exists()]
    if existing and not args.overwrite:
        existing_text = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            "Threshold sweep outputs already exist. Pass --overwrite to replace them: "
            f"{existing_text}"
        )

    patient_predictions = _load_patient_predictions(args.patient_predictions_csv)
    thresholds = pd.Series(range(args.num_thresholds)).map(
        lambda index: index / max(args.num_thresholds - 1, 1)
    ).to_numpy()
    sweep = build_threshold_sweep(patient_predictions, thresholds=thresholds)
    sweep.to_csv(output_dir / "threshold_sweep.csv", index=False)
    plot_threshold_sweep(sweep, output_path=output_dir / "threshold_sweep.png")

    summary = {
        "inputs": {
            "patient_predictions_csv": str(args.patient_predictions_csv),
        },
        "outputs": {
            "output_dir": str(output_dir),
            "threshold_sweep_csv": str(output_dir / "threshold_sweep.csv"),
            "threshold_sweep_png": str(output_dir / "threshold_sweep.png"),
        },
        "best_thresholds": {
            "youden_j": select_best_threshold(sweep, "youden_j"),
            "balanced_accuracy": select_best_threshold(sweep, "balanced_accuracy"),
            "f1": select_best_threshold(sweep, "f1"),
        },
    }
    write_json(output_dir / "threshold_summary.json", summary)

    print("Threshold Sweep Summary")
    for metric_name, payload in summary["best_thresholds"].items():
        threshold = payload["threshold"]
        value = payload["value"]
        threshold_text = "n/a" if threshold is None else f"{threshold:.3f}"
        value_text = "n/a" if value is None else f"{value:.4f}"
        print(f"  best_{metric_name}: threshold={threshold_text}, value={value_text}")

    logger.info("Saved threshold sweep to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
