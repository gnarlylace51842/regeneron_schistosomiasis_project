#!/usr/bin/env python3
"""Aggregate image-level validation predictions to patient-level metrics and outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from schisto_mobile_ai.eval.patient_level import evaluate_patient_level, format_patient_eval_summary
from schisto_mobile_ai.utils.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for patient-level evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        required=True,
        help="Path to a validation predictions CSV from a trained model run.",
    )
    parser.add_argument(
        "--patients-csv",
        type=Path,
        default=REPO_ROOT / "metadata" / "patients.csv",
        help="Path to metadata/patients.csv.",
    )
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=REPO_ROOT / "splits" / "random_patient_split.csv",
        help="Path to the split CSV used for the model run.",
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        default="mean",
        choices=("max", "mean", "noisy_or"),
        help="How to aggregate image-level probabilities to patient level.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Binary decision threshold applied to the aggregated patient probability.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where patient_predictions.csv, metrics.json, and confusion_matrix.png will be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing evaluation outputs.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce log output.",
    )
    return parser


def _default_output_dir(predictions_csv: Path, aggregation: str) -> Path:
    run_name = predictions_csv.parent.name
    return REPO_ROOT / "results" / "evaluations" / f"{run_name}_{aggregation}"


def main() -> int:
    """Run patient-level aggregation and evaluation."""
    parser = build_parser()
    args = parser.parse_args()
    logger = configure_logging(quiet=args.quiet)

    output_dir = args.output_dir or _default_output_dir(args.predictions_csv, args.aggregation)
    result = evaluate_patient_level(
        predictions_csv=args.predictions_csv,
        patients_csv=args.patients_csv,
        split_csv=args.split_csv,
        output_dir=output_dir,
        aggregation=args.aggregation,
        threshold=args.threshold,
        overwrite=args.overwrite,
    )

    print(format_patient_eval_summary(result.metrics))
    logger.info("Saved patient-level predictions to %s", result.patient_predictions_path)
    logger.info("Saved patient-level metrics to %s", result.metrics_path)
    logger.info("Saved confusion matrix plot to %s", result.confusion_matrix_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
