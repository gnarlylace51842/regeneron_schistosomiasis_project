#!/usr/bin/env python3
"""Create fixed patient-safe split definitions from metadata/patients.csv and metadata/pairs.csv."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from schisto_mobile_ai.data.splits import (
    build_split_payload,
    create_random_patient_split,
    create_study_holdout_split,
    format_split_summary,
    load_split_inputs,
    save_split_artifacts,
    summarize_splits,
    validate_contrast_balance,
    validate_no_patient_overlap,
    validate_study_holdout,
)
from schisto_mobile_ai.utils.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for split generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--patients-csv",
        type=Path,
        default=REPO_ROOT / "metadata" / "patients.csv",
        help="Path to metadata/patients.csv.",
    )
    parser.add_argument(
        "--pairs-csv",
        type=Path,
        default=REPO_ROOT / "metadata" / "pairs.csv",
        help="Path to metadata/pairs.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "splits",
        help="Directory where split JSON and CSV files will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible patient or study assignment.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Limit the split generation to a small subset of patients.",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=None,
        help="Limit the number of patients considered during split generation.",
    )
    parser.add_argument(
        "--contrast-balance-tolerance",
        type=float,
        default=0.20,
        help="Allowed absolute deviation in BF/DF availability rates across splits.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing split JSON and CSV files.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce log output.",
    )
    return parser


def _guard_output_paths(output_dir: Path, *, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_targets = [
        output_dir / "random_patient_split.json",
        output_dir / "random_patient_split.csv",
        output_dir / "study_holdout_split.json",
        output_dir / "study_holdout_split.csv",
    ]
    blocking_paths = [path for path in existing_targets if path.exists()]
    if blocking_paths and not overwrite:
        blocking_text = ", ".join(str(path) for path in blocking_paths)
        raise FileExistsError(
            "Split output files already exist. Pass --overwrite to replace them: "
            f"{blocking_text}"
        )


def _save_and_print(
    *,
    split_name: str,
    assignments,
    pairs,
    args: argparse.Namespace,
    study_validation=None,
) -> tuple[Path, Path]:
    patient_validation = validate_no_patient_overlap(assignments)
    contrast_validation = validate_contrast_balance(
        assignments,
        pairs,
        tolerance=args.contrast_balance_tolerance,
    )
    summary = summarize_splits(assignments, pairs)
    payload = build_split_payload(
        split_name=split_name,
        assignments=assignments,
        pairs=pairs,
        seed=args.seed,
        patients_csv=args.patients_csv,
        pairs_csv=args.pairs_csv,
        summary=summary,
        patient_validation=patient_validation,
        contrast_validation=contrast_validation,
        study_validation=study_validation,
    )
    json_path, csv_path = save_split_artifacts(
        output_dir=args.output_dir,
        split_name=split_name,
        assignments=assignments,
        payload=payload,
    )
    print(format_split_summary(split_name, summary, contrast_validation))
    print(f"  saved_json: {json_path}")
    print(f"  saved_csv: {csv_path}")
    return json_path, csv_path


def main() -> int:
    """Generate patient-safe split definitions and write them to disk."""
    parser = build_parser()
    args = parser.parse_args()
    logger = configure_logging(quiet=args.quiet)

    _guard_output_paths(args.output_dir, overwrite=args.overwrite)
    patients, pairs = load_split_inputs(
        args.patients_csv,
        args.pairs_csv,
        subset_size=args.subset_size,
        smoke_test=args.smoke_test,
    )

    random_assignments = create_random_patient_split(patients, seed=args.seed)
    _save_and_print(
        split_name="random_patient_split",
        assignments=random_assignments,
        pairs=pairs,
        args=args,
        study_validation=None,
    )

    try:
        study_assignments = create_study_holdout_split(patients, seed=args.seed)
        study_validation = validate_study_holdout(study_assignments)
    except ValueError as exc:
        print()
        print("Could not create study_holdout_split.")
        print(str(exc))
        print("Needed field: a non-empty 'study_id' value for every patient, available in metadata/patients.csv or inferable from metadata/pairs.csv.")
        return 1

    print()
    _save_and_print(
        split_name="study_holdout_split",
        assignments=study_assignments,
        pairs=pairs,
        args=args,
        study_validation=study_validation,
    )

    logger.info("Saved split definitions to %s", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
