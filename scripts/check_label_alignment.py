#!/usr/bin/env python3
"""Check metadata label, split, and pairing alignment before more experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from schisto_mobile_ai.data.manifest import validate_required_columns
from schisto_mobile_ai.data.schisto_dataset import parse_schisto_image_name
from schisto_mobile_ai.utils.io import ensure_dir, write_json, write_text
from schisto_mobile_ai.utils.logging import configure_logging


POSITIVE_LABELS = {"positive", "1", "true", "yes"}
NEGATIVE_LABELS = {"negative", "0", "false", "no"}


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for metadata alignment checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs-csv",
        type=Path,
        default=REPO_ROOT / "metadata" / "pairs.csv",
        help="Path to metadata/pairs.csv.",
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
        help="Path to the split CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "diagnostics" / "label_alignment",
        help="Directory where JSON and text reports will be written.",
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


def _read_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Required CSV does not exist: {csv_path}")
    return pd.read_csv(csv_path)


def _normalize_label(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def _label_to_target(value: Any) -> float | None:
    normalized = _normalize_label(value)
    if normalized in POSITIVE_LABELS:
        return 1.0
    if normalized in NEGATIVE_LABELS:
        return 0.0
    return None


def _guard_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    ensure_dir(output_dir)
    blocking = [
        output_dir / "alignment_report.json",
        output_dir / "alignment_report.txt",
    ]
    existing = [path for path in blocking if path.exists()]
    if existing and not overwrite:
        existing_text = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            "Alignment outputs already exist. Pass --overwrite to replace them: "
            f"{existing_text}"
        )


def _find_conflicting_groups(frame: pd.DataFrame, *, group_column: str, value_column: str) -> pd.DataFrame:
    working = frame[[group_column, value_column]].copy()
    working[value_column] = working[value_column].map(_normalize_label)
    working = working[working[value_column] != ""].copy()
    if working.empty:
        return pd.DataFrame(columns=[group_column, "n_unique_values", "values"])

    grouped = (
        working.groupby(group_column)[value_column]
        .agg(lambda values: sorted(set(values)))
        .reset_index(name="values")
    )
    grouped["n_unique_values"] = grouped["values"].map(len)
    return grouped[grouped["n_unique_values"] > 1].sort_values(group_column).reset_index(drop=True)


def _class_balance_by_split(frame: pd.DataFrame, *, label_column: str) -> list[dict[str, Any]]:
    working = frame.copy()
    working["target"] = working[label_column].map(_label_to_target)
    working = working.dropna(subset=["target"])
    if working.empty:
        return []

    rows = []
    for split_name, split_frame in working.groupby("split", dropna=False, sort=True):
        target = split_frame["target"].astype(float)
        rows.append(
            {
                "split": str(split_name),
                "n_samples": int(len(split_frame)),
                "n_positive": int((target == 1.0).sum()),
                "n_negative": int((target == 0.0).sum()),
                "positive_rate": float(target.mean()),
            }
        )
    return rows


def _pairing_integrity_report(pairs: pd.DataFrame) -> dict[str, Any]:
    complete_pairs = pairs[pairs["pair_status"] == "complete"].copy()
    missing_paths = complete_pairs[
        complete_pairs["brightfield_relative_path"].isna()
        | complete_pairs["darkfield_relative_path"].isna()
        | complete_pairs["brightfield_relative_path"].astype(str).str.strip().eq("")
        | complete_pairs["darkfield_relative_path"].astype(str).str.strip().eq("")
    ]

    duplicate_bf = complete_pairs["brightfield_relative_path"].astype(str)
    duplicate_df = complete_pairs["darkfield_relative_path"].astype(str)
    duplicate_bf_values = (
        duplicate_bf[duplicate_bf.duplicated(keep=False) & duplicate_bf.ne("")]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    duplicate_df_values = (
        duplicate_df[duplicate_df.duplicated(keep=False) & duplicate_df.ne("")]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    parse_issues: list[dict[str, Any]] = []
    for _, row in complete_pairs.iterrows():
        brightfield = parse_schisto_image_name(row["brightfield_relative_path"])
        darkfield = parse_schisto_image_name(row["darkfield_relative_path"])
        issues: list[str] = []
        if brightfield is None:
            issues.append("brightfield_path_not_parseable")
        if darkfield is None:
            issues.append("darkfield_path_not_parseable")
        if brightfield is not None and brightfield["contrast"] != "brightfield":
            issues.append("brightfield_path_not_brightfield")
        if darkfield is not None and darkfield["contrast"] != "darkfield":
            issues.append("darkfield_path_not_darkfield")
        if brightfield is not None and darkfield is not None:
            if brightfield["pair_key"] != darkfield["pair_key"]:
                issues.append("bf_df_pair_keys_disagree")
            if str(row["pair_key"]) != brightfield["pair_key"]:
                issues.append("row_pair_key_disagrees_with_paths")
            if str(row["patient_key"]) != brightfield["patient_key"]:
                issues.append("row_patient_key_disagrees_with_paths")
        if issues:
            parse_issues.append(
                {
                    "pair_key": str(row["pair_key"]),
                    "patient_key": str(row["patient_key"]),
                    "issues": issues,
                }
            )

    return {
        "n_complete_pairs": int(len(complete_pairs)),
        "n_complete_pairs_missing_paths": int(len(missing_paths)),
        "n_duplicate_brightfield_paths": int(len(duplicate_bf_values)),
        "n_duplicate_darkfield_paths": int(len(duplicate_df_values)),
        "n_path_parse_or_pairing_issues": int(len(parse_issues)),
        "duplicate_brightfield_path_examples": duplicate_bf_values[:10],
        "duplicate_darkfield_path_examples": duplicate_df_values[:10],
        "pairing_issue_examples": parse_issues[:10],
    }


def main() -> int:
    """Run label and split alignment diagnostics."""
    parser = build_parser()
    args = parser.parse_args()
    logger = configure_logging(quiet=args.quiet)

    _guard_output_dir(args.output_dir, overwrite=args.overwrite)

    pairs = _read_csv(args.pairs_csv)
    patients = _read_csv(args.patients_csv)
    splits = _read_csv(args.split_csv)

    validate_required_columns(
        pairs,
        [
            "pair_id",
            "pair_key",
            "study_id",
            "patient_id",
            "patient_key",
            "pair_status",
            "brightfield_relative_path",
            "darkfield_relative_path",
        ],
        table_name="pairs.csv",
    )
    validate_required_columns(
        patients,
        ["study_id", "patient_id", "patient_key", "patient_label"],
        table_name="patients.csv",
    )
    validate_required_columns(
        splits,
        ["patient_key", "split"],
        table_name="split CSV",
    )

    split_conflicts = (
        splits.groupby("patient_key")["split"]
        .nunique()
        .reset_index(name="n_splits")
    )
    split_conflicts = split_conflicts[split_conflicts["n_splits"] > 1].sort_values("patient_key").reset_index(drop=True)

    patient_level_conflicts = _find_conflicting_groups(
        pairs,
        group_column="patient_key",
        value_column="patient_level_label",
    )
    pair_label_mix = _find_conflicting_groups(
        pairs,
        group_column="patient_key",
        value_column="label",
    )

    patient_lookup = patients[["patient_key", "patient_label", "study_id", "patient_id"]].drop_duplicates("patient_key")
    pairs_with_patient = pairs.merge(
        patient_lookup,
        on="patient_key",
        how="left",
        validate="many_to_one",
        suffixes=("", "_patients"),
    )
    pair_patient_label_mismatch = pairs_with_patient[
        pairs_with_patient["patient_level_label"].map(_normalize_label).ne(
            pairs_with_patient["patient_label"].map(_normalize_label)
        )
        & pairs_with_patient["patient_level_label"].map(_normalize_label).ne("")
    ][["pair_key", "patient_key", "patient_level_label", "patient_label"]].drop_duplicates().reset_index(drop=True)

    pairs_with_split = pairs.merge(
        splits[["patient_key", "split"]].drop_duplicates("patient_key"),
        on="patient_key",
        how="left",
        validate="many_to_one",
    )
    pairs_missing_split = pairs_with_split[pairs_with_split["split"].isna()].copy()

    patients_with_split = patients.merge(
        splits[["patient_key", "split"]].drop_duplicates("patient_key"),
        on="patient_key",
        how="left",
        validate="one_to_one",
    )
    patients_missing_split = patients_with_split[patients_with_split["split"].isna()].copy()

    report = {
        "inputs": {
            "pairs_csv": str(args.pairs_csv),
            "patients_csv": str(args.patients_csv),
            "split_csv": str(args.split_csv),
        },
        "summary": {
            "n_pairs": int(len(pairs)),
            "n_patients": int(len(patients)),
            "n_split_rows": int(len(splits)),
            "n_split_conflicts": int(len(split_conflicts)),
            "n_patients_with_contradictory_patient_level_labels": int(len(patient_level_conflicts)),
            "n_patients_with_mixed_pair_labels": int(len(pair_label_mix)),
            "n_pair_to_patient_label_mismatches": int(len(pair_patient_label_mismatch)),
            "n_pairs_missing_split": int(len(pairs_missing_split)),
            "n_patients_missing_split": int(len(patients_missing_split)),
        },
        "pairing_integrity": _pairing_integrity_report(pairs),
        "class_balance_by_split": {
            "patient_level": _class_balance_by_split(patients_with_split, label_column="patient_label"),
            "pair_level": _class_balance_by_split(pairs_with_split, label_column="patient_level_label"),
        },
        "examples": {
            "split_conflicts": split_conflicts.head(10).to_dict(orient="records"),
            "contradictory_patient_level_labels": patient_level_conflicts.head(10).to_dict(orient="records"),
            "mixed_pair_labels": pair_label_mix.head(10).to_dict(orient="records"),
            "pair_patient_label_mismatches": pair_patient_label_mismatch.head(10).to_dict(orient="records"),
            "pairs_missing_split": pairs_missing_split.head(10)[["pair_key", "patient_key"]].to_dict(orient="records"),
            "patients_missing_split": patients_missing_split.head(10)[["patient_key"]].to_dict(orient="records"),
        },
    }

    lines = [
        "Label Alignment Summary",
        f"  pairs: {report['summary']['n_pairs']}",
        f"  patients: {report['summary']['n_patients']}",
        f"  split_conflicts: {report['summary']['n_split_conflicts']}",
        f"  contradictory_patient_level_labels: {report['summary']['n_patients_with_contradictory_patient_level_labels']}",
        f"  mixed_pair_labels: {report['summary']['n_patients_with_mixed_pair_labels']}",
        f"  pair_patient_label_mismatches: {report['summary']['n_pair_to_patient_label_mismatches']}",
        f"  pairs_missing_split: {report['summary']['n_pairs_missing_split']}",
        f"  patients_missing_split: {report['summary']['n_patients_missing_split']}",
        f"  complete_pairs_missing_paths: {report['pairing_integrity']['n_complete_pairs_missing_paths']}",
        f"  path_parse_or_pairing_issues: {report['pairing_integrity']['n_path_parse_or_pairing_issues']}",
    ]

    write_json(args.output_dir / "alignment_report.json", report)
    write_text(args.output_dir / "alignment_report.txt", "\n".join(lines) + "\n")

    print("\n".join(lines))
    logger.info("Saved alignment report to %s", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
