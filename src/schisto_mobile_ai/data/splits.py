"""Helpers for fold assignment and fixed patient-safe split generation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import random
from typing import Any

import numpy as np
import pandas as pd

from schisto_mobile_ai.data.manifest import normalize_optional_string, validate_required_columns
from schisto_mobile_ai.utils.io import ensure_dir, write_json


UNKNOWN_VALUE = "UNKNOWN"
DEFAULT_RANDOM_SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
DEFAULT_STUDY_SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SPLIT_ORDER = ("train", "val", "test")
PATIENT_ASSIGNMENT_COLUMNS = ["patient_key", "patient_id", "study_id", "patient_label", "split"]


def assign_folds(
    frame: pd.DataFrame,
    *,
    n_splits: int = 5,
    seed: int = 42,
    label_column: str | None = None,
    group_column: str | None = None,
    fold_column: str = "fold",
) -> pd.DataFrame:
    """Assign fold IDs without hardcoding any dataset-specific columns."""
    from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold

    if len(frame) < n_splits:
        raise ValueError("Number of rows must be at least n_splits.")

    output = frame.copy()
    output[fold_column] = -1

    index_array = np.arange(len(output))
    targets = output[label_column] if label_column else None
    groups = output[group_column] if group_column else None

    if group_column and label_column:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    elif group_column:
        splitter = GroupKFold(n_splits=n_splits)
    elif label_column:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold_id, (_, valid_index) in enumerate(splitter.split(index_array, targets, groups)):
        row_index = output.index[valid_index]
        output.loc[row_index, fold_column] = fold_id

    return output


def _normalize_label(value: Any) -> str:
    text = normalize_optional_string(value)
    return text if text else UNKNOWN_VALUE


def _read_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Required CSV does not exist: {csv_path}")
    return pd.read_csv(csv_path)


def _resolve_patient_key_column(frame: pd.DataFrame) -> str:
    if "patient_key" in frame.columns:
        return "patient_key"
    if "patient_id" in frame.columns:
        return "patient_id"
    raise ValueError("A patient identifier column is required: expected 'patient_key' or 'patient_id'.")


def _resolve_patient_label(patients: pd.DataFrame, pairs: pd.DataFrame) -> pd.Series:
    if "patient_label" in patients.columns:
        patient_labels = patients["patient_label"].map(_normalize_label)
    elif "labels_observed" in patients.columns:
        patient_labels = patients["labels_observed"].map(_normalize_label)
    else:
        patient_labels = pd.Series([UNKNOWN_VALUE] * len(patients), index=patients.index)

    patient_id_column = _resolve_patient_key_column(patients)
    pair_id_column = _resolve_patient_key_column(pairs)

    if "label" in pairs.columns:
        pair_labels = (
            pairs.assign(label=pairs["label"].map(_normalize_label))
            .groupby(pair_id_column)["label"]
            .apply(lambda values: "|".join(sorted({value for value in values if value != UNKNOWN_VALUE})) or UNKNOWN_VALUE)
        )

        missing_mask = patient_labels.eq(UNKNOWN_VALUE)
        if missing_mask.any():
            mapped = patients.loc[missing_mask, patient_id_column].map(pair_labels).fillna(UNKNOWN_VALUE)
            patient_labels.loc[missing_mask] = mapped.values

    return patient_labels


def _resolve_study_id(patients: pd.DataFrame, pairs: pd.DataFrame) -> pd.Series:
    if "study_id" in patients.columns:
        patient_studies = patients["study_id"].map(_normalize_label)
    else:
        patient_studies = pd.Series([UNKNOWN_VALUE] * len(patients), index=patients.index)

    patient_id_column = _resolve_patient_key_column(patients)
    pair_id_column = _resolve_patient_key_column(pairs)

    if "study_id" in pairs.columns:
        pair_studies = (
            pairs.assign(study_id=pairs["study_id"].map(_normalize_label))
            .groupby(pair_id_column)["study_id"]
            .apply(lambda values: next((value for value in values if value != UNKNOWN_VALUE), UNKNOWN_VALUE))
        )

        missing_mask = patient_studies.eq(UNKNOWN_VALUE)
        if missing_mask.any():
            mapped = patients.loc[missing_mask, patient_id_column].map(pair_studies).fillna(UNKNOWN_VALUE)
            patient_studies.loc[missing_mask] = mapped.values

    return patient_studies


def _limit_patients(
    patients: pd.DataFrame,
    pairs: pd.DataFrame,
    *,
    subset_size: int | None,
    smoke_test: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    effective_subset = subset_size
    if smoke_test and effective_subset is None:
        effective_subset = 30

    if effective_subset is None:
        return patients, pairs

    if effective_subset <= 0:
        raise ValueError("--subset-size must be a positive integer when provided.")

    patient_id_column = _resolve_patient_key_column(patients)
    pair_id_column = _resolve_patient_key_column(pairs)
    limited_patients = patients.sort_values([patient_id_column, "study_id"]).head(effective_subset).reset_index(drop=True)
    allowed_patients = set(limited_patients[patient_id_column])
    limited_pairs = pairs[pairs[pair_id_column].isin(allowed_patients)].copy().reset_index(drop=True)
    return limited_patients, limited_pairs


def load_split_inputs(
    patients_csv: str | Path,
    pairs_csv: str | Path,
    *,
    subset_size: int | None = None,
    smoke_test: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and normalize patient and pair metadata for split generation."""
    patients = _read_csv(patients_csv)
    pairs = _read_csv(pairs_csv)

    validate_required_columns(patients, ["patient_id"], table_name="patients.csv")
    validate_required_columns(pairs, ["patient_id"], table_name="pairs.csv")

    patients = patients.copy()
    pairs = pairs.copy()
    patients["patient_id"] = patients["patient_id"].map(_normalize_label)
    pairs["patient_id"] = pairs["patient_id"].map(_normalize_label)
    if "patient_key" not in patients.columns:
        if "study_id" in patients.columns:
            patients["patient_key"] = (
                patients["study_id"].map(_normalize_label) + "_" + patients["patient_id"].map(_normalize_label)
            )
        else:
            patients["patient_key"] = patients["patient_id"]
    else:
        patients["patient_key"] = patients["patient_key"].map(_normalize_label)

    if "patient_key" not in pairs.columns:
        if "study_id" in pairs.columns:
            pairs["patient_key"] = (
                pairs["study_id"].map(_normalize_label) + "_" + pairs["patient_id"].map(_normalize_label)
            )
        else:
            pairs["patient_key"] = pairs["patient_id"]
    else:
        pairs["patient_key"] = pairs["patient_key"].map(_normalize_label)

    if patients["patient_key"].duplicated().any():
        duplicates = patients.loc[patients["patient_key"].duplicated(), "patient_key"].tolist()
        duplicate_text = ", ".join(sorted(set(duplicates[:10])))
        raise ValueError(f"patients.csv must contain one row per patient_key. Duplicates found: {duplicate_text}")

    patients["study_id"] = _resolve_study_id(patients, pairs)
    patients["patient_label"] = _resolve_patient_label(patients, pairs)

    if "label" not in pairs.columns:
        pairs["label"] = UNKNOWN_VALUE
    pairs["label"] = pairs["label"].map(_normalize_label)

    if "study_id" not in pairs.columns:
        pairs["study_id"] = pairs["patient_key"].map(
            patients.set_index("patient_key")["study_id"].to_dict()
        ).fillna(UNKNOWN_VALUE)
    else:
        pairs["study_id"] = pairs["study_id"].map(_normalize_label)

    for column in ["brightfield_relative_path", "darkfield_relative_path", "pair_status"]:
        if column not in pairs.columns:
            pairs[column] = ""

    patients, pairs = _limit_patients(patients, pairs, subset_size=subset_size, smoke_test=smoke_test)

    pair_patient_ids = set(pairs["patient_key"])
    patient_ids = set(patients["patient_key"])
    missing_in_patients = sorted(pair_patient_ids - patient_ids)
    if missing_in_patients:
        preview = ", ".join(missing_in_patients[:10])
        raise ValueError(
            "pairs.csv contains patient identifiers that are missing from patients.csv: "
            f"{preview}"
        )

    return patients.reset_index(drop=True), pairs.reset_index(drop=True)


def _compute_split_counts(
    total_count: int,
    *,
    ratios: dict[str, float],
    require_val: bool,
) -> dict[str, int]:
    if total_count <= 0:
        raise ValueError("Cannot split an empty patient table.")

    if total_count == 1:
        return {"train": 1, "val": 0, "test": 0}
    if total_count == 2:
        return {"train": 1, "val": 0, "test": 1}

    test_count = max(1, int(round(total_count * ratios["test"])))
    val_count = max(1, int(round(total_count * ratios["val"]))) if require_val else 0
    train_count = total_count - test_count - val_count

    while train_count < 1 and val_count > 0:
        val_count -= 1
        train_count = total_count - test_count - val_count

    while train_count < 1 and test_count > 1:
        test_count -= 1
        train_count = total_count - test_count - val_count

    if train_count < 1:
        raise ValueError("Could not allocate at least one training patient.")

    return {"train": train_count, "val": val_count, "test": test_count}


def _stratify_labels(labels: pd.Series) -> pd.Series | None:
    counts = labels.value_counts()
    if len(counts) < 2:
        return None
    if counts.min() < 2:
        return None
    return labels


def _shuffle_assignments(
    patients: pd.DataFrame,
    *,
    counts: dict[str, int],
    seed: int,
) -> pd.DataFrame:
    shuffled = patients.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    train_end = counts["train"]
    val_end = counts["train"] + counts["val"]

    assignments = shuffled.copy()
    assignments["split"] = "test"
    assignments.loc[: train_end - 1, "split"] = "train"
    if counts["val"] > 0:
        assignments.loc[train_end: val_end - 1, "split"] = "val"
    return assignments


def create_random_patient_split(
    patients: pd.DataFrame,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a reproducible patient-level random split, using stratification when feasible."""
    patients = patients.copy().reset_index(drop=True)
    counts = _compute_split_counts(
        len(patients),
        ratios=DEFAULT_RANDOM_SPLIT_RATIOS,
        require_val=len(patients) >= 3,
    )

    stratify_labels = _stratify_labels(patients["patient_label"])
    if counts["test"] == 0:
        test_frame = patients.iloc[0:0].copy()
        train_val_frame = patients.copy()
    else:
        try:
            from sklearn.model_selection import train_test_split

            train_val_frame, test_frame = train_test_split(
                patients,
                test_size=counts["test"],
                random_state=seed,
                stratify=stratify_labels if stratify_labels is not None else None,
            )
        except (ImportError, ModuleNotFoundError, ValueError):
            return _shuffle_assignments(patients, counts=counts, seed=seed)[PATIENT_ASSIGNMENT_COLUMNS]

    if counts["val"] == 0:
        train_frame = train_val_frame.copy()
        val_frame = train_val_frame.iloc[0:0].copy()
    else:
        second_stratify = _stratify_labels(train_val_frame["patient_label"])
        try:
            from sklearn.model_selection import train_test_split

            train_frame, val_frame = train_test_split(
                train_val_frame,
                test_size=counts["val"],
                random_state=seed + 1,
                stratify=second_stratify if second_stratify is not None else None,
            )
        except (ImportError, ModuleNotFoundError, ValueError):
            return _shuffle_assignments(patients, counts=counts, seed=seed)[PATIENT_ASSIGNMENT_COLUMNS]

    assignments = pd.concat(
        [
            train_frame.assign(split="train"),
            val_frame.assign(split="val"),
            test_frame.assign(split="test"),
        ],
        ignore_index=True,
    )
    assignments = assignments[PATIENT_ASSIGNMENT_COLUMNS].sort_values(["study_id", "patient_id", "patient_key"]).reset_index(drop=True)
    return assignments


def _study_split_requirements(study_count: int) -> list[str]:
    if study_count < 2:
        raise ValueError(
            "study_holdout_split requires at least 2 distinct non-empty study_id values in metadata/patients.csv."
        )
    if study_count == 2:
        return ["train", "test"]
    return ["train", "val", "test"]


def create_study_holdout_split(
    patients: pd.DataFrame,
    *,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a reproducible study-holdout split with no study leakage."""
    if "study_id" not in patients.columns:
        raise ValueError(
            "study_holdout_split requires a 'study_id' column in metadata/patients.csv."
        )

    normalized = patients.copy()
    normalized["study_id"] = normalized["study_id"].map(_normalize_label)
    if normalized["study_id"].eq(UNKNOWN_VALUE).any():
        raise ValueError(
            "study_holdout_split requires every patient row to have a non-empty 'study_id' value after combining metadata/patients.csv and metadata/pairs.csv."
        )

    known_studies = sorted(study for study in normalized["study_id"].unique() if study != UNKNOWN_VALUE)
    if not known_studies:
        raise ValueError(
            "study_holdout_split requires a non-empty 'study_id' field in metadata/patients.csv."
        )

    required_splits = _study_split_requirements(len(known_studies))
    counts = _compute_split_counts(
        len(normalized),
        ratios=DEFAULT_STUDY_SPLIT_RATIOS,
        require_val="val" in required_splits,
    )

    rng = random.Random(seed)
    shuffled_studies = known_studies[:]
    rng.shuffle(shuffled_studies)

    study_sizes = normalized.groupby("study_id").size().to_dict()
    shuffled_order = {study_id: position for position, study_id in enumerate(shuffled_studies)}
    ordered_studies = sorted(
        shuffled_studies,
        key=lambda study_id: (-study_sizes[study_id], shuffled_order[study_id]),
    )

    split_to_studies = {split: [] for split in required_splits}
    split_to_count = {split: 0 for split in required_splits}
    target_counts = {split: counts[split] for split in required_splits}

    seeded_splits = sorted(required_splits, key=lambda split: (-target_counts[split], split))
    remaining_studies = ordered_studies[:]
    for split in seeded_splits:
        study_id = remaining_studies.pop(0)
        split_to_studies[split].append(study_id)
        split_to_count[split] += study_sizes[study_id]

    for study_id in remaining_studies:
        chosen_split = max(
            required_splits,
            key=lambda split: (target_counts[split] - split_to_count[split], -split_to_count[split]),
        )
        split_to_studies[chosen_split].append(study_id)
        split_to_count[chosen_split] += study_sizes[study_id]

    study_to_split = {
        study_id: split
        for split, studies in split_to_studies.items()
        for study_id in studies
    }

    assignments = normalized.copy()
    assignments["split"] = assignments["study_id"].map(study_to_split)
    assignments = assignments[PATIENT_ASSIGNMENT_COLUMNS].sort_values(["study_id", "patient_id", "patient_key"]).reset_index(drop=True)
    return assignments


def validate_no_patient_overlap(assignments: pd.DataFrame) -> dict[str, Any]:
    """Validate that each patient appears exactly once across splits."""
    if assignments["split"].isna().any() or assignments["split"].map(normalize_optional_string).eq("").any():
        missing_patients = assignments.loc[
            assignments["split"].isna() | assignments["split"].map(normalize_optional_string).eq(""),
            "patient_key",
        ].tolist()
        missing_text = ", ".join(sorted(set(missing_patients[:10])))
        raise ValueError(f"Some patients were not assigned to a split: {missing_text}")

    if assignments["patient_key"].duplicated().any():
        duplicates = assignments.loc[assignments["patient_key"].duplicated(), "patient_key"].tolist()
        duplicate_text = ", ".join(sorted(set(duplicates[:10])))
        raise ValueError(f"Patients appear in more than one split: {duplicate_text}")

    return {
        "is_valid": True,
        "n_unique_patients": int(assignments["patient_key"].nunique()),
    }


def validate_study_holdout(assignments: pd.DataFrame) -> dict[str, Any]:
    """Validate that each study is assigned to only one split."""
    if "study_id" not in assignments.columns:
        raise ValueError("study_holdout_split validation requires a study_id column.")

    normalized = assignments.copy()
    normalized["study_id"] = normalized["study_id"].map(_normalize_label)
    if normalized["study_id"].eq(UNKNOWN_VALUE).all():
        raise ValueError(
            "study_holdout_split requires a non-empty 'study_id' field that can be resolved from metadata/patients.csv or metadata/pairs.csv."
        )

    overlapping = (
        normalized.groupby("study_id")["split"].nunique().reset_index(name="n_splits")
    )
    bad_rows = overlapping[overlapping["n_splits"] > 1]
    if not bad_rows.empty:
        bad_studies = ", ".join(sorted(bad_rows["study_id"].tolist()))
        raise ValueError(f"study_holdout_split leaked studies across splits: {bad_studies}")

    study_to_split = (
        normalized.groupby("study_id")["split"]
        .first()
        .sort_index()
        .to_dict()
    )
    return {
        "is_valid": True,
        "study_to_split": study_to_split,
    }


def _pair_presence_counts(split_pairs: pd.DataFrame) -> dict[str, int]:
    bf_present = split_pairs["brightfield_relative_path"].map(lambda value: int(bool(normalize_optional_string(value)))).sum()
    df_present = split_pairs["darkfield_relative_path"].map(lambda value: int(bool(normalize_optional_string(value)))).sum()
    complete_pairs = int(split_pairs["pair_status"].eq("complete").sum()) if "pair_status" in split_pairs.columns else 0
    return {
        "n_pairs": int(len(split_pairs)),
        "n_pairs_with_brightfield": int(bf_present),
        "n_pairs_with_darkfield": int(df_present),
        "n_complete_pairs": complete_pairs,
    }


def validate_contrast_balance(
    assignments: pd.DataFrame,
    pairs: pd.DataFrame,
    *,
    tolerance: float = 0.20,
) -> dict[str, Any]:
    """Check whether BF/DF availability looks roughly similar across splits."""
    merged = pairs.merge(
        assignments[["patient_key", "split"]],
        on="patient_key",
        how="left",
        validate="many_to_one",
    )

    if merged["split"].isna().any():
        missing_patients = merged.loc[merged["split"].isna(), "patient_key"].unique().tolist()
        missing_text = ", ".join(sorted(missing_patients[:10]))
        raise ValueError(f"Pairs reference patients missing from split assignments: {missing_text}")

    overall_counts = _pair_presence_counts(merged)
    overall_pairs = max(overall_counts["n_pairs"], 1)
    overall_metrics = {
        "brightfield_rate": overall_counts["n_pairs_with_brightfield"] / overall_pairs,
        "darkfield_rate": overall_counts["n_pairs_with_darkfield"] / overall_pairs,
        "complete_pair_rate": overall_counts["n_complete_pairs"] / overall_pairs,
    }

    per_split: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    is_balanced = True

    for split_name in SPLIT_ORDER:
        split_pairs = merged[merged["split"] == split_name].copy()
        counts = _pair_presence_counts(split_pairs)
        split_total = counts["n_pairs"]
        if split_total == 0:
            metrics = {
                "brightfield_rate": None,
                "darkfield_rate": None,
                "complete_pair_rate": None,
            }
        else:
            metrics = {
                "brightfield_rate": counts["n_pairs_with_brightfield"] / split_total,
                "darkfield_rate": counts["n_pairs_with_darkfield"] / split_total,
                "complete_pair_rate": counts["n_complete_pairs"] / split_total,
            }

        deltas = {}
        for metric_name, overall_value in overall_metrics.items():
            split_value = metrics[metric_name]
            deltas[metric_name] = None if split_value is None else abs(split_value - overall_value)

        if split_total > 0:
            bad_metrics = [
                metric_name
                for metric_name, delta in deltas.items()
                if delta is not None and delta > tolerance
            ]
            if bad_metrics:
                is_balanced = False
                warnings.append(
                    f"{split_name} differs from the overall contrast profile for: {', '.join(bad_metrics)}"
                )

        per_split[split_name] = {
            **counts,
            **metrics,
            "delta_from_overall": deltas,
        }

    return {
        "is_balanced": is_balanced,
        "tolerance": tolerance,
        "overall": {
            **overall_counts,
            **overall_metrics,
        },
        "per_split": per_split,
        "warnings": warnings,
    }


def summarize_splits(assignments: pd.DataFrame, pairs: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Summarize patient counts, label balance, and pair counts for each split."""
    merged_pairs = pairs.merge(
        assignments[["patient_key", "split"]],
        on="patient_key",
        how="left",
        validate="many_to_one",
    )

    summary: dict[str, dict[str, Any]] = {}
    for split_name in SPLIT_ORDER:
        split_patients = assignments[assignments["split"] == split_name].copy()
        split_pairs = merged_pairs[merged_pairs["split"] == split_name].copy()

        patient_label_counts = (
            split_patients["patient_label"].map(_normalize_label).value_counts().sort_index().to_dict()
        )
        pair_label_counts = (
            split_pairs["label"].map(_normalize_label).value_counts().sort_index().to_dict()
            if not split_pairs.empty
            else {}
        )
        studies = sorted(
            study_id
            for study_id in split_patients["study_id"].map(_normalize_label).unique().tolist()
            if study_id != UNKNOWN_VALUE
        )

        summary[split_name] = {
            "n_patients": int(len(split_patients)),
            "n_studies": int(len(studies)),
            "studies": studies,
            "patient_label_counts": {str(key): int(value) for key, value in patient_label_counts.items()},
            "pair_label_counts": {str(key): int(value) for key, value in pair_label_counts.items()},
            **_pair_presence_counts(split_pairs),
        }

    return summary


def format_split_summary(
    split_name: str,
    summary: dict[str, dict[str, Any]],
    contrast_validation: dict[str, Any],
) -> str:
    """Build a readable terminal summary for a split definition."""
    lines = [split_name]
    for split_key in SPLIT_ORDER:
        split_summary = summary[split_key]
        label_text = ", ".join(
            f"{label}={count}" for label, count in split_summary["pair_label_counts"].items()
        ) or "none"
        patient_label_text = ", ".join(
            f"{label}={count}" for label, count in split_summary["patient_label_counts"].items()
        ) or "none"
        lines.extend(
            [
                f"  {split_key}: patients={split_summary['n_patients']}, studies={split_summary['n_studies']}, pairs={split_summary['n_pairs']}",
                f"    patient_labels: {patient_label_text}",
                f"    pair_labels: {label_text}",
                f"    contrast_presence: BF={split_summary['n_pairs_with_brightfield']}, DF={split_summary['n_pairs_with_darkfield']}, complete={split_summary['n_complete_pairs']}",
            ]
        )

    lines.append(
        f"  contrast_balance_ok: {contrast_validation['is_balanced']} (tolerance={contrast_validation['tolerance']})"
    )
    for warning in contrast_validation["warnings"]:
        lines.append(f"    warning: {warning}")
    return "\n".join(lines)


def build_split_payload(
    *,
    split_name: str,
    assignments: pd.DataFrame,
    pairs: pd.DataFrame,
    seed: int,
    patients_csv: str | Path,
    pairs_csv: str | Path,
    summary: dict[str, dict[str, Any]],
    patient_validation: dict[str, Any],
    contrast_validation: dict[str, Any],
    study_validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create the JSON payload saved for a split definition."""
    return {
        "split_name": split_name,
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "seed": seed,
        "source_files": {
            "patients_csv": str(Path(patients_csv)),
            "pairs_csv": str(Path(pairs_csv)),
        },
        "summary": summary,
        "validation": {
            "patient_overlap": patient_validation,
            "study_holdout": study_validation,
            "contrast_balance": contrast_validation,
        },
        "assignments": assignments[PATIENT_ASSIGNMENT_COLUMNS].sort_values(["study_id", "patient_id", "patient_key"]).to_dict(orient="records"),
    }


def save_split_artifacts(
    *,
    output_dir: str | Path,
    split_name: str,
    assignments: pd.DataFrame,
    payload: dict[str, Any],
) -> tuple[Path, Path]:
    """Save the JSON and CSV artifacts for a split definition."""
    destination = ensure_dir(output_dir)
    json_path = destination / f"{split_name}.json"
    csv_path = destination / f"{split_name}.csv"
    write_json(json_path, payload)
    assignments[PATIENT_ASSIGNMENT_COLUMNS].sort_values(["study_id", "patient_id", "patient_key"]).to_csv(csv_path, index=False)
    return json_path, csv_path
