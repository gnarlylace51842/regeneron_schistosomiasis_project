"""Validation helpers for generated schistosomiasis metadata tables."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from schisto_mobile_ai.data.manifest import normalize_optional_string, validate_required_columns
from schisto_mobile_ai.utils.io import write_json


ALLOWED_CONTRASTS = {"brightfield", "darkfield"}


def _normalize_scalar(value: Any) -> str:
    text = normalize_optional_string(value)
    return text if text else ""


def _normalized_unique_values(series: pd.Series) -> list[str]:
    values = {_normalize_scalar(value) for value in series.tolist()}
    return sorted(value for value in values if value)


def _numeric_unique_values(series: pd.Series) -> list[float | int]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    unique_values = []
    for value in sorted(numeric.unique().tolist()):
        if float(value).is_integer():
            unique_values.append(int(value))
        else:
            unique_values.append(float(value))
    return unique_values


def _patient_inconsistency_rows(
    images: pd.DataFrame,
    *,
    field: str,
    numeric: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for patient_key, frame in images.groupby("patient_key", sort=True):
        values = _numeric_unique_values(frame[field]) if numeric else _normalized_unique_values(frame[field])
        if len(values) <= 1:
            continue

        row: dict[str, Any] = {
            "patient_key": patient_key,
            "patient_id": _normalize_scalar(frame["patient_id"].iloc[0]),
            "study_id": _normalize_scalar(frame["study_id"].iloc[0]),
            "values": values,
            "unique_values_count": len(values),
            "n_images": int(len(frame)),
        }
        if numeric:
            row["min"] = values[0]
            row["max"] = values[-1]
        rows.append(row)
    return rows


def build_validation_report(images_csv: str | Path) -> dict[str, Any]:
    """Validate metadata invariants expected at the patient and image level."""
    images_path = Path(images_csv)
    if not images_path.exists():
        raise FileNotFoundError(f"images.csv does not exist: {images_path}")

    images = pd.read_csv(images_path)
    validate_required_columns(
        images,
        ["patient_key", "patient_id", "study_id", "patient_eggs", "patient_level_label", "contrast"],
        table_name="images.csv",
    )

    patient_eggs_inconsistencies = _patient_inconsistency_rows(
        images,
        field="patient_eggs",
        numeric=True,
    )
    patient_level_label_inconsistencies = _patient_inconsistency_rows(
        images,
        field="patient_level_label",
        numeric=False,
    )
    study_id_inconsistencies = _patient_inconsistency_rows(
        images,
        field="study_id",
        numeric=False,
    )

    invalid_contrast_rows = images[
        ~images["contrast"].map(_normalize_scalar).isin(ALLOWED_CONTRASTS)
    ].copy()
    invalid_contrast_examples = []
    for _, row in invalid_contrast_rows.head(20).iterrows():
        invalid_contrast_examples.append(
            {
                "image_id": _normalize_scalar(row.get("image_id")),
                "image_name": _normalize_scalar(row.get("image_name")),
                "patient_key": _normalize_scalar(row.get("patient_key")),
                "contrast": _normalize_scalar(row.get("contrast")),
                "relative_path": _normalize_scalar(row.get("relative_path")),
            }
        )

    report = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "images_csv": str(images_path),
        "summary": {
            "n_images": int(len(images)),
            "n_patients": int(images["patient_key"].nunique()),
            "patient_eggs_inconsistent_patients": int(len(patient_eggs_inconsistencies)),
            "patient_level_label_inconsistent_patients": int(len(patient_level_label_inconsistencies)),
            "study_id_inconsistent_patients": int(len(study_id_inconsistencies)),
            "invalid_contrast_rows": int(len(invalid_contrast_rows)),
            "is_valid": (
                len(patient_level_label_inconsistencies) == 0
                and len(study_id_inconsistencies) == 0
                and len(invalid_contrast_rows) == 0
            ),
        },
        "checks": {
            "patient_eggs_consistency": {
                "n_inconsistent_patients": int(len(patient_eggs_inconsistencies)),
                "examples": patient_eggs_inconsistencies[:20],
            },
            "patient_level_label_consistency": {
                "n_inconsistent_patients": int(len(patient_level_label_inconsistencies)),
                "examples": patient_level_label_inconsistencies[:20],
            },
            "study_id_consistency": {
                "n_inconsistent_patients": int(len(study_id_inconsistencies)),
                "examples": study_id_inconsistencies[:20],
            },
            "contrast_allowed_values": {
                "allowed_values": sorted(ALLOWED_CONTRASTS),
                "n_invalid_rows": int(len(invalid_contrast_rows)),
                "invalid_values": sorted(
                    {
                        _normalize_scalar(value)
                        for value in invalid_contrast_rows["contrast"].tolist()
                        if _normalize_scalar(value)
                    }
                ),
                "examples": invalid_contrast_examples,
            },
        },
    }
    return report


def save_validation_report(report: dict[str, Any], output_path: str | Path) -> Path:
    """Save a validation report as JSON."""
    destination = Path(output_path)
    write_json(destination, report)
    return destination


def format_validation_summary(report: dict[str, Any], *, output_path: str | Path | None = None) -> str:
    """Create a short human-readable validation summary."""
    summary = report["summary"]
    lines = [
        "Metadata Validation Summary",
        f"  Images checked: {summary['n_images']}",
        f"  Patients checked: {summary['n_patients']}",
        f"  patient_eggs inconsistent patients: {summary['patient_eggs_inconsistent_patients']}",
        f"  patient_level_label inconsistent patients: {summary['patient_level_label_inconsistent_patients']}",
        f"  study_id inconsistent patients: {summary['study_id_inconsistent_patients']}",
        f"  invalid contrast rows: {summary['invalid_contrast_rows']}",
        f"  overall_valid: {summary['is_valid']}",
    ]
    if output_path is not None:
        lines.append(f"  report_path: {output_path}")
    return "\n".join(lines)
