"""Shared dataset-audit and metadata-indexing pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from schisto_mobile_ai.data.discovery import (
    build_image_match_keys,
    classify_file,
    detect_table_columns,
    infer_contrast_from_text,
    infer_pair_key,
    infer_path_identifier,
    iter_dataset_files,
    rank_image_reference_columns,
    stable_id,
    try_load_metadata_table,
)
from schisto_mobile_ai.data.manifest import normalize_optional_string
from schisto_mobile_ai.utils.io import ensure_dir, write_json


UNKNOWN_VALUE = "UNKNOWN"

IMAGE_COLUMNS = [
    "image_id",
    "relative_path",
    "file_name",
    "file_stem",
    "parent_dir",
    "extension",
    "size_bytes",
    "contrast",
    "contrast_source",
    "contrast_raw",
    "study_id",
    "study_id_source",
    "patient_id",
    "patient_id_source",
    "label",
    "label_source",
    "metadata_pair_id",
    "metadata_pair_id_source",
    "pair_key",
    "pair_key_source",
    "metadata_source_file",
    "metadata_match_column",
    "metadata_match_status",
]

PATIENT_COLUMNS = [
    "study_id",
    "patient_id",
    "n_images",
    "n_brightfield",
    "n_darkfield",
    "n_unknown_contrast",
    "n_complete_pairs",
    "n_missing_brightfield",
    "n_missing_darkfield",
    "labels_observed",
]

PAIR_COLUMNS = [
    "pair_id",
    "pair_key",
    "study_id",
    "patient_id",
    "pair_status",
    "label",
    "brightfield_image_id",
    "brightfield_relative_path",
    "darkfield_image_id",
    "darkfield_relative_path",
]

METADATA_FILE_COLUMNS = [
    "relative_path",
    "extension",
    "parse_success",
    "parse_error",
    "n_rows",
    "n_columns",
    "image_columns",
    "patient_columns",
    "study_columns",
    "label_columns",
    "pair_columns",
    "contrast_columns",
    "matched_rows",
    "unmatched_rows",
    "ambiguous_rows",
]

SUSPICIOUS_COLUMNS = [
    "path",
    "record_type",
    "issue",
    "details",
]


@dataclass
class DatasetAuditResult:
    """Container for dataframes and report objects produced by the audit pipeline."""

    images: pd.DataFrame
    patients: pd.DataFrame
    pairs: pd.DataFrame
    metadata_files: pd.DataFrame
    suspicious_files: pd.DataFrame
    missing_pairs: pd.DataFrame
    report: dict[str, Any]


def _empty_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _safe_value(value: str) -> str:
    return normalize_optional_string(value) or UNKNOWN_VALUE


def _add_suspicious(
    records: list[dict[str, str]],
    *,
    path: str,
    record_type: str,
    issue: str,
    details: str,
) -> None:
    records.append(
        {
            "path": path,
            "record_type": record_type,
            "issue": issue,
            "details": details,
        }
    )


def _make_image_record(raw_dir: Path, path: Path) -> dict[str, Any]:
    relative_path = path.relative_to(raw_dir).as_posix()
    contrast, _, _ = infer_contrast_from_text(relative_path)
    patient_id, patient_source = infer_path_identifier(relative_path, kind="patient")
    study_id, study_source = infer_path_identifier(relative_path, kind="study")

    return {
        "image_id": stable_id("img", relative_path),
        "relative_path": relative_path,
        "file_name": path.name,
        "file_stem": path.stem,
        "parent_dir": path.parent.relative_to(raw_dir).as_posix() if path.parent != raw_dir else ".",
        "extension": path.suffix.lower(),
        "size_bytes": int(path.stat().st_size),
        "contrast": contrast,
        "contrast_source": "path" if contrast != "unknown" else "unknown",
        "contrast_raw": "",
        "study_id": study_id,
        "study_id_source": study_source,
        "patient_id": patient_id,
        "patient_id_source": patient_source,
        "label": "",
        "label_source": "unknown",
        "metadata_pair_id": "",
        "metadata_pair_id_source": "unknown",
        "pair_key": infer_pair_key(relative_path),
        "pair_key_source": "path",
        "metadata_source_file": "",
        "metadata_match_column": "",
        "metadata_match_status": "unmatched",
    }


def _build_image_lookup(image_records: list[dict[str, Any]]) -> dict[str, list[int]]:
    lookup: dict[str, list[int]] = {}
    for index, record in enumerate(image_records):
        for key in build_image_match_keys(record["relative_path"]):
            lookup.setdefault(key, []).append(index)
    return lookup


def _extract_first_value(row: pd.Series, columns: list[str]) -> tuple[str, str]:
    for column in columns:
        value = normalize_optional_string(row.get(column))
        if value:
            return value, column
    return "", ""


def _merge_field(
    record: dict[str, Any],
    *,
    field: str,
    source_field: str,
    new_value: str,
    new_source: str,
    suspicious_records: list[dict[str, str]],
) -> None:
    cleaned_value = normalize_optional_string(new_value)
    if not cleaned_value:
        return

    current_value = normalize_optional_string(record.get(field))
    current_source = normalize_optional_string(record.get(source_field))

    if not current_value:
        record[field] = cleaned_value
        record[source_field] = new_source
        return

    if current_value == cleaned_value:
        return

    if current_source.startswith("path:") and new_source.startswith("metadata:"):
        record[field] = cleaned_value
        record[source_field] = new_source
        return

    _add_suspicious(
        suspicious_records,
        path=record["relative_path"],
        record_type="image",
        issue=f"conflicting_{field}",
        details=f"Existing value '{current_value}' disagrees with '{cleaned_value}'.",
    )


def _merge_contrast(
    record: dict[str, Any],
    *,
    contrast_value: str,
    source: str,
    suspicious_records: list[dict[str, str]],
) -> None:
    cleaned_value = normalize_optional_string(contrast_value)
    if not cleaned_value:
        return

    detected_contrast, _, _ = infer_contrast_from_text(cleaned_value)
    record["contrast_raw"] = cleaned_value

    if detected_contrast == "unknown":
        return

    _merge_field(
        record,
        field="contrast",
        source_field="contrast_source",
        new_value=detected_contrast,
        new_source=source,
        suspicious_records=suspicious_records,
    )


def _match_row_to_image(
    row: pd.Series,
    image_columns: list[str],
    image_lookup: dict[str, list[int]],
) -> tuple[list[int], str, str]:
    ambiguous_match: tuple[list[int], str, str] = ([], "", "")

    for column in image_columns:
        value = normalize_optional_string(row.get(column))
        if not value:
            continue

        for key in build_image_match_keys(value):
            matches = image_lookup.get(key, [])
            if len(matches) == 1:
                return matches, column, value
            if len(matches) > 1 and not ambiguous_match[0]:
                ambiguous_match = (matches, column, value)

    return ambiguous_match


def _finalize_images(
    image_records: list[dict[str, Any]],
    suspicious_records: list[dict[str, str]],
) -> pd.DataFrame:
    for record in image_records:
        if record["size_bytes"] <= 0:
            _add_suspicious(
                suspicious_records,
                path=record["relative_path"],
                record_type="image",
                issue="empty_file",
                details="Image file has size 0 bytes.",
            )

        if record["contrast"] == "unknown":
            _add_suspicious(
                suspicious_records,
                path=record["relative_path"],
                record_type="image",
                issue="unknown_contrast",
                details="Could not infer brightfield or darkfield from path or metadata.",
            )

        record["study_id"] = _safe_value(record["study_id"])
        record["patient_id"] = _safe_value(record["patient_id"])
        record["label"] = _safe_value(record["label"])
        record["metadata_pair_id"] = normalize_optional_string(record["metadata_pair_id"])

    images = pd.DataFrame(image_records)
    if images.empty:
        return _empty_frame(IMAGE_COLUMNS)
    return images[IMAGE_COLUMNS].sort_values("relative_path").reset_index(drop=True)


def _build_pairs(
    images: pd.DataFrame,
    suspicious_records: list[dict[str, str]],
) -> pd.DataFrame:
    if images.empty:
        return _empty_frame(PAIR_COLUMNS)

    pair_records: list[dict[str, Any]] = []

    grouped = images.groupby("pair_key", dropna=False, sort=True)
    for pair_key, group in grouped:
        brightfield = group[group["contrast"] == "brightfield"].sort_values("relative_path")
        darkfield = group[group["contrast"] == "darkfield"].sort_values("relative_path")
        unknown = group[~group["contrast"].isin(["brightfield", "darkfield"])]

        if len(unknown) > 0:
            for _, row in unknown.iterrows():
                _add_suspicious(
                    suspicious_records,
                    path=row["relative_path"],
                    record_type="image",
                    issue="unpaired_unknown_contrast",
                    details=f"Image belongs to pair key '{pair_key}' but its contrast is unknown.",
                )

        if len(brightfield) > 1:
            _add_suspicious(
                suspicious_records,
                path=pair_key,
                record_type="pair",
                issue="multiple_brightfield_candidates",
                details=f"Found {len(brightfield)} brightfield images for the same pair key.",
            )
        if len(darkfield) > 1:
            _add_suspicious(
                suspicious_records,
                path=pair_key,
                record_type="pair",
                issue="multiple_darkfield_candidates",
                details=f"Found {len(darkfield)} darkfield images for the same pair key.",
            )

        pair_count = max(len(brightfield), len(darkfield))
        if pair_count == 0:
            continue

        brightfield_rows = list(brightfield.to_dict(orient="records"))
        darkfield_rows = list(darkfield.to_dict(orient="records"))

        for index in range(pair_count):
            bf_row = brightfield_rows[index] if index < len(brightfield_rows) else None
            df_row = darkfield_rows[index] if index < len(darkfield_rows) else None

            if bf_row and df_row:
                pair_status = "complete"
            elif bf_row:
                pair_status = "missing_darkfield"
            else:
                pair_status = "missing_brightfield"

            label_values = [
                row["label"]
                for row in [bf_row, df_row]
                if row is not None and row["label"] != UNKNOWN_VALUE
            ]
            label = "|".join(sorted(set(label_values))) if label_values else UNKNOWN_VALUE

            anchor_row = bf_row or df_row
            pair_id = stable_id("pair", f"{pair_key}:{index}")
            pair_records.append(
                {
                    "pair_id": pair_id,
                    "pair_key": pair_key,
                    "study_id": anchor_row["study_id"] if anchor_row else UNKNOWN_VALUE,
                    "patient_id": anchor_row["patient_id"] if anchor_row else UNKNOWN_VALUE,
                    "pair_status": pair_status,
                    "label": label,
                    "brightfield_image_id": bf_row["image_id"] if bf_row else "",
                    "brightfield_relative_path": bf_row["relative_path"] if bf_row else "",
                    "darkfield_image_id": df_row["image_id"] if df_row else "",
                    "darkfield_relative_path": df_row["relative_path"] if df_row else "",
                }
            )

    if not pair_records:
        return _empty_frame(PAIR_COLUMNS)

    return pd.DataFrame(pair_records)[PAIR_COLUMNS].sort_values("pair_key").reset_index(drop=True)


def _build_patients(images: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    if images.empty:
        return _empty_frame(PATIENT_COLUMNS)

    patient_rows: list[dict[str, Any]] = []
    pair_groups: dict[tuple[str, str], pd.DataFrame] = {}
    if not pairs.empty:
        pair_groups = {
            (study_id, patient_id): frame.copy()
            for (study_id, patient_id), frame in pairs.groupby(["study_id", "patient_id"], dropna=False)
        }

    for (study_id, patient_id), frame in images.groupby(["study_id", "patient_id"], dropna=False, sort=True):
        labels = sorted({label for label in frame["label"] if label != UNKNOWN_VALUE})
        pair_frame = pair_groups.get((study_id, patient_id), _empty_frame(PAIR_COLUMNS))

        patient_rows.append(
            {
                "study_id": study_id,
                "patient_id": patient_id,
                "n_images": int(len(frame)),
                "n_brightfield": int((frame["contrast"] == "brightfield").sum()),
                "n_darkfield": int((frame["contrast"] == "darkfield").sum()),
                "n_unknown_contrast": int((frame["contrast"] == "unknown").sum()),
                "n_complete_pairs": int((pair_frame["pair_status"] == "complete").sum())
                if not pair_frame.empty
                else 0,
                "n_missing_brightfield": int((pair_frame["pair_status"] == "missing_brightfield").sum())
                if not pair_frame.empty
                else 0,
                "n_missing_darkfield": int((pair_frame["pair_status"] == "missing_darkfield").sum())
                if not pair_frame.empty
                else 0,
                "labels_observed": "|".join(labels) if labels else UNKNOWN_VALUE,
            }
        )

    return pd.DataFrame(patient_rows)[PATIENT_COLUMNS].reset_index(drop=True)


def _value_counts(series: pd.Series) -> dict[str, int]:
    counts = series.fillna(UNKNOWN_VALUE).astype(str).value_counts(dropna=False)
    return {str(key): int(value) for key, value in counts.items()}


def _build_report(
    *,
    raw_dir: Path,
    total_files: int,
    image_files: int,
    metadata_files: pd.DataFrame,
    images: pd.DataFrame,
    patients: pd.DataFrame,
    pairs: pd.DataFrame,
    suspicious_files: pd.DataFrame,
    subset_size: int | None,
    smoke_test: bool,
    metadata_row_limit: int | None,
) -> dict[str, Any]:
    other_files = total_files - image_files - len(metadata_files)
    report = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "raw_dir": str(raw_dir),
        "smoke_test": bool(smoke_test),
        "subset_size": int(subset_size) if subset_size is not None else None,
        "metadata_row_limit": int(metadata_row_limit) if metadata_row_limit is not None else None,
        "summary": {
            "total_files_scanned": int(total_files),
            "image_files_found": int(image_files),
            "metadata_files_found": int(len(metadata_files)),
            "other_files_found": int(other_files),
            "parsed_metadata_files": int(metadata_files["parse_success"].fillna(False).sum())
            if not metadata_files.empty
            else 0,
            "images_indexed": int(len(images)),
            "patients_indexed": int(len(patients)),
            "pairs_indexed": int(len(pairs)),
            "complete_pairs": int((pairs["pair_status"] == "complete").sum()) if not pairs.empty else 0,
            "missing_brightfield_pairs": int((pairs["pair_status"] == "missing_brightfield").sum())
            if not pairs.empty
            else 0,
            "missing_darkfield_pairs": int((pairs["pair_status"] == "missing_darkfield").sum())
            if not pairs.empty
            else 0,
            "suspicious_items": int(len(suspicious_files)),
        },
        "counts": {
            "by_study": _value_counts(images["study_id"]) if not images.empty else {},
            "by_contrast": _value_counts(images["contrast"]) if not images.empty else {},
            "by_patient": _value_counts(images["patient_id"]) if not images.empty else {},
            "by_label": _value_counts(images["label"]) if not images.empty else {},
        },
        "metadata_files": metadata_files.to_dict(orient="records"),
        "suspicious_files": suspicious_files.to_dict(orient="records"),
    }
    return report


def analyze_dataset(
    raw_dir: str | Path,
    *,
    subset_size: int | None = None,
    smoke_test: bool = False,
    metadata_row_limit: int | None = None,
) -> DatasetAuditResult:
    """Audit a dataset directory and build image, patient, and pair indices."""
    root = Path(raw_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    effective_subset = subset_size
    if smoke_test and effective_subset is None:
        effective_subset = 50
    if smoke_test and metadata_row_limit is None:
        metadata_row_limit = 500

    all_files = iter_dataset_files(root)
    all_image_paths = [path for path in all_files if classify_file(path) == "image"]
    metadata_paths = [path for path in all_files if classify_file(path) == "metadata"]
    image_paths = all_image_paths

    if effective_subset is not None:
        image_paths = image_paths[:effective_subset]

    suspicious_records: list[dict[str, str]] = []
    image_records = [_make_image_record(root, path) for path in image_paths]
    image_lookup = _build_image_lookup(image_records)

    metadata_records: list[dict[str, Any]] = []
    for metadata_path in metadata_paths:
        relative_path = metadata_path.relative_to(root).as_posix()
        frame, parse_error = try_load_metadata_table(metadata_path, max_rows=metadata_row_limit)

        metadata_record = {
            "relative_path": relative_path,
            "extension": metadata_path.suffix.lower(),
            "parse_success": frame is not None,
            "parse_error": parse_error,
            "n_rows": int(len(frame)) if frame is not None else 0,
            "n_columns": int(len(frame.columns)) if frame is not None else 0,
            "image_columns": "",
            "patient_columns": "",
            "study_columns": "",
            "label_columns": "",
            "pair_columns": "",
            "contrast_columns": "",
            "matched_rows": 0,
            "unmatched_rows": 0,
            "ambiguous_rows": 0,
        }

        if frame is None:
            _add_suspicious(
                suspicious_records,
                path=relative_path,
                record_type="metadata",
                issue="metadata_parse_failed",
                details=parse_error,
            )
            metadata_records.append(metadata_record)
            continue

        detected_columns = detect_table_columns(frame)
        image_columns = rank_image_reference_columns(frame, detected_columns["image"], image_lookup)
        patient_columns = detected_columns["patient"][:3]
        study_columns = detected_columns["study"][:3]
        label_columns = detected_columns["label"][:3]
        pair_columns = detected_columns["pair"][:3]
        contrast_columns = detected_columns["contrast"][:3]

        metadata_record["image_columns"] = "|".join(image_columns)
        metadata_record["patient_columns"] = "|".join(patient_columns)
        metadata_record["study_columns"] = "|".join(study_columns)
        metadata_record["label_columns"] = "|".join(label_columns)
        metadata_record["pair_columns"] = "|".join(pair_columns)
        metadata_record["contrast_columns"] = "|".join(contrast_columns)

        if not image_columns:
            metadata_records.append(metadata_record)
            continue

        for _, row in frame.iterrows():
            match_indices, match_column, match_value = _match_row_to_image(row, image_columns, image_lookup)
            if not match_indices:
                metadata_record["unmatched_rows"] += 1
                continue
            if len(match_indices) > 1:
                metadata_record["ambiguous_rows"] += 1
                _add_suspicious(
                    suspicious_records,
                    path=relative_path,
                    record_type="metadata_row",
                    issue="ambiguous_image_match",
                    details=f"Value '{match_value}' matched {len(match_indices)} images.",
                )
                continue

            metadata_record["matched_rows"] += 1
            image_record = image_records[match_indices[0]]
            image_record["metadata_source_file"] = relative_path
            image_record["metadata_match_column"] = match_column
            image_record["metadata_match_status"] = "matched"

            patient_value, patient_column = _extract_first_value(row, patient_columns)
            study_value, study_column = _extract_first_value(row, study_columns)
            label_value, label_column = _extract_first_value(row, label_columns)
            pair_value, pair_column = _extract_first_value(row, pair_columns)
            contrast_value, contrast_column = _extract_first_value(row, contrast_columns)

            _merge_field(
                image_record,
                field="patient_id",
                source_field="patient_id_source",
                new_value=patient_value,
                new_source=f"metadata:{patient_column}" if patient_column else "metadata",
                suspicious_records=suspicious_records,
            )
            _merge_field(
                image_record,
                field="study_id",
                source_field="study_id_source",
                new_value=study_value,
                new_source=f"metadata:{study_column}" if study_column else "metadata",
                suspicious_records=suspicious_records,
            )
            _merge_field(
                image_record,
                field="label",
                source_field="label_source",
                new_value=label_value,
                new_source=f"metadata:{label_column}" if label_column else "metadata",
                suspicious_records=suspicious_records,
            )
            _merge_field(
                image_record,
                field="metadata_pair_id",
                source_field="metadata_pair_id_source",
                new_value=pair_value,
                new_source=f"metadata:{pair_column}" if pair_column else "metadata",
                suspicious_records=suspicious_records,
            )
            _merge_contrast(
                image_record,
                contrast_value=contrast_value,
                source=f"metadata:{contrast_column}" if contrast_column else "metadata",
                suspicious_records=suspicious_records,
            )

        metadata_records.append(metadata_record)

    images = _finalize_images(image_records, suspicious_records)
    pairs = _build_pairs(images, suspicious_records)
    patients = _build_patients(images, pairs)
    metadata_files = pd.DataFrame(metadata_records)
    if metadata_files.empty:
        metadata_files = _empty_frame(METADATA_FILE_COLUMNS)
    else:
        metadata_files = metadata_files[METADATA_FILE_COLUMNS].sort_values("relative_path").reset_index(drop=True)

    suspicious_files = pd.DataFrame(suspicious_records)
    if suspicious_files.empty:
        suspicious_files = _empty_frame(SUSPICIOUS_COLUMNS)
    else:
        suspicious_files = suspicious_files[SUSPICIOUS_COLUMNS].sort_values(["issue", "path"]).reset_index(drop=True)

    missing_pairs = (
        pairs[pairs["pair_status"] != "complete"].copy().reset_index(drop=True)
        if not pairs.empty
        else _empty_frame(PAIR_COLUMNS)
    )

    report = _build_report(
        raw_dir=root,
        total_files=len(all_files),
        image_files=len(all_image_paths),
        metadata_files=metadata_files,
        images=images,
        patients=patients,
        pairs=pairs,
        suspicious_files=suspicious_files,
        subset_size=effective_subset,
        smoke_test=smoke_test,
        metadata_row_limit=metadata_row_limit,
    )

    return DatasetAuditResult(
        images=images,
        patients=patients,
        pairs=pairs,
        metadata_files=metadata_files,
        suspicious_files=suspicious_files,
        missing_pairs=missing_pairs,
        report=report,
    )


def format_audit_summary(result: DatasetAuditResult) -> str:
    """Create a readable audit summary for terminal output."""
    summary = result.report["summary"]
    counts = result.report["counts"]

    def top_items(mapping: dict[str, int], limit: int = 5) -> str:
        items = list(mapping.items())[:limit]
        if not items:
            return "none"
        return ", ".join(f"{key}={value}" for key, value in items)

    lines = [
        "Dataset Audit Summary",
        f"  Files scanned: {summary['total_files_scanned']}",
        f"  Images found: {summary['image_files_found']}",
        f"  Metadata files found: {summary['metadata_files_found']}",
        f"  Other files found: {summary['other_files_found']}",
        f"  Parsed metadata files: {summary['parsed_metadata_files']}",
        f"  Indexed images: {summary['images_indexed']}",
        f"  Indexed patients: {summary['patients_indexed']}",
        f"  Indexed pairs: {summary['pairs_indexed']}",
        f"  Complete pairs: {summary['complete_pairs']}",
        f"  Missing brightfield pairs: {summary['missing_brightfield_pairs']}",
        f"  Missing darkfield pairs: {summary['missing_darkfield_pairs']}",
        f"  Suspicious items: {summary['suspicious_items']}",
        f"  Contrast counts: {top_items(counts['by_contrast'])}",
        f"  Study counts: {top_items(counts['by_study'])}",
        f"  Patient counts: {top_items(counts['by_patient'])}",
        f"  Label counts: {top_items(counts['by_label'])}",
    ]
    return "\n".join(lines)


def format_pair_examples(pairs: pd.DataFrame, *, limit: int = 20) -> str:
    """Format a small sample of inferred complete pairs for manual inspection."""
    if pairs.empty:
        return "No pairs were inferred."

    complete_pairs = pairs[pairs["pair_status"] == "complete"].head(limit)
    if complete_pairs.empty:
        return "No complete brightfield/darkfield pairs were inferred."

    lines = ["Example inferred brightfield/darkfield pairs:"]
    for _, row in complete_pairs.iterrows():
        lines.extend(
            [
                f"- Pair ID: {row['pair_id']}",
                f"  Pair key: {row['pair_key']}",
                f"  Study ID: {row['study_id']}",
                f"  Patient ID: {row['patient_id']}",
                f"  BF: {row['brightfield_relative_path']}",
                f"  DF: {row['darkfield_relative_path']}",
            ]
        )
    return "\n".join(lines)


def write_audit_outputs(result: DatasetAuditResult, output_dir: str | Path) -> None:
    """Write audit artifacts into a report directory."""
    destination = ensure_dir(output_dir)
    write_json(destination / "audit_report.json", result.report)
    result.metadata_files.to_csv(destination / "metadata_files.csv", index=False)
    result.suspicious_files.to_csv(destination / "suspicious_files.csv", index=False)
    result.missing_pairs.to_csv(destination / "missing_pairs.csv", index=False)


def write_metadata_outputs(result: DatasetAuditResult, output_dir: str | Path) -> None:
    """Write metadata tables plus audit artifacts into a metadata directory."""
    destination = ensure_dir(output_dir)
    result.images.to_csv(destination / "images.csv", index=False)
    result.patients.to_csv(destination / "patients.csv", index=False)
    result.pairs.to_csv(destination / "pairs.csv", index=False)
    result.metadata_files.to_csv(destination / "metadata_files.csv", index=False)
    result.suspicious_files.to_csv(destination / "suspicious_files.csv", index=False)
    result.missing_pairs.to_csv(destination / "missing_pairs.csv", index=False)
    write_json(destination / "audit_report.json", result.report)
