"""Dataset audit and metadata-indexing pipeline for the schistosomiasis BF/DF dataset."""

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
from schisto_mobile_ai.data.schisto_dataset import (
    aggregate_annotation_table,
    detect_schisto_table_kind,
    derive_image_label,
    derive_patient_label,
    make_pair_key,
    make_patient_key,
    map_schisto_contrast_code,
    parse_schisto_image_name,
    standardize_patient_list_table,
)
from schisto_mobile_ai.utils.io import ensure_dir, write_json


UNKNOWN_VALUE = "UNKNOWN"

IMAGE_COLUMNS = [
    "image_id",
    "image_name",
    "relative_path",
    "file_name",
    "file_stem",
    "parent_dir",
    "extension",
    "size_bytes",
    "study_id",
    "study_id_source",
    "patient_id",
    "patient_key",
    "patient_id_source",
    "frame_num",
    "frame_num_source",
    "contrast_raw",
    "contrast",
    "contrast_source",
    "pair_key",
    "pair_key_source",
    "label",
    "label_source",
    "patient_level_label",
    "patient_level_label_source",
    "patient_list_image_name",
    "patient_list_patient_num",
    "patient_list_frame_num",
    "patient_list_contrast",
    "number_eggs",
    "patient_eggs",
    "quality",
    "annotation_total_count",
    "annotation_s_haematobium_count",
    "annotation_doubtful_count",
    "annotation_object_types",
    "metadata_source_file",
    "metadata_match_column",
    "metadata_match_status",
]

PATIENT_COLUMNS = [
    "study_id",
    "patient_id",
    "patient_key",
    "n_images",
    "n_frames",
    "n_brightfield",
    "n_darkfield",
    "n_unknown_contrast",
    "n_complete_pairs",
    "n_missing_brightfield",
    "n_missing_darkfield",
    "labels_observed",
    "patient_label",
    "patient_label_source",
    "patient_eggs_min",
    "patient_eggs_max",
    "patient_eggs_unique_values_count",
    "number_eggs_sum",
    "annotation_s_haematobium_total",
    "annotation_doubtful_total",
]

PAIR_COLUMNS = [
    "pair_id",
    "pair_key",
    "study_id",
    "patient_id",
    "patient_key",
    "frame_num",
    "pair_status",
    "label",
    "label_source",
    "patient_level_label",
    "brightfield_image_id",
    "brightfield_relative_path",
    "darkfield_image_id",
    "darkfield_relative_path",
    "brightfield_number_eggs",
    "darkfield_number_eggs",
    "patient_eggs",
    "brightfield_annotation_s_haematobium_count",
    "darkfield_annotation_s_haematobium_count",
    "brightfield_annotation_doubtful_count",
    "darkfield_annotation_doubtful_count",
]

METADATA_FILE_COLUMNS = [
    "relative_path",
    "extension",
    "parse_success",
    "parse_error",
    "table_kind",
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
    "duplicate_rows_dropped",
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


def _safe_value(value: Any) -> str:
    return normalize_optional_string(value) or UNKNOWN_VALUE


def _normalize_numeric(value: Any) -> float | int | None:
    if pd.isna(value) or normalize_optional_string(value) == "":
        return None
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    if float(numeric).is_integer():
        return int(numeric)
    return float(numeric)


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


def _append_pipe_value(existing: Any, value: str) -> str:
    current = [item for item in normalize_optional_string(existing).split("|") if item]
    if value and value not in current:
        current.append(value)
    return "|".join(current)


def _select_canonical_paths(
    root: Path,
    paths: list[Path],
    *,
    key_fn,
) -> tuple[list[Path], list[tuple[str, list[Path]]]]:
    grouped: dict[str, list[Path]] = {}
    for path in paths:
        grouped.setdefault(str(key_fn(path)), []).append(path)

    canonical_paths: list[Path] = []
    duplicate_groups: list[tuple[str, list[Path]]] = []
    for key, group in sorted(grouped.items()):
        ranked = sorted(
            group,
            key=lambda item: (
                item.relative_to(root).as_posix().count("/"),
                len(item.relative_to(root).as_posix()),
                item.relative_to(root).as_posix(),
            ),
        )
        canonical_paths.append(ranked[0])
        if len(ranked) > 1:
            duplicate_groups.append((key, ranked))

    canonical_paths.sort()
    return canonical_paths, duplicate_groups


def _record_duplicate_groups(
    root: Path,
    duplicate_groups: list[tuple[str, list[Path]]],
    suspicious_records: list[dict[str, str]],
    *,
    record_type: str,
    issue: str,
) -> None:
    for key, group in duplicate_groups:
        chosen = group[0].relative_to(root).as_posix()
        duplicates = ", ".join(path.relative_to(root).as_posix() for path in group[1:])
        _add_suspicious(
            suspicious_records,
            path=chosen,
            record_type=record_type,
            issue=issue,
            details=f"Kept canonical file for key '{key}' and ignored duplicate copies: {duplicates}",
        )


def _make_image_record(raw_dir: Path, path: Path) -> dict[str, Any]:
    relative_path = path.relative_to(raw_dir).as_posix()
    parsed = parse_schisto_image_name(path.name)
    inferred_contrast, _, inferred_contrast_raw = infer_contrast_from_text(relative_path)
    patient_id, patient_source = infer_path_identifier(relative_path, kind="patient")
    study_id, study_source = infer_path_identifier(relative_path, kind="study")

    if parsed is not None:
        study_id = parsed["study_id"]
        study_source = "filename"
        patient_id = parsed["patient_id"]
        patient_source = "filename"
        frame_num = parsed["frame_num"]
        frame_source = "filename"
        contrast_raw = parsed["contrast_raw"]
        contrast = parsed["contrast"]
        contrast_source = "filename"
        pair_key = parsed["pair_key"]
    else:
        frame_num = ""
        frame_source = "unknown"
        contrast_raw = inferred_contrast_raw
        contrast = inferred_contrast
        contrast_source = "path" if inferred_contrast != "unknown" else "unknown"
        pair_key = infer_pair_key(relative_path)

    patient_key = make_patient_key(study_id, patient_id)

    return {
        "image_id": stable_id("img", relative_path),
        "image_name": path.name,
        "relative_path": relative_path,
        "file_name": path.name,
        "file_stem": path.stem,
        "parent_dir": path.parent.relative_to(raw_dir).as_posix() if path.parent != raw_dir else ".",
        "extension": path.suffix.lower(),
        "size_bytes": int(path.stat().st_size),
        "study_id": study_id,
        "study_id_source": study_source,
        "patient_id": patient_id,
        "patient_key": patient_key,
        "patient_id_source": patient_source,
        "frame_num": frame_num,
        "frame_num_source": frame_source,
        "contrast_raw": contrast_raw,
        "contrast": contrast,
        "contrast_source": contrast_source,
        "pair_key": pair_key,
        "pair_key_source": "filename" if parsed is not None else "path",
        "label": "",
        "label_source": "unknown",
        "patient_level_label": "",
        "patient_level_label_source": "unknown",
        "patient_list_image_name": "",
        "patient_list_patient_num": "",
        "patient_list_frame_num": "",
        "patient_list_contrast": "",
        "number_eggs": None,
        "patient_eggs": None,
        "quality": None,
        "annotation_total_count": None,
        "annotation_s_haematobium_count": None,
        "annotation_doubtful_count": None,
        "annotation_object_types": "",
        "metadata_source_file": "",
        "metadata_match_column": "",
        "metadata_match_status": "",
    }


def _build_image_lookup(image_records: list[dict[str, Any]]) -> dict[str, list[int]]:
    lookup: dict[str, set[int]] = {}
    for index, record in enumerate(image_records):
        candidate_values = [
            record["image_name"],
            record["relative_path"],
            record["file_stem"],
        ]
        for value in candidate_values:
            for key in build_image_match_keys(value):
                lookup.setdefault(key, set()).add(index)
    return {key: sorted(indices) for key, indices in lookup.items()}


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

    if current_source in {"unknown", "path", "filename:patient", "filename:study"} and new_source.startswith("metadata:"):
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


def _set_numeric_field(record: dict[str, Any], field: str, value: Any) -> None:
    normalized = _normalize_numeric(value)
    if normalized is None:
        return
    record[field] = normalized


def _record_metadata_trace(record: dict[str, Any], *, source_file: str, match_column: str, status: str) -> None:
    record["metadata_source_file"] = _append_pipe_value(record.get("metadata_source_file"), source_file)
    record["metadata_match_column"] = _append_pipe_value(record.get("metadata_match_column"), match_column)
    record["metadata_match_status"] = _append_pipe_value(record.get("metadata_match_status"), status)


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


def _match_image_name_to_record(
    image_name: str,
    image_lookup: dict[str, list[int]],
) -> list[int]:
    for key in build_image_match_keys(image_name):
        matches = image_lookup.get(key, [])
        if matches:
            return matches
    return []


def _apply_patient_list_rows(
    table: pd.DataFrame,
    *,
    source_file: str,
    image_lookup: dict[str, list[int]],
    image_records: list[dict[str, Any]],
    metadata_record: dict[str, Any],
    suspicious_records: list[dict[str, str]],
) -> None:
    for _, row in table.iterrows():
        matches = _match_image_name_to_record(
            normalize_optional_string(row["image_name"]),
            image_lookup,
        )
        if not matches:
            metadata_record["unmatched_rows"] += 1
            continue
        if len(matches) > 1:
            metadata_record["ambiguous_rows"] += 1
            _add_suspicious(
                suspicious_records,
                path=source_file,
                record_type="metadata_row",
                issue="ambiguous_image_match",
                details=f"Image name '{row['image_name']}' matched {len(matches)} images.",
            )
            continue

        metadata_record["matched_rows"] += 1
        record = image_records[matches[0]]
        _record_metadata_trace(record, source_file=source_file, match_column="imageName", status="patient_list")

        _merge_field(
            record,
            field="study_id",
            source_field="study_id_source",
            new_value=row["study_id"],
            new_source="metadata:imageName",
            suspicious_records=suspicious_records,
        )
        _merge_field(
            record,
            field="patient_id",
            source_field="patient_id_source",
            new_value=row["patient_id"],
            new_source="metadata:patientNum",
            suspicious_records=suspicious_records,
        )
        _merge_field(
            record,
            field="frame_num",
            source_field="frame_num_source",
            new_value=row["frame_num"],
            new_source="metadata:frameNum",
            suspicious_records=suspicious_records,
        )

        contrast_raw = normalize_optional_string(row["contrast_raw"])
        if contrast_raw:
            record["contrast_raw"] = contrast_raw
            mapped_contrast = map_schisto_contrast_code(contrast_raw)
            if mapped_contrast != "unknown":
                record["contrast"] = mapped_contrast
                record["contrast_source"] = "metadata:Contrast"

        record["patient_list_image_name"] = normalize_optional_string(row["image_name"])
        record["patient_list_patient_num"] = normalize_optional_string(row["patient_id"])
        record["patient_list_frame_num"] = normalize_optional_string(row["frame_num"])
        record["patient_list_contrast"] = contrast_raw
        _set_numeric_field(record, "number_eggs", row["number_eggs"])
        _set_numeric_field(record, "patient_eggs", row["patient_eggs"])
        _set_numeric_field(record, "quality", row["quality"])


def _apply_annotation_rows(
    table: pd.DataFrame,
    *,
    source_file: str,
    image_lookup: dict[str, list[int]],
    image_records: list[dict[str, Any]],
    metadata_record: dict[str, Any],
    suspicious_records: list[dict[str, str]],
) -> None:
    for _, row in table.iterrows():
        matches = _match_image_name_to_record(
            normalize_optional_string(row["image_name"]),
            image_lookup,
        )
        if not matches:
            metadata_record["unmatched_rows"] += 1
            continue
        if len(matches) > 1:
            metadata_record["ambiguous_rows"] += 1
            _add_suspicious(
                suspicious_records,
                path=source_file,
                record_type="metadata_row",
                issue="ambiguous_image_match",
                details=f"Image name '{row['image_name']}' matched {len(matches)} images.",
            )
            continue

        metadata_record["matched_rows"] += 1
        record = image_records[matches[0]]
        _record_metadata_trace(record, source_file=source_file, match_column="imageName", status="annotations")
        _set_numeric_field(record, "annotation_total_count", row["annotation_total_count"])
        _set_numeric_field(
            record,
            "annotation_s_haematobium_count",
            row["annotation_s_haematobium_count"],
        )
        _set_numeric_field(record, "annotation_doubtful_count", row["annotation_doubtful_count"])
        record["annotation_object_types"] = normalize_optional_string(row["annotation_object_types"])


def _apply_generic_table(
    frame: pd.DataFrame,
    *,
    source_file: str,
    image_lookup: dict[str, list[int]],
    image_records: list[dict[str, Any]],
    metadata_record: dict[str, Any],
    suspicious_records: list[dict[str, str]],
) -> None:
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
        return

    for _, row in frame.iterrows():
        match_indices, match_column, match_value = _match_row_to_image(row, image_columns, image_lookup)
        if not match_indices:
            metadata_record["unmatched_rows"] += 1
            continue
        if len(match_indices) > 1:
            metadata_record["ambiguous_rows"] += 1
            _add_suspicious(
                suspicious_records,
                path=source_file,
                record_type="metadata_row",
                issue="ambiguous_image_match",
                details=f"Value '{match_value}' matched {len(match_indices)} images.",
            )
            continue

        metadata_record["matched_rows"] += 1
        record = image_records[match_indices[0]]
        _record_metadata_trace(record, source_file=source_file, match_column=match_column, status="generic")

        patient_value, patient_column = _extract_first_value(row, patient_columns)
        study_value, study_column = _extract_first_value(row, study_columns)
        label_value, label_column = _extract_first_value(row, label_columns)
        pair_value, pair_column = _extract_first_value(row, pair_columns)
        contrast_value, contrast_column = _extract_first_value(row, contrast_columns)

        _merge_field(
            record,
            field="patient_id",
            source_field="patient_id_source",
            new_value=patient_value,
            new_source=f"metadata:{patient_column}" if patient_column else "metadata",
            suspicious_records=suspicious_records,
        )
        _merge_field(
            record,
            field="study_id",
            source_field="study_id_source",
            new_value=study_value,
            new_source=f"metadata:{study_column}" if study_column else "metadata",
            suspicious_records=suspicious_records,
        )
        if label_value and not normalize_optional_string(record["label"]):
            record["label"] = label_value
            record["label_source"] = f"metadata:{label_column}" if label_column else "metadata"
        if pair_value and not normalize_optional_string(record["pair_key"]):
            record["pair_key"] = pair_value
            record["pair_key_source"] = f"metadata:{pair_column}" if pair_column else "metadata"
        if contrast_value:
            inferred_contrast, _, _ = infer_contrast_from_text(contrast_value)
            record["contrast_raw"] = normalize_optional_string(contrast_value)
            if inferred_contrast != "unknown":
                record["contrast"] = inferred_contrast
                record["contrast_source"] = f"metadata:{contrast_column}" if contrast_column else "metadata"


def _finalize_images(
    image_records: list[dict[str, Any]],
    *,
    studies_with_annotation_tables: set[str],
    suspicious_records: list[dict[str, str]],
) -> pd.DataFrame:
    finalized_records: list[dict[str, Any]] = []
    for record in image_records:
        parsed = parse_schisto_image_name(record["image_name"])
        if parsed is not None:
            if not normalize_optional_string(record["study_id"]):
                record["study_id"] = parsed["study_id"]
                record["study_id_source"] = "filename"
            if not normalize_optional_string(record["patient_id"]):
                record["patient_id"] = parsed["patient_id"]
                record["patient_id_source"] = "filename"
            if not normalize_optional_string(record["frame_num"]):
                record["frame_num"] = parsed["frame_num"]
                record["frame_num_source"] = "filename"
            if not normalize_optional_string(record["contrast_raw"]):
                record["contrast_raw"] = parsed["contrast_raw"]
            if normalize_optional_string(record["contrast"]) in {"", "unknown"}:
                record["contrast"] = parsed["contrast"]
                record["contrast_source"] = "filename"
            if not normalize_optional_string(record["pair_key"]):
                record["pair_key"] = parsed["pair_key"]
                record["pair_key_source"] = "filename"

        if normalize_optional_string(record["contrast"]) in {"", "unknown"} and normalize_optional_string(record["contrast_raw"]):
            mapped = map_schisto_contrast_code(record["contrast_raw"])
            if mapped != "unknown":
                record["contrast"] = mapped
                record["contrast_source"] = "metadata:Contrast"

        if not normalize_optional_string(record["patient_key"]):
            record["patient_key"] = make_patient_key(record["study_id"], record["patient_id"])

        if not normalize_optional_string(record["pair_key"]):
            record["pair_key"] = make_pair_key(record["study_id"], record["patient_id"], record["frame_num"])
            if normalize_optional_string(record["pair_key"]):
                record["pair_key_source"] = "study_patient_frame"

        study_id = normalize_optional_string(record["study_id"])
        if study_id in studies_with_annotation_tables:
            if record["annotation_total_count"] is None:
                record["annotation_total_count"] = 0
            if record["annotation_s_haematobium_count"] is None:
                record["annotation_s_haematobium_count"] = 0
            if record["annotation_doubtful_count"] is None:
                record["annotation_doubtful_count"] = 0
            if not normalize_optional_string(record["annotation_object_types"]):
                record["annotation_object_types"] = "{}"

        image_label, image_label_source = derive_image_label(
            number_eggs=record["number_eggs"],
            annotation_s_haematobium_count=record["annotation_s_haematobium_count"],
        )
        record["label"] = image_label
        record["label_source"] = image_label_source

        patient_label, patient_label_source = derive_patient_label(
            patient_eggs=record["patient_eggs"],
            image_labels=[image_label],
        )
        record["patient_level_label"] = patient_label
        record["patient_level_label_source"] = patient_label_source

        if record["size_bytes"] <= 0:
            _add_suspicious(
                suspicious_records,
                path=record["relative_path"],
                record_type="image",
                issue="empty_file",
                details="Image file has size 0 bytes.",
            )

        if normalize_optional_string(record["contrast"]) in {"", "unknown"}:
            _add_suspicious(
                suspicious_records,
                path=record["relative_path"],
                record_type="image",
                issue="unknown_contrast",
                details="Could not infer brightfield or darkfield from filename or metadata.",
            )

        record["study_id"] = _safe_value(record["study_id"])
        record["patient_id"] = _safe_value(record["patient_id"])
        record["patient_key"] = _safe_value(record["patient_key"])
        record["frame_num"] = _safe_value(record["frame_num"])
        record["pair_key"] = _safe_value(record["pair_key"])
        record["contrast_raw"] = normalize_optional_string(record["contrast_raw"])
        record["contrast"] = _safe_value(record["contrast"])
        record["label"] = _safe_value(record["label"])
        record["patient_level_label"] = _safe_value(record["patient_level_label"])
        finalized_records.append(record)

    if not finalized_records:
        return _empty_frame(IMAGE_COLUMNS)

    images = pd.DataFrame(finalized_records)
    return images[IMAGE_COLUMNS].sort_values(["study_id", "patient_id", "frame_num", "contrast", "relative_path"]).reset_index(drop=True)


def _derive_frame_label(rows: list[dict[str, Any]]) -> tuple[str, str]:
    image_labels = [normalize_optional_string(row.get("label")) for row in rows if row is not None]
    if "positive" in image_labels:
        return "positive", "image_labels"
    known_labels = {label for label in image_labels if label not in {"", UNKNOWN_VALUE}}
    if known_labels == {"negative"}:
        return "negative", "image_labels"
    return "UNKNOWN", "unknown"


def _derive_pair_patient_label(rows: list[dict[str, Any]]) -> str:
    patient_labels = [
        normalize_optional_string(row.get("patient_level_label"))
        for row in rows
        if row is not None
    ]
    if "positive" in patient_labels:
        return "positive"
    known_labels = {label for label in patient_labels if label not in {"", UNKNOWN_VALUE}}
    if known_labels == {"negative"}:
        return "negative"
    return UNKNOWN_VALUE


def _build_pairs(
    images: pd.DataFrame,
    suspicious_records: list[dict[str, str]],
) -> pd.DataFrame:
    if images.empty:
        return _empty_frame(PAIR_COLUMNS)

    pair_records: list[dict[str, Any]] = []
    for pair_key, group in images.groupby("pair_key", sort=True, dropna=False):
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
                path=str(pair_key),
                record_type="pair",
                issue="multiple_brightfield_candidates",
                details=f"Found {len(brightfield)} brightfield images for pair key '{pair_key}'.",
            )
        if len(darkfield) > 1:
            _add_suspicious(
                suspicious_records,
                path=str(pair_key),
                record_type="pair",
                issue="multiple_darkfield_candidates",
                details=f"Found {len(darkfield)} darkfield images for pair key '{pair_key}'.",
            )

        bf_row = brightfield.iloc[0].to_dict() if not brightfield.empty else None
        df_row = darkfield.iloc[0].to_dict() if not darkfield.empty else None

        if bf_row and df_row:
            pair_status = "complete"
        elif bf_row:
            pair_status = "missing_darkfield"
        elif df_row:
            pair_status = "missing_brightfield"
        else:
            continue

        anchor_row = bf_row or df_row
        frame_label, frame_label_source = _derive_frame_label([bf_row, df_row])
        pair_records.append(
            {
                "pair_id": stable_id("pair", str(pair_key)),
                "pair_key": pair_key,
                "study_id": anchor_row["study_id"],
                "patient_id": anchor_row["patient_id"],
                "patient_key": anchor_row["patient_key"],
                "frame_num": anchor_row["frame_num"],
                "pair_status": pair_status,
                "label": frame_label,
                "label_source": frame_label_source,
                "patient_level_label": _derive_pair_patient_label([bf_row, df_row]),
                "brightfield_image_id": bf_row["image_id"] if bf_row else "",
                "brightfield_relative_path": bf_row["relative_path"] if bf_row else "",
                "darkfield_image_id": df_row["image_id"] if df_row else "",
                "darkfield_relative_path": df_row["relative_path"] if df_row else "",
                "brightfield_number_eggs": bf_row["number_eggs"] if bf_row else None,
                "darkfield_number_eggs": df_row["number_eggs"] if df_row else None,
                "patient_eggs": (
                    bf_row["patient_eggs"]
                    if bf_row and bf_row["patient_eggs"] is not None
                    else df_row["patient_eggs"] if df_row else None
                ),
                "brightfield_annotation_s_haematobium_count": bf_row["annotation_s_haematobium_count"] if bf_row else None,
                "darkfield_annotation_s_haematobium_count": df_row["annotation_s_haematobium_count"] if df_row else None,
                "brightfield_annotation_doubtful_count": bf_row["annotation_doubtful_count"] if bf_row else None,
                "darkfield_annotation_doubtful_count": df_row["annotation_doubtful_count"] if df_row else None,
            }
        )

    if not pair_records:
        return _empty_frame(PAIR_COLUMNS)

    return pd.DataFrame(pair_records)[PAIR_COLUMNS].sort_values(["study_id", "patient_id", "frame_num"]).reset_index(drop=True)


def _build_patients(images: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    if images.empty:
        return _empty_frame(PATIENT_COLUMNS)

    pair_groups = {
        patient_key: frame.copy()
        for patient_key, frame in pairs.groupby("patient_key", dropna=False, sort=False)
    } if not pairs.empty else {}

    patient_rows: list[dict[str, Any]] = []
    for patient_key, frame in images.groupby("patient_key", sort=True, dropna=False):
        pair_frame = pair_groups.get(patient_key, _empty_frame(PAIR_COLUMNS))
        image_labels = [
            normalize_optional_string(label)
            for label in frame["label"].tolist()
            if normalize_optional_string(label) and normalize_optional_string(label) != UNKNOWN_VALUE
        ]

        patient_eggs_series = pd.to_numeric(frame["patient_eggs"], errors="coerce")
        patient_eggs_non_null = patient_eggs_series.dropna()
        patient_eggs_unique_values = sorted(
            {
                _normalize_numeric(value)
                for value in patient_eggs_non_null.tolist()
                if _normalize_numeric(value) is not None
            }
        )
        patient_label, patient_label_source = derive_patient_label(
            patient_eggs=patient_eggs_non_null.max() if not patient_eggs_non_null.empty else None,
            image_labels=image_labels,
        )

        patient_rows.append(
            {
                "study_id": frame["study_id"].iloc[0],
                "patient_id": frame["patient_id"].iloc[0],
                "patient_key": patient_key,
                "n_images": int(len(frame)),
                "n_frames": int(frame["frame_num"].nunique()),
                "n_brightfield": int((frame["contrast"] == "brightfield").sum()),
                "n_darkfield": int((frame["contrast"] == "darkfield").sum()),
                "n_unknown_contrast": int((frame["contrast"] == "UNKNOWN").sum()),
                "n_complete_pairs": int((pair_frame["pair_status"] == "complete").sum()) if not pair_frame.empty else 0,
                "n_missing_brightfield": int((pair_frame["pair_status"] == "missing_brightfield").sum()) if not pair_frame.empty else 0,
                "n_missing_darkfield": int((pair_frame["pair_status"] == "missing_darkfield").sum()) if not pair_frame.empty else 0,
                "labels_observed": "|".join(sorted(set(image_labels))) if image_labels else UNKNOWN_VALUE,
                "patient_label": patient_label,
                "patient_label_source": patient_label_source,
                "patient_eggs_min": _normalize_numeric(patient_eggs_non_null.min()) if not patient_eggs_non_null.empty else None,
                "patient_eggs_max": _normalize_numeric(patient_eggs_non_null.max()) if not patient_eggs_non_null.empty else None,
                "patient_eggs_unique_values_count": len(patient_eggs_unique_values),
                "number_eggs_sum": _normalize_numeric(pd.to_numeric(frame["number_eggs"], errors="coerce").sum(skipna=True)),
                "annotation_s_haematobium_total": int(pd.to_numeric(frame["annotation_s_haematobium_count"], errors="coerce").fillna(0).sum()),
                "annotation_doubtful_total": int(pd.to_numeric(frame["annotation_doubtful_count"], errors="coerce").fillna(0).sum()),
            }
        )

    return pd.DataFrame(patient_rows)[PATIENT_COLUMNS].sort_values(["study_id", "patient_id"]).reset_index(drop=True)


def _value_counts(series: pd.Series) -> dict[str, int]:
    counts = series.fillna(UNKNOWN_VALUE).astype(str).value_counts(dropna=False)
    return {str(key): int(value) for key, value in counts.items()}


def _build_report(
    *,
    raw_dir: Path,
    total_files: int,
    total_image_files: int,
    total_metadata_files: int,
    metadata_files: pd.DataFrame,
    images: pd.DataFrame,
    patients: pd.DataFrame,
    pairs: pd.DataFrame,
    suspicious_files: pd.DataFrame,
    subset_size: int | None,
    smoke_test: bool,
    metadata_row_limit: int | None,
) -> dict[str, Any]:
    report = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "raw_dir": str(raw_dir),
        "smoke_test": bool(smoke_test),
        "subset_size": int(subset_size) if subset_size is not None else None,
        "metadata_row_limit": int(metadata_row_limit) if metadata_row_limit is not None else None,
        "summary": {
            "total_files_scanned": int(total_files),
            "image_files_found": int(total_image_files),
            "metadata_files_found": int(total_metadata_files),
            "other_files_found": int(total_files - total_image_files - total_metadata_files),
            "parsed_metadata_files": int(metadata_files["parse_success"].fillna(False).sum()) if not metadata_files.empty else 0,
            "images_indexed": int(len(images)),
            "patients_indexed": int(len(patients)),
            "pairs_indexed": int(len(pairs)),
            "complete_pairs": int((pairs["pair_status"] == "complete").sum()) if not pairs.empty else 0,
            "missing_brightfield_pairs": int((pairs["pair_status"] == "missing_brightfield").sum()) if not pairs.empty else 0,
            "missing_darkfield_pairs": int((pairs["pair_status"] == "missing_darkfield").sum()) if not pairs.empty else 0,
            "suspicious_items": int(len(suspicious_files)),
        },
        "counts": {
            "by_study": _value_counts(images["study_id"]) if not images.empty else {},
            "by_contrast": _value_counts(images["contrast"]) if not images.empty else {},
            "by_patient": _value_counts(images["patient_key"]) if not images.empty else {},
            "by_label": _value_counts(patients["patient_label"]) if not patients.empty else {},
            "by_pair_status": _value_counts(pairs["pair_status"]) if not pairs.empty else {},
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
    all_metadata_paths = [path for path in all_files if classify_file(path) == "metadata"]

    suspicious_records: list[dict[str, str]] = []
    image_paths, duplicate_images = _select_canonical_paths(root, all_image_paths, key_fn=lambda path: path.name)
    metadata_paths, duplicate_metadata = _select_canonical_paths(root, all_metadata_paths, key_fn=lambda path: path.name)
    _record_duplicate_groups(
        root,
        duplicate_images,
        suspicious_records,
        record_type="image",
        issue="duplicate_image_name",
    )
    _record_duplicate_groups(
        root,
        duplicate_metadata,
        suspicious_records,
        record_type="metadata",
        issue="duplicate_metadata_name",
    )

    if effective_subset is not None:
        image_paths = image_paths[:effective_subset]

    image_records = [_make_image_record(root, path) for path in image_paths]
    image_lookup = _build_image_lookup(image_records)

    metadata_records: list[dict[str, Any]] = []
    studies_with_annotation_tables: set[str] = set()
    for metadata_path in metadata_paths:
        relative_path = metadata_path.relative_to(root).as_posix()
        frame, parse_error = try_load_metadata_table(metadata_path, max_rows=metadata_row_limit)

        metadata_record = {
            "relative_path": relative_path,
            "extension": metadata_path.suffix.lower(),
            "parse_success": frame is not None,
            "parse_error": parse_error,
            "table_kind": "unknown",
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
            "duplicate_rows_dropped": 0,
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

        table_kind = detect_schisto_table_kind(frame, metadata_path)
        metadata_record["table_kind"] = table_kind

        if table_kind == "patient_list":
            standardized = standardize_patient_list_table(frame, relative_path)
            metadata_record["image_columns"] = "imageName"
            metadata_record["patient_columns"] = "patientNum"
            metadata_record["study_columns"] = "filename_prefix"
            metadata_record["label_columns"] = "numberEggs|patientEggs"
            metadata_record["pair_columns"] = "frameNum"
            metadata_record["contrast_columns"] = "Contrast"
            metadata_record["duplicate_rows_dropped"] = int(len(frame) - len(standardized))
            _apply_patient_list_rows(
                standardized,
                source_file=relative_path,
                image_lookup=image_lookup,
                image_records=image_records,
                metadata_record=metadata_record,
                suspicious_records=suspicious_records,
            )
        elif table_kind == "annotations":
            aggregated = aggregate_annotation_table(frame, relative_path)
            study_hint = normalize_optional_string(metadata_path.name.split("_")[0])
            if study_hint:
                studies_with_annotation_tables.add(study_hint)
            metadata_record["image_columns"] = "imageName"
            metadata_record["label_columns"] = "objectType"
            metadata_record["duplicate_rows_dropped"] = max(0, int(len(frame) - aggregated["annotation_total_count"].sum())) if not aggregated.empty else 0
            _apply_annotation_rows(
                aggregated,
                source_file=relative_path,
                image_lookup=image_lookup,
                image_records=image_records,
                metadata_record=metadata_record,
                suspicious_records=suspicious_records,
            )
        else:
            _apply_generic_table(
                frame,
                source_file=relative_path,
                image_lookup=image_lookup,
                image_records=image_records,
                metadata_record=metadata_record,
                suspicious_records=suspicious_records,
            )

        metadata_records.append(metadata_record)

    images = _finalize_images(
        image_records,
        studies_with_annotation_tables=studies_with_annotation_tables,
        suspicious_records=suspicious_records,
    )
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
        total_image_files=len(all_image_paths),
        total_metadata_files=len(all_metadata_paths),
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
    """Format a sample of inferred complete pairs for manual inspection."""
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
                f"  Frame: {row['frame_num']}",
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
