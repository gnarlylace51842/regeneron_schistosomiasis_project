"""Dataset-specific helpers for the schistosomiasis BF/DF microscopy release."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from schisto_mobile_ai.data.manifest import normalize_optional_string


SCHISTO_IMAGE_PATTERN = re.compile(
    r"^(?P<study>[A-Za-z0-9]+)_(?P<patient>\d+)_(?P<frame>\d+)_(?P<contrast>\d+)$"
)
SCHISTO_CONTRAST_MAP = {
    "0": "brightfield",
    "2": "darkfield",
}


def _normalize_column_name(value: Any) -> str:
    text = normalize_optional_string(value).lower()
    return re.sub(r"[^a-z0-9]+", "", text)


def normalize_patient_token(value: Any) -> str:
    """Normalize patient numbers while preserving the dataset's 3-digit style."""
    text = normalize_optional_string(value)
    if not text:
        return ""
    if text.isdigit():
        return text.zfill(3)
    return text


def normalize_frame_token(value: Any) -> str:
    """Normalize frame identifiers into compact string tokens."""
    text = normalize_optional_string(value)
    if not text:
        return ""
    if text.isdigit():
        return str(int(text))
    return text


def normalize_image_name(value: Any) -> str:
    """Return just the image file name used by the metadata tables."""
    text = normalize_optional_string(value)
    if not text:
        return ""
    return Path(text).name


def map_schisto_contrast_code(value: Any) -> str:
    """Map numeric contrast tokens used by the dataset into readable names."""
    text = normalize_optional_string(value)
    return SCHISTO_CONTRAST_MAP.get(text, "unknown")


def make_patient_key(study_id: Any, patient_id: Any) -> str:
    """Build a study-aware patient key so repeated patient IDs do not collide."""
    study = normalize_optional_string(study_id)
    patient = normalize_patient_token(patient_id)
    if not study or not patient:
        return ""
    return f"{study}_{patient}"


def make_pair_key(study_id: Any, patient_id: Any, frame_num: Any) -> str:
    """Build the canonical pair key for one BF/DF frame pair."""
    study = normalize_optional_string(study_id)
    patient = normalize_patient_token(patient_id)
    frame = normalize_frame_token(frame_num)
    if not study or not patient or not frame:
        return ""
    return f"{study}_{patient}_{frame}"


def parse_schisto_image_name(value: Any) -> dict[str, str] | None:
    """Parse filenames like nov2021_001_0_2.jpg into their structural parts."""
    image_name = normalize_image_name(value)
    if not image_name:
        return None

    match = SCHISTO_IMAGE_PATTERN.match(Path(image_name).stem)
    if not match:
        return None

    study_id = normalize_optional_string(match.group("study"))
    patient_id = normalize_patient_token(match.group("patient"))
    frame_num = normalize_frame_token(match.group("frame"))
    contrast_raw = normalize_optional_string(match.group("contrast"))
    contrast = map_schisto_contrast_code(contrast_raw)

    return {
        "image_name": image_name,
        "study_id": study_id,
        "patient_id": patient_id,
        "patient_key": make_patient_key(study_id, patient_id),
        "frame_num": frame_num,
        "contrast_raw": contrast_raw,
        "contrast": contrast,
        "pair_key": make_pair_key(study_id, patient_id, frame_num),
    }


def looks_like_patient_list(frame: pd.DataFrame) -> bool:
    """Return True when a table matches the patient-list schema."""
    normalized = {_normalize_column_name(column) for column in frame.columns}
    required = {"imagename", "patientnum", "framenum", "contrast"}
    return required.issubset(normalized)


def looks_like_annotation_table(frame: pd.DataFrame) -> bool:
    """Return True when a table matches the annotation schema."""
    normalized = {_normalize_column_name(column) for column in frame.columns}
    required = {"imagename", "objecttype"}
    return required.issubset(normalized)


def detect_schisto_table_kind(frame: pd.DataFrame, path: str | Path) -> str:
    """Classify a metadata table into a known schistosomiasis table type when possible."""
    path_text = str(path).lower()
    if looks_like_patient_list(frame):
        return "patient_list"
    if looks_like_annotation_table(frame):
        return "annotations"
    if "patients_list" in path_text:
        return "patient_list"
    if "annotations" in path_text:
        return "annotations"
    return "generic"


def _rename_columns(frame: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    for column in frame.columns:
        normalized = _normalize_column_name(column)
        if normalized in mapping:
            rename_map[column] = mapping[normalized]
    return frame.rename(columns=rename_map)


def _to_numeric(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype="float64")
    return pd.to_numeric(series, errors="coerce")


def standardize_patient_list_table(frame: pd.DataFrame, source_path: str | Path) -> pd.DataFrame:
    """Normalize a patient-list table into a stable joinable structure."""
    standardized = _rename_columns(
        frame.copy(),
        {
            "imagename": "image_name",
            "patientnum": "patient_num",
            "framenum": "frame_num_table",
            "contrast": "contrast_raw_table",
            "numbereggs": "number_eggs",
            "patienteggs": "patient_eggs",
            "quality": "quality",
        },
    )

    if "image_name" not in standardized.columns:
        return pd.DataFrame()

    standardized["image_name"] = standardized["image_name"].map(normalize_image_name)
    standardized = standardized[standardized["image_name"].ne("")].copy()

    parsed = standardized["image_name"].map(parse_schisto_image_name)
    standardized["study_id"] = parsed.map(lambda item: item["study_id"] if item else "")
    standardized["patient_id_from_filename"] = parsed.map(
        lambda item: item["patient_id"] if item else ""
    )
    standardized["frame_num_from_filename"] = parsed.map(
        lambda item: item["frame_num"] if item else ""
    )
    standardized["contrast_raw_from_filename"] = parsed.map(
        lambda item: item["contrast_raw"] if item else ""
    )

    if "patient_num" in standardized.columns:
        standardized["patient_num"] = standardized["patient_num"].map(normalize_patient_token)
    else:
        standardized["patient_num"] = standardized["patient_id_from_filename"]

    if "frame_num_table" in standardized.columns:
        standardized["frame_num_table"] = standardized["frame_num_table"].map(normalize_frame_token)
    else:
        standardized["frame_num_table"] = standardized["frame_num_from_filename"]

    if "contrast_raw_table" in standardized.columns:
        standardized["contrast_raw_table"] = standardized["contrast_raw_table"].map(
            lambda value: normalize_optional_string(value)
        )
    else:
        standardized["contrast_raw_table"] = standardized["contrast_raw_from_filename"]

    standardized["study_id"] = standardized["study_id"].map(normalize_optional_string)
    standardized["patient_id"] = standardized["patient_num"].where(
        standardized["patient_num"].ne(""),
        standardized["patient_id_from_filename"],
    )
    standardized["frame_num"] = standardized["frame_num_table"].where(
        standardized["frame_num_table"].ne(""),
        standardized["frame_num_from_filename"],
    )
    standardized["contrast_raw"] = standardized["contrast_raw_table"].where(
        standardized["contrast_raw_table"].ne(""),
        standardized["contrast_raw_from_filename"],
    )
    standardized["contrast"] = standardized["contrast_raw"].map(map_schisto_contrast_code)
    standardized["patient_key"] = standardized.apply(
        lambda row: make_patient_key(row["study_id"], row["patient_id"]),
        axis=1,
    )
    standardized["pair_key"] = standardized.apply(
        lambda row: make_pair_key(row["study_id"], row["patient_id"], row["frame_num"]),
        axis=1,
    )

    standardized["number_eggs"] = _to_numeric(standardized.get("number_eggs"))
    standardized["patient_eggs"] = _to_numeric(standardized.get("patient_eggs"))
    standardized["quality"] = _to_numeric(standardized.get("quality"))
    standardized["source_file"] = str(source_path)

    standardized = standardized.drop_duplicates(subset=["image_name"], keep="first").reset_index(drop=True)
    return standardized[
        [
            "image_name",
            "study_id",
            "patient_id",
            "patient_key",
            "frame_num",
            "contrast_raw",
            "contrast",
            "pair_key",
            "number_eggs",
            "patient_eggs",
            "quality",
            "source_file",
            "patient_num",
            "frame_num_table",
            "contrast_raw_table",
        ]
    ]


def aggregate_annotation_table(frame: pd.DataFrame, source_path: str | Path) -> pd.DataFrame:
    """Aggregate object annotations by image name and object type counts."""
    standardized = _rename_columns(
        frame.copy(),
        {
            "imagename": "image_name",
            "objecttype": "object_type",
            "xcoord": "x_coord",
            "ycoord": "y_coord",
        },
    )

    if "image_name" not in standardized.columns or "object_type" not in standardized.columns:
        return pd.DataFrame()

    standardized["image_name"] = standardized["image_name"].map(normalize_image_name)
    standardized["object_type"] = standardized["object_type"].map(normalize_optional_string)
    standardized = standardized[
        standardized["image_name"].ne("") & standardized["object_type"].ne("")
    ].copy()
    standardized = standardized.drop_duplicates().reset_index(drop=True)

    if standardized.empty:
        return pd.DataFrame(
            columns=[
                "image_name",
                "annotation_total_count",
                "annotation_s_haematobium_count",
                "annotation_doubtful_count",
                "annotation_object_types",
                "source_file",
            ]
        )

    def build_counts(values: pd.Series) -> tuple[int, int, int, str]:
        counts = values.value_counts().to_dict()
        normalized_counts = {str(key): int(value) for key, value in counts.items()}
        lower_lookup = {str(key).strip().lower(): int(value) for key, value in counts.items()}
        s_haematobium = lower_lookup.get("s.haematobium", 0)
        doubtful = lower_lookup.get("doubtful", 0)
        total = int(sum(normalized_counts.values()))
        return total, s_haematobium, doubtful, json.dumps(normalized_counts, sort_keys=True)

    grouped = (
        standardized.groupby("image_name")["object_type"]
        .apply(build_counts)
        .reset_index(name="counts")
    )
    grouped["annotation_total_count"] = grouped["counts"].map(lambda item: item[0])
    grouped["annotation_s_haematobium_count"] = grouped["counts"].map(lambda item: item[1])
    grouped["annotation_doubtful_count"] = grouped["counts"].map(lambda item: item[2])
    grouped["annotation_object_types"] = grouped["counts"].map(lambda item: item[3])
    grouped["source_file"] = str(source_path)
    return grouped[
        [
            "image_name",
            "annotation_total_count",
            "annotation_s_haematobium_count",
            "annotation_doubtful_count",
            "annotation_object_types",
            "source_file",
        ]
    ].reset_index(drop=True)


def derive_image_label(
    *,
    number_eggs: Any = None,
    annotation_s_haematobium_count: Any = None,
) -> tuple[str, str]:
    """Derive an image-level label from the most direct available signal."""
    if pd.notna(number_eggs):
        return ("positive", "number_eggs") if float(number_eggs) > 0 else ("negative", "number_eggs")

    if pd.notna(annotation_s_haematobium_count) and float(annotation_s_haematobium_count) > 0:
        return "positive", "annotation_s_haematobium_count"

    return "UNKNOWN", "unknown"


def derive_patient_label(
    *,
    patient_eggs: Any = None,
    image_labels: list[str] | None = None,
) -> tuple[str, str]:
    """Derive a patient-level label from patient-wide egg counts or image labels."""
    if pd.notna(patient_eggs):
        return ("positive", "patient_eggs") if float(patient_eggs) > 0 else ("negative", "patient_eggs")

    labels = [label for label in (image_labels or []) if normalize_optional_string(label) and label != "UNKNOWN"]
    if "positive" in labels:
        return "positive", "image_labels"
    if labels and set(labels) == {"negative"}:
        return "negative", "image_labels"

    return "UNKNOWN", "unknown"
