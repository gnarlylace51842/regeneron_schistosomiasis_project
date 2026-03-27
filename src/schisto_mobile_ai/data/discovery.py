"""Filesystem and metadata-discovery helpers for a flexible dataset audit pipeline."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd

from schisto_mobile_ai.data.manifest import TABULAR_METADATA_SUFFIXES, load_table, normalize_optional_string


IMAGE_SUFFIXES = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
METADATA_SUFFIXES = TABULAR_METADATA_SUFFIXES | {".xml", ".yaml", ".yml"}

BRIGHTFIELD_TOKENS = {"bf", "bright", "brightfield"}
DARKFIELD_TOKENS = {"dark", "darkfield", "df"}

ROLE_PATTERNS = {
    "image": [
        "image_path",
        "filepath",
        "file_path",
        "filename",
        "file_name",
        "image",
        "img",
        "path",
        "file",
        "name",
        "photo",
        "slide",
    ],
    "patient": [
        "patient_id",
        "patient",
        "subject_id",
        "subject",
        "participant_id",
        "participant",
        "case_id",
        "case",
        "person_id",
    ],
    "study": [
        "study_id",
        "study",
        "cohort",
        "site",
        "project",
        "trial",
    ],
    "label": [
        "label",
        "class",
        "target",
        "diagnosis",
        "status",
        "infection",
        "result",
        "outcome",
        "egg_count",
        "schisto",
    ],
    "pair": [
        "pair_id",
        "pair",
        "sample_id",
        "sample",
        "specimen_id",
        "specimen",
        "field_id",
        "field",
        "view_id",
        "view",
        "capture_id",
        "capture",
    ],
    "contrast": [
        "contrast",
        "illumination",
        "modality",
        "channel",
        "view_type",
        "image_type",
    ],
}

PATH_PATTERNS = {
    "patient": [
        re.compile(r"(?:patient|subject|participant|case|person|child|individual)[\W_]*([a-z0-9]+)"),
        re.compile(r"(?:pt)[\W_]*([a-z0-9]+)"),
    ],
    "study": [
        re.compile(r"(?:study|cohort|site|project|trial)[\W_]*([a-z0-9]+)"),
    ],
}


def iter_dataset_files(root_dir: str | Path) -> list[Path]:
    """Return all files beneath a root directory in deterministic order."""
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    return sorted(path for path in root.rglob("*") if path.is_file())


def classify_file(path: str | Path) -> str:
    """Classify a file as image, metadata, or other using its suffix."""
    suffix = Path(path).suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in METADATA_SUFFIXES:
        return "metadata"
    return "other"


def stable_id(prefix: str, text: str) -> str:
    """Build a short deterministic identifier from a string."""
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def normalize_text(value: Any) -> str:
    """Normalize text for matching and tokenization."""
    text = normalize_optional_string(value).lower()
    return text.replace("\\", "/")


def tokenize(value: Any) -> list[str]:
    """Split text into lowercase alphanumeric tokens."""
    text = normalize_text(value)
    return [token for token in re.split(r"[^a-z0-9]+", text) if token]


def normalize_identifier(value: Any) -> str:
    """Normalize a value into a snake-case-like identifier string."""
    return "_".join(tokenize(value))


def infer_contrast_from_text(value: Any) -> tuple[str, str, str]:
    """Infer brightfield or darkfield from file paths or metadata values."""
    text = normalize_text(value)
    tokens = tokenize(text)

    bf_hits: list[str] = []
    df_hits: list[str] = []

    if "brightfield" in text or "bright-field" in text or "bright_field" in text:
        bf_hits.append("brightfield")
    if "darkfield" in text or "dark-field" in text or "dark_field" in text:
        df_hits.append("darkfield")

    bf_hits.extend(token for token in tokens if token in BRIGHTFIELD_TOKENS)
    df_hits.extend(token for token in tokens if token in DARKFIELD_TOKENS)

    if bf_hits and not df_hits:
        return "brightfield", "token", bf_hits[0]
    if df_hits and not bf_hits:
        return "darkfield", "token", df_hits[0]
    if bf_hits and df_hits:
        return "unknown", "ambiguous", ",".join(sorted(set(bf_hits + df_hits)))
    return "unknown", "unknown", ""


def _strip_contrast_tokens(part: str) -> str:
    tokens = tokenize(part)
    kept = [token for token in tokens if token not in BRIGHTFIELD_TOKENS | DARKFIELD_TOKENS]
    return "_".join(kept)


def infer_pair_key(relative_path: str | Path) -> str:
    """Infer a contrast-agnostic pairing key from the relative path."""
    path = Path(relative_path)
    parts: list[str] = []
    for part in path.with_suffix("").parts:
        cleaned = _strip_contrast_tokens(part)
        if cleaned:
            parts.append(cleaned)

    if not parts:
        parts.append(normalize_identifier(path.stem))

    return "/".join(parts)


def infer_path_identifier(relative_path: str | Path, *, kind: str) -> tuple[str, str]:
    """Infer likely patient or study identifiers from path parts when possible."""
    if kind not in PATH_PATTERNS:
        raise ValueError(f"Unsupported identifier kind: {kind}")

    path = Path(relative_path)
    for part in reversed(path.parts):
        normalized_part = normalize_text(part)
        for pattern in PATH_PATTERNS[kind]:
            match = pattern.search(normalized_part)
            if match:
                return normalize_identifier(match.group(1)), f"path:{kind}"
    return "", "unknown"


def build_image_match_keys(value: Any) -> list[str]:
    """Create several normalized keys for matching metadata rows to image files."""
    text = normalize_text(value)
    if not text:
        return []

    path = PurePosixPath(text)
    keys = [
        text,
        path.name,
    ]

    if len(path.parts) >= 2:
        keys.append("/".join(path.parts[-2:]))

    stem = path.name.rsplit(".", maxsplit=1)[0]
    if stem:
        keys.append(stem)

    seen: set[str] = set()
    unique_keys: list[str] = []
    for key in keys:
        if key and key not in seen:
            unique_keys.append(key)
            seen.add(key)
    return unique_keys


def rank_columns(columns: list[str], patterns: list[str]) -> list[str]:
    """Rank columns by how strongly their names resemble target patterns."""
    scored: list[tuple[int, str]] = []
    for column in columns:
        normalized = normalize_identifier(column)
        best_score = 0
        for rank, pattern in enumerate(patterns):
            pattern_norm = normalize_identifier(pattern)
            if not pattern_norm:
                continue
            if normalized == pattern_norm:
                best_score = max(best_score, 100 - rank)
            elif normalized.startswith(pattern_norm + "_") or normalized.endswith("_" + pattern_norm):
                best_score = max(best_score, 80 - rank)
            elif pattern_norm in normalized:
                best_score = max(best_score, 60 - rank)
        if best_score > 0:
            scored.append((best_score, column))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [column for _, column in scored]


def detect_table_columns(frame: pd.DataFrame) -> dict[str, list[str]]:
    """Return candidate columns for image references, IDs, labels, and contrast."""
    columns = [str(column) for column in frame.columns]
    return {
        role: rank_columns(columns, patterns)
        for role, patterns in ROLE_PATTERNS.items()
    }


def rank_image_reference_columns(
    frame: pd.DataFrame,
    candidate_columns: list[str],
    image_lookup: dict[str, list[int]],
    *,
    sample_rows: int = 200,
) -> list[str]:
    """Prefer columns that actually match observed image paths or names."""
    if not candidate_columns:
        return []

    preview = frame.head(sample_rows)
    scored: list[tuple[int, str]] = []
    for column in candidate_columns:
        matches = 0
        for value in preview[column]:
            keys = build_image_match_keys(value)
            if any(key in image_lookup for key in keys):
                matches += 1
        if matches > 0:
            scored.append((matches, column))

    if not scored:
        return candidate_columns[:3]

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [column for _, column in scored]


def try_load_metadata_table(
    path: str | Path,
    *,
    max_rows: int | None = None,
) -> tuple[pd.DataFrame | None, str]:
    """Attempt to load a metadata table and return an error string on failure."""
    try:
        frame = load_table(path, max_rows=max_rows)
    except Exception as exc:
        return None, str(exc)
    return frame, ""
