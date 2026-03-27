"""Helpers for reading tabular metadata files without assuming a fixed schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


TABULAR_METADATA_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".tsv",
    ".txt",
    ".xls",
    ".xlsx",
}


def is_tabular_metadata_file(path: str | Path) -> bool:
    """Return True when a path looks like a readable table file."""
    return Path(path).suffix.lower() in TABULAR_METADATA_SUFFIXES


def _load_json_table(path: Path) -> pd.DataFrame:
    """Load JSON into a dataframe using a permissive normalization strategy."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        return pd.json_normalize(payload)
    if isinstance(payload, dict):
        return pd.json_normalize(payload)
    return pd.DataFrame({"value": [payload]})


def load_table(path: str | Path, max_rows: int | None = None) -> pd.DataFrame:
    """Load a tabular metadata file based on its suffix."""
    table_path = Path(path)
    if not table_path.exists():
        raise FileNotFoundError(f"Metadata table not found: {table_path}")

    suffix = table_path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(table_path, nrows=max_rows)
    elif suffix == ".tsv":
        frame = pd.read_csv(table_path, sep="\t", nrows=max_rows)
    elif suffix == ".txt":
        frame = pd.read_csv(table_path, sep=None, engine="python", nrows=max_rows)
    elif suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(table_path, nrows=max_rows)
    elif suffix == ".json":
        frame = _load_json_table(table_path)
    elif suffix == ".jsonl":
        frame = pd.read_json(table_path, lines=True, nrows=max_rows)
    else:
        raise ValueError(
            f"Unsupported table format for {table_path}. "
            "Use .csv, .tsv, .txt, .json, .jsonl, .xlsx, or .xls."
        )

    if max_rows is not None and len(frame) > max_rows:
        frame = frame.head(max_rows).copy()

    frame.columns = [str(column) for column in frame.columns]
    return frame


def validate_required_columns(
    frame: pd.DataFrame,
    required_columns: Iterable[str],
    *,
    table_name: str = "metadata table",
) -> None:
    """Raise an error if required columns are missing from a dataframe."""
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns in {table_name}: {missing_text}")


def maybe_limit_rows(frame: pd.DataFrame, subset_size: int | None) -> pd.DataFrame:
    """Return a deterministic head subset for quick debugging runs."""
    if subset_size is None:
        return frame
    if subset_size <= 0:
        raise ValueError("subset_size must be a positive integer when provided.")
    return frame.head(subset_size).copy()


def normalize_optional_string(value: Any) -> str:
    """Convert a scalar value into a clean string or an empty string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if pd.isna(value):
        return ""
    return str(value).strip()


def first_non_empty(values: Iterable[Any]) -> str:
    """Return the first non-empty scalar value from an iterable."""
    for value in values:
        normalized = normalize_optional_string(value)
        if normalized:
            return normalized
    return ""
