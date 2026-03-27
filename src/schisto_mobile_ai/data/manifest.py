"""Generic helpers for metadata tables without assuming a specific schema."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def load_table(path: str | Path) -> pd.DataFrame:
    """Load a tabular metadata file based on its suffix."""
    table_path = Path(path)
    if not table_path.exists():
        raise FileNotFoundError(f"Metadata table not found: {table_path}")

    suffix = table_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(table_path)
    if suffix == ".tsv":
        return pd.read_csv(table_path, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(table_path)
    if suffix == ".jsonl":
        return pd.read_json(table_path, lines=True)

    raise ValueError(
        f"Unsupported table format for {table_path}. "
        "Use .csv, .tsv, .parquet, or .jsonl."
    )


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

