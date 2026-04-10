"""Metadata-driven paired-image helpers for always-on BF+DF training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from schisto_mobile_ai.data.classification import build_image_transform
from schisto_mobile_ai.data.manifest import validate_required_columns


POSITIVE_LABELS = {"positive", "1", "true", "yes"}
NEGATIVE_LABELS = {"negative", "0", "false", "no"}


@dataclass
class DualContrastDataBundle:
    """Prepared train/validation/test pair frames plus metadata about the target source."""

    train_frame: pd.DataFrame
    val_frame: pd.DataFrame
    test_frame: pd.DataFrame
    label_column: str
    validation_split_name: str
    metadata: dict[str, Any]


class PairedContrastDataset(Dataset):
    """Simple dataset that loads brightfield/darkfield pairs from metadata rows."""

    def __init__(self, frame: pd.DataFrame, *, image_size: int, train: bool) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.transform = build_image_transform(image_size=image_size, train=train)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        brightfield_path = Path(row["brightfield_path"])
        darkfield_path = Path(row["darkfield_path"])
        brightfield_image = _load_rgb_image(brightfield_path)
        darkfield_image = _load_rgb_image(darkfield_path)

        return {
            "brightfield_image": self.transform(brightfield_image),
            "darkfield_image": self.transform(darkfield_image),
            "target": torch.tensor(float(row["target"]), dtype=torch.float32),
            "image_id": str(row["pair_id"]),
            "pair_id": str(row["pair_id"]),
            "pair_key": str(row["pair_key"]),
            "patient_key": str(row["patient_key"]),
            "patient_id": str(row["patient_id"]),
            "study_id": str(row["study_id"]),
            "split": str(row["split"]),
            "brightfield_relative_path": str(row["brightfield_relative_path"]),
            "darkfield_relative_path": str(row["darkfield_relative_path"]),
        }


def _load_rgb_image(path: Path) -> Image.Image:
    try:
        with Image.open(path) as image:
            return ImageOps.exif_transpose(image).convert("RGB")
    except Exception as exc:
        raise RuntimeError(f"Failed to load image '{path}': {exc}") from exc


def _read_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Required CSV does not exist: {csv_path}")
    return pd.read_csv(csv_path, dtype=str)


def _label_to_target(value: Any) -> float | None:
    if pd.isna(value):
        return None
    normalized = str(value).strip().lower()
    if normalized in POSITIVE_LABELS:
        return 1.0
    if normalized in NEGATIVE_LABELS:
        return 0.0
    return None


def _resolve_validation_split(split_frame: pd.DataFrame) -> str:
    available = set(split_frame["split"].dropna().tolist())
    if "val" in available:
        return "val"
    if "test" in available:
        return "test"
    raise ValueError("The split CSV must contain either a 'val' or 'test' split for validation.")


def _balanced_limit(frame: pd.DataFrame, *, max_samples: int | None, seed: int) -> pd.DataFrame:
    if max_samples is None or len(frame) <= max_samples:
        return frame.copy()
    if max_samples <= 0:
        raise ValueError("Sample limits must be positive integers when provided.")

    fractions = frame["target"].value_counts(normalize=True).to_dict()
    sampled_parts: list[pd.DataFrame] = []
    remaining = max_samples
    class_targets = sorted(frame["target"].dropna().unique().tolist())
    for index, class_value in enumerate(class_targets):
        class_frame = frame[frame["target"] == class_value].copy()
        if index == len(class_targets) - 1:
            class_limit = min(len(class_frame), remaining)
        else:
            class_limit = max(1, int(round(max_samples * fractions.get(class_value, 0.0))))
            class_limit = min(len(class_frame), class_limit)
        remaining -= class_limit
        sampled_parts.append(class_frame.sample(n=class_limit, random_state=seed + index))

    sampled = pd.concat(sampled_parts, ignore_index=True)
    if len(sampled) > max_samples:
        sampled = sampled.sample(n=max_samples, random_state=seed)
    return sampled.sort_values(["study_id", "patient_key", "pair_key"]).reset_index(drop=True)


def load_dual_contrast_data(
    *,
    pairs_csv: str | Path,
    patients_csv: str | Path,
    split_csv: str | Path,
    raw_dir: str | Path,
    label_source: str = "auto",
    smoke_test: bool = False,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    seed: int = 42,
) -> DualContrastDataBundle:
    """Load train/validation pair frames for always-on dual-contrast training.

    label_source controls which label column is used as the training target:
      "image"   – use pair-level 'label' (egg-detection; correct for triage pipeline).
      "patient" – use 'patient_level_label' (patient diagnosis).
      "auto"    – legacy: prefer 'patient_level_label', fall back to 'label'.
    """
    pairs = _read_csv(pairs_csv)
    patients = _read_csv(patients_csv)
    splits = _read_csv(split_csv)

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
        ["patient_key", "patient_label"],
        table_name="patients.csv",
    )
    validate_required_columns(
        splits,
        ["patient_key", "split"],
        table_name="split CSV",
    )

    pairs = pairs[pairs["pair_status"] == "complete"].copy()
    if pairs.empty:
        raise ValueError("pairs.csv does not contain any complete BF/DF pairs.")

    patients = patients[["patient_key", "patient_label"]].drop_duplicates(subset=["patient_key"]).copy()
    pairs = pairs.merge(patients, on="patient_key", how="left", validate="many_to_one")
    pairs = pairs.merge(splits[["patient_key", "split"]], on="patient_key", how="inner", validate="many_to_one")

    if label_source == "image":
        preferred_cols = ["label", "patient_level_label", "patient_label"]
    elif label_source == "patient":
        preferred_cols = ["patient_level_label", "patient_label", "label"]
    else:  # auto – legacy behaviour
        preferred_cols = ["patient_level_label", "patient_label", "label"]

    label_column: str | None = None
    for col in preferred_cols:
        if col in pairs.columns:
            candidate_targets = pairs[col].map(_label_to_target)
            if candidate_targets.notna().any():
                label_column = col
                break
    if label_column is None:
        raise ValueError("No usable label column found in pairs.csv. Expected 'label', 'patient_level_label', or 'patient_label'.")

    pairs["target"] = pairs[label_column].map(_label_to_target)
    # Fallback to patient_label when patient_level_label was chosen but has NaN rows
    if label_column == "patient_level_label" and pairs["target"].isna().any() and "patient_label" in pairs.columns:
        fallback_mask = pairs["target"].isna()
        pairs.loc[fallback_mask, "target"] = pairs.loc[fallback_mask, "patient_label"].map(_label_to_target)

    pairs = pairs[pairs["target"].notna()].copy()
    if pairs.empty:
        raise ValueError("No usable target labels were found for complete BF/DF pairs.")

    raw_root = Path(raw_dir)
    pairs["brightfield_path"] = pairs["brightfield_relative_path"].map(lambda value: str(raw_root / Path(value)))
    pairs["darkfield_path"] = pairs["darkfield_relative_path"].map(lambda value: str(raw_root / Path(value)))
    pairs["brightfield_exists"] = pairs["brightfield_path"].map(lambda value: Path(value).exists())
    pairs["darkfield_exists"] = pairs["darkfield_path"].map(lambda value: Path(value).exists())

    missing_files = pairs[~pairs["brightfield_exists"] | ~pairs["darkfield_exists"]].copy()
    pairs = pairs[pairs["brightfield_exists"] & pairs["darkfield_exists"]].copy()
    if pairs.empty:
        raise ValueError("No complete BF/DF pairs remain after checking image file existence.")

    validation_split_name = _resolve_validation_split(splits)
    train_frame = pairs[pairs["split"] == "train"].copy()
    val_frame = pairs[pairs["split"] == validation_split_name].copy()
    # Test split: separate held-out set (distinct from validation)
    test_split_name = "test" if validation_split_name == "val" else "val"
    test_frame = pairs[pairs["split"] == test_split_name].copy()
    if train_frame.empty:
        raise ValueError("No training pairs were found after applying the split file.")
    if val_frame.empty:
        raise ValueError("No validation pairs were found after applying the split file.")

    if smoke_test:
        if max_train_samples is None:
            max_train_samples = 48
        if max_val_samples is None:
            max_val_samples = 24

    train_frame = _balanced_limit(train_frame, max_samples=max_train_samples, seed=seed)
    val_frame = _balanced_limit(val_frame, max_samples=max_val_samples, seed=seed + 1)
    # test_frame is never balanced/limited — always full held-out set

    metadata = {
        "label_column": label_column,
        "label_source": label_source,
        "validation_split_name": validation_split_name,
        "test_split_name": test_split_name,
        "n_train_pairs": int(len(train_frame)),
        "n_val_pairs": int(len(val_frame)),
        "n_test_pairs": int(len(test_frame)),
        "n_train_patients": int(train_frame["patient_key"].nunique()),
        "n_val_patients": int(val_frame["patient_key"].nunique()),
        "n_test_patients": int(test_frame["patient_key"].nunique()),
        "train_label_counts": {
            str(int(key)): int(value)
            for key, value in train_frame["target"].value_counts().sort_index().to_dict().items()
        },
        "val_label_counts": {
            str(int(key)): int(value)
            for key, value in val_frame["target"].value_counts().sort_index().to_dict().items()
        },
        "test_label_counts": {
            str(int(key)): int(value)
            for key, value in test_frame["target"].value_counts().sort_index().to_dict().items()
        } if not test_frame.empty else {},
        "missing_pair_rows_dropped": int(len(missing_files)),
    }

    columns = [
        "pair_id",
        "pair_key",
        "patient_key",
        "patient_id",
        "study_id",
        "split",
        "brightfield_relative_path",
        "darkfield_relative_path",
        "brightfield_path",
        "darkfield_path",
        "target",
    ]
    if "frame_num" in pairs.columns:
        columns.append("frame_num")

    test_cols = [c for c in columns if c in test_frame.columns]
    return DualContrastDataBundle(
        train_frame=train_frame[columns].reset_index(drop=True),
        val_frame=val_frame[columns].reset_index(drop=True),
        test_frame=test_frame[test_cols].reset_index(drop=True) if not test_frame.empty else pd.DataFrame(columns=columns),
        label_column=label_column,
        validation_split_name=validation_split_name,
        metadata=metadata,
    )
