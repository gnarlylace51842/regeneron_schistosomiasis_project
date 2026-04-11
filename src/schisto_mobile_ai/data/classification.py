"""Metadata-driven image classification helpers for single-contrast training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from schisto_mobile_ai.data.manifest import validate_required_columns


CONTRAST_ALIASES = {
    "bf": "brightfield",
    "brightfield": "brightfield",
    "df": "darkfield",
    "darkfield": "darkfield",
}
POSITIVE_LABELS = {"positive", "1", "true", "yes"}
NEGATIVE_LABELS = {"negative", "0", "false", "no"}
DEFAULT_MEAN = (0.5, 0.5, 0.5)
DEFAULT_STD = (0.5, 0.5, 0.5)


@dataclass
class SingleContrastDataBundle:
    """Prepared train/validation frames plus metadata about the chosen label setup."""

    train_frame: pd.DataFrame
    val_frame: pd.DataFrame
    label_column: str
    validation_split_name: str
    metadata: dict[str, Any]


class MetadataImageDataset(Dataset):
    """Simple dataset that reads images from metadata rows and returns tensors plus metadata."""

    def __init__(self, frame: pd.DataFrame, *, image_size: int, train: bool) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.transform = build_image_transform(image_size=image_size, train=train)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        image_path = Path(row["image_path"])
        try:
            with Image.open(image_path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Failed to load image '{image_path}': {exc}") from exc

        return {
            "image": self.transform(image),
            "target": torch.tensor(float(row["target"]), dtype=torch.float32),
            "image_id": str(row["image_id"]),
            "patient_key": str(row["patient_key"]),
            "patient_id": str(row["patient_id"]),
            "study_id": str(row["study_id"]),
            "contrast": str(row["contrast"]),
            "relative_path": str(row["relative_path"]),
            "split": str(row["split"]),
        }


def normalize_contrast_name(value: str) -> str:
    """Map short or long contrast names to the canonical metadata values."""
    normalized = str(value).strip().lower()
    if normalized not in CONTRAST_ALIASES:
        allowed = ", ".join(sorted(CONTRAST_ALIASES))
        raise ValueError(f"Unsupported contrast '{value}'. Choose from: {allowed}")
    return CONTRAST_ALIASES[normalized]


class SimpleImageTransform:
    """Microscopy-physics-aware transform pipeline.

    Training augmentations are grounded in the actual failure modes of field
    microscopy in resource-scarce settings:

    - Random rotation (any angle): schistosoma eggs have no biological orientation;
      field technicians place slides arbitrarily on the stage.
    - Gaussian blur (defocus simulation): cheap field microscopes are frequently
      out of focus. Blur radius drawn from a distribution matching typical
      defocus ranges (sigma 0.5-1.5px at 224px).
    - Uneven illumination (vignette): Köhler illumination is inconsistently
      calibrated in low-resource labs. Simulated by multiplying a radial
      Gaussian falloff with random centre offset.
    - Brightness/contrast jitter: stain batch variability and lamp intensity
      differences across microscope units.
    - Horizontal + vertical flip: orientation-free (as above).
    """

    # D4 test-time augmentation: 4 rotations (0, 90, 180, 270) × 2 flips (none, H-flip)
    # Schistosome eggs are orientation-free → all 8 views are valid clinical reads.
    # We deliberately exclude defocus and vignette from TTA: these are training
    # robustness augmentations that simulate real degradation, not orientation
    # invariances. Averaging in artificially blurred/vignetted views hurts AUC.
    D4_TTA_VIEWS: int = 8

    def __init__(self, *, image_size: int, train: bool, tta_view: int | None = None) -> None:
        if image_size <= 0:
            raise ValueError("image_size must be a positive integer.")
        self.image_size = image_size
        self.train = train
        self.tta_view = tta_view  # None = standard; 0-7 = specific D4 view
        self.mean = np.asarray(DEFAULT_MEAN, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(DEFAULT_STD, dtype=np.float32).reshape(1, 1, 3)
        # Pre-compute vignette base grid once — reused for every call to avoid
        # allocating 224×224 meshgrids per image (major GC pressure over epochs).
        h, w = image_size, image_size
        xs = np.linspace(0, w - 1, w, dtype=np.float32)
        ys = np.linspace(0, h - 1, h, dtype=np.float32)
        self._vx, self._vy = np.meshgrid(xs, ys)   # (H, W) each, float32
        self._vmax_dist = float(np.sqrt((w / 2) ** 2 + (h / 2) ** 2))

    def _simulate_defocus(self, array: np.ndarray) -> np.ndarray:
        """Gaussian blur to simulate out-of-focus field microscope."""
        from PIL import ImageFilter
        sigma = random.uniform(0.4, 1.5)
        img = Image.fromarray((array * 255).clip(0, 255).astype(np.uint8))
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return np.asarray(img, dtype=np.float32) / 255.0

    def _simulate_vignette(self, array: np.ndarray) -> np.ndarray:
        """Radial falloff to simulate uneven Köhler illumination.

        Uses pre-computed grid (self._vx, self._vy) to avoid per-call meshgrid
        allocation which would cause multi-GB GC pressure over training epochs.
        """
        h, w = array.shape[:2]
        cx = w / 2 + random.uniform(-0.15, 0.15) * w
        cy = h / 2 + random.uniform(-0.15, 0.15) * h
        strength = random.uniform(0.3, 0.7)
        dist = np.sqrt((self._vx - cx) ** 2 + (self._vy - cy) ** 2)
        vignette = np.clip(1.0 - strength * (dist / self._vmax_dist), 0.0, 1.0)[:, :, None]
        return np.clip(array * vignette, 0.0, 1.0)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)

        # D4 TTA mode: deterministic orientation transform (no degradation augments)
        if self.tta_view is not None:
            v = self.tta_view % self.D4_TTA_VIEWS
            if v >= 4:          # views 4-7: horizontal flip first
                image = ImageOps.mirror(image)
            rot = (v % 4) * 90  # 0, 90, 180, 270 degrees
            if rot > 0:
                image = image.rotate(rot, resample=Image.Resampling.BILINEAR, expand=False)
            array = np.asarray(image, dtype=np.float32) / 255.0
            if array.ndim == 2:
                array = np.repeat(array[:, :, None], 3, axis=2)
            normalized = (array - self.mean) / self.std
            return torch.from_numpy(np.transpose(normalized, (2, 0, 1))).float()

        if self.train:
            # Full 360° rotation — eggs have no orientation
            if random.random() < 0.8:
                angle = random.uniform(0, 360)
                image = image.rotate(angle, resample=Image.Resampling.BILINEAR,
                                     expand=False, fillcolor=0)
            # Horizontal flip
            if random.random() < 0.5:
                image = ImageOps.mirror(image)
            # Vertical flip
            if random.random() < 0.5:
                image = ImageOps.flip(image)

        array = np.asarray(image, dtype=np.float32) / 255.0
        if array.ndim == 2:
            array = np.repeat(array[:, :, None], 3, axis=2)

        if self.train:
            # Defocus blur — simulates cheap field microscopes
            if random.random() < 0.4:
                array = self._simulate_defocus(array)

            # Uneven illumination — simulates poor Köhler calibration
            if random.random() < 0.35:
                array = self._simulate_vignette(array)

            # Brightness/contrast jitter — stain batch variability
            if random.random() < 0.5:
                array = np.clip(array * random.uniform(0.75, 1.25), 0.0, 1.0)
            if random.random() < 0.4:
                mean = array.mean()
                array = np.clip((array - mean) * random.uniform(0.8, 1.3) + mean, 0.0, 1.0)

        normalized = (array - self.mean) / self.std
        return torch.from_numpy(np.transpose(normalized, (2, 0, 1))).float()


def build_image_transform(
    *,
    image_size: int,
    train: bool,
    tta_view: int | None = None,
) -> SimpleImageTransform:
    """Build a microscopy-physics-aware transform pipeline.

    tta_view=None: standard mode (val=clean, train=augmented).
    tta_view=0..7: deterministic D4 orientation transform for TTA inference
                   (no degradation augments — avoids averaging in noise).
    """
    if image_size <= 0:
        raise ValueError("image_size must be a positive integer.")
    return SimpleImageTransform(image_size=image_size, train=train, tta_view=tta_view)


def _normalize_label_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def _target_from_label(value: Any) -> float | None:
    normalized = _normalize_label_value(value)
    if normalized in POSITIVE_LABELS:
        return 1.0
    if normalized in NEGATIVE_LABELS:
        return 0.0
    return None


def choose_label_column(images: pd.DataFrame, label_source: str = "auto") -> str:
    """Select the training label column based on label_source.

    label_source options:
      "image"   – force image-level 'label' column (egg-detection target; correct
                  for the cross-contrast SSL and triage pipeline).
      "patient" – force patient-level 'patient_level_label' column.
      "auto"    – legacy behaviour: prefer 'patient_level_label', fall back to 'label'.
    """
    if label_source == "image":
        candidates = ["label", "patient_level_label"]
    elif label_source == "patient":
        candidates = ["patient_level_label", "label"]
    elif label_source == "auto":
        candidates = ["patient_level_label", "label"]
    else:
        raise ValueError(f"label_source must be 'image', 'patient', or 'auto', got '{label_source}'")

    for column in candidates:
        if column not in images.columns:
            continue
        mapped = images[column].map(_target_from_label)
        if mapped.notna().any():
            return column
    raise ValueError(
        "Could not find a usable binary label column in images.csv. "
        "Expected 'patient_level_label' or 'label' with positive/negative values."
    )


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
    return sampled.sort_values(["study_id", "patient_key", "relative_path"]).reset_index(drop=True)


def _read_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Required CSV does not exist: {csv_path}")
    return pd.read_csv(csv_path, dtype=str)


def _resolve_validation_split(split_frame: pd.DataFrame) -> str:
    available = set(split_frame["split"].dropna().tolist())
    if "val" in available:
        return "val"
    if "test" in available:
        return "test"
    raise ValueError("The split CSV must contain either a 'val' or 'test' split for validation.")


def load_single_contrast_data(
    *,
    images_csv: str | Path,
    split_csv: str | Path,
    raw_dir: str | Path,
    contrast: str,
    label_source: str = "auto",
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    smoke_test: bool = False,
    seed: int = 42,
) -> SingleContrastDataBundle:
    """Load train/validation frames for one contrast using metadata and patient-safe splits."""
    images = _read_csv(images_csv)
    splits = _read_csv(split_csv)
    validate_required_columns(
        images,
        ["image_id", "study_id", "patient_id", "patient_key", "contrast", "relative_path"],
        table_name="images.csv",
    )
    validate_required_columns(
        splits,
        ["patient_key", "split"],
        table_name="split CSV",
    )

    contrast_name = normalize_contrast_name(contrast)
    label_column = choose_label_column(images, label_source=label_source)
    validation_split_name = _resolve_validation_split(splits)

    merged = images.merge(
        splits[["patient_key", "split"]],
        on="patient_key",
        how="inner",
        validate="many_to_one",
    )
    merged = merged[merged["contrast"].str.lower() == contrast_name].copy()
    merged["target"] = merged[label_column].map(_target_from_label)
    merged = merged[merged["target"].notna()].copy()

    raw_root = Path(raw_dir)
    merged["image_path"] = merged["relative_path"].map(lambda value: str(raw_root / Path(value)))
    merged["file_exists"] = merged["image_path"].map(lambda value: Path(value).exists())

    missing_files = merged[~merged["file_exists"]].copy()
    merged = merged[merged["file_exists"]].copy()
    if merged.empty:
        raise ValueError("No usable image rows remain after filtering by contrast, labels, and existing files.")

    train_frame = merged[merged["split"] == "train"].copy()
    val_frame = merged[merged["split"] == validation_split_name].copy()
    if train_frame.empty:
        raise ValueError("No training rows were found for the requested contrast.")
    if val_frame.empty:
        raise ValueError("No validation rows were found for the requested contrast.")

    if smoke_test:
        if max_train_samples is None:
            max_train_samples = 64
        if max_val_samples is None:
            max_val_samples = 32

    train_frame = _balanced_limit(train_frame, max_samples=max_train_samples, seed=seed)
    val_frame = _balanced_limit(val_frame, max_samples=max_val_samples, seed=seed + 1)

    metadata = {
        "contrast": contrast_name,
        "label_column": label_column,
        "label_source": label_source,
        "validation_split_name": validation_split_name,
        "n_train_images": int(len(train_frame)),
        "n_val_images": int(len(val_frame)),
        "n_train_patients": int(train_frame["patient_key"].nunique()),
        "n_val_patients": int(val_frame["patient_key"].nunique()),
        "train_label_counts": {
            str(int(key)): int(value)
            for key, value in train_frame["target"].value_counts().sort_index().to_dict().items()
        },
        "val_label_counts": {
            str(int(key)): int(value)
            for key, value in val_frame["target"].value_counts().sort_index().to_dict().items()
        },
        "missing_image_rows_dropped": int(len(missing_files)),
    }

    columns = [
        "image_id",
        "patient_key",
        "patient_id",
        "study_id",
        "contrast",
        "relative_path",
        "split",
        "image_path",
        "target",
    ]
    if "frame_num" in merged.columns:
        columns.append("frame_num")
    if "patient_level_label" in merged.columns:
        columns.append("patient_level_label")

    return SingleContrastDataBundle(
        train_frame=train_frame[columns].reset_index(drop=True),
        val_frame=val_frame[columns].reset_index(drop=True),
        label_column=label_column,
        validation_split_name=validation_split_name,
        metadata=metadata,
    )
