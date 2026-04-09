"""Unlabelled BF/DF pair dataset for cross-contrast SSL pre-training.

For pre-training we need pairs, not labels. We use only train-split patients
to maintain strict separation: the val/test patients' images never enter the
pre-training representation, even though SSL uses no labels. This is the
conservative, reviewer-safe choice.

Augmentation strategy for SSL pre-training:
    Each image gets a mild stochastic augmentation BEFORE being passed through
    the encoder. This prevents the encoder from using trivial pixel-level cues
    to identify the matching view and forces it to learn structural features.

    We keep augmentation mild (no aggressive colour jitter or grayscale) because:
    1. BF/DF already differ substantially in appearance — the cross-contrast
       signal is strong without needing heavy within-contrast augmentation.
    2. Egg morphology is our target feature; aggressive distortion could destroy
       the very signal we want the encoder to learn.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

from schisto_mobile_ai.data.manifest import validate_required_columns


# Mild augmentation for SSL: flip + small rotation only. No colour jitter.
# Colour jitter would destroy the BF/DF contrast difference that IS the signal.
_SSL_AUG_FLIP_PROB = 0.5
_SSL_AUG_VFLIP_PROB = 0.2


@dataclass
class SSLPairBundle:
    """Pre-training pair frames plus dataset statistics."""

    train_frame: pd.DataFrame
    metadata: dict[str, Any]


class CrossContrastPairDataset(Dataset):
    """Dataset of (augmented_BF, augmented_DF) pairs for contrastive pre-training.

    No labels are loaded. Each __getitem__ returns two mildly augmented views
    of the same slide under different illumination — the natural positive pair.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        image_size: int = 224,
    ) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.frame.iloc[index]
        bf_image = _load_and_augment(Path(row["brightfield_path"]), size=self.image_size)
        df_image = _load_and_augment(Path(row["darkfield_path"]), size=self.image_size)
        return {
            "brightfield": bf_image,
            "darkfield": df_image,
            "pair_key": str(row["pair_key"]),
            "patient_key": str(row["patient_key"]),
        }


def _load_and_augment(path: Path, *, size: int) -> torch.Tensor:
    """Load an image, apply mild stochastic augmentation, return normalised tensor."""
    try:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception as exc:
        raise RuntimeError(f"Failed to load image '{path}': {exc}") from exc

    img = img.resize((size, size), Image.Resampling.BILINEAR)

    if random.random() < _SSL_AUG_FLIP_PROB:
        img = ImageOps.mirror(img)
    if random.random() < _SSL_AUG_VFLIP_PROB:
        img = ImageOps.flip(img)

    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, np.newaxis], 3, axis=2)
    # Normalise to [-1, 1] (same as supervised training transforms)
    arr = (arr - 0.5) / 0.5
    return torch.from_numpy(np.transpose(arr, (2, 0, 1))).float()


def _read_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Required CSV does not exist: {csv_path}")
    return pd.read_csv(csv_path, dtype=str)


def load_ssl_pairs(
    *,
    pairs_csv: str | Path,
    split_csv: str | Path,
    raw_dir: str | Path,
    splits_to_include: tuple[str, ...] = ("train",),
    smoke_test: bool = False,
    seed: int = 42,
) -> SSLPairBundle:
    """Load BF/DF pairs for SSL pre-training.

    Args:
        pairs_csv: Path to metadata/pairs.csv.
        split_csv: Path to a patient-safe split CSV.
        raw_dir: Root directory for resolving relative image paths.
        splits_to_include: Which patient splits to include. Default is ('train',)
            to maintain strict val/test separation even though SSL uses no labels.
        smoke_test: If True, cap at 48 pairs for a quick end-to-end check.
        seed: Random seed for smoke-test sampling.

    Returns:
        SSLPairBundle with a train_frame of complete, file-verified pairs.
    """
    pairs = _read_csv(pairs_csv)
    splits = _read_csv(split_csv)

    validate_required_columns(
        pairs,
        [
            "pair_key", "patient_key", "study_id", "patient_id",
            "pair_status", "brightfield_relative_path", "darkfield_relative_path",
        ],
        table_name="pairs.csv",
    )
    validate_required_columns(splits, ["patient_key", "split"], table_name="split CSV")

    # Only complete pairs
    pairs = pairs[pairs["pair_status"] == "complete"].copy()
    if pairs.empty:
        raise ValueError("No complete BF/DF pairs found in pairs.csv.")

    # Join splits and filter to requested splits
    pairs = pairs.merge(
        splits[["patient_key", "split"]].drop_duplicates("patient_key"),
        on="patient_key",
        how="inner",
        validate="many_to_one",
    )
    pairs = pairs[pairs["split"].isin(splits_to_include)].copy()
    if pairs.empty:
        raise ValueError(f"No pairs found in splits {splits_to_include}.")

    # Resolve file paths and verify existence
    raw_root = Path(raw_dir)
    pairs["brightfield_path"] = pairs["brightfield_relative_path"].map(
        lambda v: str(raw_root / Path(v))
    )
    pairs["darkfield_path"] = pairs["darkfield_relative_path"].map(
        lambda v: str(raw_root / Path(v))
    )
    pairs["bf_exists"] = pairs["brightfield_path"].map(lambda v: Path(v).exists())
    pairs["df_exists"] = pairs["darkfield_path"].map(lambda v: Path(v).exists())
    missing = pairs[~pairs["bf_exists"] | ~pairs["df_exists"]]
    pairs = pairs[pairs["bf_exists"] & pairs["df_exists"]].copy()
    if pairs.empty:
        raise ValueError("No complete pairs remain after checking file existence.")

    if smoke_test:
        pairs = pairs.sample(n=min(48, len(pairs)), random_state=seed).reset_index(drop=True)

    columns = [
        "pair_key", "patient_key", "patient_id", "study_id", "split",
        "brightfield_relative_path", "darkfield_relative_path",
        "brightfield_path", "darkfield_path",
    ]

    return SSLPairBundle(
        train_frame=pairs[columns].reset_index(drop=True),
        metadata={
            "n_pairs": int(len(pairs)),
            "n_patients": int(pairs["patient_key"].nunique()),
            "splits_included": list(splits_to_include),
            "n_missing_dropped": int(len(missing)),
            "studies": sorted(pairs["study_id"].dropna().unique().tolist()),
        },
    )


def build_ssl_loader(
    bundle: SSLPairBundle,
    *,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
) -> DataLoader:
    """Build a DataLoader for SSL pre-training."""
    dataset = CrossContrastPairDataset(bundle.train_frame, image_size=image_size)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,  # NT-Xent needs consistent batch size
        generator=generator,
    )
