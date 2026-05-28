#!/usr/bin/env python3
"""Cross-contrast self-supervised pre-training (BF↔DF alignment).

Each FOV in the SchistoScope dataset has a paired brightfield (BF) and
darkfield (DF) image. The biology is identical between the two — same eggs,
same debris, same patient — but the contrast mechanism is fundamentally
different (light absorption vs. light scattering).

We treat each BF↔DF pair as two "views" of the same underlying biological
signal and pre-train an encoder with a SimCLR-style InfoNCE loss to align
their representations. We do this on TILES (not full images), using the same
6×5 grid as the detection task. Importantly, this includes unlabeled tiles
from the training set only — the test patients (nov2021) are never touched
during pre-training, so we do not leak holdout information.

The resulting encoder weights are loaded as the YOLOv8 backbone for the
downstream detection fine-tune. The hypothesis: contrast-invariant
representations will transfer better across studies (different operators,
slightly different illumination, etc.).

Usage:
    python scripts/cross_contrast_ssl_pretrain.py --epochs 50 --batch 128
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR   = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from schisto_mobile_ai.utils.io import ensure_dir
from schisto_mobile_ai.utils.reproducibility import resolve_device, seed_everything

PAIRS_CSV = REPO_ROOT / "metadata" / "pairs.csv"
SPLIT_CSV = REPO_ROOT / "splits"   / "mar_to_nov_split.csv"
RAW_DIR   = REPO_ROOT / "data"     / "raw"
OUT_DIR   = ensure_dir(REPO_ROOT / "runs" / "cross_contrast_ssl")

TILE_SIZE   = 640
TILE_COLS   = 6
TILE_ROWS   = 5
IMG_W, IMG_H = 4032, 3024
INPUT_SIZE  = 224
PROJ_DIM    = 128
SEED        = 42

COL_STARTS = np.linspace(0, IMG_W - TILE_SIZE, TILE_COLS, dtype=int).tolist()
ROW_STARTS = np.linspace(0, IMG_H - TILE_SIZE, TILE_ROWS, dtype=int).tolist()


def _tile_positions() -> list[tuple[int, int]]:
    return [(c, r) for r in ROW_STARTS for c in COL_STARTS]


class PairedTileDataset(Dataset):
    """Yields (bf_tile, df_tile) pairs from the same FOV + tile location.

    Only includes patients in the SSL training set (mar2020 train + val split).
    The nov2021 test set is never seen during pre-training.
    """

    def __init__(self, pairs_csv: Path, split_csv: Path) -> None:
        pairs = pd.read_csv(pairs_csv)
        split = pd.read_csv(split_csv)

        # Use only train+val patients (no test leakage)
        ssl_keys = set(split[split["split"].isin(["train", "val"])]["patient_key"])
        pairs = pairs[
            (pairs["patient_key"].isin(ssl_keys)) &
            (pairs["pair_status"] == "complete")
        ].reset_index(drop=True)

        positions = _tile_positions()
        rows = []
        for _, p in pairs.iterrows():
            for ti, (tx, ty) in enumerate(positions):
                rows.append({
                    "bf_path": p["brightfield_relative_path"],
                    "df_path": p["darkfield_relative_path"],
                    "tile_x":  tx,
                    "tile_y":  ty,
                    "tile_id": ti,
                })
        self.records = rows

        self.tf_train = transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        print(f"Paired tiles for SSL: {len(self.records):,}")

    def __len__(self) -> int:
        return len(self.records)

    def _crop_tile(self, rel_path: str, tx: int, ty: int) -> Image.Image:
        path = RAW_DIR / rel_path
        with Image.open(path) as im:
            im = im.convert("RGB")
            return im.crop((tx, ty, tx + TILE_SIZE, ty + TILE_SIZE))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        r = self.records[idx]
        bf = self._crop_tile(r["bf_path"], r["tile_x"], r["tile_y"])
        df = self._crop_tile(r["df_path"], r["tile_x"], r["tile_y"])
        return self.tf_train(bf), self.tf_train(df)


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = PROJ_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


def build_encoder() -> tuple[nn.Module, int]:
    """ResNet-50 backbone, ImageNet pre-init, classifier stripped.

    Returns (encoder, feature_dim). We use ResNet-50 because (a) ultralytics
    YOLO model surgery is messy, and (b) we can transfer ResNet features to a
    custom detector later. For a v1 SSL experiment, a clean ResNet baseline is
    easier to reason about than fighting with YOLO's backbone export.
    """
    import torchvision.models as models
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    feat_dim = m.fc.in_features
    m.fc = nn.Identity()
    return m, feat_dim


def info_nce_loss(z_bf: torch.Tensor, z_df: torch.Tensor, temp: float) -> torch.Tensor:
    """Symmetric InfoNCE (SimCLR-style) over a batch of 2N samples.

    Each BF tile's positive is its paired DF tile; negatives are everything else.
    """
    n = z_bf.size(0)
    z = torch.cat([z_bf, z_df], dim=0)                  # (2N, D)
    sim = z @ z.t() / temp                              # (2N, 2N)
    mask = torch.eye(2 * n, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float("-inf"))
    targets = torch.arange(2 * n, device=z.device)
    targets = (targets + n) % (2 * n)                   # positive index
    return F.cross_entropy(sim, targets)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch",  type=int, default=128)
    ap.add_argument("--lr",     type=float, default=1e-3)
    ap.add_argument("--wd",     type=float, default=1e-4)
    ap.add_argument("--temp",   type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    seed_everything(SEED)
    device = resolve_device("auto")
    print(f"Device: {device}")

    ds = PairedTileDataset(PAIRS_CSV, SPLIT_CSV)
    loader = DataLoader(
        ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
    )

    encoder, feat_dim = build_encoder()
    proj = ProjectionHead(feat_dim)
    encoder = encoder.to(device)
    proj    = proj.to(device)

    opt = torch.optim.AdamW(
        list(encoder.parameters()) + list(proj.parameters()),
        lr=args.lr, weight_decay=args.wd,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * len(loader)
    )

    best_loss = float("inf")
    for ep in range(1, args.epochs + 1):
        encoder.train(); proj.train()
        total, count = 0.0, 0
        for bf, df in loader:
            bf, df = bf.to(device, non_blocking=True), df.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            z_bf = proj(encoder(bf))
            z_df = proj(encoder(df))
            loss = info_nce_loss(z_bf, z_df, args.temp)
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss.item()) * bf.size(0)
            count += bf.size(0)
        avg = total / count
        print(f"  epoch {ep:2d}/{args.epochs}  loss={avg:.4f}")
        if avg < best_loss:
            best_loss = avg
            torch.save({
                "encoder": encoder.state_dict(),
                "proj":    proj.state_dict(),
                "epoch":   ep,
                "loss":    avg,
            }, OUT_DIR / "best_encoder.pt")

    torch.save({
        "encoder": encoder.state_dict(),
        "proj":    proj.state_dict(),
        "epoch":   args.epochs,
        "loss":    avg,
    }, OUT_DIR / "final_encoder.pt")
    print(f"\nSaved: {OUT_DIR / 'best_encoder.pt'}")
    print(f"Saved: {OUT_DIR / 'final_encoder.pt'}")


if __name__ == "__main__":
    main()
