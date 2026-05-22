#!/usr/bin/env python3
"""Cross-study tiled classification experiment.

Root cause identified: full-image resize (4032×3024 → 224×224) shrinks
schistosoma eggs to ~1-2px, forcing the model to learn brightness/background
shortcuts instead of egg morphology.

Fix: cut each image into 30 non-overlapping 640×640 tiles (6 cols × 5 rows),
matching Delahunt et al. (PLOS NTDs 2025). Eggs remain ~50-100px in each tile.
Resize tiles to 224×224 for MobileNetV2 input — eggs are still ~17-35px,
learnable by a convolutional classifier.

Train on nov2021 BF tiles → zero-shot test on mar2020 BF tiles.

Outputs:
  results/cross_study_tiled/results.csv
  results/cross_study_tiled/tiled_vs_baseline.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR   = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from schisto_mobile_ai.data.classification import build_image_transform
from schisto_mobile_ai.utils.io import ensure_dir
from schisto_mobile_ai.utils.reproducibility import resolve_device, seed_everything

CROSS_STUDY_SPLIT = REPO_ROOT / "splits"  / "cross_study_split.csv"
IMAGES_CSV        = REPO_ROOT / "metadata" / "images.csv"
RAW_DIR           = REPO_ROOT / "data"     / "raw"
ANN_MAR = RAW_DIR / "mar2020" / "mar2020_docs" / "mar2020_annotations_08262024.csv"
ANN_NOV = RAW_DIR / "nov2021" / "nov2021_docs" / "nov2021_annotations_08262024.csv"
OUT_DIR  = ensure_dir(REPO_ROOT / "results" / "cross_study_tiled")
RUN_DIR  = ensure_dir(REPO_ROOT / "runs"    / "cross_study_tiled")

TILE_SIZE = 640
TILE_COLS = 6
TILE_ROWS = 5
IMG_W, IMG_H = 4032, 3024
IMG_SIZE  = 224    # model input after resize
EPOCHS    = 15
BATCH     = 32
LR        = 3e-4
WD        = 1e-4
N_BOOT    = 2000
SEED      = 42


# ---------------------------------------------------------------------------
# Tile grid
# ---------------------------------------------------------------------------

COL_STARTS = np.linspace(0, IMG_W - TILE_SIZE, TILE_COLS, dtype=int).tolist()
ROW_STARTS = np.linspace(0, IMG_H - TILE_SIZE, TILE_ROWS, dtype=int).tolist()
N_TILES    = TILE_COLS * TILE_ROWS   # 30


def _tile_positions() -> list[tuple[int, int]]:
    return [(c, r) for r in ROW_STARTS for c in COL_STARTS]


# ---------------------------------------------------------------------------
# Build tile metadata DataFrame
# ---------------------------------------------------------------------------

def build_tile_df(split_name: str) -> pd.DataFrame:
    """Build a per-tile DataFrame for one split (train / val / test)."""
    meta = pd.read_csv(IMAGES_CSV)
    spl  = pd.read_csv(CROSS_STUDY_SPLIT)

    # Which patients are in this split?
    patient_keys = set(spl[spl["split"] == split_name]["patient_key"])

    # BF images only
    imgs = meta[
        meta["patient_key"].isin(patient_keys) &
        (meta["contrast"] == "brightfield")
    ].copy()

    # Load annotations for this split's studies
    study_ids = set(spl[spl["split"] == split_name]["study_id"])
    ann_parts = []
    if "mar2020" in study_ids:
        ann_parts.append(pd.read_csv(ANN_MAR))
    if "nov2021" in study_ids:
        ann_parts.append(pd.read_csv(ANN_NOV))

    ann = pd.concat(ann_parts, ignore_index=True) if ann_parts else pd.DataFrame()

    # Keep only confirmed S.haematobium annotations
    if not ann.empty:
        ann = ann[ann["objectType"] == "S.haematobium"]

    # Build per-tile rows
    positions = _tile_positions()
    rows = []
    for _, img_row in imgs.iterrows():
        img_name = img_row["image_name"]
        patient_key = img_row["patient_key"]
        rel_path    = img_row["relative_path"]

        # Annotations for this image
        if not ann.empty:
            img_ann = ann[ann["imageName"] == img_name]
            ann_xy  = img_ann[["xCoord", "yCoord"]].values
        else:
            ann_xy = np.empty((0, 2), dtype=int)

        for tile_x, tile_y in positions:
            # Tile label: positive if any egg center falls within tile bounds
            if len(ann_xy) > 0:
                in_tile = (
                    (ann_xy[:, 0] >= tile_x) & (ann_xy[:, 0] < tile_x + TILE_SIZE) &
                    (ann_xy[:, 1] >= tile_y) & (ann_xy[:, 1] < tile_y + TILE_SIZE)
                )
                has_egg = int(in_tile.any())
            else:
                has_egg = 0

            rows.append({
                "patient_key": patient_key,
                "image_name":  img_name,
                "relative_path": rel_path,
                "tile_x": tile_x,
                "tile_y": tile_y,
                "has_egg": has_egg,
                "split": split_name,
            })

    df = pd.DataFrame(rows)
    # Patient-level label (for aggregation at eval time)
    pat_label = spl.set_index("patient_key")["patient_label"].map(
        {"positive": 1, "negative": 0}
    )
    df["patient_label"] = df["patient_key"].map(pat_label)
    return df


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TileDataset(Dataset):
    def __init__(self, tile_df: pd.DataFrame, *, train: bool) -> None:
        self.tiles = tile_df.reset_index(drop=True)
        self.transform = build_image_transform(image_size=IMG_SIZE, train=train)

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> dict:
        row = self.tiles.iloc[idx]
        img_path = RAW_DIR / row["relative_path"]
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            x, y = int(row["tile_x"]), int(row["tile_y"])
            tile = img.crop((x, y, x + TILE_SIZE, y + TILE_SIZE))

        return {
            "image":       self.transform(tile),
            "target":      torch.tensor(float(row["has_egg"]),      dtype=torch.float32),
            "pat_target":  torch.tensor(float(row["patient_label"]), dtype=torch.float32),
            "patient_key": str(row["patient_key"]),
        }


# ---------------------------------------------------------------------------
# AUC + bootstrap
# ---------------------------------------------------------------------------

def _wilcoxon_auc(t: np.ndarray, s: np.ndarray) -> float:
    pos = t == 1; neg = ~pos
    pc, nc = int(pos.sum()), int(neg.sum())
    if pc == 0 or nc == 0:
        return float("nan")
    ranks = pd.Series(s).rank(method="average").values
    return float((ranks[pos].sum() - pc * (pc + 1) / 2.0) / (pc * nc))


def bootstrap_auc(t: np.ndarray, s: np.ndarray) -> tuple[float, float, float]:
    pt  = _wilcoxon_auc(t, s)
    rng = np.random.default_rng(SEED)
    n   = len(t)
    boot = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        a = _wilcoxon_auc(t[idx], s[idx])
        if np.isfinite(a):
            boot.append(a)
    if len(boot) < 10:
        return pt, float("nan"), float("nan")
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return pt, float(lo), float(hi)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model() -> nn.Module:
    import torchvision.models as models
    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    m.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(m.classifier[1].in_features, 1))
    return m


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def _train_epoch(model, loader, *, optimizer, criterion, device) -> None:
    model.train()
    for batch in loader:
        imgs = batch["image"].to(device)
        tgt  = batch["target"].to(device)
        optimizer.zero_grad(set_to_none=True)
        criterion(model(imgs).squeeze(1), tgt).backward()
        optimizer.step()


def _patient_auc(model, loader, *, device) -> float:
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            probs = torch.sigmoid(model(batch["image"].to(device)).squeeze(1)).cpu().numpy()
            for pk, pt, p in zip(batch["patient_key"],
                                  batch["pat_target"].numpy(), probs):
                rows.append({"patient_key": pk, "pat_target": float(pt), "prob": float(p)})
    df = pd.DataFrame(rows)
    pat = df.groupby("patient_key").agg(
        target=("pat_target", "max"), score=("prob", "max")
    ).reset_index()
    return _wilcoxon_auc(pat["target"].values, pat["score"].values)


def run_inference(model, loader, *, device) -> pd.DataFrame:
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            probs = torch.sigmoid(model(batch["image"].to(device)).squeeze(1)).cpu().numpy()
            for pk, pt, p in zip(batch["patient_key"],
                                  batch["pat_target"].numpy(), probs):
                rows.append({"patient_key": pk, "pat_target": float(pt), "prob": float(p)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    seed_everything(SEED)
    device = resolve_device("auto")
    print(f"Device:    {device}")
    print(f"Tile size: {TILE_SIZE}×{TILE_SIZE}  grid: {TILE_COLS}×{TILE_ROWS} = {N_TILES} tiles/image")
    print(f"Model input: {IMG_SIZE}×{IMG_SIZE} (resized from tile)\n")

    # Build tile DataFrames
    print("Building tile metadata...")
    train_df = build_tile_df("train")
    val_df   = build_tile_df("val")
    test_df  = build_tile_df("test")

    n_pos_train = int(train_df["has_egg"].sum())
    print(f"  Train tiles: {len(train_df):,}  ({n_pos_train:,} positive = {100*n_pos_train/len(train_df):.1f}%)")
    print(f"  Val tiles:   {len(val_df):,}")
    print(f"  Test tiles:  {len(test_df):,}")
    print()

    train_ds = TileDataset(train_df, train=True)
    val_ds   = TileDataset(val_df,   train=False)
    test_ds  = TileDataset(test_df,  train=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=0)

    model = build_model().to(device)

    # Prior bias init
    pos_rate = float(train_df["has_egg"].mean())
    if 0 < pos_rate < 1:
        bias = float(np.log(pos_rate / (1 - pos_rate)))
        for m in reversed(list(model.modules())):
            if isinstance(m, nn.Linear) and m.out_features == 1:
                with torch.no_grad():
                    m.bias.fill_(bias)
                break

    pos = float(train_df["has_egg"].sum())
    neg = float(len(train_df) - pos)
    pw  = torch.tensor([(neg / pos) ** 0.5], dtype=torch.float32).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR * 0.01)

    best_val, best_state = float("-inf"), None
    for epoch in range(1, EPOCHS + 1):
        _train_epoch(model, train_loader, optimizer=opt, criterion=crit, device=device)
        val_auc = _patient_auc(model, val_loader, device=device)
        sched.step()
        print(f"  epoch {epoch:2d}/{EPOCHS}  val_patient_auc={val_auc:.4f}")
        if np.isfinite(val_auc) and val_auc > best_val:
            best_val = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
        torch.save(best_state, RUN_DIR / "best_model.pt")

    # Test evaluation (mar2020 zero-shot)
    preds = run_inference(model, test_loader, device=device)
    pat   = preds.groupby("patient_key").agg(
        target=("pat_target", "max"), score=("prob", "max")
    ).reset_index()
    pt, lo, hi = bootstrap_auc(pat["target"].values, pat["score"].values)

    print(f"\n=== Tiled MobileNetV2 Results ===")
    print(f"  nov2021 val patient AUC:       {best_val:.4f}")
    print(f"  mar2020 zero-shot patient AUC: {pt:.4f}  [{lo:.4f}, {hi:.4f}]")

    print(f"\n--- Baseline (full-image resize) reference ---")
    print(f"  nov2021 val:  0.8008")
    print(f"  mar2020 test: 0.5920  [0.5246, 0.6628]")

    results = pd.DataFrame([{
        "model": "MobileNetV2 tiled (640×640)",
        "nov2021_val_auc": round(best_val, 4),
        "mar2020_auc": round(pt, 4),
        "mar2020_lo": round(lo, 4),
        "mar2020_hi": round(hi, 4),
    }])
    results.to_csv(OUT_DIR / "results.csv", index=False)
    preds.to_csv(OUT_DIR / "test_predictions.csv", index=False)
    print(f"\nSaved: {OUT_DIR / 'results.csv'}")

    _plot(best_val, pt, lo, hi, OUT_DIR / "tiled_vs_baseline.png")
    print(f"Saved: {OUT_DIR / 'tiled_vs_baseline.png'}")


def _plot(val_auc, test_auc, lo, hi, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    x = [0, 1]
    labels = ["nov2021 val", "mar2020 zero-shot"]

    ax.plot(x, [0.8008, 0.5920], "s--", color="#aaaaaa", linewidth=1.5,
            label="MobileNetV2 baseline (full-image resize)")
    ax.plot(x, [val_auc, test_auc], "o-", color="#2ca02c", linewidth=2.5,
            label="MobileNetV2 tiled (640×640 tiles)")
    ax.errorbar([1], [test_auc],
                yerr=[[test_auc - lo], [hi - test_auc]],
                fmt="none", color="#2ca02c", capsize=6)
    ax.errorbar([1], [0.5920],
                yerr=[[0.5920 - 0.5246], [0.6628 - 0.5920]],
                fmt="none", color="#aaaaaa", capsize=5)

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="Random (AUC=0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Patient-level AUC  (95% bootstrap CI)")
    ax.set_title("Tiling Fix: Does preserving egg scale restore cross-study transfer?\n"
                 "Train on nov2021 → Zero-shot test on mar2020")
    ax.set_ylim(0.3, 1.0)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
