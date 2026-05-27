#!/usr/bin/env python3
"""Mar2020 → Nov2021 tiled classification experiment.

Matches de Leon Derby et al. (2025) direction: train on mar2020, test on nov2021.
Allows direct comparison of tiled MobileNetV2 against their reported results.

Outputs:
  results/mar_to_nov_tiled/results.csv
  results/mar_to_nov_tiled/predictions.csv
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

SPLIT_CSV = REPO_ROOT / "splits"  / "mar_to_nov_split.csv"
IMAGES_CSV = REPO_ROOT / "metadata" / "images.csv"
RAW_DIR    = REPO_ROOT / "data"     / "raw"
ANN_MAR = RAW_DIR / "mar2020" / "mar2020_docs" / "mar2020_annotations_08262024.csv"
ANN_NOV = RAW_DIR / "nov2021" / "nov2021_docs" / "nov2021_annotations_08262024.csv"
OUT_DIR  = ensure_dir(REPO_ROOT / "results" / "mar_to_nov_tiled")
RUN_DIR  = ensure_dir(REPO_ROOT / "runs"    / "mar_to_nov_tiled")

TILE_SIZE = 640
TILE_COLS = 6
TILE_ROWS = 5
IMG_W, IMG_H = 4032, 3024
IMG_SIZE  = 224
EPOCHS    = 15
BATCH     = 32
LR        = 3e-4
WD        = 1e-4
N_BOOT    = 2000
SEED      = 42

COL_STARTS = np.linspace(0, IMG_W - TILE_SIZE, TILE_COLS, dtype=int).tolist()
ROW_STARTS = np.linspace(0, IMG_H - TILE_SIZE, TILE_ROWS, dtype=int).tolist()
N_TILES    = TILE_COLS * TILE_ROWS


def _tile_positions() -> list[tuple[int, int]]:
    return [(c, r) for r in ROW_STARTS for c in COL_STARTS]


def build_tile_df(split_name: str) -> pd.DataFrame:
    meta = pd.read_csv(IMAGES_CSV)
    spl  = pd.read_csv(SPLIT_CSV)

    patient_keys = set(spl[spl["split"] == split_name]["patient_key"])
    imgs = meta[
        meta["patient_key"].isin(patient_keys) &
        (meta["contrast"] == "brightfield")
    ].copy()

    study_ids = set(spl[spl["split"] == split_name]["study_id"])
    ann_parts = []
    if "mar2020" in study_ids:
        ann_parts.append(pd.read_csv(ANN_MAR))
    if "nov2021" in study_ids:
        ann_parts.append(pd.read_csv(ANN_NOV))

    ann = pd.concat(ann_parts, ignore_index=True) if ann_parts else pd.DataFrame()
    if not ann.empty:
        ann = ann[ann["objectType"] == "S.haematobium"]

    positions = _tile_positions()
    rows = []
    for _, img_row in imgs.iterrows():
        img_name    = img_row["image_name"]
        patient_key = img_row["patient_key"]
        rel_path    = img_row["relative_path"]

        if not ann.empty:
            img_ann = ann[ann["imageName"] == img_name]
            ann_xy  = img_ann[["xCoord", "yCoord"]].values
        else:
            ann_xy = np.empty((0, 2), dtype=int)

        for tile_x, tile_y in positions:
            if len(ann_xy) > 0:
                in_tile = (
                    (ann_xy[:, 0] >= tile_x) & (ann_xy[:, 0] < tile_x + TILE_SIZE) &
                    (ann_xy[:, 1] >= tile_y) & (ann_xy[:, 1] < tile_y + TILE_SIZE)
                )
                has_egg = int(in_tile.any())
            else:
                has_egg = 0

            rows.append({
                "patient_key":   patient_key,
                "image_name":    img_name,
                "relative_path": rel_path,
                "tile_x":        tile_x,
                "tile_y":        tile_y,
                "has_egg":       has_egg,
                "split":         split_name,
            })

    df = pd.DataFrame(rows)
    pat_label = spl.set_index("patient_key")["patient_label"].map(
        {"positive": 1, "negative": 0}
    )
    df["patient_label"] = df["patient_key"].map(pat_label)
    return df


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
            "target":      torch.tensor(float(row["has_egg"]),       dtype=torch.float32),
            "pat_target":  torch.tensor(float(row["patient_label"]), dtype=torch.float32),
            "patient_key": str(row["patient_key"]),
        }


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


def build_model() -> nn.Module:
    import torchvision.models as models
    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    m.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(m.classifier[1].in_features, 1))
    return m


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


def main() -> None:
    seed_everything(SEED)
    device = resolve_device("auto")
    print(f"Device:    {device}")
    print(f"Direction: train on mar2020 → zero-shot test on nov2021")
    print(f"Tile size: {TILE_SIZE}×{TILE_SIZE}  grid: {TILE_COLS}×{TILE_ROWS} = {N_TILES} tiles/image\n")

    print("Building tile metadata...")
    train_df = build_tile_df("train")
    val_df   = build_tile_df("val")
    test_df  = build_tile_df("test")

    n_pos_train = int(train_df["has_egg"].sum())
    print(f"  Train tiles: {len(train_df):,}  ({n_pos_train:,} positive = {100*n_pos_train/len(train_df):.1f}%)")
    print(f"  Val tiles:   {len(val_df):,}")
    print(f"  Test tiles:  {len(test_df):,}\n")

    train_ds = TileDataset(train_df, train=True)
    val_ds   = TileDataset(val_df,   train=False)
    test_ds  = TileDataset(test_df,  train=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=0)

    model = build_model().to(device)

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

    preds = run_inference(model, test_loader, device=device)
    pat   = preds.groupby("patient_key").agg(
        target=("pat_target", "max"), score=("prob", "max")
    ).reset_index()
    pt, lo, hi = bootstrap_auc(pat["target"].values, pat["score"].values)

    print(f"\n=== Mar2020→Nov2021 Tiled MobileNetV2 Results ===")
    print(f"  mar2020 val patient AUC:       {best_val:.4f}")
    print(f"  nov2021 zero-shot patient AUC: {pt:.4f}  [{lo:.4f}, {hi:.4f}]")

    results = pd.DataFrame([{
        "model":           "MobileNetV2 tiled (640×640) mar→nov",
        "mar2020_val_auc": round(best_val, 4),
        "nov2021_auc":     round(pt,       4),
        "nov2021_lo":      round(lo,       4),
        "nov2021_hi":      round(hi,       4),
    }])
    results.to_csv(OUT_DIR / "results.csv", index=False)
    preds.to_csv(OUT_DIR / "predictions.csv", index=False)
    print(f"\nSaved: {OUT_DIR / 'results.csv'}")


if __name__ == "__main__":
    main()
