#!/usr/bin/env python3
"""Tiled conditional routing experiment.

Reproduces the compute-sensitivity tradeoff on the tiled pipeline:
  - Trains tiled BF model  (loads from cache if already done)
  - Trains tiled DF model
  - Sweeps BF-confidence threshold for routing:
      confident BF  →  BF only  (save DF compute)
      uncertain BF  →  fuse BF + DF
  - Outputs tradeoff curve: patient AUC vs % DF compute used
  - Evaluated on mar2020 zero-shot (cross-study)

Outputs:
  results/tiled_routing/tradeoff_curve.csv
  results/tiled_routing/tradeoff_curve.png
  results/tiled_routing/operating_points.json
"""

from __future__ import annotations

import json
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
OUT_DIR  = ensure_dir(REPO_ROOT / "results" / "tiled_routing")
RUN_DIR  = ensure_dir(REPO_ROOT / "runs"    / "tiled_routing")

TILE_SIZE = 640
TILE_COLS, TILE_ROWS = 6, 5
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


# ---------------------------------------------------------------------------
# Tile metadata
# ---------------------------------------------------------------------------

def build_tile_df(split_name: str, contrast: str) -> pd.DataFrame:
    meta = pd.read_csv(IMAGES_CSV)
    spl  = pd.read_csv(CROSS_STUDY_SPLIT)
    contrast_full = {"bf": "brightfield", "df": "darkfield"}[contrast]

    patient_keys = set(spl[spl["split"] == split_name]["patient_key"])
    study_ids    = set(spl[spl["split"] == split_name]["study_id"])

    imgs = meta[
        meta["patient_key"].isin(patient_keys) &
        (meta["contrast"] == contrast_full)
    ].copy()

    ann_parts = []
    if "mar2020" in study_ids:
        ann_parts.append(pd.read_csv(ANN_MAR))
    if "nov2021" in study_ids:
        ann_parts.append(pd.read_csv(ANN_NOV))
    ann = pd.concat(ann_parts, ignore_index=True) if ann_parts else pd.DataFrame()
    if not ann.empty:
        ann = ann[ann["objectType"] == "S.haematobium"]

    positions = [(c, r) for r in ROW_STARTS for c in COL_STARTS]
    rows = []
    for _, img_row in imgs.iterrows():
        img_name    = img_row["image_name"]
        patient_key = img_row["patient_key"]
        rel_path    = img_row["relative_path"]

        img_ann = ann[ann["imageName"] == img_name] if not ann.empty else pd.DataFrame()
        ann_xy  = img_ann[["xCoord", "yCoord"]].values if len(img_ann) > 0 else np.empty((0, 2))

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
                "patient_key":  patient_key,
                "image_name":   img_name,
                "relative_path": rel_path,
                "tile_x": tile_x,
                "tile_y": tile_y,
                "has_egg": has_egg,
                "contrast": contrast,
            })

    df = pd.DataFrame(rows)
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
        with Image.open(RAW_DIR / row["relative_path"]) as img:
            img = img.convert("RGB")
            x, y = int(row["tile_x"]), int(row["tile_y"])
            tile = img.crop((x, y, x + TILE_SIZE, y + TILE_SIZE))
        return {
            "image":        self.transform(tile),
            "target":       torch.tensor(float(row["has_egg"]),       dtype=torch.float32),
            "patient_label": torch.tensor(float(row["patient_label"]), dtype=torch.float32),
            "patient_key":  str(row["patient_key"]),
        }


# ---------------------------------------------------------------------------
# AUC
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
    boot = [b for _ in range(N_BOOT)
            if np.isfinite(b := _wilcoxon_auc(t[idx := rng.integers(0, n, n)], s[idx]))]
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
# Train / infer
# ---------------------------------------------------------------------------

def _train_epoch(model, loader, *, optimizer, criterion, device) -> None:
    model.train()
    for batch in loader:
        imgs = batch["image"].to(device)
        tgt  = batch["target"].to(device)
        optimizer.zero_grad(set_to_none=True)
        criterion(model(imgs).squeeze(1), tgt).backward()
        optimizer.step()


def _val_patient_auc(model, loader, *, device) -> float:
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            probs = torch.sigmoid(model(batch["image"].to(device)).squeeze(1)).cpu().numpy()
            for pk, pl, p in zip(batch["patient_key"],
                                  batch["patient_label"].numpy(), probs):
                rows.append({"patient_key": pk, "patient_label": float(pl), "prob": float(p)})
    df = pd.DataFrame(rows)
    pat = df.groupby("patient_key").agg(
        target=("patient_label", "max"), score=("prob", "max")
    ).reset_index()
    return _wilcoxon_auc(pat["target"].values, pat["score"].values)


def infer_patient_scores(model, loader, *, device) -> pd.DataFrame:
    """Return patient-level max scores."""
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            probs = torch.sigmoid(model(batch["image"].to(device)).squeeze(1)).cpu().numpy()
            for pk, pl, p in zip(batch["patient_key"],
                                  batch["patient_label"].numpy(), probs):
                rows.append({"patient_key": pk, "patient_label": float(pl), "prob": float(p)})
    df = pd.DataFrame(rows)
    return df.groupby("patient_key").agg(
        target=("patient_label", "max"),
        score=("prob", "max"),
    ).reset_index()


def train_contrast(contrast: str, device) -> nn.Module:
    saved = RUN_DIR / f"best_model_{contrast}.pt"
    model = build_model().to(device)

    print(f"\n--- {contrast.upper()} model ---")
    if saved.exists():
        print(f"  Loading from {saved}")
        model.load_state_dict(torch.load(saved, map_location=device))
        return model

    train_df = build_tile_df("train", contrast)
    val_df   = build_tile_df("val",   contrast)
    print(f"  Train tiles: {len(train_df):,}  ({int(train_df['has_egg'].sum()):,} positive)")

    train_ds = TileDataset(train_df, train=True)
    val_ds   = TileDataset(val_df,   train=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)

    pos_rate = float(train_df["has_egg"].mean())
    if 0 < pos_rate < 1:
        bias = float(np.log(pos_rate / (1 - pos_rate)))
        for m in reversed(list(model.modules())):
            if isinstance(m, nn.Linear) and m.out_features == 1:
                with torch.no_grad():
                    m.bias.fill_(bias)
                break

    pos  = float(train_df["has_egg"].sum())
    neg  = float(len(train_df) - pos)
    pw   = torch.tensor([(neg / pos) ** 0.5], dtype=torch.float32).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR * 0.01)

    best_val, best_state = float("-inf"), None
    for epoch in range(1, EPOCHS + 1):
        _train_epoch(model, train_loader, optimizer=opt, criterion=crit, device=device)
        val_auc = _val_patient_auc(model, val_loader, device=device)
        sched.step()
        print(f"  epoch {epoch:2d}/{EPOCHS}  val_auc={val_auc:.4f}")
        if np.isfinite(val_auc) and val_auc > best_val:
            best_val = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
        torch.save(best_state, saved)
    return model


# ---------------------------------------------------------------------------
# Conditional routing sweep
# ---------------------------------------------------------------------------

def routing_sweep(bf_pat: pd.DataFrame, df_pat: pd.DataFrame) -> pd.DataFrame:
    """Sweep BF confidence threshold and compute AUC + % DF compute."""
    merged = bf_pat.merge(df_pat, on=["patient_key", "target"],
                          suffixes=("_bf", "_df"))
    merged["bf_conf"] = (merged["score_bf"] - 0.5).abs()
    merged["fused"]   = (merged["score_bf"] + merged["score_df"]) / 2.0

    thresholds = np.linspace(0.0, 0.5, 101)
    rows = []
    for t in thresholds:
        use_df  = merged["bf_conf"] < t
        scores  = np.where(use_df, merged["fused"], merged["score_bf"])
        auc     = _wilcoxon_auc(merged["target"].values, scores)
        df_frac = float(use_df.mean())
        rows.append({"threshold": float(t), "patient_auc": float(auc),
                     "df_fraction": df_frac})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    seed_everything(SEED)
    device = resolve_device("auto")
    print(f"Device: {device}")

    # Train / load BF and DF models
    # BF model may already exist from cross_study_tiled_experiment.py
    bf_model_path = REPO_ROOT / "runs" / "cross_study_tiled" / "best_model.pt"
    bf_model = build_model().to(device)
    if bf_model_path.exists():
        print(f"\n--- BF model ---")
        print(f"  Loading existing weights from {bf_model_path}")
        bf_model.load_state_dict(torch.load(bf_model_path, map_location=device))
    else:
        bf_model = train_contrast("bf", device)

    df_model = train_contrast("df", device)

    # Build test tile DataFrames
    print("\nBuilding test tile DataFrames (mar2020)...")
    test_bf_df = build_tile_df("test", "bf")
    test_df_df = build_tile_df("test", "df")

    test_bf_loader = DataLoader(TileDataset(test_bf_df, train=False),
                                 batch_size=BATCH, shuffle=False, num_workers=0)
    test_df_loader = DataLoader(TileDataset(test_df_df, train=False),
                                 batch_size=BATCH, shuffle=False, num_workers=0)

    print("Running BF inference on mar2020...")
    bf_pat = infer_patient_scores(bf_model, test_bf_loader, device=device)

    print("Running DF inference on mar2020...")
    df_pat = infer_patient_scores(df_model, test_df_loader, device=device)

    # Standalone AUCs
    bf_auc, bf_lo, bf_hi = bootstrap_auc(bf_pat["target"].values, bf_pat["score"].values)
    df_auc, df_lo, df_hi = bootstrap_auc(df_pat["target"].values, df_pat["score"].values)

    # Always-fused AUC
    fused_scores = (bf_pat.set_index("patient_key")["score"] +
                    df_pat.set_index("patient_key")["score"]) / 2.0
    fused_scores = fused_scores.reindex(bf_pat["patient_key"]).values
    fused_auc, fused_lo, fused_hi = bootstrap_auc(bf_pat["target"].values, fused_scores)

    print(f"\n=== Standalone AUCs (mar2020 zero-shot) ===")
    print(f"  BF-only  AUC: {bf_auc:.4f}  [{bf_lo:.4f}, {bf_hi:.4f}]")
    print(f"  DF-only  AUC: {df_auc:.4f}  [{df_lo:.4f}, {df_hi:.4f}]")
    print(f"  Always-fused: {fused_auc:.4f}  [{fused_lo:.4f}, {fused_hi:.4f}]")

    # Routing sweep
    print("\nRunning conditional routing sweep...")
    curve = routing_sweep(bf_pat, df_pat)
    curve.to_csv(OUT_DIR / "tradeoff_curve.csv", index=False)

    peak_idx  = int(curve["patient_auc"].idxmax())
    peak_row  = curve.iloc[peak_idx]

    print(f"\n=== Conditional Routing Results ===")
    print(f"  Peak conditional AUC: {peak_row['patient_auc']:.4f} "
          f"at {peak_row['df_fraction']*100:.1f}% DF compute")
    print(f"  BF-only  (0% DF):  {curve[curve['df_fraction']<0.01]['patient_auc'].max():.4f}")
    print(f"  Always-fused (100% DF): {fused_auc:.4f}")

    # Operating points
    ops = {
        "bf_only_auc":     round(bf_auc, 4),
        "df_only_auc":     round(df_auc, 4),
        "always_fused_auc": round(fused_auc, 4),
        "peak_conditional_auc": round(float(peak_row["patient_auc"]), 4),
        "peak_df_fraction":     round(float(peak_row["df_fraction"]), 4),
    }
    with open(OUT_DIR / "operating_points.json", "w") as f:
        json.dump(ops, f, indent=2)
    print(f"\nSaved: {OUT_DIR / 'operating_points.json'}")

    _plot(curve, bf_auc, df_auc, fused_auc, OUT_DIR / "tradeoff_curve.png")
    print(f"Saved: {OUT_DIR / 'tradeoff_curve.png'}")


def _plot(curve: pd.DataFrame, bf_auc: float, df_auc: float,
          fused_auc: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(curve["df_fraction"] * 100, curve["patient_auc"],
            color="#2ca02c", linewidth=2.5, label="Conditional routing (tiled)")

    ax.axhline(bf_auc,    color="#1f77b4", linestyle="--", linewidth=1.5,
               label=f"BF-only  (AUC={bf_auc:.3f})")
    ax.axhline(df_auc,    color="#ff7f0e", linestyle="--", linewidth=1.5,
               label=f"DF-only  (AUC={df_auc:.3f})")
    ax.axhline(fused_auc, color="#9467bd", linestyle=":",  linewidth=1.5,
               label=f"Always-fused  (AUC={fused_auc:.3f})")

    ax.set_xlabel("% patients routed to DF (compute cost)")
    ax.set_ylabel("Patient-level AUC  (mar2020 zero-shot)")
    ax.set_title("Tiled Conditional Routing: Compute-Sensitivity Tradeoff\n"
                 "Train on nov2021 → Zero-shot test on mar2020")
    ax.set_xlim(0, 100)
    ax.set_ylim(0.5, 1.0)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
