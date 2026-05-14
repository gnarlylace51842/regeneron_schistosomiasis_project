#!/usr/bin/env python3
"""Cross-study generalization experiment for ImageNet-pretrained baselines.

Trains MobileNetV2 and EfficientNet-B0 on nov2021 only, evaluates zero-shot on
mar2020 (using splits/cross_study_split.csv). Answers whether the near-random
cross-study AUC we observed for TinyConv is a model-capacity issue or a genuine
transfer failure.

Outputs:
  results/cross_study_pretrained/results.csv
  results/cross_study_pretrained/cross_study_pretrained_curve.png
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
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from schisto_mobile_ai.data.classification import MetadataImageDataset, load_single_contrast_data
from schisto_mobile_ai.models.patient_aggregation import aggregate_patient_predictions
from schisto_mobile_ai.utils.io import ensure_dir
from schisto_mobile_ai.utils.reproducibility import resolve_device, seed_everything

CROSS_STUDY_SPLIT = REPO_ROOT / "splits" / "cross_study_split.csv"
IMAGES_CSV        = REPO_ROOT / "metadata" / "images.csv"
RAW_DIR           = REPO_ROOT / "data" / "raw"
OUT_DIR           = ensure_dir(REPO_ROOT / "results" / "cross_study_pretrained")
RUN_DIR           = ensure_dir(REPO_ROOT / "runs" / "cross_study_pretrained")

ARCHS   = ["mobilenet_v2", "efficientnet_b0"]
EPOCHS  = 20
IMG_SIZE = 224
LR      = 3e-4
WD      = 1e-4
BATCH   = 16
N_BOOT  = 2000
SEED    = 42


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(arch: str) -> nn.Module:
    import torchvision.models as models
    if arch == "mobilenet_v2":
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        m.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(m.classifier[1].in_features, 1))
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(m.classifier[1].in_features, 1))
    else:
        raise ValueError(arch)
    return m


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _build_test_frame(images_csv: Path, split_csv: Path, contrast: str) -> pd.DataFrame:
    """Build a dataframe of test-split images for inference."""
    img = pd.read_csv(images_csv)
    spl = pd.read_csv(split_csv)
    test_keys = set(spl[spl["split"] == "test"]["patient_key"])
    contrast_full = {"bf": "brightfield", "df": "darkfield"}[contrast]
    sub = img[
        img["patient_key"].isin(test_keys) &
        (img["contrast"] == contrast_full)
    ].copy()
    # target column: image-level label
    sub["target"] = (sub["label"] == "positive").astype(float)
    sub["split"] = "test"
    sub["image_path"] = sub["relative_path"].apply(lambda p: str(RAW_DIR / p))
    return sub.reset_index(drop=True)


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


def bootstrap_auc(t: np.ndarray, s: np.ndarray, seed: int = SEED) -> tuple[float, float, float]:
    pt = _wilcoxon_auc(t, s)
    rng = np.random.default_rng(seed)
    n = len(t)
    boot = [b for _ in range(N_BOOT)
            if np.isfinite(b := _wilcoxon_auc(t[idx := rng.integers(0, n, n)], s[idx]))]
    if len(boot) < 10:
        return pt, float("nan"), float("nan")
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return pt, float(lo), float(hi)


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def _train_epoch(model, loader, *, optimizer, criterion, device):
    model.train()
    for batch in loader:
        imgs = batch["image"].to(device)
        tgt  = batch["target"].to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(imgs).squeeze(1), tgt)
        loss.backward()
        optimizer.step()


def _eval_auc(model, loader, *, device) -> float:
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            probs = torch.sigmoid(model(imgs).squeeze(1)).cpu().numpy()
            for i, (pk, t, p) in enumerate(zip(
                    batch["patient_key"], batch["target"].numpy(), probs)):
                rows.append({"patient_key": pk, "target": float(t), "prob": float(p)})
    df = pd.DataFrame(rows)
    pat = df.groupby("patient_key").agg(
        target=("target", "max"), prob=("prob", "max")
    ).reset_index()
    return _wilcoxon_auc(pat["target"].values, pat["prob"].values)


def _run_inference(model, loader, *, device) -> pd.DataFrame:
    """Run inference and return per-image rows."""
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            probs = torch.sigmoid(model(imgs).squeeze(1)).cpu().numpy()
            for pk, t, p in zip(batch["patient_key"], batch["target"].numpy(), probs):
                rows.append({"patient_key": pk, "target": float(t), "prob": float(p)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train_and_eval(arch: str, device) -> dict:
    seed_everything(SEED)
    run_dir = ensure_dir(RUN_DIR / arch)

    # --- training data (nov2021) ---
    data = load_single_contrast_data(
        images_csv=IMAGES_CSV,
        split_csv=CROSS_STUDY_SPLIT,
        raw_dir=RAW_DIR,
        contrast="bf",
        label_source="image",
        seed=SEED,
    )
    train_ds = MetadataImageDataset(data.train_frame, image_size=IMG_SIZE, train=True)
    val_ds   = MetadataImageDataset(data.val_frame,   image_size=IMG_SIZE, train=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)

    # --- test data (mar2020) ---
    test_frame = _build_test_frame(IMAGES_CSV, CROSS_STUDY_SPLIT, "bf")
    test_ds    = MetadataImageDataset(test_frame, image_size=IMG_SIZE, train=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    model = build_model(arch).to(device)

    # Prior bias init
    pos_rate = float(data.train_frame["target"].mean())
    if 0 < pos_rate < 1:
        bias = float(np.log(pos_rate / (1 - pos_rate)))
        for m in reversed(list(model.modules())):
            if isinstance(m, nn.Linear) and m.out_features == 1:
                with torch.no_grad():
                    m.bias.fill_(bias)
                break

    pos  = float(data.train_frame["target"].sum())
    neg  = float(len(data.train_frame) - pos)
    pw   = torch.tensor([(neg / pos) ** 0.5], dtype=torch.float32).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt  = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR * 0.01)

    saved_model = run_dir / "best_model.pt"
    if saved_model.exists():
        print(f"  [{arch}] Loading existing weights from {saved_model}")
        model.load_state_dict(torch.load(saved_model, map_location=device))
        best_val = _eval_auc(model, val_loader, device=device)
        print(f"  [{arch}] val_auc (loaded) = {best_val:.4f}")
    else:
        best_val, best_state = float("-inf"), None
        for epoch in range(1, EPOCHS + 1):
            _train_epoch(model, train_loader, optimizer=opt, criterion=crit, device=device)
            val_auc = _eval_auc(model, val_loader, device=device)
            sched.step()
            print(f"  [{arch}] epoch {epoch}/{EPOCHS}  val_auc={val_auc:.4f}")
            if np.isfinite(val_auc) and val_auc > best_val:
                best_val = val_auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)
            torch.save(best_state, saved_model)

    # --- test evaluation (mar2020 zero-shot) ---
    preds = _run_inference(model, test_loader, device=device)
    pat   = preds.groupby("patient_key").agg(
        target=("target", "max"), prob=("prob", "max")
    ).reset_index()
    pt, lo, hi = bootstrap_auc(pat["target"].values, pat["prob"].values)

    print(f"\n  {arch} results:")
    print(f"    nov2021 val AUC:      {best_val:.4f}")
    print(f"    mar2020 zero-shot AUC: {pt:.4f}  [{lo:.4f}, {hi:.4f}]")

    preds.to_csv(run_dir / "test_predictions.csv", index=False)

    return {
        "arch": arch,
        "nov2021_val_auc": round(best_val, 4),
        "mar2020_auc": round(pt, 4),
        "mar2020_lo":  round(lo, 4),
        "mar2020_hi":  round(hi, 4),
    }


def main() -> None:
    device = resolve_device("auto")
    print(f"Device: {device}")
    print(f"Split:  {CROSS_STUDY_SPLIT}")
    print(f"Archs:  {ARCHS}\n")

    results = []
    for arch in ARCHS:
        print(f"\n{'='*55}")
        print(f"  {arch}")
        print(f"{'='*55}")
        results.append(train_and_eval(arch, device))

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "results.csv", index=False)
    print(f"\nSaved: {OUT_DIR / 'results.csv'}")

    print("\n=== Cross-Study Pretrained Baselines ===")
    print(df.to_string(index=False))

    # Also print TinyConv reference for comparison
    print("\n--- TinyConv reference (from earlier experiment) ---")
    print("  scratch BF:  nov2021 val 0.697 | mar2020 zero-shot 0.547 [0.482, 0.615]")
    print("  BYOL BF:     nov2021 val 0.704 | mar2020 zero-shot 0.530 [0.461, 0.598]")

    _plot(df, OUT_DIR / "cross_study_pretrained_curve.png")
    print(f"\nSaved: {OUT_DIR / 'cross_study_pretrained_curve.png'}")


def _plot(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    tinyconv_ref = [
        ("TinyConv scratch BF", 0.697, 0.547, 0.482, 0.615, "#aec7e8"),
        ("TinyConv BYOL BF",    0.704, 0.530, 0.461, 0.598, "#c5b0d5"),
    ]
    colors = {"mobilenet_v2": "#2ca02c", "efficientnet_b0": "#d62728"}

    x    = [0, 1]
    xlabels = ["nov2021 val", "mar2020 zero-shot"]

    for _, row in df.iterrows():
        c = colors.get(row["arch"], "gray")
        ax.plot(x, [row["nov2021_val_auc"], row["mar2020_auc"]],
                "o-", color=c, linewidth=2.5, label=row["arch"])
        ax.fill_between([1, 1],
                        [row["mar2020_lo"]], [row["mar2020_hi"]],
                        alpha=0, color=c)
        ax.errorbar([1], [row["mar2020_auc"]],
                    yerr=[[row["mar2020_auc"] - row["mar2020_lo"]],
                          [row["mar2020_hi"] - row["mar2020_auc"]]],
                    fmt="none", color=c, capsize=5)

    for label, val_auc, test_auc, lo, hi, c in tinyconv_ref:
        ax.plot(x, [val_auc, test_auc], "s--", color=c, linewidth=1.5,
                alpha=0.7, label=label)
        ax.errorbar([1], [test_auc],
                    yerr=[[test_auc - lo], [hi - test_auc]],
                    fmt="none", color=c, capsize=4, alpha=0.7)

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="Random (AUC=0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_ylabel("Patient-level AUC  (95% bootstrap CI)")
    ax.set_title("Cross-Study Generalization: ImageNet Pretrained vs TinyConv\n"
                 "Train on nov2021 → Zero-shot test on mar2020")
    ax.set_ylim(0.3, 1.0)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
