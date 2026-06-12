#!/usr/bin/env python3
"""Patient-level ROC curve with bootstrap 95% CI band for a YOLOv8 run.

Reads the patient_scores.csv produced by eval_yolov8.py / train_yolov8_baseline.py
and plots the ROC with a bootstrap confidence band and the two TPP operating
points (M&E = 96.5% spec, TI&S = 99.5% spec) marked.

Usage:
    python scripts/plot_patient_roc.py --contrast bf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
SEED = 42
N_BOOT = 2000


def _roc_on_grid(y: np.ndarray, s: np.ndarray, fpr_grid: np.ndarray) -> np.ndarray:
    """TPR interpolated onto a common FPR grid (for averaging bootstrap curves)."""
    P, N = int(y.sum()), int((y == 0).sum())
    if P == 0 or N == 0:
        return np.full_like(fpr_grid, np.nan, dtype=float)
    thr = np.concatenate([[np.inf], np.sort(np.unique(s))[::-1]])
    fpr = np.empty(len(thr)); tpr = np.empty(len(thr))
    for i, t in enumerate(thr):
        pred = s >= t
        tpr[i] = np.sum(pred & (y == 1)) / P
        fpr[i] = np.sum(pred & (y == 0)) / N
    order = np.argsort(fpr)
    return np.interp(fpr_grid, fpr[order], tpr[order])


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    pos = y == 1
    pc, nc = int(pos.sum()), int((~pos).sum())
    if pc == 0 or nc == 0:
        return float("nan")
    ranks = pd.Series(s).rank(method="average").values
    return float((ranks[pos].sum() - pc * (pc + 1) / 2.0) / (pc * nc))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contrast", choices=["bf", "df"], default="bf")
    args = ap.parse_args()

    res_dir = REPO_ROOT / "results" / f"yolov8_{args.contrast}_mar_to_nov"
    df = pd.read_csv(res_dir / "patient_scores.csv")
    y = df["target"].values.astype(int)
    s = df["score"].values.astype(float)

    fpr_grid = np.linspace(0, 1, 201)
    base_tpr = _roc_on_grid(y, s, fpr_grid)
    auc = _auc(y, s)

    rng = np.random.default_rng(SEED)
    boots, aucs = [], []
    n = len(y)
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        if y[idx].sum() == 0 or (y[idx] == 0).sum() == 0:
            continue
        boots.append(_roc_on_grid(y[idx], s[idx], fpr_grid))
        aucs.append(_auc(y[idx], s[idx]))
    boots = np.array(boots)
    lo_band = np.percentile(boots, 2.5, axis=0)
    hi_band = np.percentile(boots, 97.5, axis=0)
    auc_lo, auc_hi = np.percentile(aucs, [2.5, 97.5])

    # Operating points (from summary.json targets): (fpr, tpr)
    me_pt = (1 - 0.966, 0.738)   # M&E:  >=96.5% spec
    tis_pt = (1 - 0.997, 0.557)  # TI&S: >=99.5% spec

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.fill_between(fpr_grid, lo_band, hi_band, color="#4C72B0", alpha=0.20,
                    label="95% bootstrap CI")
    ax.plot(fpr_grid, base_tpr, color="#1A3C6E", lw=2.2,
            label=f"YOLOv8s detector (AUC = {auc:.3f} [{auc_lo:.3f}, {auc_hi:.3f}])")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="chance (AUC = 0.50)")

    ax.scatter(*me_pt, color="#C44E52", zorder=5, s=55)
    ax.annotate("M&E TPP\n(96.5% spec, sens 0.74)", me_pt,
                textcoords="offset points", xytext=(12, -6), fontsize=8.5, color="#C44E52")
    ax.scatter(*tis_pt, color="#55A868", zorder=5, s=55)
    ax.annotate("TI&S TPP\n(99.5% spec, sens 0.56)", tis_pt,
                textcoords="offset points", xytext=(12, 8), fontsize=8.5, color="#55A868")

    ax.set_xlabel("False positive rate (1 − specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title("Patient-level cross-study detection (mar2020 → nov2021, BF)\n"
                 "Schistosoma egg detector — 380 patients (61 pos / 319 neg)", fontsize=10)
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.02)
    ax.legend(loc="lower right", fontsize=8.5, frameon=True)
    ax.grid(alpha=0.25)
    fig.tight_layout()

    out_png = res_dir / "patient_roc.png"
    fig.savefig(out_png, dpi=200)
    print(f"AUC = {auc:.4f}  [{auc_lo:.4f}, {auc_hi:.4f}]")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
