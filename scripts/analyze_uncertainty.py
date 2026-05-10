#!/usr/bin/env python3
"""Bootstrap confidence interval analysis for all key model results.

Answers:
  1. Are the val/test AUC differences statistically significant?
  2. Which model comparisons survive 95% CI overlap?
  3. What are honest CIs for every number in the paper?

Outputs:
  results/uncertainty_analysis/bootstrap_ci_table.csv
  results/uncertainty_analysis/forest_plot.png
  results/uncertainty_analysis/val_test_gap_table.txt
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

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

from schisto_mobile_ai.data.classification import MetadataImageDataset
from schisto_mobile_ai.utils.io import ensure_dir
from schisto_mobile_ai.utils.reproducibility import resolve_device, seed_everything

N_BOOT = 2000
SEED = 42
CI_LOW_PCT = 2.5
CI_HIGH_PCT = 97.5


# ---------------------------------------------------------------------------
# Core AUC + bootstrap — operate on pre-aggregated patient arrays (fast)
# ---------------------------------------------------------------------------

def _wilcoxon_auc(targets: np.ndarray, scores: np.ndarray) -> float:
    pos = targets == 1
    neg = ~pos
    pc, nc = int(pos.sum()), int(neg.sum())
    if pc == 0 or nc == 0:
        return float("nan")
    ranks = pd.Series(scores).rank(method="average").values
    return float((ranks[pos].sum() - pc * (pc + 1) / 2.0) / (pc * nc))


def bootstrap_auc(
    targets: np.ndarray,
    scores: np.ndarray,
    *,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> tuple[float, float, float]:
    """Return (point, ci_low, ci_high). Patient-level arrays — each row is one patient."""
    point = _wilcoxon_auc(targets, scores)
    rng = np.random.default_rng(seed)
    n = len(targets)
    boot: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        a = _wilcoxon_auc(targets[idx], scores[idx])
        if np.isfinite(a):
            boot.append(a)
    if len(boot) < 10:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(boot, [CI_LOW_PCT, CI_HIGH_PCT])
    return point, float(lo), float(hi)


# ---------------------------------------------------------------------------
# Aggregation: pair_scores.csv → patient arrays
# ---------------------------------------------------------------------------

def pair_scores_to_patient_arrays(
    ps: pd.DataFrame, score_col: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return (targets, scores) arrays — one entry per patient."""
    g = ps.groupby("patient_key").agg(
        target=("target", "max"),
        score=(score_col, "max"),
    ).reset_index()
    return g["target"].values.astype(float), g["score"].values.astype(float)


def image_preds_to_patient_arrays(preds: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    g = preds.groupby("patient_key").agg(
        target=("target", "max"),
        score=("probability", "max"),
    ).reset_index()
    return g["target"].values.astype(float), g["score"].values.astype(float)


# ---------------------------------------------------------------------------
# Conditional sweep peak — vectorized
# ---------------------------------------------------------------------------

def _build_patient_score_matrix(
    ps: pd.DataFrame,
    thresholds: np.ndarray,
    gate_col: str = "bf_confidence",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Returns:
      patient_targets  shape (n_patients,)
      patient_scores   shape (n_patients, n_thresholds)
      patient_keys     list of patient keys
    """
    patient_keys = sorted(ps["patient_key"].unique())
    n_thresh = len(thresholds)
    n_patients = len(patient_keys)
    key_to_idx = {k: i for i, k in enumerate(patient_keys)}

    patient_targets = np.zeros(n_patients)
    patient_scores = np.zeros((n_patients, n_thresh))

    for pk, grp in ps.groupby("patient_key"):
        i = key_to_idx[pk]
        patient_targets[i] = float(grp["target"].max())
        gate = grp[gate_col].values          # shape (n_pairs,)
        p_bf = grp["p_bf"].values
        p_fused = grp["p_fused"].values

        # For each threshold: route uncertain pairs to DF, certain to BF
        # pair_score[t] = p_fused if gate < thresh else p_bf
        # patient_score = max over pairs
        for t_idx, thresh in enumerate(thresholds):
            use_df = gate < thresh
            pair_scores = np.where(use_df, p_fused, p_bf)
            patient_scores[i, t_idx] = float(pair_scores.max())

    return patient_targets, patient_scores, patient_keys


def bootstrap_conditional_peak(
    ps: pd.DataFrame,
    *,
    n_thresholds: int = 51,
    n_boot: int = N_BOOT,
    seed: int = SEED,
    gate_col: str = "bf_confidence",
) -> tuple[float, float, float, float]:
    """Return (point_peak_auc, ci_low, ci_high, val_optimal_threshold).

    Pre-builds a (n_patients, n_thresholds) score matrix so each bootstrap
    iteration only needs array indexing — no pandas inside the loop.
    """
    gate_min = float(ps[gate_col].min())
    gate_max = float(ps[gate_col].max())
    thresholds = np.linspace(gate_min, gate_max, n_thresholds)

    targets, score_matrix, patient_keys = _build_patient_score_matrix(ps, thresholds, gate_col)
    n_patients = len(patient_keys)

    # Point estimate
    aucs = np.array([_wilcoxon_auc(targets, score_matrix[:, t]) for t in range(len(thresholds))])
    peak_idx = int(np.nanargmax(aucs))
    point_peak = float(aucs[peak_idx])
    optimal_threshold = float(thresholds[peak_idx])

    # Bootstrap — sample patient indices, recompute AUC per threshold, take max
    rng = np.random.default_rng(seed)
    boot_peaks: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_patients, size=n_patients)
        t_boot = targets[idx]
        s_boot = score_matrix[idx, :]
        boot_aucs = np.array([_wilcoxon_auc(t_boot, s_boot[:, t]) for t in range(len(thresholds))])
        peak = np.nanmax(boot_aucs)
        if np.isfinite(peak):
            boot_peaks.append(float(peak))

    if len(boot_peaks) < 10:
        return point_peak, float("nan"), float("nan"), optimal_threshold
    lo, hi = np.percentile(boot_peaks, [CI_LOW_PCT, CI_HIGH_PCT])
    return point_peak, float(lo), float(hi), optimal_threshold


def conditional_auc_fixed_threshold(
    ps: pd.DataFrame,
    threshold: float,
    *,
    n_boot: int = N_BOOT,
    seed: int = SEED,
    gate_col: str = "bf_confidence",
) -> tuple[float, float, float]:
    """AUC + CI at a fixed threshold — no optimization on this set."""
    thresholds = np.array([threshold])
    targets, score_matrix, _ = _build_patient_score_matrix(ps, thresholds, gate_col)
    scores = score_matrix[:, 0]
    return bootstrap_auc(targets, scores, n_boot=n_boot, seed=seed)


# ---------------------------------------------------------------------------
# Pretrained baseline test evaluation
# ---------------------------------------------------------------------------

def build_pretrained_model(arch: str) -> nn.Module:
    import torchvision.models as models
    if arch == "resnet18":
        model = models.resnet18(weights=None)
        in_f = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_f, 1))
    elif arch == "mobilenet_v2":
        model = models.mobilenet_v2(weights=None)
        in_f = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_f, 1))
    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_f = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_f, 1))
    else:
        raise ValueError(arch)
    return model


def _build_single_contrast_frame(split: str, contrast: str = "brightfield") -> pd.DataFrame:
    """Build image frame for a given split directly from metadata CSVs."""
    images = pd.read_csv(REPO_ROOT / "metadata" / "images.csv")
    splits = pd.read_csv(REPO_ROOT / "splits" / "random_patient_split.csv")
    raw_root = REPO_ROOT / "data" / "raw"

    merged = images.merge(splits[["patient_key", "split"]], on="patient_key", how="inner")
    merged = merged[merged["contrast"].str.lower() == contrast].copy()
    merged = merged[merged["split"] == split].copy()

    label_col = "label" if "label" in merged.columns else "patient_level_label"
    merged["target"] = merged[label_col].map({"positive": 1.0, "negative": 0.0})
    merged = merged[merged["target"].notna()].copy()
    merged["image_path"] = merged["relative_path"].map(lambda p: str(raw_root / p))
    merged = merged[merged["image_path"].map(lambda p: Path(p).exists())].copy()
    return merged.reset_index(drop=True)


@torch.no_grad()
def eval_pretrained_on_split(
    arch: str,
    split: str,
    *,
    device: str,
    img_size: int = 224,
    batch_size: int = 64,
) -> pd.DataFrame:
    ckpt = REPO_ROOT / "runs" / "baselines" / arch / "best_model.pt"
    if not ckpt.exists():
        print(f"  WARNING: {ckpt} not found, skipping {arch}/{split}")
        return pd.DataFrame()

    model = build_pretrained_model(arch)
    state = torch.load(ckpt, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    frame = _build_single_contrast_frame(split)
    ds = MetadataImageDataset(frame, image_size=img_size, train=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    rows: list[dict[str, Any]] = []
    for batch in loader:
        imgs = batch["image"].to(device)
        probs = torch.sigmoid(model(imgs).squeeze(1)).cpu().numpy()
        for i in range(len(probs)):
            rows.append({
                "image_id": batch["image_id"][i],
                "patient_key": batch["patient_key"][i],
                "target": float(batch["target"][i].item()),
                "probability": float(probs[i]),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    seed_everything(SEED)
    device = resolve_device("auto")
    out_dir = ensure_dir(REPO_ROOT / "results" / "uncertainty_analysis")
    results: list[dict[str, Any]] = []

    def record(model: str, split: str, metric: str, pt: float, lo: float, hi: float, **kw):
        results.append({"model": model, "split": split, "metric": metric,
                        "point": pt, "ci_low": lo, "ci_high": hi, **kw})
        print(f"  {model} [{split}] {metric}: {pt:.4f}  [{lo:.4f}, {hi:.4f}]")

    # ------------------------------------------------------------------
    # 1. TinyConv BYOL + TTA (headline)
    # ------------------------------------------------------------------
    val_optimal_thresh: float | None = None
    for split_name, run_name in [("val", "byol_tta8_val"), ("test", "byol_tta8_test")]:
        ps_path = REPO_ROOT / "results" / "conditional_inference" / run_name / "pair_scores.csv"
        if not ps_path.exists():
            print(f"Missing: {ps_path}")
            continue
        ps = pd.read_csv(ps_path)
        n_pat = ps["patient_key"].nunique()
        n_pos = int(ps.groupby("patient_key")["target"].max().sum())
        print(f"\n[{run_name}] n_patients={n_pat}, n_pos={n_pos}")

        for col, label in [("p_bf", "BF-only"), ("p_df", "DF-only"), ("p_fused", "Always-fused")]:
            t, s = pair_scores_to_patient_arrays(ps, col)
            pt, lo, hi = bootstrap_auc(t, s)
            record(f"TinyConv BYOL ({label}, TTA=8)", split_name, "Patient AUC", pt, lo, hi)

        pt, lo, hi, opt_thresh = bootstrap_conditional_peak(ps)
        record(f"TinyConv BYOL (Conditional peak, TTA=8)", split_name,
               "Patient AUC (cond. peak)", pt, lo, hi,
               note=f"thresh={opt_thresh:.3f} sweep-optimized on same split")

        if split_name == "val":
            val_optimal_thresh = opt_thresh

    # Fixed val threshold applied to test (honest cross-set metric)
    test_ps_path = REPO_ROOT / "results" / "conditional_inference" / "byol_tta8_test" / "pair_scores.csv"
    if test_ps_path.exists() and val_optimal_thresh is not None:
        test_ps = pd.read_csv(test_ps_path)
        pt, lo, hi = conditional_auc_fixed_threshold(test_ps, val_optimal_thresh)
        record("TinyConv BYOL (Cond, val-thresh→test, TTA=8)", "test",
               "Patient AUC (fixed thresh)", pt, lo, hi,
               note=f"val-optimal thresh={val_optimal_thresh:.3f}")

    # ------------------------------------------------------------------
    # 2. TinyConv BYOL no TTA (ablation)
    # ------------------------------------------------------------------
    print()
    for split_name, run_name in [("val", "byol_matched_final_val"), ("test", "byol_matched_final_test")]:
        ps_path = REPO_ROOT / "results" / "conditional_inference" / run_name / "pair_scores.csv"
        if not ps_path.exists():
            continue
        ps = pd.read_csv(ps_path)
        t, s = pair_scores_to_patient_arrays(ps, "p_bf")
        pt, lo, hi = bootstrap_auc(t, s)
        record("TinyConv BYOL (BF-only, no TTA)", split_name, "Patient AUC", pt, lo, hi)
        pt_c, lo_c, hi_c, _ = bootstrap_conditional_peak(ps)
        record("TinyConv BYOL (Conditional peak, no TTA)", split_name,
               "Patient AUC (cond. peak)", pt_c, lo_c, hi_c)

    # ------------------------------------------------------------------
    # 3. TinyConv scratch (no SSL)
    # ------------------------------------------------------------------
    print()
    scratch_path = REPO_ROOT / "results" / "conditional_inference" / "scratch_bf_routing" / "pair_scores.csv"
    if scratch_path.exists():
        ps = pd.read_csv(scratch_path)
        t, s = pair_scores_to_patient_arrays(ps, "p_bf")
        pt, lo, hi = bootstrap_auc(t, s)
        record("TinyConv (scratch, BF-only)", "val", "Patient AUC", pt, lo, hi)
        pt_c, lo_c, hi_c, _ = bootstrap_conditional_peak(ps)
        record("TinyConv (scratch, Conditional peak)", "val", "Patient AUC (cond. peak)", pt_c, lo_c, hi_c)

    # ------------------------------------------------------------------
    # 4. Pretrained baselines — val (cached) + test (evaluated inline)
    # ------------------------------------------------------------------
    arch_labels = {
        "resnet18": "ResNet-18 (11.2M, ImageNet)",
        "mobilenet_v2": "MobileNetV2 (3.4M, ImageNet)",
        "efficientnet_b0": "EfficientNet-B0 (5.3M, ImageNet)",
    }
    print("\n[pretrained baselines]")
    for arch, label in arch_labels.items():
        for split_name in ("val", "test"):
            cache = REPO_ROOT / "runs" / "baselines" / arch / f"{split_name}_predictions.csv"
            if cache.exists():
                preds = pd.read_csv(cache)
            else:
                print(f"  Evaluating {arch} on {split_name}...")
                preds = eval_pretrained_on_split(arch, split_name, device=device)
                if not preds.empty:
                    preds.to_csv(cache, index=False)
            if preds.empty:
                continue
            t, s = image_preds_to_patient_arrays(preds)
            pt, lo, hi = bootstrap_auc(t, s)
            record(label, split_name, "Patient AUC", pt, lo, hi)

    # ------------------------------------------------------------------
    # Save table + text summary
    # ------------------------------------------------------------------
    df = pd.DataFrame(results)
    df = df.sort_values(["split", "point"], ascending=[True, False])
    df["ci_width"] = (df["ci_high"] - df["ci_low"]).round(4)
    for col in ["point", "ci_low", "ci_high"]:
        df[col] = df[col].round(4)
    df.to_csv(out_dir / "bootstrap_ci_table.csv", index=False)
    print(f"\nSaved: {out_dir / 'bootstrap_ci_table.csv'}")

    # Gap analysis text
    _write_gap_summary(df, out_dir / "val_test_gap_table.txt")

    # Forest plot
    _plot_forest(df, out_dir / "forest_plot.png")
    print(f"Saved: {out_dir / 'forest_plot.png'}")


def _write_gap_summary(df: pd.DataFrame, path: Path) -> None:
    lines = [
        "Val/Test Gap Analysis — Bootstrap 95% CIs",
        "=" * 60,
        f"N_BOOT={N_BOOT}, CI={100-CI_LOW_PCT*2:.0f}%, patient-level resampling.",
        "CIs overlap → gap not statistically significant at 95%.",
        "",
    ]
    focus = [
        ("TinyConv BYOL (BF-only, TTA=8)", "Patient AUC"),
        ("TinyConv BYOL (Conditional peak, TTA=8)", "Patient AUC (cond. peak)"),
        ("TinyConv BYOL (Cond, val-thresh→test, TTA=8)", "Patient AUC (fixed thresh)"),
        ("ResNet-18 (11.2M, ImageNet)", "Patient AUC"),
        ("MobileNetV2 (3.4M, ImageNet)", "Patient AUC"),
        ("EfficientNet-B0 (5.3M, ImageNet)", "Patient AUC"),
    ]
    for model, metric in focus:
        sub = df[(df["model"] == model) & (df["metric"] == metric)]
        if sub.empty:
            continue
        lines.append(f"{model}  [{metric}]")
        for _, row in sub.iterrows():
            lines.append(f"  {row['split']:4s}: {row['point']:.4f}  [{row['ci_low']:.4f}, {row['ci_high']:.4f}]")
        val_r = sub[sub["split"] == "val"]
        test_r = sub[sub["split"] == "test"]
        if not val_r.empty and not test_r.empty:
            v, t = val_r.iloc[0], test_r.iloc[0]
            overlap = min(v["ci_high"], t["ci_high"]) - max(v["ci_low"], t["ci_low"])
            sig = "NOT significant" if overlap > 0 else "SIGNIFICANT"
            lines.append(f"  Gap: {v['point']-t['point']:+.4f}  CI overlap={overlap:.4f}  → {sig}")
        lines.append("")

    text = "\n".join(lines)
    path.write_text(text)
    print("\n" + text)


def _plot_forest(df: pd.DataFrame, path: Path) -> None:
    # One row per (model, split) — prefer cond. peak metric, fallback to Patient AUC
    metric_pref = ["Patient AUC (cond. peak)", "Patient AUC (fixed thresh)", "Patient AUC"]
    plot_rows: list[dict] = []
    for model in df["model"].unique():
        for split in ["val", "test"]:
            sub = df[(df["model"] == model) & (df["split"] == split)]
            for m in metric_pref:
                hit = sub[sub["metric"] == m]
                if not hit.empty:
                    r = hit.iloc[0].to_dict()
                    r["label"] = f"{model}  [{split}]"
                    plot_rows.append(r)
                    break

    plot_df = pd.DataFrame(plot_rows).dropna(subset=["ci_low", "ci_high"])
    plot_df = plot_df.sort_values("point", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11, max(6, len(plot_df) * 0.42)))
    colors = {"val": "#1f77b4", "test": "#d62728"}

    for i, row in plot_df.iterrows():
        col = colors.get(row["split"], "#888888")
        ax.errorbar(
            row["point"], i,
            xerr=[[row["point"] - row["ci_low"]], [row["ci_high"] - row["point"]]],
            fmt="o", color=col, ecolor=col, elinewidth=1.8, capsize=4, markersize=7,
        )
        ax.text(
            row["ci_high"] + 0.003, i,
            f"{row['point']:.3f}  [{row['ci_low']:.3f}, {row['ci_high']:.3f}]",
            va="center", ha="left", fontsize=7.5,
        )

    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df["label"], fontsize=8)
    ax.set_xlabel("Patient-level AUC  (95% bootstrap CI, patient-level resampling)")
    ax.set_title(
        "Model Comparison with Bootstrap CIs\n"
        "Blue = val  |  Red = test  |  Overlapping CIs = not statistically distinguishable"
    )
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0.3, 1.05)

    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4", markersize=8, label="Val set"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728", markersize=8, label="Test set"),
    ], loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
