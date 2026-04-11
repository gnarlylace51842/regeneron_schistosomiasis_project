"""
Calibration analysis + bootstrap CIs for the routing quality gap.

Produces two figures:
  results/calibration_analysis.png  — reliability diagrams + ECE bars
  results/bootstrap_routing_gap.png — bootstrap distribution of BYOL - Scratch routing AUC gap
"""

import json
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.calibration import calibration_curve  # type: ignore
from sklearn.metrics import roc_auc_score  # type: ignore

RESULTS = pathlib.Path("results")
COND_DIR = RESULTS / "conditional_inference"

# ── colours ──────────────────────────────────────────────────────────────────
SCRATCH_COL = "#e07b39"   # orange
BYOL_COL    = "#4a90d9"   # blue
BYOL_TTA_COL= "#1a5fa8"   # dark blue
IDEAL_COL   = "#888888"

RNG = np.random.default_rng(42)
N_BOOTSTRAP = 2000


# ── helpers ──────────────────────────────────────────────────────────────────

def load_pairs(run_name: str) -> pd.DataFrame:
    path = COND_DIR / run_name / "pair_scores.csv"
    return pd.read_csv(path)


def load_tradeoff(run_name: str) -> pd.DataFrame:
    path = COND_DIR / run_name / "tradeoff_curve.csv"
    return pd.read_csv(path)


def patient_auc_from_pairs(df: pd.DataFrame, prob_col: str = "p_bf") -> float:
    """Max-aggregate pair probs to patient level, then compute AUC."""
    pat = df.groupby("patient_key").agg(
        score=(prob_col, "max"),
        label=("target", "max"),
    ).reset_index()
    if pat["label"].nunique() < 2:
        return float("nan")
    return roc_auc_score(pat["label"], pat["score"])


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc  = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += mask.sum() / n * abs(acc - conf)
    return ece


def conditional_auc_peak(tradeoff: pd.DataFrame) -> float:
    """Peak patient AUC from tradeoff curve."""
    return tradeoff["patient_auc"].max()


def bootstrap_conditional_auc(
    pairs: pd.DataFrame, tradeoff: pd.DataFrame, n: int = N_BOOTSTRAP
) -> np.ndarray:
    """
    Vectorized bootstrap over patients.

    Pre-build patient-level max arrays, then sweep gate thresholds with numpy
    broadcasting — avoids pandas groupby inside the loop.
    """
    patients = pairs["patient_key"].unique()
    n_pat = len(patients)
    pat_index = {p: i for i, p in enumerate(patients)}

    # Assign each pair a patient index
    pair_pat_idx = np.array([pat_index[k] for k in pairs["patient_key"]])

    # Array shapes: (n_pairs,)
    p_bf   = pairs["p_bf"].values.astype(np.float32)
    p_fuse = pairs["p_fused"].values.astype(np.float32)
    conf   = pairs["bf_confidence"].values.astype(np.float32)
    labels = pairs["target"].values.astype(np.float32)

    # Patient-level labels (max over pairs): (n_patients,)
    pat_labels = np.zeros(n_pat, dtype=np.float32)
    for pi, lab in zip(pair_pat_idx, labels):
        if lab > pat_labels[pi]:
            pat_labels[pi] = lab

    # Gate thresholds from tradeoff curve — thin down to 50 representative values
    gate_vals = tradeoff["threshold"].values
    if len(gate_vals) > 50:
        gate_vals = gate_vals[np.linspace(0, len(gate_vals) - 1, 50, dtype=int)]

    def _auc_np(scores: np.ndarray, labels_1d: np.ndarray) -> float:
        """Compute AUC without sklearn (Mann-Whitney)."""
        pos = scores[labels_1d == 1]
        neg = scores[labels_1d == 0]
        if len(pos) == 0 or len(neg) == 0:
            return float("nan")
        # Count pairs where pos > neg
        count = np.sum(pos[:, None] > neg[None, :])
        ties  = np.sum(pos[:, None] == neg[None, :])
        return float(count + 0.5 * ties) / (len(pos) * len(neg))

    aucs = np.empty(n)
    for i in range(n):
        # Sample patient indices with replacement
        samp_idx = RNG.integers(0, n_pat, size=n_pat)
        # For each pair, find how many times its patient was sampled
        # Then compute patient-level max score for each gate

        best = -np.inf
        for gate in gate_vals:
            # Choose BF or fused per pair
            p_use = np.where(conf < gate, p_fuse, p_bf)

            # Patient max score in bootstrap sample
            boot_pat_scores = np.full(n_pat, -1.0, dtype=np.float32)
            for pi, score in zip(pair_pat_idx, p_use):
                if score > boot_pat_scores[pi]:
                    boot_pat_scores[pi] = score

            # Extract bootstrap sample (patients may appear multiple times)
            s = boot_pat_scores[samp_idx]
            lab = pat_labels[samp_idx]

            auc = _auc_np(s, lab)
            if not np.isnan(auc) and auc > best:
                best = auc

        aucs[i] = best if best > -np.inf else float("nan")
    return aucs


# ── load data ─────────────────────────────────────────────────────────────────

scratch_pairs    = load_pairs("scratch_bf_routing")
byol_pairs       = load_pairs("byol_matched_final_val")    # no TTA, apples-to-apples
byol_tta_pairs   = load_pairs("byol_tta8_val")             # with TTA, headline

scratch_tradeoff = load_tradeoff("scratch_bf_routing")
byol_tradeoff    = load_tradeoff("byol_matched_final_val")
byol_tta_tradeoff= load_tradeoff("byol_tta8_val")


# ── calibration curves ────────────────────────────────────────────────────────

def _pair_calibration(pairs: pd.DataFrame):
    y = pairs["target"].values
    p = pairs["p_bf"].values
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    ece = expected_calibration_error(y, p)
    return frac_pos, mean_pred, ece


scratch_cal  = _pair_calibration(scratch_pairs)
byol_cal     = _pair_calibration(byol_pairs)
byol_tta_cal = _pair_calibration(byol_tta_pairs)


# ── bootstrap routing gap ─────────────────────────────────────────────────────

print("Running bootstrap for scratch conditional AUC...")
scratch_boot = bootstrap_conditional_auc(scratch_pairs, scratch_tradeoff)
print("Running bootstrap for BYOL (no TTA) conditional AUC...")
byol_boot = bootstrap_conditional_auc(byol_pairs, byol_tradeoff)
print("Running bootstrap for BYOL (TTA=8) conditional AUC...")
byol_tta_boot = bootstrap_conditional_auc(byol_tta_pairs, byol_tta_tradeoff)

gap_boot = byol_tta_boot - scratch_boot

scratch_peak  = conditional_auc_peak(scratch_tradeoff)
byol_peak     = conditional_auc_peak(byol_tradeoff)
byol_tta_peak = conditional_auc_peak(byol_tta_tradeoff)
gap_observed  = byol_tta_peak - scratch_peak

ci_lo, ci_hi = np.nanpercentile(gap_boot, [2.5, 97.5])
p_val = np.mean(gap_boot <= 0)

print(f"\nObserved gap (BYOL+TTA vs Scratch): {gap_observed:+.4f}")
print(f"Bootstrap 95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]")
print(f"P(gap ≤ 0): {p_val:.4f}")
print(f"Scratch peak: {scratch_peak:.4f}  (95% CI [{np.nanpercentile(scratch_boot,2.5):.4f}, {np.nanpercentile(scratch_boot,97.5):.4f}])")
print(f"BYOL peak:    {byol_peak:.4f}  (95% CI [{np.nanpercentile(byol_boot,2.5):.4f}, {np.nanpercentile(byol_boot,97.5):.4f}])")
print(f"BYOL+TTA peak:{byol_tta_peak:.4f}  (95% CI [{np.nanpercentile(byol_tta_boot,2.5):.4f}, {np.nanpercentile(byol_tta_boot,97.5):.4f}])")


# ── Figure 1: Calibration analysis ───────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
fig.suptitle("BF Model Calibration: Scratch vs BYOL SSL Pre-Training",
             fontsize=13, fontweight="bold", y=1.02)

datasets = [
    (scratch_cal,  SCRATCH_COL, "Scratch (no SSL)",           "scratch_bf_routing"),
    (byol_cal,     BYOL_COL,    "BYOL SSL (no TTA)",          "byol_matched_final_val"),
    (byol_tta_cal, BYOL_TTA_COL,"BYOL SSL + D4 TTA (TTA=8)",  "byol_tta8_val"),
]

for ax, (cal_data, col, label, _) in zip(axes, datasets):
    frac_pos, mean_pred, ece = cal_data
    # Ideal line
    ax.plot([0, 1], [0, 1], "--", color=IDEAL_COL, lw=1.5, label="Ideal", zorder=1)
    # Gap fill
    ax.fill_between(mean_pred, mean_pred, frac_pos,
                    alpha=0.18, color=col, zorder=2)
    # Calibration curve
    ax.plot(mean_pred, frac_pos, "o-", color=col, lw=2, ms=6,
            label=f"ECE = {ece:.3f}", zorder=3)
    ax.set_xlabel("Mean Predicted Probability", fontsize=10)
    ax.set_ylabel("Fraction of Positives", fontsize=10)
    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    # ECE bar annotation
    ax.text(0.97, 0.05, f"ECE = {ece:.3f}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=10, color=col, fontweight="bold")

plt.tight_layout()
cal_path = RESULTS / "calibration_analysis.png"
plt.savefig(cal_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved calibration figure to {cal_path}")


# ── Figure 2: Bootstrap routing gap ──────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.suptitle("Bootstrap Confidence Intervals: Conditional Routing AUC",
             fontsize=13, fontweight="bold")

# Left: distributions of individual conditional AUCs
ax = axes[0]
bins = np.linspace(0.55, 0.85, 35)
ax.hist(scratch_boot[~np.isnan(scratch_boot)], bins=bins,
        color=SCRATCH_COL, alpha=0.6, label=f"Scratch  (peak={scratch_peak:.3f})", density=True)
ax.hist(byol_boot[~np.isnan(byol_boot)], bins=bins,
        color=BYOL_COL, alpha=0.6, label=f"BYOL     (peak={byol_peak:.3f})", density=True)
ax.hist(byol_tta_boot[~np.isnan(byol_tta_boot)], bins=bins,
        color=BYOL_TTA_COL, alpha=0.6, label=f"BYOL+TTA (peak={byol_tta_peak:.3f})", density=True)

# Vertical lines at observed peaks
ax.axvline(scratch_peak,  color=SCRATCH_COL, lw=2, ls="--")
ax.axvline(byol_peak,     color=BYOL_COL,    lw=2, ls="--")
ax.axvline(byol_tta_peak, color=BYOL_TTA_COL, lw=2, ls="--")

ax.set_xlabel("Conditional Peak AUC (bootstrap)", fontsize=10)
ax.set_ylabel("Density", fontsize=10)
ax.set_title("Bootstrap Distributions", fontsize=10, fontweight="bold")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Right: gap distribution (BYOL+TTA - Scratch)
ax = axes[1]
valid_gap = gap_boot[~np.isnan(gap_boot)]
bins_gap = np.linspace(-0.15, 0.25, 35)
ax.hist(valid_gap, bins=bins_gap, color="#6a5acd", alpha=0.75,
        label=f"Gap = BYOL+TTA − Scratch", density=True)
ax.axvline(gap_observed, color="#6a5acd", lw=2.5, ls="-",
           label=f"Observed gap = {gap_observed:+.3f}")
ax.axvline(0, color="black", lw=1.5, ls="--", label="Zero gap")
ax.axvspan(ci_lo, ci_hi, alpha=0.15, color="#6a5acd",
           label=f"95% CI [{ci_lo:+.3f}, {ci_hi:+.3f}]")
ax.text(0.97, 0.95, f"p(gap≤0) = {p_val:.3f}", transform=ax.transAxes,
        ha="right", va="top", fontsize=10, color="#6a5acd", fontweight="bold")
ax.set_xlabel("AUC Gap (BYOL+TTA − Scratch)", fontsize=10)
ax.set_ylabel("Density", fontsize=10)
ax.set_title("Routing Quality Gap", fontsize=10, fontweight="bold")
ax.legend(fontsize=8, loc="upper left")
ax.grid(True, alpha=0.3)

plt.tight_layout()
boot_path = RESULTS / "bootstrap_routing_gap.png"
plt.savefig(boot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved bootstrap figure to {boot_path}")


# ── ECE summary ───────────────────────────────────────────────────────────────

print("\n── Calibration Summary ──")
for (cal_data, _col, label, _run), _lbl in zip(datasets, ["Scratch (no SSL)", "BYOL SSL (no TTA)", "BYOL SSL + D4 TTA"]):
    _, _, ece = cal_data
    print(f"  {_lbl:30s}  ECE = {ece:.4f}")
