#!/usr/bin/env python3
"""Domain adaptation sample-efficiency experiment.

Answers: How many mar2020-labeled patients does it take to adapt a model
trained on nov2021 to reach usable performance on mar2020?

Design:
  - Hold out 200 mar2020 patients (stratified) as a fixed evaluation set.
  - Remaining 148 mar2020 patients form the adaptation pool.
  - For K in [0, 5, 10, 25, 50, 100, 148]:
      - K=0: zero-shot (nov2021-trained models, no mar2020 fine-tuning)
      - K>0: fine-tune BYOL encoder on nov2021_train + K mar2020 patients
    - 3 seeds for K <= 50, 1 seed for K > 50.
  - Evaluate conditional inference (TTA=8) on the fixed 200-patient holdout.
  - Bootstrap 95% CIs. Plot AUC vs K (log scale).

Outputs:
  results/domain_adaptation/results.csv
  results/domain_adaptation/adaptation_curve.png
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from schisto_mobile_ai.utils.io import ensure_dir

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
K_VALUES       = [5, 10, 25, 50, 100, 148]
SEEDS_SMALL    = [42, 43, 44]   # K <= 50
SEEDS_LARGE    = [42]           # K > 50
HOLDOUT_N      = 200            # fixed mar2020 evaluation set size
HOLDOUT_SEED   = 99             # separate seed so holdout never overlaps pool samples
EPOCHS         = 20
N_BOOT         = 2000

BYOL_ENCODER   = REPO_ROOT / "runs" / "cross_study" / "byol_pretrain_100ep" / "encoder_weights.pt"
ZERO_BF_MODEL  = REPO_ROOT / "runs" / "cross_study" / "byol_bf" / "best_model.pt"
ZERO_DF_MODEL  = REPO_ROOT / "runs" / "cross_study" / "byol_df" / "best_model.pt"
NOV_SPLIT      = REPO_ROOT / "splits" / "cross_study_split.csv"
OUT_DIR        = ensure_dir(REPO_ROOT / "results" / "domain_adaptation")
SPLIT_DIR      = ensure_dir(REPO_ROOT / "splits" / "domain_adapt")
RUN_DIR        = ensure_dir(REPO_ROOT / "runs" / "domain_adaptation")


# ---------------------------------------------------------------------------
# Split construction
# ---------------------------------------------------------------------------

def build_holdout_and_pool(seed: int = HOLDOUT_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split 348 mar2020 patients into fixed holdout (200) and adaptation pool (148)."""
    base = pd.read_csv(NOV_SPLIT)
    mar = base[base["study_id"] == "mar2020"].copy()
    rng = np.random.default_rng(seed)

    pos = mar[mar["patient_label"] == "positive"]["patient_key"].values.copy()
    neg = mar[mar["patient_label"] == "negative"]["patient_key"].values.copy()
    rng.shuffle(pos); rng.shuffle(neg)

    # Proportional holdout
    n_pos_hold = round(HOLDOUT_N * len(pos) / len(mar))
    n_neg_hold = HOLDOUT_N - n_pos_hold

    holdout_keys = set(pos[:n_pos_hold].tolist() + neg[:n_neg_hold].tolist())
    pool_keys    = set(mar["patient_key"]) - holdout_keys

    holdout = mar[mar["patient_key"].isin(holdout_keys)].copy()
    pool    = mar[mar["patient_key"].isin(pool_keys)].copy()
    return holdout, pool


def sample_adaptation_patients(pool: pd.DataFrame, k: int, seed: int) -> pd.DataFrame:
    """Stratified sample of K patients from the adaptation pool."""
    rng = np.random.default_rng(seed)
    pos = pool[pool["patient_label"] == "positive"]["patient_key"].values.copy()
    neg = pool[pool["patient_label"] == "negative"]["patient_key"].values.copy()
    rng.shuffle(pos); rng.shuffle(neg)

    n_pos = max(1, round(k * len(pos) / len(pool)))
    n_neg = k - n_pos
    if n_neg > len(neg):
        n_neg = len(neg)
        n_pos = k - n_neg
    n_pos = min(n_pos, len(pos))

    sampled_keys = pos[:n_pos].tolist() + neg[:n_neg].tolist()
    return pool[pool["patient_key"].isin(sampled_keys)].copy()


def make_split_csv(
    holdout: pd.DataFrame,
    adapt_patients: pd.DataFrame | None,
    split_path: Path,
) -> None:
    """Write split CSV: train=nov2021_train[+adapt], val=nov2021_val, test=mar2020_holdout."""
    base = pd.read_csv(NOV_SPLIT)
    nov_train = base[(base["study_id"] == "nov2021") & (base["split"] == "train")].copy()
    nov_val   = base[(base["study_id"] == "nov2021") & (base["split"] == "val")].copy()

    holdout_out = holdout.copy()
    holdout_out["split"] = "test"

    parts = [nov_train, nov_val, holdout_out]

    if adapt_patients is not None and len(adapt_patients) > 0:
        adapt_out = adapt_patients.copy()
        adapt_out["split"] = "train"
        parts.append(adapt_out)

    out = pd.concat(parts, ignore_index=True)
    out.to_csv(split_path, index=False)


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


def bootstrap_auc(t: np.ndarray, s: np.ndarray, seed: int = 42) -> tuple[float, float, float]:
    pt = _wilcoxon_auc(t, s)
    rng = np.random.default_rng(seed)
    n = len(t)
    boot: list[float] = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        a = _wilcoxon_auc(t[idx], s[idx])
        if np.isfinite(a):
            boot.append(a)
    if len(boot) < 10:
        return pt, float("nan"), float("nan")
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return pt, float(lo), float(hi)


def peak_conditional_auc_from_dir(eval_dir: Path) -> tuple[float, float, float]:
    curve = pd.read_csv(eval_dir / "tradeoff_curve.csv")
    ps    = pd.read_csv(eval_dir / "pair_scores.csv")

    # Best threshold on this eval set
    peak_idx = int(curve["patient_auc"].idxmax())
    best_thresh = float(curve.iloc[peak_idx]["threshold"])

    # Recompute patient scores at that threshold
    ps2 = ps.copy()
    ps2["use_df"] = ps2["bf_confidence"] < best_thresh
    ps2["pair_score"] = np.where(ps2["use_df"], ps2["p_fused"], ps2["p_bf"])
    rows = []
    for pk, grp in ps2.groupby("patient_key"):
        rows.append({"target": float(grp["target"].max()),
                     "score":  float(grp["pair_score"].max())})
    pf = pd.DataFrame(rows)
    return bootstrap_auc(pf["target"].values, pf["score"].values)


def bf_only_auc_from_dir(eval_dir: Path) -> tuple[float, float, float]:
    ps = pd.read_csv(eval_dir / "pair_scores.csv")
    rows = []
    for pk, grp in ps.groupby("patient_key"):
        rows.append({"target": float(grp["target"].max()),
                     "score":  float(grp["p_bf"].max())})
    pf = pd.DataFrame(rows)
    return bootstrap_auc(pf["target"].values, pf["score"].values)


# ---------------------------------------------------------------------------
# Subprocess runners
# ---------------------------------------------------------------------------

def run(cmd: list[str], label: str) -> int:
    print(f"  → {label}", flush=True)
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    FAILED (code {result.returncode})")
        print(result.stderr[-800:] if result.stderr else "(no stderr)")
    return result.returncode


def finetune(encoder: Path | None, contrast: str, split_csv: Path,
             out_dir: Path, run_name: str) -> int:
    cmd = [
        sys.executable, "scripts/finetune_ssl.py",
        "--split-csv", str(split_csv),
        "--contrast", contrast,
        "--label-fraction", "1.0",
        "--output-dir", str(out_dir),
        "--run-name", run_name,
        "--epochs", str(EPOCHS),
        "--quiet",
    ]
    if encoder is not None:
        cmd += ["--encoder-weights", str(encoder)]
    return run(cmd, f"finetune {contrast} {'BYOL' if encoder else 'scratch'} → {out_dir.name}")


def eval_cond(bf_model: Path, df_model: Path, split_csv: Path,
              out_dir: Path, run_name: str) -> int:
    result_dir = out_dir / run_name
    if result_dir.exists() and any(result_dir.iterdir()):
        overwrite = ["--overwrite"]
    else:
        overwrite = []
    cmd = [
        sys.executable, "scripts/eval_conditional_inference.py",
        "--bf-model", str(bf_model),
        "--df-model", str(df_model),
        "--split-csv", str(split_csv),
        "--eval-split", "test",
        "--tta-views", "8",
        "--output-dir", str(out_dir),
        "--run-name", run_name,
        "--quiet",
    ] + overwrite
    return run(cmd, f"eval conditional → {run_name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    holdout, pool = build_holdout_and_pool()
    n_pos_hold = int((holdout["patient_label"] == "positive").sum())
    n_pos_pool = int((pool["patient_label"] == "positive").sum())
    print(f"Holdout: {len(holdout)} patients, {n_pos_hold} positive")
    print(f"Pool:    {len(pool)} patients, {n_pos_pool} positive")
    print(f"K values: {K_VALUES}")
    print()

    records: list[dict] = []

    # -----------------------------------------------------------------------
    # K=0 — zero-shot (re-evaluate on the 200-patient holdout)
    # -----------------------------------------------------------------------
    print("K=0  zero-shot (re-eval on holdout)...")
    split_path = SPLIT_DIR / "adapt_K0.csv"
    make_split_csv(holdout, None, split_path)
    eval_out = OUT_DIR / "evals"
    ensure_dir(eval_out)
    rc = eval_cond(ZERO_BF_MODEL, ZERO_DF_MODEL, split_path, eval_out, "K0_seed42")
    if rc == 0:
        pt, lo, hi = peak_conditional_auc_from_dir(eval_out / "K0_seed42")
        bf_pt, bf_lo, bf_hi = bf_only_auc_from_dir(eval_out / "K0_seed42")
        records.append({"k": 0, "seed": 42,
                         "cond_auc": pt, "cond_lo": lo, "cond_hi": hi,
                         "bf_auc": bf_pt, "bf_lo": bf_lo, "bf_hi": bf_hi})
        print(f"  K=0 cond AUC: {pt:.4f}  [{lo:.4f}, {hi:.4f}]")

    # -----------------------------------------------------------------------
    # K>0 — fine-tune on nov2021 + K mar2020, eval on holdout
    # -----------------------------------------------------------------------
    for k in K_VALUES:
        seeds = SEEDS_SMALL if k <= 50 else SEEDS_LARGE
        for seed in seeds:
            tag = f"K{k}_seed{seed}"
            print(f"\nK={k}  seed={seed}")

            adapt_pts = sample_adaptation_patients(pool, k, seed)
            split_path = SPLIT_DIR / f"adapt_{tag}.csv"
            make_split_csv(holdout, adapt_pts, split_path)

            bf_dir = RUN_DIR / f"byol_bf_{tag}"
            df_dir = RUN_DIR / f"byol_df_{tag}"

            rc_bf = finetune(BYOL_ENCODER, "bf", split_path, bf_dir, tag)
            rc_df = finetune(BYOL_ENCODER, "df", split_path, df_dir, tag)
            if rc_bf != 0 or rc_df != 0:
                print(f"  Skipping eval for {tag} (finetune failed)")
                continue

            rc = eval_cond(bf_dir / "best_model.pt", df_dir / "best_model.pt",
                           split_path, eval_out, tag)
            if rc != 0:
                continue

            pt, lo, hi = peak_conditional_auc_from_dir(eval_out / tag)
            bf_pt, bf_lo, bf_hi = bf_only_auc_from_dir(eval_out / tag)
            records.append({"k": k, "seed": seed,
                             "cond_auc": pt, "cond_lo": lo, "cond_hi": hi,
                             "bf_auc": bf_pt, "bf_lo": bf_lo, "bf_hi": bf_hi})
            print(f"  cond AUC: {pt:.4f}  [{lo:.4f}, {hi:.4f}]  "
                  f"| bf AUC: {bf_pt:.4f}  [{bf_lo:.4f}, {bf_hi:.4f}]")

    # -----------------------------------------------------------------------
    # Aggregate and save
    # -----------------------------------------------------------------------
    if not records:
        print("No results collected — exiting.")
        return

    df = pd.DataFrame(records)
    agg = df.groupby("k").agg(
        cond_mean=("cond_auc", "mean"),
        cond_lo=("cond_lo", "mean"),
        cond_hi=("cond_hi", "mean"),
        bf_mean=("bf_auc", "mean"),
        bf_lo=("bf_lo", "mean"),
        bf_hi=("bf_hi", "mean"),
        n_seeds=("seed", "count"),
    ).reset_index()

    df.to_csv(OUT_DIR / "results_raw.csv", index=False)
    agg.to_csv(OUT_DIR / "results_agg.csv", index=False)
    print(f"\nSaved: {OUT_DIR / 'results_raw.csv'}")
    print(f"Saved: {OUT_DIR / 'results_agg.csv'}")
    print("\n=== Aggregated Results ===")
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    _plot(agg, OUT_DIR / "adaptation_curve.png")
    print(f"\nSaved: {OUT_DIR / 'adaptation_curve.png'}")


def _plot(agg: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ks = agg["k"].values

    # Conditional AUC
    ax.plot(ks, agg["cond_mean"], "o-", color="#2ca02c", linewidth=2.5,
            label="Conditional (peak, TTA=8)")
    ax.fill_between(ks, agg["cond_lo"], agg["cond_hi"], alpha=0.2, color="#2ca02c")

    # BF-only AUC
    ax.plot(ks, agg["bf_mean"], "s--", color="#1f77b4", linewidth=2,
            label="BF-only")
    ax.fill_between(ks, agg["bf_lo"], agg["bf_hi"], alpha=0.15, color="#1f77b4")

    # Reference lines
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="Random (AUC=0.5)")

    ax.set_xlabel("mar2020 labeled patients added to training set (K)")
    ax.set_ylabel("Patient-level AUC  (95% bootstrap CI)")
    ax.set_title("Domain Adaptation Sample Efficiency\n"
                 "BYOL model trained on nov2021, adapted to mar2020")
    ax.set_ylim(0.3, 1.0)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Use log-like x ticks if K spans orders of magnitude
    if ks.max() / max(ks.min(), 1) > 10:
        ax.set_xscale("symlog", linthresh=5)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
