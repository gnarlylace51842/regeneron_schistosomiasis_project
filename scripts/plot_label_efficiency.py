#!/usr/bin/env python3
"""Plot label efficiency curves: BYOL vs SimCLR vs Scratch.

Reads all fine-tuning run directories, aggregates across seeds (mean ± std),
and generates the primary paper figure:
  val_patient_auc_max vs label_fraction for each initialization method.

Usage:
    python scripts/plot_label_efficiency.py
    python scripts/plot_label_efficiency.py --output results/label_efficiency.png
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
FINETUNE_DIR = REPO_ROOT / "runs" / "ssl" / "finetune"
RESULTS_DIR = REPO_ROOT / "results"


def _load_run_result(run_dir: Path) -> dict | None:
    """Extract key metrics from a fine-tuning run directory."""
    history_path = run_dir / "history.csv"
    config_path = run_dir / "config.json"
    if not history_path.exists():
        return None

    df = pd.read_csv(history_path)
    if "val_patient_auc_max" not in df.columns:
        return None

    best_idx = int(df["val_patient_auc_max"].fillna(-1).idxmax())
    best_patient_auc = float(df.loc[best_idx, "val_patient_auc_max"])
    best_pair_auc = float(df.loc[best_idx, "val_pair_auc"])

    label_fraction = None
    ssl_method = None
    seed = 42  # default

    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        label_fraction = cfg.get("data", {}).get("label_fraction")
        init_mode = cfg.get("ssl", {}).get("init_mode")
        enc_weights = cfg.get("ssl", {}).get("encoder_weights") or ""
        if "byol" in enc_weights.lower():
            ssl_method = "byol"
        elif init_mode == "ssl_pretrained":
            ssl_method = "simclr"
        else:
            ssl_method = "scratch"

    # Parse seed and label fraction from directory name as fallback
    name = run_dir.name
    if "smoke" in name:
        return None

    if label_fraction is None:
        for frac_str, frac_val in [("0p10", 0.10), ("0p25", 0.25), ("0p50", 0.50), ("1p00", 1.00)]:
            if frac_str in name:
                label_fraction = frac_val
                break

    if ssl_method is None:
        if "byol" in name:
            ssl_method = "byol"
        elif "ssl_ft" in name or ("simclr" in name):
            ssl_method = "simclr"
        elif "scratch" in name:
            ssl_method = "scratch"

    # Parse seed from run name (_s1, _s2, etc.)
    import re
    seed_match = re.search(r"_s(\d+)(?:_|$)", name)
    if seed_match:
        seed = int(seed_match.group(1))

    if label_fraction is None or ssl_method is None:
        return None

    return {
        "run_dir": str(run_dir),
        "label_fraction": label_fraction,
        "ssl_method": ssl_method,
        "seed": seed,
        "best_epoch": best_idx + 1,
        "val_patient_auc_max": best_patient_auc,
        "val_pair_auc": best_pair_auc,
    }


def collect_results() -> pd.DataFrame:
    rows = []
    if not FINETUNE_DIR.exists():
        return pd.DataFrame()
    for d in sorted(FINETUNE_DIR.iterdir()):
        if not d.is_dir():
            continue
        result = _load_run_result(d)
        if result is not None:
            rows.append(result)
    return pd.DataFrame(rows)


def aggregate_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate across seeds: compute mean and std per (method, label_fraction)."""
    agg = (
        df.groupby(["ssl_method", "label_fraction"])["val_patient_auc_max"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg.columns = ["ssl_method", "label_fraction", "mean_auc", "std_auc", "n_seeds"]
    agg["std_auc"] = agg["std_auc"].fillna(0.0)
    return agg


def print_table(raw: pd.DataFrame, agg: pd.DataFrame) -> None:
    print("\nLabel Efficiency Results (mean ± std across seeds)")
    print("=" * 72)
    print(f"{'Method':<12} {'Labels':>8} {'Mean AUC':>10} {'Std':>7} {'N seeds':>8}")
    print("-" * 72)
    method_order = ["scratch", "simclr", "byol"]
    for method in method_order:
        sub = agg[agg["ssl_method"] == method].sort_values("label_fraction")
        if sub.empty:
            continue
        for _, row in sub.iterrows():
            print(f"{method:<12} {row['label_fraction']:>8.0%} "
                  f"{row['mean_auc']:>10.4f} ±{row['std_auc']:>6.4f} "
                  f"{int(row['n_seeds']):>8d}")
        print()

    print("SSL Gains over Scratch (mean AUC delta)")
    print("-" * 50)
    scratch = agg[agg["ssl_method"] == "scratch"].set_index("label_fraction")["mean_auc"]
    for method in ["simclr", "byol"]:
        sub = agg[agg["ssl_method"] == method].sort_values("label_fraction")
        if sub.empty:
            continue
        print(f"  {method}:")
        for _, row in sub.iterrows():
            frac = row["label_fraction"]
            gain = row["mean_auc"] - scratch.get(frac, float("nan"))
            print(f"    {frac:.0%}: {gain:+.4f} (std={row['std_auc']:.4f}, n={int(row['n_seeds'])})")
        print()


def plot_curves(agg: pd.DataFrame, output_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5.5))

    style = {
        "scratch": dict(color="#555555", linestyle="--", marker="o",
                        label="From Scratch"),
        "simclr": dict(color="#2196F3", linestyle="-", marker="s",
                       label="SimCLR (cross-contrast SSL)"),
        "byol": dict(color="#E91E63", linestyle="-", marker="^",
                     label="BYOL (cross-contrast SSL)"),
    }

    for method, kw in style.items():
        sub = agg[agg["ssl_method"] == method].sort_values("label_fraction")
        if sub.empty:
            continue
        x = sub["label_fraction"].values * 100
        y = sub["mean_auc"].values
        yerr = sub["std_auc"].values
        color = kw["color"]

        ax.plot(x, y, linewidth=2.2, markersize=8, **kw)
        if (yerr > 0).any():
            ax.fill_between(x, y - yerr, y + yerr, alpha=0.15, color=color)

    # BF stage-1 supervised baseline
    ax.axhline(y=0.643, color="#999999", linestyle=":", linewidth=1.2,
               label="BF supervised baseline (100% labels)")

    ax.set_xlabel("Labelled Training Data Used (%)", fontsize=12)
    ax.set_ylabel("Patient-Level AUC (mean ± std)", fontsize=12)
    ax.set_title("Label Efficiency: Cross-Contrast SSL Pre-Training vs Scratch\n"
                 "Schistosomiasis Egg Detection (BF contrast, 3 seeds)", fontsize=11)
    ax.set_xlim(5, 105)
    ax.set_ylim(0.55, 0.76)
    ax.set_xticks([10, 25, 50, 100])
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {output_path}")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=RESULTS_DIR / "label_efficiency.png")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    raw = collect_results()
    if raw.empty:
        print("No fine-tuning results found.")
        return 1

    agg = aggregate_seeds(raw)
    print_table(raw, agg)

    # Save both raw and aggregated CSVs
    summary_path = RESULTS_DIR / "label_efficiency_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    raw.sort_values(["ssl_method", "label_fraction", "seed"]).to_csv(
        RESULTS_DIR / "label_efficiency_raw.csv", index=False)
    agg.sort_values(["ssl_method", "label_fraction"]).to_csv(summary_path, index=False)
    print(f"Saved raw CSV to {RESULTS_DIR / 'label_efficiency_raw.csv'}")
    print(f"Saved summary CSV to {summary_path}")

    if not args.no_plot:
        plot_curves(agg, args.output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
