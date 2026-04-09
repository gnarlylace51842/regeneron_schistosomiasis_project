#!/usr/bin/env python3
"""Plot label efficiency curves: BYOL vs SimCLR vs Scratch.

Reads all fine-tuning run directories and generates the primary paper figure:
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
import sys

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
    init_mode = None
    ssl_method = None

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

    # Fallback: parse from directory name
    if label_fraction is None:
        name = run_dir.name
        for frac_str, frac_val in [("0p10", 0.10), ("0p25", 0.25), ("0p50", 0.50), ("1p00", 1.00)]:
            if frac_str in name:
                label_fraction = frac_val
                break

    if init_mode is None:
        name = run_dir.name
        if "byol" in name:
            ssl_method = "byol"
        elif "ssl_ft" in name:
            ssl_method = "simclr"
        elif "scratch" in name:
            ssl_method = "scratch"

    if label_fraction is None or ssl_method is None:
        return None
    if "smoke" in run_dir.name:
        return None

    return {
        "run_dir": str(run_dir),
        "label_fraction": label_fraction,
        "ssl_method": ssl_method,
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


def print_table(df: pd.DataFrame) -> None:
    print("\nLabel Efficiency Results")
    print("=" * 70)
    print(f"{'Method':<12} {'Labels':>8} {'PatAUC(max)':>12} {'PairAUC':>10}")
    print("-" * 70)
    method_order = ["scratch", "simclr", "byol"]
    for method in method_order:
        sub = df[df["ssl_method"] == method].sort_values("label_fraction")
        if sub.empty:
            continue
        for _, row in sub.iterrows():
            print(f"{method:<12} {row['label_fraction']:>8.0%} "
                  f"{row['val_patient_auc_max']:>12.4f} "
                  f"{row['val_pair_auc']:>10.4f}")
        print()

    # Print relative gains of SSL methods over scratch at each fraction
    print("SSL Gains over Scratch (patient AUC delta)")
    print("-" * 50)
    scratch = df[df["ssl_method"] == "scratch"].set_index("label_fraction")["val_patient_auc_max"]
    for method in ["simclr", "byol"]:
        sub = df[df["ssl_method"] == method].sort_values("label_fraction")
        if sub.empty:
            continue
        print(f"  {method}:")
        for _, row in sub.iterrows():
            frac = row["label_fraction"]
            gain = row["val_patient_auc_max"] - scratch.get(frac, float("nan"))
            print(f"    {frac:.0%}: {gain:+.4f}")
        print()


def plot_curves(df: pd.DataFrame, output_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    style = {
        "scratch": dict(color="#555555", linestyle="--", marker="o", label="Scratch"),
        "simclr": dict(color="#2196F3", linestyle="-", marker="s", label="SimCLR (cross-contrast)"),
        "byol": dict(color="#E91E63", linestyle="-", marker="^", label="BYOL (cross-contrast)"),
    }

    for method, kw in style.items():
        sub = df[df["ssl_method"] == method].sort_values("label_fraction")
        if sub.empty:
            continue
        ax.plot(
            sub["label_fraction"] * 100,
            sub["val_patient_auc_max"],
            linewidth=2, markersize=7, **kw,
        )

    ax.set_xlabel("Labelled Training Data Used (%)", fontsize=12)
    ax.set_ylabel("Patient-Level AUC (max aggregation)", fontsize=12)
    ax.set_title("Label Efficiency: SSL Pre-Training vs Scratch\n"
                 "Schistosomiasis Egg Detection (BF contrast)", fontsize=12)
    ax.set_xlim(0, 105)
    ax.set_ylim(0.55, 0.75)
    ax.axhline(y=0.643, color="#999999", linestyle=":", linewidth=1, label="BF stage-1 baseline")
    ax.set_xticks([10, 25, 50, 100])
    ax.legend(fontsize=10)
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

    df = collect_results()
    if df.empty:
        print("No fine-tuning results found.")
        return 1

    print_table(df)

    # Save summary CSV
    summary_path = RESULTS_DIR / "label_efficiency_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values(["ssl_method", "label_fraction"]).to_csv(summary_path, index=False)
    print(f"Saved summary CSV to {summary_path}")

    if not args.no_plot:
        plot_curves(df, args.output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
