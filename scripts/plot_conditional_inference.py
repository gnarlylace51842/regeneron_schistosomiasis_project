#!/usr/bin/env python3
"""Generate the definitive conditional inference figure for the paper.

Two-panel figure:
  Left:  Patient AUC vs % DF compute — BYOL BF vs Scratch BF (same DF model)
         Proves SSL improves routing quality, not just accuracy.
  Right: Sensitivity@80%spec vs % DF compute — same comparison.

Plus annotated operating points for the paper's key claims.

Usage:
    python scripts/plot_conditional_inference.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"


def _load_curve(run_name: str, curve_file: str = "tradeoff_curve.csv") -> pd.DataFrame | None:
    path = RESULTS_DIR / "conditional_inference" / run_name / curve_file
    if not path.exists():
        return None
    return pd.read_csv(path)


def _load_ops(run_name: str) -> dict:
    import json
    path = RESULTS_DIR / "conditional_inference" / run_name / "operating_points.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def main() -> int:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required.")
        return 1

    # Load curves — prefer TTA results if available
    _byol_run = "byol_tta8_val" if (RESULTS_DIR / "conditional_inference" / "byol_tta8_val" / "tradeoff_curve.csv").exists() else "byol_matched_final_val"
    byol_conf = _load_curve(_byol_run, "tradeoff_curve.csv")
    byol_align = _load_curve(_byol_run, "tradeoff_curve_alignment.csv")
    scratch_conf = _load_curve("scratch_bf_routing", "tradeoff_curve.csv")
    # Original no-TTA curve for TTA ablation (if different from main)
    no_tta_conf = _load_curve("byol_matched_final_val", "tradeoff_curve.csv") if _byol_run != "byol_matched_final_val" else None

    byol_ops = _load_ops(_byol_run)
    scratch_ops = _load_ops("scratch_bf_routing")

    if byol_conf is None or scratch_conf is None:
        print("Missing curve files. Run eval_conditional_inference.py first.")
        return 1

    byol_bf_auc = byol_ops.get("baselines", {}).get("bf_only_patient_auc", 0.692)
    scratch_bf_auc = scratch_ops.get("baselines", {}).get("bf_only_patient_auc", 0.669)
    df_only_auc = byol_ops.get("baselines", {}).get("df_only_patient_auc", 0.644)
    always_fused_auc = byol_ops.get("baselines", {}).get("always_fused_naive_patient_auc", 0.711)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # ── Left panel: Patient AUC vs DF compute ──
    ax = axes[0]
    ax.plot(byol_conf["df_fraction"] * 100, byol_conf["patient_auc"],
            color="#E91E63", linewidth=2.5, label="BYOL BF → conditional routing")
    ax.plot(scratch_conf["df_fraction"] * 100, scratch_conf["patient_auc"],
            color="#555555", linewidth=2.0, linestyle="--",
            label="Scratch BF → conditional routing")
    if byol_align is not None:
        ax.plot(byol_align["df_fraction"] * 100, byol_align["patient_auc"],
                color="#E91E63", linewidth=1.8, linestyle="-.",
                alpha=0.7, label="BYOL BF → alignment gate")
    if no_tta_conf is not None:
        ax.plot(no_tta_conf["df_fraction"] * 100, no_tta_conf["patient_auc"],
                color="#FF9800", linewidth=1.8, linestyle=":",
                alpha=0.8, label="BYOL BF → conditional (no TTA)")

    # Baselines
    ax.axhline(byol_bf_auc, color="#E91E63", linestyle=":", linewidth=1.3, alpha=0.7,
               label=f"BYOL BF-only  AUC={byol_bf_auc:.3f}")
    ax.axhline(scratch_bf_auc, color="#555555", linestyle=":", linewidth=1.3, alpha=0.7,
               label=f"Scratch BF-only  AUC={scratch_bf_auc:.3f}")
    ax.axhline(always_fused_auc, color="#2196F3", linestyle="--", linewidth=1.5,
               label=f"Always-fused dual  AUC={always_fused_auc:.3f}")

    # Annotate BYOL peak
    byol_peak_idx = int(byol_conf["patient_auc"].idxmax())
    byol_peak_auc = float(byol_conf.loc[byol_peak_idx, "patient_auc"])
    byol_peak_df = float(byol_conf.loc[byol_peak_idx, "df_fraction"])
    ax.annotate(
        f"Peak: AUC={byol_peak_auc:.3f}\n({byol_peak_df:.0%} DF compute)",
        xy=(byol_peak_df * 100, byol_peak_auc),
        xytext=(byol_peak_df * 100 + 15, byol_peak_auc - 0.025),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#E91E63", lw=1.5),
        color="#E91E63",
    )

    ax.set_xlabel("% Pairs Routed to Darkfield (Compute Cost)", fontsize=11)
    ax.set_ylabel("Patient-Level AUC", fontsize=11)
    ax.set_title("Conditional Routing: AUC vs Compute\n"
                 "SSL Pre-Training Improves Routing Quality", fontsize=11)
    ax.set_xlim(0, 100)
    ax.set_ylim(0.60, 0.80)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(True, alpha=0.3)

    # ── Right panel: Sensitivity@80%spec vs DF compute ──
    ax = axes[1]
    sens_col = "sensitivity_at_spec0p8"
    if sens_col in byol_conf.columns:
        ax.plot(byol_conf["df_fraction"] * 100,
                pd.to_numeric(byol_conf[sens_col], errors="coerce"),
                color="#E91E63", linewidth=2.5, label="BYOL BF → conditional")
    if sens_col in scratch_conf.columns:
        ax.plot(scratch_conf["df_fraction"] * 100,
                pd.to_numeric(scratch_conf[sens_col], errors="coerce"),
                color="#555555", linewidth=2.0, linestyle="--",
                label="Scratch BF → conditional")
    if byol_align is not None and sens_col in byol_align.columns:
        ax.plot(byol_align["df_fraction"] * 100,
                pd.to_numeric(byol_align[sens_col], errors="coerce"),
                color="#E91E63", linewidth=1.8, linestyle="-.", alpha=0.7,
                label="BYOL BF → alignment gate")

    # Annotate peak sensitivity point for BYOL
    if sens_col in byol_conf.columns:
        peak_sens_idx = int(pd.to_numeric(byol_conf[sens_col], errors="coerce").idxmax())
        peak_sens = float(byol_conf.loc[peak_sens_idx, sens_col])
        peak_df_s = float(byol_conf.loc[peak_sens_idx, "df_fraction"])
        ax.annotate(
            f"Sensitivity={peak_sens:.2f}\n({peak_df_s:.0%} DF compute)",
            xy=(peak_df_s * 100, peak_sens),
            xytext=(peak_df_s * 100 + 10, peak_sens - 0.12),
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color="#E91E63", lw=1.5),
            color="#E91E63",
        )

    ax.set_xlabel("% Pairs Routed to Darkfield (Compute Cost)", fontsize=11)
    ax.set_ylabel("Patient Sensitivity @ 80% Specificity", fontsize=11)
    ax.set_title("Conditional Routing: Sensitivity vs Compute\n"
                 "BYOL BF Achieves Higher Sensitivity at Lower Cost", fontsize=11)
    ax.set_xlim(0, 100)
    ax.set_ylim(0.20, 0.85)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Resource-Aware Conditional Dual-Contrast Schistosomiasis Diagnosis\n"
        "BYOL Cross-Contrast SSL Pre-Training Enables Smarter Compute Routing",
        fontsize=12, y=1.01,
    )

    out = RESULTS_DIR / "conditional_inference_figure.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Saved figure to {out}")

    # Print the key paper claim numbers
    print("\n── Key Numbers for Paper ──")
    print(f"BF-only (scratch):         AUC={scratch_bf_auc:.4f}")
    print(f"BF-only (BYOL SSL):        AUC={byol_bf_auc:.4f}  (+{byol_bf_auc-scratch_bf_auc:+.4f})")
    print(f"Always-fused dual:         AUC={always_fused_auc:.4f}")
    byol_peak_row = byol_conf.loc[byol_conf["patient_auc"].idxmax()]
    scratch_peak_row = scratch_conf.loc[scratch_conf["patient_auc"].idxmax()]
    print(f"Conditional BYOL peak:     AUC={byol_peak_row['patient_auc']:.4f}  "
          f"at {byol_peak_row['df_fraction']:.0%} DF  "
          f"(+{byol_peak_row['patient_auc']-always_fused_auc:+.4f} vs always-fused)")
    print(f"Conditional Scratch peak:  AUC={scratch_peak_row['patient_auc']:.4f}  "
          f"at {scratch_peak_row['df_fraction']:.0%} DF")
    print(f"SSL routing quality gain:  "
          f"{byol_peak_row['patient_auc']-scratch_peak_row['patient_auc']:+.4f} "
          f"at peak operating point")
    if sens_col in byol_conf.columns:
        byol_sens_peak = pd.to_numeric(byol_conf[sens_col], errors="coerce").max()
        scratch_sens_peak = pd.to_numeric(scratch_conf[sens_col], errors="coerce").max() if sens_col in scratch_conf.columns else float("nan")
        print(f"Peak sensitivity@80%spec (BYOL):   {byol_sens_peak:.4f}")
        print(f"Peak sensitivity@80%spec (Scratch): {scratch_sens_peak:.4f}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
