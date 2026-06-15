#!/usr/bin/env python3
"""Method pulse-check figure: locality recovers cross-study robustness with cheap labels.

Forest plot of patient-level cross-study (mar->nov) AUC for three approaches that differ
in annotation cost and receptive field. Numbers from the local-MIL run + prior baselines.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "mil_locality_pulsecheck" / "method_pulsecheck.png"

# (label, auc, lo, hi, color, labelcost) -- plotted bottom-to-top
ROWS = [
    ("YOLOv8 detector\n(box labels — expensive)", 0.914, 0.859, 0.961, "#1A3C6E"),
    ("Local-MIL classifier — OURS\n(image labels — cheap)", 0.711, 0.636, 0.786, "#2E8B57"),
    ("Whole-image classifier\n(image labels — cheap)", 0.517, 0.432, 0.600, "#C44E52"),
]


def main() -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    for i, (label, auc, lo, hi, color) in enumerate(ROWS):
        ax.errorbar(auc, i, xerr=[[auc - lo], [hi - auc]], fmt="o", color=color,
                    capsize=5, markersize=10, lw=2)
        ax.annotate(f"{auc:.3f}  [{lo:.3f}, {hi:.3f}]", (auc, i), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=9.5, color=color, fontweight="bold")

    ax.axvline(0.5, color="grey", ls=":", lw=1.2)
    ax.annotate("chance", (0.5, -0.55), ha="center", fontsize=8.5, color="grey")
    # arrow showing the recovery
    ax.annotate("", xy=(0.711, 1), xytext=(0.517, 1),
                arrowprops=dict(arrowstyle="->", color="#2E8B57", lw=1.6))
    ax.annotate("locality, same cheap labels\n→ recovers ~half the gap", (0.614, 1),
                textcoords="offset points", xytext=(0, -34), ha="center", fontsize=8, color="#2E8B57")

    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels([r[0] for r in ROWS], fontsize=9)
    ax.set_ylim(-0.7, len(ROWS) - 0.3)
    ax.set_xlim(0.40, 1.0)
    ax.set_xlabel("Patient-level cross-study AUC (mar2020 → nov2021, 95% bootstrap CI)")
    ax.set_title("Forcing locality recovers cross-study robustness — without box annotations\n"
                 "380 nov2021 patients · same MobileNet backbone, same image-level labels", fontsize=10.5)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT, dpi=200)
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
