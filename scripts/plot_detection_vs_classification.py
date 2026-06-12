#!/usr/bin/env python3
"""Headline figure: detection vs classification under the mar->nov study shift.

One forest plot telling the whole story at patient level:
  1. Classifier, WITHIN-study (works)        -- reference from project_status
  2. Classifier, cross-study mar->nov (collapses toward chance)
  3. YOLOv8 detector, cross-study mar->nov (holds)

Reads:
  results/cross_study_mar_to_nov/results.csv   (classifier mar->nov, this repo)
  results/yolov8_bf_mar_to_nov/summary.json    (detector mar->nov)

Output:
  results/yolov8_bf_mar_to_nov/detection_vs_classification.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    clf_csv = REPO_ROOT / "results" / "cross_study_mar_to_nov" / "results.csv"
    det_json = REPO_ROOT / "results" / "yolov8_bf_mar_to_nov" / "summary.json"
    if not clf_csv.exists():
        raise FileNotFoundError(f"Missing classifier results: {clf_csv} (run cross_study_mar_to_nov_classifier.py)")
    if not det_json.exists():
        raise FileNotFoundError(f"Missing detector summary: {det_json}")

    clf = pd.read_csv(clf_csv)
    best = clf.loc[clf["nov2021_auc"].idxmax()]  # strongest classifier = fairest comparison
    det = json.loads(det_json.read_text())

    # rows: (label, auc, lo, hi, color)  -- plotted bottom-to-top; lo/hi None = point only
    rows = [
        (f"Detector — cross-study\n(YOLOv8s, mar→nov)", det["patient_auc"], det["auc_lo"], det["auc_hi"], "#1A3C6E"),
        (f"Classifier — cross-study\n({best['arch']}, mar→nov)", float(best["nov2021_auc"]), float(best["nov2021_lo"]), float(best["nov2021_hi"]), "#C44E52"),
        (f"Classifier — within-study\n({best['arch']}, mar val)", float(best["mar2020_val_auc"]), None, None, "#888888"),
    ]

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for i, (label, auc, lo, hi, color) in enumerate(rows):
        if lo is None or hi is None:
            ax.plot(auc, i, "o", color=color, markersize=9)
            ax.annotate(f"{auc:.3f}", (auc, i), textcoords="offset points",
                        xytext=(0, 11), ha="center", fontsize=9, color=color, fontweight="bold")
        else:
            ax.errorbar(auc, i, xerr=[[auc - lo], [hi - auc]], fmt="o", color=color,
                        capsize=5, markersize=9, lw=2)
            ax.annotate(f"{auc:.3f}  [{lo:.3f}, {hi:.3f}]", (auc, i),
                        textcoords="offset points", xytext=(0, 11), ha="center",
                        fontsize=9, color=color, fontweight="bold")

    ax.axvline(0.5, color="grey", ls=":", lw=1.2)
    ax.annotate("chance", (0.5, -0.55), ha="center", fontsize=8.5, color="grey")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=9.5)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlim(0.40, 1.0)
    ax.set_xlabel("Patient-level AUC  (95% bootstrap CI)")
    ax.set_title("Localize, don't classify: detection survives the mar2020→nov2021 study shift\n"
                 "that collapses classification  (380 nov2021 patients, 61 positive)", fontsize=10.5)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()

    out = REPO_ROOT / "results" / "yolov8_bf_mar_to_nov" / "detection_vs_classification.png"
    fig.savefig(out, dpi=200)
    print("Classifier (best, mar→nov):", best["arch"],
          f"{best['nov2021_auc']:.4f} [{best['nov2021_lo']:.4f}, {best['nov2021_hi']:.4f}]")
    print("Detector (mar→nov):       ",
          f"{det['patient_auc']:.4f} [{det['auc_lo']:.4f}, {det['auc_hi']:.4f}]")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
