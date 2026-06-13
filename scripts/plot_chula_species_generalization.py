#!/usr/bin/env python3
"""Chula-ParasiteEgg-11: held-out-species generalization figure.

A single-class 'parasite_egg' YOLOv8s detector trained on 8 species and tested on
3 species it never saw. Numbers from the Kaggle run (ultralytics val), recorded
here for the repo. Shows detection (localization) generalizes across species ->
the second pillar of the "localize the invariant target" thesis.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "results" / "chula_species_generalization"
OUT.mkdir(parents=True, exist_ok=True)

RES = {
    "model": "YOLOv8s, single class 'parasite_egg', 60 epochs",
    "held_out_unseen_species": ["Ascaris lumbricoides", "Capillaria philippinensis", "Trichuris trichiura"],
    "train_species": ["Enterobius vermicularis", "Fasciolopsis buski", "Hookworm egg",
                       "Hymenolepis diminuta", "Hymenolepis nana", "Opisthorchis viverrine",
                       "Paragonimus spp", "Taenia spp. egg"],
    "in_dist_8_seen":   {"map50": 0.992, "map50_95": 0.943},
    "ood_3_unseen":     {"map50": 0.993, "map50_95": 0.896},
    "source": "Kaggle commit, ultralytics best.val() on held-out-species test split",
}
(OUT / "results.json").write_text(json.dumps(RES, indent=2))


def main() -> None:
    metrics = ["mAP@50", "mAP@[.5:.95]"]
    seen = [RES["in_dist_8_seen"]["map50"], RES["in_dist_8_seen"]["map50_95"]]
    unseen = [RES["ood_3_unseen"]["map50"], RES["ood_3_unseen"]["map50_95"]]
    x = np.arange(2); w = 0.36

    fig, ax = plt.subplots(figsize=(6.6, 4.7))
    b1 = ax.bar(x - w / 2, seen, w, label="8 seen species (in-distribution)", color="#9aa0a6")
    b2 = ax.bar(x + w / 2, unseen, w, label="3 UNSEEN species (held out)", color="#1A3C6E")
    for b in list(b1) + list(b2):
        ax.annotate(f"{b.get_height():.3f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9.5, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(metrics)
    ax.set_ylim(0.6, 1.03); ax.set_ylabel("Detection mAP")
    ax.set_title("Class-agnostic egg detector generalizes to UNSEEN parasite species\n"
                 "Chula-ParasiteEgg-11: train on 8 species → test on 3 held-out species", fontsize=10)
    ax.legend(fontsize=8.5, loc="lower left"); ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "chula_species_generalization.png", dpi=200)
    print("Saved:", OUT / "chula_species_generalization.png")


if __name__ == "__main__":
    main()
