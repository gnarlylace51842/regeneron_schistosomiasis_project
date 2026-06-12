#!/usr/bin/env python3
"""Mechanism panel: Grad-CAM of the (failed) mar->nov classifier on nov2021 images.

The classifier trained on mar2020 collapses to chance on nov2021. This visualizes
WHY: Grad-CAM shows it spreads attention over global background / illumination
rather than the small egg (~8 px in a 224-px view) that the detector localizes.
This is the mechanism behind "localize, don't classify."

Usage:
    python scripts/classifier_saliency_panel.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cross_study_pretrained_experiment import build_model, _build_test_frame, IMG_SIZE
from schisto_mobile_ai.data.classification import MetadataImageDataset

WEIGHTS    = REPO_ROOT / "runs" / "cross_study_mar_to_nov" / "mobilenet_v2" / "best_model.pt"
IMAGES_CSV = REPO_ROOT / "metadata" / "images.csv"
SPLIT      = REPO_ROOT / "splits" / "mar_to_nov_split.csv"
OUT        = REPO_ROOT / "results" / "cross_study_mar_to_nov" / "classifier_saliency_nov.png"
N_IMG      = 6


def gradcam(model, x, target_layer):
    acts, grads = {}, {}
    h1 = target_layer.register_forward_hook(lambda m, i, o: acts.__setitem__("v", o))
    h2 = target_layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("v", go[0].detach()))
    model.zero_grad()
    logit = model(x).squeeze()
    logit.backward()
    h1.remove(); h2.remove()
    A = acts["v"][0]                      # C,h,w
    G = grads["v"][0]                     # C,h,w
    w = G.mean(dim=(1, 2))               # C
    cam = F.relu((w[:, None, None] * A).sum(0))
    cam = cam / (cam.max() + 1e-8)
    cam = F.interpolate(cam[None, None], size=(IMG_SIZE, IMG_SIZE),
                        mode="bilinear", align_corners=False)[0, 0]
    return cam.detach().cpu().numpy(), float(torch.sigmoid(logit.detach()))


def main() -> None:
    device = "cpu"
    model = build_model("mobilenet_v2")
    model.load_state_dict(torch.load(WEIGHTS, map_location=device))
    model.eval().to(device)
    target_layer = model.features[-1]

    frame = _build_test_frame(IMAGES_CSV, SPLIT, "bf")
    pos = frame[frame["target"] == 1].sort_values("image_name").head(N_IMG).reset_index(drop=True)
    ds = MetadataImageDataset(pos, image_size=IMG_SIZE, train=False)

    fig, axes = plt.subplots(2, len(pos), figsize=(2.7 * len(pos), 5.6))
    for i in range(len(pos)):
        x = ds[i]["image"].unsqueeze(0).to(device)
        cam, prob = gradcam(model, x, target_layer)
        img = Image.open(pos.loc[i, "image_path"]).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        axes[0, i].imshow(img); axes[0, i].axis("off")
        axes[0, i].set_title(f"nov2021 (egg present)\nclf p(pos)={prob:.2f}", fontsize=8)
        axes[1, i].imshow(img); axes[1, i].imshow(cam, cmap="jet", alpha=0.45)
        axes[1, i].axis("off"); axes[1, i].set_title("classifier Grad-CAM", fontsize=8)

    fig.suptitle(
        "Mechanism: the mar→nov classifier attends to global background / illumination, not the egg\n"
        "(an egg is ~8 px in this 224-px view) — the invariant cue the detector localizes",
        fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160)
    print("Saved", OUT)
    print("classifier p(pos) on these egg-present nov images:",
          "(diffuse/background attention = the failure mode)")


if __name__ == "__main__":
    main()
