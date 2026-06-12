#!/usr/bin/env python3
"""Same-direction (mar2020 -> nov2021) ImageNet-pretrained classifier baseline.

The prior cross-study classifier evidence (cross_study_pretrained_experiment.py)
was nov2021 -> mar2020. The YOLOv8 detector headline is mar2020 -> nov2021. To
compare the two approaches head-to-head on the SAME shift and the SAME test
patients, this trains MobileNetV2 / EfficientNet-B0 on mar2020 BF and evaluates
zero-shot on the nov2021 test patients (the exact 380 patients / 61 positives the
detector was scored on).

Outputs:
  results/cross_study_mar_to_nov/results.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from schisto_mobile_ai.data.classification import MetadataImageDataset, load_single_contrast_data
from schisto_mobile_ai.utils.io import ensure_dir
from schisto_mobile_ai.utils.reproducibility import resolve_device, seed_everything

# Reuse the exact training/eval/bootstrap helpers from the nov->mar experiment.
from cross_study_pretrained_experiment import (
    build_model, _train_epoch, _eval_auc, _run_inference, bootstrap_auc,
    _build_test_frame, IMG_SIZE, BATCH, EPOCHS, LR, WD, SEED,
)

SPLIT      = REPO_ROOT / "splits" / "mar_to_nov_split.csv"
IMAGES_CSV = REPO_ROOT / "metadata" / "images.csv"
RAW_DIR    = REPO_ROOT / "data" / "raw"
OUT_DIR    = ensure_dir(REPO_ROOT / "results" / "cross_study_mar_to_nov")
RUN_DIR    = ensure_dir(REPO_ROOT / "runs" / "cross_study_mar_to_nov")
ARCHS      = ["mobilenet_v2", "efficientnet_b0"]


def train_and_eval(arch: str, device) -> dict:
    seed_everything(SEED)
    run_dir = ensure_dir(RUN_DIR / arch)

    # train/val = mar2020
    data = load_single_contrast_data(
        images_csv=IMAGES_CSV, split_csv=SPLIT, raw_dir=RAW_DIR,
        contrast="bf", label_source="image", seed=SEED,
    )
    train_loader = DataLoader(MetadataImageDataset(data.train_frame, image_size=IMG_SIZE, train=True),
                              batch_size=BATCH, shuffle=True, num_workers=0)
    val_loader = DataLoader(MetadataImageDataset(data.val_frame, image_size=IMG_SIZE, train=False),
                            batch_size=BATCH, shuffle=False, num_workers=0)

    # test = nov2021 (zero-shot) — same 380 patients as the detector
    test_frame = _build_test_frame(IMAGES_CSV, SPLIT, "bf")
    test_loader = DataLoader(MetadataImageDataset(test_frame, image_size=IMG_SIZE, train=False),
                             batch_size=BATCH, shuffle=False, num_workers=0)

    model = build_model(arch).to(device)

    # prior-bias init on the final 1-logit head
    pos_rate = float(data.train_frame["target"].mean())
    if 0 < pos_rate < 1:
        bias = float(np.log(pos_rate / (1 - pos_rate)))
        for m in reversed(list(model.modules())):
            if isinstance(m, nn.Linear) and m.out_features == 1:
                with torch.no_grad():
                    m.bias.fill_(bias)
                break

    pos = float(data.train_frame["target"].sum())
    neg = float(len(data.train_frame) - pos)
    pw = torch.tensor([(neg / pos) ** 0.5], dtype=torch.float32).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=LR * 0.01)

    best_val, best_state = float("-inf"), None
    for epoch in range(1, EPOCHS + 1):
        _train_epoch(model, train_loader, optimizer=opt, criterion=crit, device=device)
        val_auc = _eval_auc(model, val_loader, device=device)
        sched.step()
        print(f"  [{arch}] epoch {epoch}/{EPOCHS}  mar_val_auc={val_auc:.4f}", flush=True)
        if np.isfinite(val_auc) and val_auc > best_val:
            best_val = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, run_dir / "best_model.pt")

    preds = _run_inference(model, test_loader, device=device)
    pat = preds.groupby("patient_key").agg(target=("target", "max"), prob=("prob", "max")).reset_index()
    pt, lo, hi = bootstrap_auc(pat["target"].values, pat["prob"].values)

    print(f"\n  {arch}:  mar2020 val AUC {best_val:.4f}  |  nov2021 zero-shot AUC {pt:.4f} [{lo:.4f}, {hi:.4f}]\n")
    preds.to_csv(run_dir / "test_predictions.csv", index=False)
    return {
        "arch": arch,
        "mar2020_val_auc": round(best_val, 4),
        "nov2021_auc": round(pt, 4),
        "nov2021_lo": round(lo, 4),
        "nov2021_hi": round(hi, 4),
        "n_test_patients": int(len(pat)),
        "n_test_pos": int(pat["target"].sum()),
    }


def main() -> None:
    device = resolve_device("auto")
    print(f"Device: {device}")
    print(f"Split:  {SPLIT.name}  (train/val=mar2020, test=nov2021)\n")
    results = [train_and_eval(a, device) for a in ARCHS]
    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "results.csv", index=False)
    print("=== mar->nov classifier (SAME direction as YOLOv8 detector) ===")
    print(df.to_string(index=False))
    print("\nDetector reference (mar->nov, same 380 test patients): patient AUC 0.914 [0.859, 0.961]")
    print(f"\nSaved: {OUT_DIR / 'results.csv'}")


if __name__ == "__main__":
    main()
