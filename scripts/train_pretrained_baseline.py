#!/usr/bin/env python3
"""Fine-tune ImageNet-pretrained models on schistosomiasis egg detection.

Compares our lightweight TinyConvEncoder (trained from scratch and with BYOL SSL)
against standard transfer-learning baselines:
  - ResNet-18     (11.2M params) — standard CV benchmark
  - MobileNetV2   (3.4M params)  — designed for resource-constrained mobile deployment
  - EfficientNet-B0 (5.3M params) — state-of-the-art efficient architecture

All three are fine-tuned from ImageNet weights on our BF data, same protocol as
our SSL fine-tuning: 20 epochs, AdamW, sqrt pos_weight, prior bias init.

This answers: "Do heavyweight pretrained models outperform our tiny SSL approach,
and at what parameter cost?"

Usage:
    python scripts/train_pretrained_baseline.py --arch resnet18
    python scripts/train_pretrained_baseline.py --arch mobilenet_v2
    python scripts/train_pretrained_baseline.py --arch efficientnet_b0
    python scripts/train_pretrained_baseline.py --arch all
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from schisto_mobile_ai.data.classification import load_single_contrast_data, MetadataImageDataset
from schisto_mobile_ai.models.patient_aggregation import aggregate_patient_predictions
from schisto_mobile_ai.utils.io import ensure_dir
from schisto_mobile_ai.utils.logging import configure_logging
from schisto_mobile_ai.utils.reproducibility import resolve_device, seed_everything
from schisto_mobile_ai.utils.script_base import resolve_output_dir


ARCHITECTURES = ["resnet18", "mobilenet_v2", "efficientnet_b0"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", type=str, default="all",
                        choices=ARCHITECTURES + ["all"],
                        help="Architecture to train. 'all' runs all three sequentially.")
    parser.add_argument("--contrast", type=str, default="bf", choices=("bf", "df"))
    parser.add_argument("--images-csv", type=Path,
                        default=REPO_ROOT / "metadata" / "images.csv")
    parser.add_argument("--split-csv", type=Path,
                        default=REPO_ROOT / "splits" / "random_patient_split.csv")
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path,
                        default=REPO_ROOT / "runs" / "baselines")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Freeze all but final classifier (linear probe mode).")
    parser.add_argument("--device", type=str, choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--smoke", dest="smoke_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def _count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(arch: str, *, freeze_backbone: bool = False) -> nn.Module:
    """Build ImageNet-pretrained model with a binary classification head."""
    import torchvision.models as models

    if arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        model.fc = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_features, 1))

    elif arch == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        model.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_features, 1))

    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        model.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_features, 1))

    else:
        raise ValueError(f"Unknown arch: {arch}")

    return model


def _safe_auc(targets: list[float], probs: list[float]) -> float:
    if len(set(targets)) < 2:
        return float("nan")
    frame = pd.DataFrame({"t": targets, "p": probs})
    pos = frame["t"] >= 0.5
    pc, nc = int(pos.sum()), int((~pos).sum())
    if pc == 0 or nc == 0:
        return float("nan")
    ranks = frame["p"].rank(method="average")
    return float((ranks[pos].sum() - pc * (pc + 1) / 2.0) / (pc * nc))


def _compute_metrics(targets: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    """Compute AUC, F1, sensitivity, specificity, AUPRC at optimal threshold."""
    if len(np.unique(targets)) < 2:
        return {k: float("nan") for k in ["auc", "f1", "sensitivity", "specificity",
                                           "ppv", "npv", "auprc", "threshold"]}
    # AUC
    auc = _safe_auc(targets.tolist(), probs.tolist())

    # AUPRC (Average Precision) — sklearn-style interpolation
    order = np.argsort(-probs)
    sorted_t = targets[order]
    tp_cum = np.cumsum(sorted_t)
    total_pos = max(float(sorted_t.sum()), 1)
    precision_arr = tp_cum / (np.arange(len(sorted_t)) + 1)
    recall_arr = tp_cum / total_pos
    # Prepend (recall=0, precision=1) sentinel
    recall_arr = np.concatenate([[0.0], recall_arr])
    precision_arr = np.concatenate([[1.0], precision_arr])
    # Trapezoid integration over ascending recall
    auprc = float(np.trapz(precision_arr, recall_arr))

    # Find optimal F1 threshold (sweep)
    best_f1, best_thresh, best_sens, best_spec, best_ppv, best_npv = 0.0, 0.5, 0.0, 0.0, 0.0, 0.0
    thresholds = np.linspace(0.05, 0.95, 181)
    pos_mask = targets == 1
    neg_mask = ~pos_mask
    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()
    for t in thresholds:
        preds = probs >= t
        tp = float((preds & pos_mask).sum())
        fp = float((preds & neg_mask).sum())
        fn = float((~preds & pos_mask).sum())
        tn = float((~preds & neg_mask).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(t)
            best_sens = rec
            best_spec = tn / n_neg if n_neg > 0 else 0.0
            best_ppv = prec
            best_npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return {
        "auc": float(auc),
        "auprc": float(auprc),
        "f1": float(best_f1),
        "sensitivity": float(best_sens),
        "specificity": float(best_spec),
        "ppv": float(best_ppv),
        "npv": float(best_npv),
        "threshold": float(best_thresh),
    }


def _train_epoch(model, loader, *, optimizer, criterion, device) -> dict[str, float]:
    model.train()
    total_loss, all_t, all_p = 0.0, [], []
    for batch in loader:
        imgs = batch["image"].to(device)
        targets = batch["target"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs).squeeze(1)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        probs = torch.sigmoid(logits.detach()).cpu().numpy().tolist()
        t = targets.detach().cpu().numpy().tolist()
        total_loss += float(loss.item()) * len(t)
        all_t.extend(float(v) for v in t)
        all_p.extend(float(v) for v in probs)
    n = max(len(all_t), 1)
    return {"loss": total_loss / n, "auc": _safe_auc(all_t, all_p)}


def _eval_epoch(model, loader, *, criterion, device) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    total_loss, all_t, all_p, rows = 0.0, [], [], []
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            targets = batch["target"].to(device)
            logits = model(imgs).squeeze(1)
            loss = criterion(logits, targets)
            probs = torch.sigmoid(logits).cpu().numpy().tolist()
            t = targets.cpu().numpy().tolist()
            total_loss += float(loss.item()) * len(t)
            all_t.extend(float(v) for v in t)
            all_p.extend(float(v) for v in probs)
            for i in range(len(t)):
                rows.append({
                    "image_id": batch["image_id"][i],
                    "patient_key": batch["patient_key"][i],
                    "target": float(t[i]),
                    "probability": float(probs[i]),
                    "contrast": batch["contrast"][i],
                    "split": batch["split"][i],
                })
    preds = pd.DataFrame(rows)
    n = max(len(all_t), 1)
    pair_auc = _safe_auc(all_t, all_p)
    patient_frame = aggregate_patient_predictions(preds, patient_target_aggregation="max")
    patient_probs = patient_frame["patient_probability_max"].values
    patient_targets = patient_frame["target"].values
    patient_auc = _safe_auc(patient_targets.tolist(), patient_probs.tolist())
    return {
        "loss": total_loss / n,
        "val_pair_auc": pair_auc,
        "val_patient_auc_max": patient_auc,
    }, preds


def train_one_arch(arch: str, args: argparse.Namespace) -> dict[str, Any]:
    """Train a single architecture and return its best metrics."""
    logger = configure_logging(quiet=args.quiet)
    seed_everything(args.seed)
    device = resolve_device(args.device)

    epochs = min(args.epochs, 2) if args.smoke_test else args.epochs
    batch_size = min(args.batch_size, 8) if args.smoke_test else args.batch_size
    img_size = min(args.img_size, 128) if args.smoke_test else args.img_size

    output_dir = ensure_dir(args.output_dir / arch)

    data = load_single_contrast_data(
        images_csv=args.images_csv,
        split_csv=args.split_csv,
        raw_dir=args.raw_dir,
        contrast=args.contrast,
        label_source="image",
        smoke_test=args.smoke_test,
        seed=args.seed,
    )

    train_ds = MetadataImageDataset(data.train_frame, image_size=img_size, train=True)
    val_ds = MetadataImageDataset(data.val_frame, image_size=img_size, train=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=args.num_workers)

    model = build_model(arch, freeze_backbone=args.freeze_backbone).to(device)

    total_params = _count_parameters(model)
    trainable_params = _count_trainable_parameters(model)

    # Prior bias init
    train_pos_rate = float(data.train_frame["target"].sum()) / max(len(data.train_frame), 1)
    if 0.0 < train_pos_rate < 1.0:
        prior_bias = float(np.log(train_pos_rate / (1.0 - train_pos_rate)))
        # Find the last linear layer and init its bias
        for module in reversed(list(model.modules())):
            if isinstance(module, nn.Linear) and module.out_features == 1:
                with torch.no_grad():
                    module.bias.fill_(prior_bias)
                break

    pos = float(data.train_frame["target"].sum())
    neg = float(len(data.train_frame) - pos)
    pos_weight = torch.tensor([(neg / pos) ** 0.5], dtype=torch.float32).to(device) if pos > 0 else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=args.lr * 0.01)

    history_rows: list[dict] = []
    best_score = float("-inf")
    best_preds = pd.DataFrame()

    # Measure inference time (before training, on CPU for fair comparison)
    dummy = torch.randn(1, 3, img_size, img_size).to(device)
    model.eval()
    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(50):
            _ = model(dummy)
        inference_ms = (time.perf_counter() - t0) / 50 * 1000

    for epoch in range(1, epochs + 1):
        train_m = _train_epoch(model, train_loader, optimizer=optimizer,
                               criterion=criterion, device=device)
        val_m, val_preds = _eval_epoch(model, val_loader, criterion=criterion, device=device)
        scheduler.step()

        row = {"epoch": epoch, **train_m, **val_m}
        history_rows.append(row)
        pd.DataFrame(history_rows).to_csv(output_dir / "history.csv", index=False)

        score = val_m["val_patient_auc_max"]
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_preds = val_preds.copy()
            torch.save(model.state_dict(), output_dir / "best_model.pt")

        logger.info("[%s] Epoch %d/%d | train_loss=%.4f | val_patient_auc=%.4f",
                    arch, epoch, epochs, train_m["loss"], val_m["val_patient_auc_max"])

    if not best_preds.empty:
        best_preds.to_csv(output_dir / "val_predictions.csv", index=False)

    # Compute full metrics on best predictions
    patient_frame = aggregate_patient_predictions(best_preds, patient_target_aggregation="max")
    patient_metrics = _compute_metrics(
        patient_frame["target"].values,
        patient_frame["patient_probability_max"].values,
    )

    result = {
        "arch": arch,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "inference_ms_per_image": round(inference_ms, 3),
        "best_val_patient_auc": round(best_score, 4),
        **{f"patient_{k}": round(v, 4) if np.isfinite(v) else None
           for k, v in patient_metrics.items()},
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    archs = ARCHITECTURES if args.arch == "all" else [args.arch]

    all_results = []
    for arch in archs:
        print(f"\n{'='*60}")
        print(f"Training {arch}...")
        print(f"{'='*60}")
        result = train_one_arch(arch, args)
        all_results.append(result)
        print(f"  {arch}: AUC={result['best_val_patient_auc']:.4f} | "
              f"params={result['total_params']:,} | "
              f"inf={result['inference_ms_per_image']:.1f}ms")

    summary_path = args.output_dir / "pretrained_baselines_summary.json"
    ensure_dir(args.output_dir)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
