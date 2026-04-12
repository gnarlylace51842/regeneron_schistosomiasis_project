#!/usr/bin/env python3
"""Fine-tune a cross-contrast SSL pre-trained encoder for egg detection.

Takes encoder weights from pretrain_cross_contrast.py, attaches a linear
classification head, and fine-tunes on a specified fraction of the labelled
training data. Run this multiple times at different label fractions to generate
the label efficiency curve — the primary result figure of the project.

The label efficiency experiment:
    For label_fraction in [0.10, 0.25, 0.50, 1.00]:
        finetune_ssl.py --encoder-weights ... --label-fraction <frac> --run-name ssl_ft_<frac>
        finetune_ssl.py --from-scratch --label-fraction <frac> --run-name scratch_ft_<frac>

    Plot: val_patient_auc_max vs label_fraction, two curves (SSL vs scratch).
    Expected: SSL curve reaches scratch ceiling at ~25-50% of labels.

This generates Figure 1 of the paper and directly answers the scientific question:
"Does physics-grounded cross-contrast pre-training reduce annotation requirements
for schistosomiasis egg detection in low-resource settings?"
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from schisto_mobile_ai.config import load_config, save_config_snapshot
from schisto_mobile_ai.data.classification import load_single_contrast_data, MetadataImageDataset
from schisto_mobile_ai.models.cross_contrast_ssl import CrossContrastSSLModel
from schisto_mobile_ai.models.simple_cnn import TinyConvEncoder
from schisto_mobile_ai.models.patient_aggregation import aggregate_patient_predictions
from schisto_mobile_ai.utils.io import ensure_dir
from schisto_mobile_ai.utils.logging import configure_logging
from schisto_mobile_ai.utils.reproducibility import resolve_device, seed_everything
from schisto_mobile_ai.utils.script_base import resolve_output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder-weights", type=Path, default=None,
                        help="Path to encoder_weights.pt from pretrain_cross_contrast.py. "
                             "If omitted, trains from scratch (the supervised baseline).")
    parser.add_argument("--contrast", type=str, default="bf",
                        choices=("bf", "df", "brightfield", "darkfield"),
                        help="Which contrast to fine-tune on.")
    parser.add_argument("--label-fraction", type=float, default=1.0,
                        help="Fraction of training labels to use (0.0-1.0). "
                             "Subsampled with stratification to preserve class ratio.")
    parser.add_argument("--images-csv", type=Path,
                        default=REPO_ROOT / "metadata" / "images.csv")
    parser.add_argument("--split-csv", type=Path,
                        default=REPO_ROOT / "splits" / "random_patient_split.csv")
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-name", type=str, default="ssl_finetune")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--base-channels", type=int, default=32,
                        help="Must match the pre-trained encoder's base_channels.")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="Freeze encoder weights during fine-tuning (linear probe mode).")
    parser.add_argument("--device", type=str, choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--smoke", dest="smoke_test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--focal-gamma", type=float, default=0.0,
                        help="Focal loss gamma (0 = standard BCE). Try 1.0 or 2.0 for "
                             "class-imbalanced DF training.")
    parser.add_argument("--cosine-lr", action="store_true",
                        help="Use cosine annealing LR schedule (recommended with physics aug).")
    parser.add_argument("--warmup-epochs", type=int, default=3,
                        help="Linear warmup epochs before cosine decay (default 3).")
    parser.add_argument("--freeze-epochs", type=int, default=0,
                        help="Freeze encoder for this many epochs first (two-stage fine-tuning).")
    return parser


class FocalBCELoss(nn.Module):
    """Binary focal loss: FL(p) = -α(1-p)^γ log(p).

    gamma=0 → standard BCE. gamma=1-2 → down-weights easy negatives.
    pos_weight mirrors BCEWithLogitsLoss for class imbalance.
    """
    def __init__(self, gamma: float = 2.0, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )
        probs = torch.sigmoid(logits)
        # p_t = probability of the true class
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


class SSLFineTuneClassifier(nn.Module):
    """Encoder (optionally pre-trained) + linear classification head."""

    def __init__(
        self,
        *,
        base_channels: int = 32,
        num_classes: int = 1,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = TinyConvEncoder(base_channels=base_channels)
        self.head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.encoder.feature_dim, num_classes),
        )
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(image))


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


def _subsample_by_fraction(
    frame: pd.DataFrame,
    *,
    fraction: float,
    seed: int,
) -> pd.DataFrame:
    """Stratified subsample of training data by label fraction.

    Subsampling is done at the PATIENT level (not image level) to preserve
    patient-safe separation and avoid inflating the effective sample size.
    The result is all pairs for the selected patients.
    """
    if fraction >= 1.0:
        return frame.copy()
    if fraction <= 0.0:
        raise ValueError("label_fraction must be > 0.")

    # Get unique patients and their majority label
    patient_targets = (
        frame.groupby("patient_key")["target"]
        .max()  # patient positive if any image is positive
        .reset_index()
    )
    rng = np.random.default_rng(seed)
    selected_patients: list[str] = []
    for label_val in [0.0, 1.0]:
        group = patient_targets[patient_targets["target"] == label_val]["patient_key"].tolist()
        n_select = max(1, math.ceil(len(group) * fraction))
        n_select = min(n_select, len(group))
        chosen = rng.choice(group, size=n_select, replace=False).tolist()
        selected_patients.extend(chosen)

    return frame[frame["patient_key"].isin(selected_patients)].copy().reset_index(drop=True)


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
            meta = batch
            for i in range(len(t)):
                rows.append({
                    "image_id": meta["image_id"][i],
                    "patient_key": meta["patient_key"][i],
                    "target": float(t[i]),
                    "probability": float(probs[i]),
                    "contrast": meta["contrast"][i],
                    "split": meta["split"][i],
                })
    preds = pd.DataFrame(rows)
    n = max(len(all_t), 1)
    pair_auc = _safe_auc(all_t, all_p)
    # Patient-level metrics with max aggregation
    patient_frame = aggregate_patient_predictions(preds, patient_target_aggregation="max")
    patient_auc_max = _safe_auc(
        patient_frame["target"].tolist(),
        patient_frame["patient_probability_max"].tolist(),
    )
    patient_auc_mean = _safe_auc(
        patient_frame["target"].tolist(),
        patient_frame["patient_probability_mean"].tolist(),
    )
    return {
        "loss": total_loss / n,
        "val_pair_auc": pair_auc,
        "val_patient_auc_max": patient_auc_max,
        "val_patient_auc_mean": patient_auc_mean,
    }, preds


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    logger = configure_logging(quiet=args.quiet)

    from_scratch = args.encoder_weights is None
    init_mode = "scratch" if from_scratch else "ssl_pretrained"

    args.subset_size = None
    output_dir = resolve_output_dir(
        args=args,
        task_name="finetune_ssl",
        default_output_subdir="runs/ssl/finetune",
    )

    seed_everything(args.seed)
    device = resolve_device(args.device)

    epochs = min(args.epochs, 2) if args.smoke_test else args.epochs
    batch_size = min(args.batch_size, 8) if args.smoke_test else args.batch_size
    img_size = min(args.img_size, 128) if args.smoke_test else args.img_size

    data = load_single_contrast_data(
        images_csv=args.images_csv,
        split_csv=args.split_csv,
        raw_dir=args.raw_dir,
        contrast=args.contrast,
        label_source="image",
        smoke_test=args.smoke_test,
        seed=args.seed,
    )

    # Subsample training data by patient-level label fraction
    train_frame = _subsample_by_fraction(
        data.train_frame, fraction=args.label_fraction, seed=args.seed,
    )
    logger.info(
        "Label fraction %.2f: using %d/%d train pairs (%d patients)",
        args.label_fraction, len(train_frame), len(data.train_frame),
        train_frame["patient_key"].nunique(),
    )

    train_ds = MetadataImageDataset(train_frame, image_size=img_size, train=True)
    val_ds = MetadataImageDataset(data.val_frame, image_size=img_size, train=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=False)

    model = SSLFineTuneClassifier(
        base_channels=args.base_channels,
        freeze_encoder=args.freeze_encoder,
    ).to(device)

    # Load pre-trained encoder weights if provided
    if not from_scratch:
        enc_path = args.encoder_weights
        if not enc_path.exists():
            print(f"ERROR: encoder weights not found: {enc_path}", file=sys.stderr)
            return 1
        state = torch.load(enc_path, map_location=device)
        model.encoder.load_state_dict(state)
        logger.info("Loaded SSL pre-trained encoder from %s", enc_path)
    else:
        logger.info("Training from scratch (no SSL pre-training)")

    # Prior bias initialisation
    train_pos_rate = float(train_frame["target"].sum()) / max(len(train_frame), 1)
    if 0.0 < train_pos_rate < 1.0:
        prior_bias = float(np.log(train_pos_rate / (1.0 - train_pos_rate)))
        with torch.no_grad():
            model.head[-1].bias.fill_(prior_bias)

    pos = float(train_frame["target"].sum())
    neg = float(len(train_frame) - pos)
    pos_weight = torch.tensor([(neg / pos) ** 0.5], dtype=torch.float32).to(device) if pos > 0 else None
    if args.focal_gamma > 0:
        criterion = FocalBCELoss(gamma=args.focal_gamma, pos_weight=pos_weight)
        logger.info("Using focal loss with gamma=%.1f", args.focal_gamma)
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Two-stage fine-tuning: freeze encoder for first N epochs, then unfreeze
    freeze_epochs = min(args.freeze_epochs, epochs) if hasattr(args, "freeze_epochs") else 0
    if freeze_epochs > 0 and not from_scratch:
        for param in model.encoder.parameters():
            param.requires_grad = False
        logger.info("Encoder frozen for first %d epochs (two-stage fine-tuning)", freeze_epochs)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )

    # Cosine LR schedule with linear warmup
    use_cosine = getattr(args, "cosine_lr", False)
    warmup_epochs = getattr(args, "warmup_epochs", 3)
    if use_cosine:
        def _lr_lambda(ep: int) -> float:
            # ep is 0-indexed
            if ep < warmup_epochs:
                return float(ep + 1) / float(max(warmup_epochs, 1))
            progress = (ep - warmup_epochs) / max(epochs - warmup_epochs, 1)
            return 0.5 * (1.0 + np.cos(np.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
        logger.info("Using cosine LR schedule with %d warmup epochs", warmup_epochs)
    else:
        scheduler = None

    history_rows: list[dict[str, Any]] = []
    history_path = output_dir / "history.csv"
    best_score = float("-inf")
    best_preds = pd.DataFrame()

    for epoch in range(1, epochs + 1):
        # Unfreeze encoder after freeze_epochs
        if freeze_epochs > 0 and epoch == freeze_epochs + 1 and not from_scratch:
            for param in model.encoder.parameters():
                param.requires_grad = True
            # Re-create optimizer with all params at lower LR
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr * 0.3, weight_decay=args.weight_decay,
            )
            if use_cosine:
                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    lr_lambda=lambda ep: 0.5 * (1.0 + np.cos(np.pi * ep / max(epochs - epoch, 1))),
                )
            logger.info("Encoder unfrozen at epoch %d (full model fine-tuning, lr=%.2e)",
                        epoch, args.lr * 0.3)

        train_m = _train_epoch(model, train_loader, optimizer=optimizer,
                               criterion=criterion, device=device)
        val_m, val_preds = _eval_epoch(model, val_loader, criterion=criterion, device=device)

        if scheduler is not None:
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_m["loss"],
            "train_auc": train_m["auc"],
            "val_loss": val_m["loss"],
            "val_pair_auc": val_m["val_pair_auc"],
            "val_patient_auc_max": val_m["val_patient_auc_max"],
            "val_patient_auc_mean": val_m["val_patient_auc_mean"],
        }
        history_rows.append(row)
        pd.DataFrame(history_rows).to_csv(history_path, index=False)

        score = val_m["val_patient_auc_max"] if np.isfinite(val_m["val_patient_auc_max"]) else val_m["val_pair_auc"]
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_preds = val_preds.copy()
            torch.save(model.state_dict(), output_dir / "best_model.pt")

        logger.info(
            "Epoch %d/%d | lr=%.2e | train_loss=%.4f | val_pair_auc=%.4f | val_patient_auc_max=%.4f",
            epoch, epochs, current_lr, train_m["loss"], val_m["val_pair_auc"],
            val_m["val_patient_auc_max"],
        )

    if not best_preds.empty:
        best_preds.to_csv(output_dir / "val_predictions.csv", index=False)

    # Save config for log_experiment.py compatibility
    history_df = pd.DataFrame(history_rows)
    best_idx = int(history_df["val_patient_auc_max"].fillna(-1).idxmax())
    config_snapshot = {
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "img_size": img_size,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "backbone": f"ssl_finetune_base{args.base_channels}",
        },
        "data": {
            "contrast": args.contrast,
            "label_source": "image",
            "label_column": "label",
            "label_fraction": args.label_fraction,
            "images_csv": str(args.images_csv),
            "split_csv": str(args.split_csv),
            "raw_dir": str(args.raw_dir),
        },
        "dataset_summary": {
            "contrast": args.contrast,
            "label_source": "image",
            "label_column": "label",
            "label_fraction": args.label_fraction,
            "n_train_images": len(train_frame),
            "n_val_images": len(data.val_frame),
            "n_train_patients": int(train_frame["patient_key"].nunique()),
            "n_val_patients": int(data.val_frame["patient_key"].nunique()),
            "train_label_counts": {
                str(int(k)): int(v)
                for k, v in train_frame["target"].value_counts().sort_index().items()
            },
            "val_label_counts": {
                str(int(k)): int(v)
                for k, v in data.val_frame["target"].value_counts().sort_index().items()
            },
        },
        "ssl": {
            "init_mode": init_mode,
            "encoder_weights": str(args.encoder_weights) if args.encoder_weights else None,
            "freeze_encoder": args.freeze_encoder,
        },
        "runtime": {"resolved_device": device},
        "outputs": {
            "output_dir": str(output_dir),
            "best_model_path": str(output_dir / "best_model.pt"),
            "history_csv_path": str(history_path),
            "val_predictions_csv_path": str(output_dir / "val_predictions.csv"),
        },
    }
    import json
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_snapshot, f, indent=2)

    print("SSL Fine-Tuning Summary")
    print(f"  init_mode:        {init_mode}")
    print(f"  contrast:         {args.contrast}")
    print(f"  label_fraction:   {args.label_fraction:.2f}")
    print(f"  train_pairs:      {len(train_frame)}")
    print(f"  best_epoch:       {best_idx + 1}")
    print(f"  best_patient_auc_max: {best_score:.4f}")
    print(f"  output_dir:       {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
