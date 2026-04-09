#!/usr/bin/env python3
"""Train a minimal metadata-driven single-contrast schistosomiasis classifier."""

from __future__ import annotations

import argparse
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
from schisto_mobile_ai.data.classification import MetadataImageDataset, load_single_contrast_data
from schisto_mobile_ai.models.patient_aggregation import aggregate_patient_predictions
from schisto_mobile_ai.models.simple_cnn import TinyConvClassifier
from schisto_mobile_ai.utils.logging import configure_logging
from schisto_mobile_ai.utils.reproducibility import resolve_device, seed_everything
from schisto_mobile_ai.utils.script_base import resolve_output_dir


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for single-contrast training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "default.yaml",
        help="Path to a YAML or JSON config file.",
    )
    parser.add_argument(
        "--images-csv",
        type=Path,
        default=REPO_ROOT / "metadata" / "images.csv",
        help="Path to metadata/images.csv.",
    )
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=REPO_ROOT / "splits" / "random_patient_split.csv",
        help="Path to a patient-safe split CSV.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=REPO_ROOT / "data" / "raw",
        help="Root directory used with images.csv relative paths.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional explicit output directory.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="baseline",
        help="Short label added to the output folder name.",
    )
    parser.add_argument(
        "--contrast",
        type=str,
        required=True,
        choices=("bf", "df", "brightfield", "darkfield"),
        help="Which single contrast to train on.",
    )
    parser.add_argument(
        "--smoke",
        dest="smoke_test",
        action="store_true",
        help="Run a tiny end-to-end training pass for environment verification.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optionally cap the number of training images.",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
        help="Optionally cap the number of validation images.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Mini-batch size.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=None,
        help="Square resize used for both training and validation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=("auto", "cpu", "mps"),
        default="auto",
        help="Execution device.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing non-empty output directory.",
    )
    parser.add_argument(
        "--label-source",
        type=str,
        choices=("auto", "image", "patient"),
        default="image",
        help=(
            "Which label column to use as the training target. "
            "'image' uses the pair/image-level egg-detection label (recommended). "
            "'patient' uses the patient-level diagnosis label. "
            "'auto' falls back to legacy behaviour (prefers patient-level)."
        ),
    )
    parser.add_argument(
        "--pos-weight",
        type=float,
        default=None,
        help=(
            "Explicit positive-class weight for BCEWithLogitsLoss. "
            "Default (None) uses sqrt(neg/pos), which moderates class imbalance "
            "without recreating the gradient saddle point caused by the exact neg/pos ratio."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce log output.",
    )
    return parser


def _safe_auc(targets: list[float], probabilities: list[float]) -> float:
    if len(set(targets)) < 2:
        return float("nan")

    frame = pd.DataFrame(
        {
            "target": pd.Series(targets, dtype=float),
            "probability": pd.Series(probabilities, dtype=float),
        }
    )
    positive_mask = frame["target"] >= 0.5
    negative_mask = ~positive_mask
    positive_count = int(positive_mask.sum())
    negative_count = int(negative_mask.sum())
    if positive_count == 0 or negative_count == 0:
        return float("nan")

    ranks = frame["probability"].rank(method="average")
    positive_rank_sum = float(ranks[positive_mask].sum())
    auc = (
        positive_rank_sum
        - (positive_count * (positive_count + 1) / 2.0)
    ) / (positive_count * negative_count)
    return float(auc)


def _binary_accuracy(targets: list[float], probabilities: list[float]) -> float:
    if not targets:
        return float("nan")
    predictions = [1.0 if value >= 0.5 else 0.0 for value in probabilities]
    matches = [float(prediction == target) for prediction, target in zip(predictions, targets)]
    return float(np.mean(matches))


def _compute_metrics(targets: list[float], probabilities: list[float], average_loss: float) -> dict[str, float]:
    return {
        "loss": float(average_loss),
        "accuracy": _binary_accuracy(targets, probabilities),
        "auc": _safe_auc(targets, probabilities),
    }


def _patient_level_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    # patient_target_aggregation="max": patient is positive if ANY image is positive.
    # This is clinically correct for egg detection — a patient is infected if at
    # least one slide shows eggs — and handles mixed per-image targets that arise
    # when training with image-level labels.
    patient_frame = aggregate_patient_predictions(predictions, patient_target_aggregation="max")
    metrics: dict[str, float] = {}
    if patient_frame.empty or "target" not in patient_frame.columns:
        for method in ("max", "mean", "noisy_or"):
            metrics[f"patient_auc_{method}"] = float("nan")
        return metrics

    for method in ("max", "mean", "noisy_or"):
        metrics[f"patient_auc_{method}"] = _safe_auc(
            patient_frame["target"].astype(float).tolist(),
            patient_frame[f"patient_probability_{method}"].astype(float).tolist(),
        )
    return metrics


def _move_batch_to_device(batch: dict[str, Any], device: str) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    image = batch["image"].to(device)
    target = batch["target"].to(device)
    metadata = {key: value for key, value in batch.items() if key not in {"image", "target"}}
    return image, target, metadata


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> dict[str, float]:
    """Run one training epoch and return aggregate metrics."""
    model.train()
    total_loss = 0.0
    all_targets: list[float] = []
    all_probabilities: list[float] = []

    for batch in loader:
        images, targets, _ = _move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images).squeeze(1)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        probabilities = torch.sigmoid(logits.detach()).cpu().numpy().tolist()
        batch_targets = targets.detach().cpu().numpy().tolist()
        total_loss += float(loss.item()) * len(batch_targets)
        all_targets.extend(float(value) for value in batch_targets)
        all_probabilities.extend(float(value) for value in probabilities)

    average_loss = total_loss / max(len(all_targets), 1)
    return _compute_metrics(all_targets, all_probabilities, average_loss)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    criterion: nn.Module,
    device: str,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate on the validation loader and return metrics plus per-image predictions."""
    model.eval()
    total_loss = 0.0
    rows: list[dict[str, Any]] = []
    all_targets: list[float] = []
    all_probabilities: list[float] = []

    with torch.no_grad():
        for batch in loader:
            images, targets, metadata = _move_batch_to_device(batch, device)
            logits = model(images).squeeze(1)
            loss = criterion(logits, targets)
            probabilities = torch.sigmoid(logits).cpu().numpy().tolist()
            batch_targets = targets.cpu().numpy().tolist()

            total_loss += float(loss.item()) * len(batch_targets)
            all_targets.extend(float(value) for value in batch_targets)
            all_probabilities.extend(float(value) for value in probabilities)

            for index in range(len(batch_targets)):
                rows.append(
                    {
                        "image_id": metadata["image_id"][index],
                        "patient_key": metadata["patient_key"][index],
                        "patient_id": metadata["patient_id"][index],
                        "study_id": metadata["study_id"][index],
                        "contrast": metadata["contrast"][index],
                        "relative_path": metadata["relative_path"][index],
                        "split": metadata["split"][index],
                        "target": float(batch_targets[index]),
                        "probability": float(probabilities[index]),
                        "predicted_label": int(probabilities[index] >= 0.5),
                    }
                )

    predictions = pd.DataFrame(rows)
    metrics = _compute_metrics(all_targets, all_probabilities, total_loss / max(len(all_targets), 1))
    metrics.update(_patient_level_metrics(predictions))
    return metrics, predictions


def _pos_weight_from_frame(train_frame: pd.DataFrame, explicit_pos_weight: float | None = None) -> torch.Tensor | None:
    positives = float(train_frame["target"].sum())
    negatives = float(len(train_frame) - positives)
    if positives <= 0 or negatives <= 0:
        return None
    if explicit_pos_weight is not None:
        return torch.tensor([explicit_pos_weight], dtype=torch.float32)
    # Use sqrt(neg/pos) instead of neg/pos. The exact ratio creates a gradient
    # saddle point where constant predictions minimise the loss. sqrt moderates
    # imbalance while keeping gradients informative for both classes.
    return torch.tensor([float(negatives / positives) ** 0.5], dtype=torch.float32)


def _build_config_snapshot(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    resolved_device: str,
    output_dir: Path,
    data_bundle_metadata: dict[str, Any],
    backbone_name: str,
    epochs: int,
    batch_size: int,
    img_size: int,
) -> dict[str, Any]:
    snapshot = dict(config)
    snapshot.setdefault("runtime", {})
    snapshot["runtime"]["resolved_device"] = resolved_device
    snapshot["runtime"]["num_workers"] = args.num_workers
    snapshot["training"] = {
        **snapshot.get("training", {}),
        "task_type": "binary_image_classification",
        "backbone": backbone_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "img_size": img_size,
    }
    snapshot["data"] = {
        **snapshot.get("data", {}),
        "images_csv": str(args.images_csv),
        "split_csv": str(args.split_csv),
        "raw_dir": str(args.raw_dir),
        "contrast": args.contrast,
        "label_column": data_bundle_metadata["label_column"],
        "label_source": data_bundle_metadata.get("label_source", "auto"),
    }
    snapshot["outputs"] = {
        "output_dir": str(output_dir),
        "best_model_path": str(output_dir / "best_model.pt"),
        "history_csv_path": str(output_dir / "history.csv"),
        "val_predictions_csv_path": str(output_dir / "val_predictions.csv"),
    }
    snapshot["dataset_summary"] = data_bundle_metadata
    return snapshot


def main() -> int:
    """Train the single-contrast image classifier."""
    parser = build_parser()
    args = parser.parse_args()
    args.subset_size = None
    logger = configure_logging(quiet=args.quiet)
    config = load_config(args.config)
    output_dir = resolve_output_dir(
        args=args,
        task_name="train_single_contrast",
        default_output_subdir="runs/experiments/train_single_contrast",
    )

    seed_everything(args.seed)
    device = resolve_device(args.device)

    training_config = config.get("training", {})
    runtime_config = config.get("runtime", {})
    backbone_name = "tiny_cnn"

    epochs = args.epochs if args.epochs is not None else int(training_config.get("epochs", 20))
    batch_size = args.batch_size if args.batch_size is not None else int(training_config.get("batch_size", 8))
    img_size = args.img_size if args.img_size is not None else int(config.get("data", {}).get("resize") or 224)

    if args.smoke_test:
        epochs = min(epochs, 2)
        batch_size = min(batch_size, 8)
        img_size = min(img_size, 160)

    data_bundle = load_single_contrast_data(
        images_csv=args.images_csv,
        split_csv=args.split_csv,
        raw_dir=args.raw_dir,
        contrast=args.contrast,
        label_source=args.label_source,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        smoke_test=args.smoke_test,
        seed=args.seed,
    )
    data_bundle.metadata["label_column"] = data_bundle.label_column

    train_dataset = MetadataImageDataset(data_bundle.train_frame, image_size=img_size, train=True)
    val_dataset = MetadataImageDataset(data_bundle.val_frame, image_size=img_size, train=False)
    num_workers = args.num_workers if args.num_workers is not None else int(runtime_config.get("num_workers", 0))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    model = TinyConvClassifier(num_classes=1).to(device)

    # Prior bias initialization: start the output logit at log(p/(1-p)) where p is
    # the training positive rate. This ensures the model begins at the correct class
    # prior rather than at logit=0 (prob=0.5), which is a gradient saddle point
    # when pos_weight equals the exact neg/pos ratio.
    train_pos_rate = float(data_bundle.train_frame["target"].sum()) / max(len(data_bundle.train_frame), 1)
    if 0.0 < train_pos_rate < 1.0:
        prior_bias = float(np.log(train_pos_rate / (1.0 - train_pos_rate)))
        with torch.no_grad():
            model.head[-1].bias.fill_(prior_bias)
        logger.info(
            "Prior bias init: %.4f  (train positive rate: %.3f, label_source: %s)",
            prior_bias, train_pos_rate, args.label_source,
        )

    pos_weight = _pos_weight_from_frame(data_bundle.train_frame, explicit_pos_weight=args.pos_weight)
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)
        logger.info("pos_weight: %.4f", float(pos_weight.item()))
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 3e-4)),
        weight_decay=float(training_config.get("weight_decay", 1e-4)),
    )

    best_score = float("-inf")
    best_predictions = pd.DataFrame()
    history_rows: list[dict[str, Any]] = []
    checkpoint_path = output_dir / "best_model.pt"
    history_path = output_dir / "history.csv"
    predictions_path = output_dir / "val_predictions.csv"
    config_path = output_dir / "config.json"

    resolved_config = _build_config_snapshot(
        args=args,
        config=config,
        resolved_device=device,
        output_dir=output_dir,
        data_bundle_metadata=data_bundle.metadata,
        backbone_name=backbone_name,
        epochs=epochs,
        batch_size=batch_size,
        img_size=img_size,
    )
    save_config_snapshot(resolved_config, config_path)

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        val_metrics, val_predictions = evaluate(
            model,
            val_loader,
            criterion=criterion,
            device=device,
        )

        history_row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_auc": train_metrics["auc"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_auc": val_metrics["auc"],
            "val_patient_auc_max": val_metrics["patient_auc_max"],
            "val_patient_auc_mean": val_metrics["patient_auc_mean"],
            "val_patient_auc_noisy_or": val_metrics["patient_auc_noisy_or"],
        }
        history_rows.append(history_row)
        pd.DataFrame(history_rows).to_csv(history_path, index=False)

        selection_score = val_metrics["auc"] if np.isfinite(val_metrics["auc"]) else -val_metrics["loss"]
        if selection_score > best_score:
            best_score = selection_score
            best_predictions = val_predictions.copy()
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "resolved_config": resolved_config,
                    "epoch": epoch,
                    "selection_score": selection_score,
                },
                checkpoint_path,
            )

        logger.info(
            "Epoch %s/%s | train_loss=%.4f | val_loss=%.4f | val_auc=%s",
            epoch,
            epochs,
            train_metrics["loss"],
            val_metrics["loss"],
            f"{val_metrics['auc']:.4f}" if np.isfinite(val_metrics["auc"]) else "nan",
        )

    if best_predictions.empty:
        raise RuntimeError("Training finished without producing validation predictions.")

    patient_predictions = aggregate_patient_predictions(best_predictions)
    merge_columns = ["patient_key", "patient_probability_max", "patient_probability_mean", "patient_probability_noisy_or"]
    best_predictions = best_predictions.merge(
        patient_predictions[merge_columns],
        on="patient_key",
        how="left",
    )
    best_predictions.to_csv(predictions_path, index=False)

    print("Single-Contrast Training Summary")
    print(f"  contrast: {data_bundle.metadata['contrast']}")
    print(f"  label_source: {args.label_source}")
    print(f"  label_column: {data_bundle.label_column}")
    print(f"  device: {device}")
    print(f"  train_images: {data_bundle.metadata['n_train_images']}")
    print(f"  val_images: {data_bundle.metadata['n_val_images']}")
    print(f"  train_pos_rate: {float(data_bundle.train_frame['target'].mean()):.4f}")
    print(f"  pos_weight: {float(pos_weight.item()) if pos_weight is not None else 'none'}")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  history_csv: {history_path}")
    print(f"  val_predictions_csv: {predictions_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
