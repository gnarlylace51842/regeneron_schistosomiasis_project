#!/usr/bin/env python3
"""Append one completed training run to the project's master experiment log.

This is the canonical data-recording step after every training run. Every
experiment in the project — from the broken baseline to the cross-contrast
SSL pipeline — gets one row in results/experiment_log.csv, enabling a clean
longitudinal view of the research progression.

Usage:
    python scripts/log_experiment.py \\
        --run-dir runs/experiments/train_single_contrast/bf_image_level \\
        --stage stage1_fixed_baseline \\
        --notes "BF-only with image-level labels and prior bias init"
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import json
from schisto_mobile_ai.utils.io import ensure_dir

LOG_PATH = REPO_ROOT / "results" / "experiment_log.csv"

LOG_COLUMNS = [
    "timestamp",
    "stage",
    "run_name",
    "run_dir",
    # config
    "contrast",
    "label_source",
    "label_column",
    "label_fraction",
    "epochs_configured",
    "best_epoch",
    "pos_weight",
    "img_size",
    "batch_size",
    "learning_rate",
    # data
    "n_train",
    "n_val",
    "n_train_patients",
    "n_val_patients",
    "train_pos_rate",
    "val_pos_rate",
    # pair/image-level metrics at best epoch
    "val_pair_auc",
    "val_pair_accuracy",
    # patient-level metrics at best epoch
    "val_patient_auc_max",
    "val_patient_auc_mean",
    "val_patient_auc_noisy_or",
    # training metrics at best epoch
    "train_auc",
    "train_loss",
    "val_loss",
    # prediction diagnostics
    "pred_mean",
    "pred_std",
    "pred_range",
    "is_collapsed",
    # free-text
    "notes",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to the completed training run directory (must contain config.json and history.csv).",
    )
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        help=(
            "Short stage label for this row, e.g. 'stage0_broken_baseline', "
            "'stage1_bf_fixed', 'stage2_ssl_pretrain'. Use consistent names."
        ),
    )
    parser.add_argument(
        "--label-fraction",
        type=float,
        default=1.0,
        help="Fraction of training labels used (1.0 = all; <1.0 for label-efficiency experiments).",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Free-text note logged with this row.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=LOG_PATH,
        help="Path to the master experiment log CSV.",
    )
    parser.add_argument(
        "--overwrite-run",
        action="store_true",
        help="If a row for this run_dir already exists, replace it.",
    )
    return parser


def _read_config(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {run_dir}")
    with open(config_path) as f:
        return json.load(f)


def _read_history(run_dir: Path) -> pd.DataFrame:
    history_path = run_dir / "history.csv"
    if not history_path.exists():
        raise FileNotFoundError(f"history.csv not found in {run_dir}")
    return pd.read_csv(history_path)


def _read_predictions(run_dir: Path) -> pd.DataFrame | None:
    pred_path = run_dir / "val_predictions.csv"
    if not pred_path.exists():
        return None
    return pd.read_csv(pred_path)


def _find_best_epoch(history: pd.DataFrame) -> int:
    """Return the epoch index (0-based) with best val_patient_auc_mean, or val_auc fallback."""
    for col in ("val_patient_auc_mean", "val_auc"):
        if col in history.columns:
            series = pd.to_numeric(history[col], errors="coerce")
            if series.notna().any():
                return int(series.idxmax())
    return len(history) - 1


def _prediction_diagnostics(predictions: pd.DataFrame | None) -> dict[str, Any]:
    if predictions is None or "probability" not in predictions.columns:
        return {"pred_mean": None, "pred_std": None, "pred_range": None, "is_collapsed": None}
    probs = pd.to_numeric(predictions["probability"], errors="coerce").dropna()
    if probs.empty:
        return {"pred_mean": None, "pred_std": None, "pred_range": None, "is_collapsed": None}
    mean = float(probs.mean())
    std = float(probs.std(ddof=0))
    rng = float(probs.max() - probs.min())
    # Effectively constant: std < 0.01 OR range < 0.05 OR >95% within ±0.01 of 0.5
    near_center = float((probs.sub(0.5).abs() <= 0.01).mean())
    is_collapsed = bool(std < 0.01 or rng < 0.05 or near_center >= 0.95)
    return {"pred_mean": round(mean, 6), "pred_std": round(std, 6), "pred_range": round(rng, 6), "is_collapsed": is_collapsed}


def _extract_row(
    *,
    run_dir: Path,
    stage: str,
    label_fraction: float,
    notes: str,
    config: dict[str, Any],
    history: pd.DataFrame,
    predictions: pd.DataFrame | None,
) -> dict[str, Any]:
    best_idx = _find_best_epoch(history)
    best_row = history.iloc[best_idx]

    training_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    dataset_summary = config.get("dataset_summary", {})
    runtime_cfg = config.get("runtime", {})

    contrast = data_cfg.get("contrast", dataset_summary.get("contrast", ""))
    label_source = data_cfg.get("label_source", dataset_summary.get("label_source", "auto"))
    label_column = data_cfg.get("label_column", dataset_summary.get("label_column", ""))

    n_train = dataset_summary.get("n_train_pairs", dataset_summary.get("n_train_images", None))
    n_val = dataset_summary.get("n_val_pairs", dataset_summary.get("n_val_images", None))
    n_train_patients = dataset_summary.get("n_train_patients", None)
    n_val_patients = dataset_summary.get("n_val_patients", None)

    train_counts = dataset_summary.get("train_label_counts", {})
    val_counts = dataset_summary.get("val_label_counts", {})
    train_pos = int(train_counts.get("1", 0))
    train_neg = int(train_counts.get("0", 0))
    val_pos = int(val_counts.get("1", 0))
    val_neg = int(val_counts.get("0", 0))
    train_pos_rate = train_pos / (train_pos + train_neg) if (train_pos + train_neg) > 0 else None
    val_pos_rate = val_pos / (val_pos + val_neg) if (val_pos + val_neg) > 0 else None

    diag = _prediction_diagnostics(predictions)

    def _get(row: pd.Series, *keys: str) -> float | None:
        for key in keys:
            val = row.get(key)
            if val is not None and pd.notna(val):
                return round(float(val), 6)
        return None

    return {
        "timestamp": datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stage": stage,
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "contrast": contrast,
        "label_source": label_source,
        "label_column": label_column,
        "label_fraction": label_fraction,
        "epochs_configured": int(training_cfg.get("epochs", len(history))),
        "best_epoch": int(best_row.get("epoch", best_idx + 1)),
        "pos_weight": None,  # not stored in config yet; fill manually if needed
        "img_size": int(training_cfg.get("img_size", 0)) or None,
        "batch_size": int(training_cfg.get("batch_size", 0)) or None,
        "learning_rate": training_cfg.get("learning_rate", None),
        "n_train": n_train,
        "n_val": n_val,
        "n_train_patients": n_train_patients,
        "n_val_patients": n_val_patients,
        "train_pos_rate": round(train_pos_rate, 4) if train_pos_rate is not None else None,
        "val_pos_rate": round(val_pos_rate, 4) if val_pos_rate is not None else None,
        "val_pair_auc": _get(best_row, "val_auc"),
        "val_pair_accuracy": _get(best_row, "val_accuracy"),
        "val_patient_auc_max": _get(best_row, "val_patient_auc_max"),
        "val_patient_auc_mean": _get(best_row, "val_patient_auc_mean"),
        "val_patient_auc_noisy_or": _get(best_row, "val_patient_auc_noisy_or"),
        "train_auc": _get(best_row, "train_auc"),
        "train_loss": _get(best_row, "train_loss"),
        "val_loss": _get(best_row, "val_loss"),
        "pred_mean": diag["pred_mean"],
        "pred_std": diag["pred_std"],
        "pred_range": diag["pred_range"],
        "is_collapsed": diag["is_collapsed"],
        "notes": notes,
    }


def _append_to_log(log_path: Path, row: dict[str, Any], *, overwrite_run: bool) -> None:
    ensure_dir(log_path.parent)
    existing_rows: list[dict[str, Any]] = []
    if log_path.exists():
        with open(log_path, newline="") as f:
            reader = csv.DictReader(f)
            for existing in reader:
                if overwrite_run and existing.get("run_dir") == row["run_dir"]:
                    continue
                existing_rows.append(existing)

    existing_rows.append(row)
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing_rows)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        print(f"ERROR: run directory does not exist: {run_dir}", file=sys.stderr)
        return 1

    config = _read_config(run_dir)
    history = _read_history(run_dir)
    predictions = _read_predictions(run_dir)

    row = _extract_row(
        run_dir=run_dir,
        stage=args.stage,
        label_fraction=args.label_fraction,
        notes=args.notes,
        config=config,
        history=history,
        predictions=predictions,
    )

    _append_to_log(args.log_path, row, overwrite_run=args.overwrite_run)

    print(f"Logged experiment to {args.log_path}")
    print(f"  stage:             {row['stage']}")
    print(f"  run_name:          {row['run_name']}")
    print(f"  contrast:          {row['contrast']}")
    print(f"  label_source:      {row['label_source']}")
    print(f"  best_epoch:        {row['best_epoch']}")
    print(f"  val_pair_auc:      {row['val_pair_auc']}")
    print(f"  val_patient_auc_max:  {row['val_patient_auc_max']}")
    print(f"  val_patient_auc_mean: {row['val_patient_auc_mean']}")
    print(f"  pred_std:          {row['pred_std']}")
    print(f"  is_collapsed:      {row['is_collapsed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
