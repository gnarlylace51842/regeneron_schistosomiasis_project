#!/usr/bin/env python3
"""Run a tiny sample-efficiency ablation across BF, DF, and dual models."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from schisto_mobile_ai.data.classification import load_single_contrast_data
from schisto_mobile_ai.data.paired_classification import load_dual_contrast_data
from schisto_mobile_ai.eval.patient_level import evaluate_patient_level
from schisto_mobile_ai.utils.io import ensure_dir, write_json
from schisto_mobile_ai.utils.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for tiny ablation runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images-csv",
        type=Path,
        default=REPO_ROOT / "metadata" / "images.csv",
        help="Path to metadata/images.csv.",
    )
    parser.add_argument(
        "--pairs-csv",
        type=Path,
        default=REPO_ROOT / "metadata" / "pairs.csv",
        help="Path to metadata/pairs.csv.",
    )
    parser.add_argument(
        "--patients-csv",
        type=Path,
        default=REPO_ROOT / "metadata" / "patients.csv",
        help="Path to metadata/patients.csv.",
    )
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=REPO_ROOT / "splits" / "random_patient_split.csv",
        help="Path to the split CSV.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=REPO_ROOT / "data" / "raw",
        help="Root directory for image paths.",
    )
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.25],
        help="Training-set fractions to evaluate.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="One to three random seeds.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Epoch count for each tiny ablation run.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Mini-batch size.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=128,
        help="Image resize used across all runs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=("auto", "cpu", "mps"),
        default="auto",
        help="Execution device.",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=128,
        help="Optional cap on validation samples to keep runs fast.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "diagnostics" / "ablation_sample_efficiency",
        help="Directory where run artifacts and summary tables will be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing outputs.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce log output.",
    )
    return parser


def _guard_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    ensure_dir(output_dir)
    blocking = [
        output_dir / "summary.csv",
        output_dir / "summary.json",
    ]
    existing = [path for path in blocking if path.exists()]
    if existing and not overwrite:
        existing_text = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            "Ablation outputs already exist. Pass --overwrite to replace them: "
            f"{existing_text}"
        )


def _resolve_train_counts(args: argparse.Namespace) -> dict[str, dict[str, int]]:
    bf_bundle = load_single_contrast_data(
        images_csv=args.images_csv,
        split_csv=args.split_csv,
        raw_dir=args.raw_dir,
        contrast="bf",
        seed=args.seeds[0],
    )
    df_bundle = load_single_contrast_data(
        images_csv=args.images_csv,
        split_csv=args.split_csv,
        raw_dir=args.raw_dir,
        contrast="df",
        seed=args.seeds[0],
    )
    dual_bundle = load_dual_contrast_data(
        pairs_csv=args.pairs_csv,
        patients_csv=args.patients_csv,
        split_csv=args.split_csv,
        raw_dir=args.raw_dir,
        seed=args.seeds[0],
    )
    return {
        "bf": {
            "train": int(len(bf_bundle.train_frame)),
            "val": int(len(bf_bundle.val_frame)),
        },
        "df": {
            "train": int(len(df_bundle.train_frame)),
            "val": int(len(df_bundle.val_frame)),
        },
        "dual": {
            "train": int(len(dual_bundle.train_frame)),
            "val": int(len(dual_bundle.val_frame)),
        },
    }


def _train_command(
    *,
    modality: str,
    args: argparse.Namespace,
    train_limit: int,
    val_limit: int,
    seed: int,
    output_dir: Path,
) -> list[str]:
    python_bin = sys.executable
    if modality in {"bf", "df"}:
        return [
            python_bin,
            "scripts/train_single_contrast.py",
            "--images-csv",
            str(args.images_csv),
            "--split-csv",
            str(args.split_csv),
            "--raw-dir",
            str(args.raw_dir),
            "--contrast",
            modality,
            "--max-train-samples",
            str(train_limit),
            "--max-val-samples",
            str(val_limit),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--img-size",
            str(args.img_size),
            "--device",
            args.device,
            "--seed",
            str(seed),
            "--output-dir",
            str(output_dir),
            "--overwrite",
        ]

    return [
        python_bin,
        "scripts/train_dual_contrast.py",
        "--pairs-csv",
        str(args.pairs_csv),
        "--patients-csv",
        str(args.patients_csv),
        "--split-csv",
        str(args.split_csv),
        "--raw-dir",
        str(args.raw_dir),
        "--max-train-samples",
        str(train_limit),
        "--max-val-samples",
        str(val_limit),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--img-size",
        str(args.img_size),
        "--device",
        args.device,
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir),
        "--overwrite",
    ]


def main() -> int:
    """Run a tiny ablation to see whether any formulation picks up signal quickly."""
    parser = build_parser()
    args = parser.parse_args()
    logger = configure_logging(quiet=args.quiet)

    if len(args.seeds) > 3:
        raise ValueError("Please keep --seeds to at most 3 values for this lightweight ablation.")
    if any(fraction <= 0 or fraction > 1 for fraction in args.fractions):
        raise ValueError("All --fractions values must be in the interval (0, 1].")

    _guard_output_dir(args.output_dir, overwrite=args.overwrite)
    counts = _resolve_train_counts(args)
    summary_rows: list[dict[str, Any]] = []

    for fraction in args.fractions:
        for seed in args.seeds:
            for modality in ("bf", "df", "dual"):
                train_limit = max(1, int(round(counts[modality]["train"] * fraction)))
                val_limit = min(counts[modality]["val"], args.max_val_samples)
                train_output_dir = args.output_dir / "train_runs" / f"{modality}_frac{int(round(fraction * 100)):02d}_seed{seed}"
                eval_output_dir = args.output_dir / "eval_runs" / f"{modality}_frac{int(round(fraction * 100)):02d}_seed{seed}"
                command = _train_command(
                    modality=modality,
                    args=args,
                    train_limit=train_limit,
                    val_limit=val_limit,
                    seed=seed,
                    output_dir=train_output_dir,
                )
                logger.info("Running %s fraction %.2f seed %s", modality, fraction, seed)
                subprocess.run(command, check=True, cwd=REPO_ROOT)

                metrics = evaluate_patient_level(
                    predictions_csv=train_output_dir / "val_predictions.csv",
                    patients_csv=args.patients_csv,
                    split_csv=args.split_csv,
                    output_dir=eval_output_dir,
                    aggregation="mean",
                    overwrite=True,
                ).metrics
                history = pd.read_csv(train_output_dir / "history.csv")
                last_history = history.iloc[-1].to_dict()
                summary_rows.append(
                    {
                        "modality": modality,
                        "fraction": float(fraction),
                        "seed": int(seed),
                        "train_samples": int(train_limit),
                        "val_samples": int(val_limit),
                        "train_loss": last_history.get("train_loss"),
                        "val_loss": last_history.get("val_loss"),
                        "model_level_val_auc": last_history.get("val_auc"),
                        "patient_level_accuracy_mean": metrics["metrics"]["accuracy"],
                        "patient_level_roc_auc_mean": metrics["metrics"]["roc_auc"],
                        "train_output_dir": str(train_output_dir),
                        "eval_output_dir": str(eval_output_dir),
                    }
                )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["fraction", "modality", "seed"]
    ).reset_index(drop=True)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    write_json(
        args.output_dir / "summary.json",
        {
            "inputs": {
                "images_csv": str(args.images_csv),
                "pairs_csv": str(args.pairs_csv),
                "patients_csv": str(args.patients_csv),
                "split_csv": str(args.split_csv),
                "raw_dir": str(args.raw_dir),
                "fractions": args.fractions,
                "seeds": args.seeds,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "img_size": args.img_size,
                "device": args.device,
                "max_val_samples": args.max_val_samples,
            },
            "summary_rows": summary_rows,
        },
    )

    print("Sample-Efficiency Ablation Summary")
    print(summary.to_string(index=False))
    logger.info("Saved ablation summary to %s", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
