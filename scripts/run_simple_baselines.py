#!/usr/bin/env python3
"""Run very cheap metadata-based baselines to sanity-check label signal."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from schisto_mobile_ai.eval.metrics import compute_binary_classification_metrics
from schisto_mobile_ai.utils.io import ensure_dir, write_json
from schisto_mobile_ai.utils.logging import configure_logging


POSITIVE_LABELS = {"positive", "1", "true", "yes"}
NEGATIVE_LABELS = {"negative", "0", "false", "no"}


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for cheap baseline evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--image-quality-csv",
        type=Path,
        default=REPO_ROOT / "metadata" / "image_quality.csv",
        help="Optional path to metadata/image_quality.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "diagnostics" / "simple_baselines",
        help="Directory where baseline outputs will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stochastic baselines.",
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


def _read_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Required CSV does not exist: {csv_path}")
    return pd.read_csv(csv_path)


def _normalize_label(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def _label_to_target(value: Any) -> float | None:
    normalized = _normalize_label(value)
    if normalized in POSITIVE_LABELS:
        return 1.0
    if normalized in NEGATIVE_LABELS:
        return 0.0
    return None


def _resolve_eval_split(splits: pd.DataFrame) -> str:
    values = set(splits["split"].dropna().astype(str))
    if "val" in values:
        return "val"
    if "test" in values:
        return "test"
    raise ValueError("Split CSV must contain either a 'val' or 'test' split.")


def _scale_feature(eval_values: pd.Series, train_values: pd.Series) -> pd.Series:
    train_numeric = pd.to_numeric(train_values, errors="coerce").dropna()
    eval_numeric = pd.to_numeric(eval_values, errors="coerce")
    if train_numeric.empty:
        return pd.Series(np.nan, index=eval_values.index, dtype=float)

    train_min = float(train_numeric.min())
    train_max = float(train_numeric.max())
    if train_max <= train_min:
        return eval_numeric.map(lambda value: float(value > 0.0) if pd.notna(value) else np.nan)

    scaled = (eval_numeric - train_min) / (train_max - train_min)
    return scaled.clip(lower=0.0, upper=1.0)


def _confusion_counts(targets: pd.Series, predictions: pd.Series) -> dict[str, int]:
    target_values = targets.astype(int)
    prediction_values = predictions.astype(int)
    return {
        "tp": int(((target_values == 1) & (prediction_values == 1)).sum()),
        "tn": int(((target_values == 0) & (prediction_values == 0)).sum()),
        "fp": int(((target_values == 0) & (prediction_values == 1)).sum()),
        "fn": int(((target_values == 1) & (prediction_values == 0)).sum()),
    }


def _save_baseline_run(
    *,
    baseline_name: str,
    patient_frame: pd.DataFrame,
    patient_scores: pd.Series,
    output_root: Path,
    evaluation_split_name: str,
    inputs: dict[str, str],
    details: dict[str, Any],
) -> dict[str, Any]:
    output_dir = ensure_dir(output_root / baseline_name)
    working = patient_frame.copy()
    working["patient_probability"] = pd.to_numeric(patient_scores, errors="coerce")
    working = working.dropna(subset=["patient_probability"]).reset_index(drop=True)
    if working.empty:
        raise ValueError(f"Baseline '{baseline_name}' did not produce any valid patient scores.")

    working["predicted_label"] = (working["patient_probability"] >= 0.5).astype(int)
    patient_predictions = working[
        [
            "patient_key",
            "patient_id",
            "study_id",
            "split",
            "patient_label",
            "target",
            "patient_probability",
            "predicted_label",
        ]
    ].copy()
    patient_predictions["baseline_name"] = baseline_name
    patient_predictions.to_csv(output_dir / "patient_predictions.csv", index=False)

    metrics = compute_binary_classification_metrics(
        patient_predictions["target"].astype(int).tolist(),
        patient_predictions["patient_probability"].astype(float).tolist(),
        threshold=0.5,
    )
    confusion = _confusion_counts(patient_predictions["target"], patient_predictions["predicted_label"])
    payload = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "inputs": inputs,
        "outputs": {
            "output_dir": str(output_dir),
            "patient_predictions_csv": str(output_dir / "patient_predictions.csv"),
            "metrics_json": str(output_dir / "metrics.json"),
        },
        "evaluation": {
            "aggregation": baseline_name,
            "threshold": 0.5,
            "evaluation_split_name": evaluation_split_name,
            "n_patients": int(len(patient_predictions)),
            "n_positive_patients": int((patient_predictions["target"] == 1).sum()),
            "n_negative_patients": int((patient_predictions["target"] == 0).sum()),
        },
        "metrics": metrics,
        "confusion_matrix": confusion,
        "baseline_details": details,
    }
    write_json(output_dir / "metrics.json", payload)
    return {
        "baseline_name": baseline_name,
        "n_patients": int(len(patient_predictions)),
        "accuracy": metrics["accuracy"],
        "sensitivity": metrics["sensitivity"],
        "specificity": metrics["specificity"],
        "precision": metrics["precision"],
        "f1": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
        "output_dir": str(output_dir),
    }


def _build_egg_feature(pairs: pd.DataFrame) -> tuple[pd.Series | None, str | None]:
    if "patient_eggs" in pairs.columns and pd.to_numeric(pairs["patient_eggs"], errors="coerce").notna().any():
        values = pd.to_numeric(pairs["patient_eggs"], errors="coerce")
        feature = pairs.assign(patient_eggs_numeric=values).groupby("patient_key")["patient_eggs_numeric"].max()
        return feature, "patient_eggs"

    egg_columns = [column for column in ["brightfield_number_eggs", "darkfield_number_eggs"] if column in pairs.columns]
    if egg_columns:
        working = pairs[["patient_key", *egg_columns]].copy()
        for column in egg_columns:
            working[column] = pd.to_numeric(working[column], errors="coerce")
        working["pair_egg_score"] = working[egg_columns].mean(axis=1)
        feature = working.groupby("patient_key")["pair_egg_score"].mean()
        return feature, "mean_pair_number_eggs"

    return None, None


def _build_quality_feature(
    *,
    pairs: pd.DataFrame,
    image_quality: pd.DataFrame,
    train_patients: pd.DataFrame,
    eval_patients: pd.DataFrame,
) -> tuple[pd.Series | None, dict[str, Any] | None]:
    quality_columns = [
        column
        for column in ["blur_score", "brightness_std", "contrast_std", "edge_density"]
        if column in image_quality.columns
    ]
    if not quality_columns:
        return None, None

    quality = image_quality[["image_id", *quality_columns]].copy()
    bf_quality = quality.rename(columns={column: f"bf_{column}" for column in quality_columns})
    df_quality = quality.rename(columns={column: f"df_{column}" for column in quality_columns})

    pair_quality = pairs[["patient_key", "brightfield_image_id", "darkfield_image_id"]].copy()
    pair_quality = pair_quality.merge(
        bf_quality,
        left_on="brightfield_image_id",
        right_on="image_id",
        how="left",
    ).drop(columns="image_id")
    pair_quality = pair_quality.merge(
        df_quality,
        left_on="darkfield_image_id",
        right_on="image_id",
        how="left",
    ).drop(columns="image_id")

    pair_feature_columns: list[str] = []
    for column in quality_columns:
        bf_column = f"bf_{column}"
        df_column = f"df_{column}"
        pair_column = f"pair_mean_{column}"
        pair_quality[pair_column] = pair_quality[[bf_column, df_column]].apply(
            lambda row: pd.to_numeric(row, errors="coerce").mean(),
            axis=1,
        )
        pair_feature_columns.append(pair_column)

    patient_quality = pair_quality.groupby("patient_key")[pair_feature_columns].mean().reset_index()
    patient_quality = patient_quality.merge(
        pd.concat([train_patients, eval_patients], ignore_index=True)[["patient_key", "target"]].drop_duplicates("patient_key"),
        on="patient_key",
        how="inner",
    )

    best_feature = None
    best_auc = -1.0
    best_sign = 1.0
    for column in pair_feature_columns:
        train_feature = train_patients[["patient_key", "target"]].merge(
            patient_quality[["patient_key", column]],
            on="patient_key",
            how="left",
        )
        train_feature[column] = pd.to_numeric(train_feature[column], errors="coerce")
        train_feature = train_feature.dropna(subset=[column])
        if train_feature.empty or train_feature[column].nunique() < 2 or train_feature["target"].nunique() < 2:
            continue
        auc = compute_binary_classification_metrics(
            train_feature["target"].astype(int).tolist(),
            train_feature[column].astype(float).tolist(),
            threshold=float(train_feature[column].median()),
        )["roc_auc"]
        if auc is None:
            continue
        effective_auc = float(auc) if auc >= 0.5 else float(1.0 - auc)
        sign = 1.0 if auc >= 0.5 else -1.0
        if effective_auc > best_auc:
            best_auc = effective_auc
            best_feature = column
            best_sign = sign

    if best_feature is None:
        return None, None

    train_values = train_patients[["patient_key"]].merge(
        patient_quality[["patient_key", best_feature]],
        on="patient_key",
        how="left",
    )[best_feature].astype(float) * best_sign
    eval_values = eval_patients[["patient_key"]].merge(
        patient_quality[["patient_key", best_feature]],
        on="patient_key",
        how="left",
    )[best_feature].astype(float) * best_sign
    scores = _scale_feature(eval_values, train_values)
    return scores, {
        "selected_feature": best_feature,
        "effective_train_auc": best_auc,
        "sign": "higher_is_more_positive" if best_sign > 0 else "lower_is_more_positive",
    }


def main() -> int:
    """Run cheap baselines on the validation or test split."""
    parser = build_parser()
    args = parser.parse_args()
    logger = configure_logging(quiet=args.quiet)

    ensure_dir(args.output_dir)
    summary_path = args.output_dir / "baseline_summary.csv"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output directory already contains files: {args.output_dir}. Pass --overwrite to replace them."
        )

    pairs = _read_csv(args.pairs_csv)
    patients = _read_csv(args.patients_csv)
    splits = _read_csv(args.split_csv)
    image_quality = _read_csv(args.image_quality_csv) if args.image_quality_csv.exists() else pd.DataFrame()

    patient_frame = patients[["patient_key", "patient_id", "study_id", "patient_label"]].copy()
    patient_frame["target"] = patient_frame["patient_label"].map(_label_to_target)
    patient_frame = patient_frame.dropna(subset=["target"]).reset_index(drop=True)
    patient_frame = patient_frame.merge(
        splits[["patient_key", "split"]].drop_duplicates("patient_key"),
        on="patient_key",
        how="inner",
        validate="one_to_one",
    )

    evaluation_split_name = _resolve_eval_split(splits)
    train_patients = patient_frame[patient_frame["split"] == "train"].copy().reset_index(drop=True)
    eval_patients = patient_frame[patient_frame["split"] == evaluation_split_name].copy().reset_index(drop=True)
    if train_patients.empty or eval_patients.empty:
        raise ValueError("Could not build non-empty train/eval patient tables from patients.csv and split CSV.")

    inputs = {
        "pairs_csv": str(args.pairs_csv),
        "patients_csv": str(args.patients_csv),
        "split_csv": str(args.split_csv),
        "image_quality_csv": str(args.image_quality_csv),
    }
    rng = np.random.default_rng(args.seed)
    train_positive_rate = float(train_patients["target"].mean())

    summary_rows: list[dict[str, Any]] = []
    summary_rows.append(
        _save_baseline_run(
            baseline_name="always_negative",
            patient_frame=eval_patients,
            patient_scores=pd.Series(0.0, index=eval_patients.index),
            output_root=args.output_dir,
            evaluation_split_name=evaluation_split_name,
            inputs=inputs,
            details={"rule": "constant_zero"},
        )
    )
    summary_rows.append(
        _save_baseline_run(
            baseline_name="always_positive",
            patient_frame=eval_patients,
            patient_scores=pd.Series(1.0, index=eval_patients.index),
            output_root=args.output_dir,
            evaluation_split_name=evaluation_split_name,
            inputs=inputs,
            details={"rule": "constant_one"},
        )
    )
    summary_rows.append(
        _save_baseline_run(
            baseline_name="random_class_prior",
            patient_frame=eval_patients,
            patient_scores=pd.Series(
                rng.binomial(1, train_positive_rate, size=len(eval_patients)).astype(float),
                index=eval_patients.index,
            ),
            output_root=args.output_dir,
            evaluation_split_name=evaluation_split_name,
            inputs=inputs,
            details={"seed": args.seed, "train_positive_rate": train_positive_rate},
        )
    )

    egg_feature, egg_source = _build_egg_feature(pairs)
    if egg_feature is not None:
        train_egg = train_patients["patient_key"].map(egg_feature)
        eval_egg = eval_patients["patient_key"].map(egg_feature)
        summary_rows.append(
            _save_baseline_run(
                baseline_name="egg_count_heuristic",
                patient_frame=eval_patients,
                patient_scores=_scale_feature(eval_egg, train_egg),
                output_root=args.output_dir,
                evaluation_split_name=evaluation_split_name,
                inputs=inputs,
                details={"source_column": egg_source},
            )
        )

    if not image_quality.empty:
        quality_scores, quality_details = _build_quality_feature(
            pairs=pairs,
            image_quality=image_quality,
            train_patients=train_patients,
            eval_patients=eval_patients,
        )
        if quality_scores is not None and quality_details is not None:
            summary_rows.append(
                _save_baseline_run(
                    baseline_name="image_quality_heuristic",
                    patient_frame=eval_patients,
                    patient_scores=quality_scores,
                    output_root=args.output_dir,
                    evaluation_split_name=evaluation_split_name,
                    inputs=inputs,
                    details=quality_details,
                )
            )

    summary = pd.DataFrame(summary_rows).sort_values("roc_auc", ascending=False, na_position="last").reset_index(drop=True)
    summary.to_csv(summary_path, index=False)
    write_json(args.output_dir / "baseline_summary.json", summary_rows)

    print("Simple Baseline Summary")
    print(summary.to_string(index=False))
    logger.info("Saved baseline diagnostics to %s", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
