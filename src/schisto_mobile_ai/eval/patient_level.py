"""Patient-level evaluation helpers for metadata-driven schistosomiasis experiments."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from schisto_mobile_ai.data.manifest import validate_required_columns
from schisto_mobile_ai.eval.metrics import aggregate_group_scores, compute_binary_classification_metrics
from schisto_mobile_ai.utils.io import ensure_dir, write_json


POSITIVE_LABELS = {"positive", "1", "true", "yes"}
NEGATIVE_LABELS = {"negative", "0", "false", "no"}


@dataclass
class PatientLevelEvalResult:
    """Outputs produced by one patient-level evaluation run."""

    patient_predictions: pd.DataFrame
    metrics: dict[str, Any]
    patient_predictions_path: Path
    metrics_path: Path
    confusion_matrix_path: Path


def _label_to_target(value: Any) -> float | None:
    if pd.isna(value):
        return None
    normalized = str(value).strip().lower()
    if normalized in POSITIVE_LABELS:
        return 1.0
    if normalized in NEGATIVE_LABELS:
        return 0.0
    return None


def _read_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Required CSV does not exist: {csv_path}")
    return pd.read_csv(csv_path, dtype=str)


def load_patient_level_inputs(
    *,
    predictions_csv: str | Path,
    patients_csv: str | Path,
    split_csv: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the prediction, patient, and split tables required for evaluation."""
    predictions = _read_csv(predictions_csv)
    patients = _read_csv(patients_csv)
    splits = _read_csv(split_csv)

    validate_required_columns(
        predictions,
        ["image_id", "patient_key", "patient_id", "study_id", "split", "target", "probability"],
        table_name="validation predictions CSV",
    )
    validate_required_columns(
        patients,
        ["patient_key", "patient_id", "study_id", "patient_label"],
        table_name="patients.csv",
    )
    validate_required_columns(
        splits,
        ["patient_key", "split"],
        table_name="split CSV",
    )

    predictions = predictions.copy()
    predictions["target"] = pd.to_numeric(predictions["target"], errors="coerce")
    predictions["probability"] = pd.to_numeric(predictions["probability"], errors="coerce")
    predictions = predictions.dropna(subset=["target", "probability"]).reset_index(drop=True)
    if predictions.empty:
        raise ValueError("No valid prediction rows were found after parsing target/probability columns.")

    patients = patients.copy()
    patients["patient_target"] = patients["patient_label"].map(_label_to_target)
    patients = patients.dropna(subset=["patient_target"]).reset_index(drop=True)
    if patients.empty:
        raise ValueError("No valid patient-level labels were found in patients.csv.")

    return predictions, patients, splits


def build_patient_prediction_frame(
    predictions: pd.DataFrame,
    *,
    patients: pd.DataFrame,
    splits: pd.DataFrame,
    aggregation: str,
    threshold: float,
) -> tuple[pd.DataFrame, str]:
    """Aggregate image-level predictions to patient level and merge ground-truth metadata."""
    grouped = aggregate_group_scores(
        predictions["patient_key"].tolist(),
        predictions["target"].astype(float).tolist(),
        predictions["probability"].astype(float).tolist(),
        reduction=aggregation,
    ).rename(
        columns={
            "group_id": "patient_key",
            "y_true": "target_from_predictions",
            "y_score": "patient_probability",
            "n_items": "n_images",
        }
    )

    patient_info = patients[["patient_key", "patient_id", "study_id", "patient_label", "patient_target"]].drop_duplicates(
        subset=["patient_key"]
    )
    split_info = splits[["patient_key", "split"]].drop_duplicates(subset=["patient_key"])

    merged = grouped.merge(patient_info, on="patient_key", how="left", validate="one_to_one")
    merged = merged.merge(split_info, on="patient_key", how="left", validate="one_to_one")
    if merged["patient_target"].isna().any():
        missing_keys = merged.loc[merged["patient_target"].isna(), "patient_key"].tolist()[:10]
        raise ValueError(
            "Some evaluated patients are missing from patients.csv or have unusable labels: "
            + ", ".join(missing_keys)
        )

    prediction_targets = merged["target_from_predictions"].astype(float)
    patient_targets = merged["patient_target"].astype(float)
    mismatched = merged[prediction_targets != patient_targets]
    if not mismatched.empty:
        bad_keys = mismatched["patient_key"].tolist()[:10]
        raise ValueError(
            "Prediction targets disagree with patient-level labels for these patients: "
            + ", ".join(bad_keys)
        )

    prediction_splits = sorted(set(predictions["split"].dropna().tolist()))
    evaluation_split_name = prediction_splits[0] if prediction_splits else "unknown"
    if len(prediction_splits) > 1:
        evaluation_split_name = ",".join(prediction_splits)

    merged["predicted_label"] = (merged["patient_probability"] >= threshold).astype(int)
    merged["target"] = merged["patient_target"].astype(int)
    merged["aggregation"] = aggregation

    patient_predictions = merged[
        [
            "patient_key",
            "patient_id",
            "study_id",
            "split",
            "patient_label",
            "target",
            "n_images",
            "patient_probability",
            "predicted_label",
            "aggregation",
        ]
    ].sort_values(["study_id", "patient_id", "patient_key"]).reset_index(drop=True)
    return patient_predictions, evaluation_split_name


def _confusion_counts(targets: pd.Series, predictions: pd.Series) -> dict[str, int]:
    targets_int = targets.astype(int)
    predictions_int = predictions.astype(int)
    tp = int(((targets_int == 1) & (predictions_int == 1)).sum())
    tn = int(((targets_int == 0) & (predictions_int == 0)).sum())
    fp = int(((targets_int == 0) & (predictions_int == 1)).sum())
    fn = int(((targets_int == 1) & (predictions_int == 0)).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _plot_confusion_matrix(
    confusion: dict[str, int],
    *,
    title: str,
    output_path: Path,
) -> None:
    matrix = np.asarray(
        [
            [confusion["tn"], confusion["fp"]],
            [confusion["fn"], confusion["tp"]],
        ],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(4.5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1], labels=["True 0", "True 1"])
    ax.set_title(title)
    for row in range(2):
        for column in range(2):
            ax.text(
                column,
                row,
                f"{int(matrix[row, column])}",
                ha="center",
                va="center",
                color="black",
                fontsize=12,
            )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_metrics_payload(
    patient_predictions: pd.DataFrame,
    *,
    predictions_csv: Path,
    patients_csv: Path,
    split_csv: Path,
    output_dir: Path,
    aggregation: str,
    threshold: float,
    evaluation_split_name: str,
    confusion_matrix_path: Path,
) -> dict[str, Any]:
    """Create the JSON payload saved for patient-level evaluation."""
    metrics = compute_binary_classification_metrics(
        patient_predictions["target"].astype(int).tolist(),
        patient_predictions["patient_probability"].astype(float).tolist(),
        threshold=threshold,
    )
    confusion = _confusion_counts(
        patient_predictions["target"],
        patient_predictions["predicted_label"],
    )

    return {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "inputs": {
            "predictions_csv": str(predictions_csv),
            "patients_csv": str(patients_csv),
            "split_csv": str(split_csv),
        },
        "outputs": {
            "output_dir": str(output_dir),
            "patient_predictions_csv": str(output_dir / "patient_predictions.csv"),
            "metrics_json": str(output_dir / "metrics.json"),
            "confusion_matrix_png": str(confusion_matrix_path),
        },
        "evaluation": {
            "aggregation": aggregation,
            "threshold": float(threshold),
            "evaluation_split_name": evaluation_split_name,
            "n_patients": int(len(patient_predictions)),
            "n_positive_patients": int((patient_predictions["target"] == 1).sum()),
            "n_negative_patients": int((patient_predictions["target"] == 0).sum()),
        },
        "metrics": metrics,
        "confusion_matrix": confusion,
    }


def format_patient_eval_summary(metrics_payload: dict[str, Any]) -> str:
    """Create a concise terminal summary table."""
    evaluation = metrics_payload["evaluation"]
    metrics = metrics_payload["metrics"]
    confusion = metrics_payload["confusion_matrix"]

    def format_value(value: Any) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    rows = [
        ("aggregation", evaluation["aggregation"]),
        ("split", evaluation["evaluation_split_name"]),
        ("n_patients", evaluation["n_patients"]),
        ("accuracy", metrics["accuracy"]),
        ("sensitivity", metrics["sensitivity"]),
        ("specificity", metrics["specificity"]),
        ("precision", metrics["precision"]),
        ("f1", metrics["f1"]),
        ("roc_auc", metrics["roc_auc"]),
        ("tp", confusion["tp"]),
        ("tn", confusion["tn"]),
        ("fp", confusion["fp"]),
        ("fn", confusion["fn"]),
    ]
    label_width = max(len(label) for label, _ in rows)
    lines = ["Patient-Level Evaluation Summary"]
    for label, value in rows:
        lines.append(f"  {label.ljust(label_width)}  {format_value(value)}")
    return "\n".join(lines)


def evaluate_patient_level(
    *,
    predictions_csv: str | Path,
    patients_csv: str | Path,
    split_csv: str | Path,
    output_dir: str | Path,
    aggregation: str,
    threshold: float = 0.5,
    overwrite: bool = False,
) -> PatientLevelEvalResult:
    """Run patient-level aggregation, metrics, and confusion-matrix generation."""
    predictions_path = Path(predictions_csv)
    patients_path = Path(patients_csv)
    split_path = Path(split_csv)
    destination = Path(output_dir)
    ensure_dir(destination)

    patient_predictions_path = destination / "patient_predictions.csv"
    metrics_path = destination / "metrics.json"
    confusion_matrix_path = destination / "confusion_matrix.png"

    blocking_paths = [path for path in [patient_predictions_path, metrics_path, confusion_matrix_path] if path.exists()]
    if blocking_paths and not overwrite:
        blocking_text = ", ".join(str(path) for path in blocking_paths)
        raise FileExistsError(
            "Evaluation outputs already exist. Pass --overwrite to replace them: "
            f"{blocking_text}"
        )

    predictions, patients, splits = load_patient_level_inputs(
        predictions_csv=predictions_path,
        patients_csv=patients_path,
        split_csv=split_path,
    )
    patient_predictions, evaluation_split_name = build_patient_prediction_frame(
        predictions,
        patients=patients,
        splits=splits,
        aggregation=aggregation,
        threshold=threshold,
    )
    patient_predictions.to_csv(patient_predictions_path, index=False)

    metrics_payload = build_metrics_payload(
        patient_predictions,
        predictions_csv=predictions_path,
        patients_csv=patients_path,
        split_csv=split_path,
        output_dir=destination,
        aggregation=aggregation,
        threshold=threshold,
        evaluation_split_name=evaluation_split_name,
        confusion_matrix_path=confusion_matrix_path,
    )
    _plot_confusion_matrix(
        metrics_payload["confusion_matrix"],
        title=f"{aggregation} aggregation ({evaluation_split_name})",
        output_path=confusion_matrix_path,
    )
    write_json(metrics_path, metrics_payload)

    return PatientLevelEvalResult(
        patient_predictions=patient_predictions,
        metrics=metrics_payload,
        patient_predictions_path=patient_predictions_path,
        metrics_path=metrics_path,
        confusion_matrix_path=confusion_matrix_path,
    )


def compare_metrics_payloads(metrics_payloads: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a side-by-side comparison table from several metrics.json payloads."""
    rows = []
    for payload in metrics_payloads:
        evaluation = payload.get("evaluation", {})
        metrics = payload.get("metrics", {})
        rows.append(
            {
                "run": Path(payload.get("inputs", {}).get("predictions_csv", "")).parent.name
                or Path(payload.get("outputs", {}).get("output_dir", "")).name,
                "aggregation": evaluation.get("aggregation"),
                "split": evaluation.get("evaluation_split_name"),
                "n_patients": evaluation.get("n_patients"),
                "accuracy": metrics.get("accuracy"),
                "sensitivity": metrics.get("sensitivity"),
                "specificity": metrics.get("specificity"),
                "precision": metrics.get("precision"),
                "f1": metrics.get("f1"),
                "roc_auc": metrics.get("roc_auc"),
                "metrics_json": payload.get("outputs", {}).get("metrics_json"),
            }
        )
    return pd.DataFrame(rows)
