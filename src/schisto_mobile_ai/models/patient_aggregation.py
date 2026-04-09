"""Simple patient-level aggregation rules for image-level probabilities."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def noisy_or(probabilities: Iterable[float]) -> float:
    """Combine independent image-level probabilities into one patient-level score."""
    values = np.clip(np.asarray(list(probabilities), dtype=np.float64), 1e-6, 1.0 - 1e-6)
    if values.size == 0:
        return float("nan")
    return float(1.0 - np.prod(1.0 - values))


def aggregate_probabilities(probabilities: Iterable[float], method: str) -> float:
    """Aggregate a collection of probabilities with one supported rule."""
    values = np.asarray(list(probabilities), dtype=np.float64)
    if values.size == 0:
        return float("nan")

    if method == "max":
        return float(np.max(values))
    if method == "mean":
        return float(np.mean(values))
    if method == "noisy_or":
        return noisy_or(values)

    raise ValueError("method must be one of: max, mean, noisy_or")


def aggregate_patient_predictions(
    predictions: pd.DataFrame,
    *,
    patient_key_column: str = "patient_key",
    probability_column: str = "probability",
    target_column: str = "target",
    methods: tuple[str, ...] = ("max", "mean", "noisy_or"),
    patient_target_aggregation: str = "max",
) -> pd.DataFrame:
    """Aggregate image-level predictions into one row per patient.

    patient_target_aggregation controls how per-image targets are resolved to a
    patient-level label when images within a patient have different targets (e.g.,
    when training with image-level egg-detection labels where some slides are
    egg-free even for a positive patient):
      "max"    – patient is positive if ANY image is positive (clinically correct
                 for egg detection: a patient is infected if at least one slide
                 shows eggs).
      "strict" – raises ValueError when a patient has inconsistent image targets
                 (original behaviour, useful as a sanity check when every image
                 should share the same patient-level label).
    """
    required = {patient_key_column, probability_column}
    missing = required - set(predictions.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Predictions frame is missing required columns: {missing_text}")

    if patient_target_aggregation not in ("max", "strict"):
        raise ValueError("patient_target_aggregation must be 'max' or 'strict'")

    grouped_rows = []
    for patient_key, frame in predictions.groupby(patient_key_column, sort=True):
        row = {
            patient_key_column: patient_key,
            "n_images": int(len(frame)),
        }
        if target_column in frame.columns:
            targets = sorted(set(frame[target_column].astype(float).tolist()))
            if len(targets) > 1:
                if patient_target_aggregation == "strict":
                    raise ValueError(
                        f"Patient '{patient_key}' has inconsistent targets in the prediction "
                        f"frame: {targets}. Use patient_target_aggregation='max' when training "
                        "with image-level labels where not every slide is egg-positive."
                    )
                row[target_column] = float(max(targets))
            else:
                row[target_column] = float(targets[0])

        probabilities = frame[probability_column].astype(float).tolist()
        for method in methods:
            row[f"patient_probability_{method}"] = aggregate_probabilities(probabilities, method)
        grouped_rows.append(row)

    return pd.DataFrame(grouped_rows).sort_values(patient_key_column).reset_index(drop=True)
