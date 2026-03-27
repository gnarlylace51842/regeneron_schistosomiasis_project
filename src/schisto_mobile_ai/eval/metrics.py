"""Metric utilities that do not assume a fixed metadata schema."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    """Return ROC-AUC when both classes are present, otherwise None."""
    unique = np.unique(y_true)
    if unique.size < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def _safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    """Return average precision when both classes are present, otherwise None."""
    unique = np.unique(y_true)
    if unique.size < 2:
        return None
    return float(average_precision_score(y_true, y_score))


def compute_binary_classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float | int | None]:
    """Compute a compact binary classification metric dictionary."""
    labels = np.asarray(y_true)
    scores = np.asarray(y_score)
    predictions = (scores >= threshold).astype(int)

    return {
        "n_samples": int(labels.size),
        "positive_rate": float(labels.mean()) if labels.size else 0.0,
        "threshold": float(threshold),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": _safe_roc_auc(labels, scores),
        "average_precision": _safe_average_precision(labels, scores),
    }


def aggregate_group_scores(
    group_ids: Sequence[str] | Sequence[int],
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    *,
    reduction: str = "mean",
) -> pd.DataFrame:
    """Aggregate image-level predictions to generic group-level predictions."""
    frame = pd.DataFrame(
        {
            "group_id": group_ids,
            "y_true": y_true,
            "y_score": y_score,
        }
    )

    if reduction != "mean":
        raise ValueError("Only 'mean' reduction is implemented in the starter scaffold.")

    aggregated = (
        frame.groupby("group_id", as_index=False)
        .agg(y_true=("y_true", "max"), y_score=("y_score", "mean"))
        .sort_values("group_id")
        .reset_index(drop=True)
    )
    return aggregated

