"""Metric utilities that do not assume a fixed metadata schema."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    """Return ROC-AUC when both classes are present, otherwise None."""
    unique = np.unique(y_true)
    if unique.size < 2:
        return None
    frame = pd.DataFrame(
        {
            "y_true": pd.Series(y_true, dtype=float),
            "y_score": pd.Series(y_score, dtype=float),
        }
    )
    positive_mask = frame["y_true"] >= 0.5
    negative_mask = ~positive_mask
    positive_count = int(positive_mask.sum())
    negative_count = int(negative_mask.sum())
    if positive_count == 0 or negative_count == 0:
        return None

    ranks = frame["y_score"].rank(method="average")
    positive_rank_sum = float(ranks[positive_mask].sum())
    auc = (
        positive_rank_sum
        - (positive_count * (positive_count + 1) / 2.0)
    ) / (positive_count * negative_count)
    return float(auc)


def _safe_divide(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _confusion_counts(labels: np.ndarray, predictions: np.ndarray) -> dict[str, int]:
    tp = int(((labels == 1) & (predictions == 1)).sum())
    tn = int(((labels == 0) & (predictions == 0)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _noisy_or(scores: np.ndarray) -> float:
    clipped = np.clip(scores.astype(float), 1e-6, 1.0 - 1e-6)
    return float(1.0 - np.prod(1.0 - clipped))


def compute_binary_classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float | int | None]:
    """Compute a compact binary classification metric dictionary."""
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    predictions = (scores >= threshold).astype(int)
    confusion = _confusion_counts(labels, predictions)
    precision = _safe_divide(confusion["tp"], confusion["tp"] + confusion["fp"])
    sensitivity = _safe_divide(confusion["tp"], confusion["tp"] + confusion["fn"])
    specificity = _safe_divide(confusion["tn"], confusion["tn"] + confusion["fp"])
    accuracy = _safe_divide(confusion["tp"] + confusion["tn"], int(labels.size))
    if precision is None or sensitivity is None or (precision + sensitivity) == 0:
        f1 = None
    else:
        f1 = float(2.0 * precision * sensitivity / (precision + sensitivity))

    return {
        "n_samples": int(labels.size),
        "positive_rate": float(labels.mean()) if labels.size else 0.0,
        "threshold": float(threshold),
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "roc_auc": _safe_roc_auc(labels, scores),
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

    if reduction not in {"mean", "max", "noisy_or"}:
        raise ValueError("reduction must be one of: mean, max, noisy_or")

    per_group_rows = []
    for group_id, group_frame in frame.groupby("group_id", sort=True):
        unique_targets = sorted(set(group_frame["y_true"].astype(float).tolist()))
        if len(unique_targets) > 1:
            raise ValueError(
                f"Group '{group_id}' has inconsistent y_true values: {unique_targets}"
            )

        scores = group_frame["y_score"].astype(float).to_numpy()
        if reduction == "mean":
            score = float(np.mean(scores))
        elif reduction == "max":
            score = float(np.max(scores))
        else:
            score = _noisy_or(scores)

        per_group_rows.append(
            {
                "group_id": group_id,
                "n_items": int(len(group_frame)),
                "y_true": float(unique_targets[0]),
                "y_score": score,
            }
        )

    return pd.DataFrame(per_group_rows).sort_values("group_id").reset_index(drop=True)
