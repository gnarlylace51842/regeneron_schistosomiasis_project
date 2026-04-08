"""Compact diagnostic helpers for prediction collapse and threshold analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from schisto_mobile_ai.eval.metrics import compute_binary_classification_metrics


PROBABILITY_BANDS = (0.001, 0.005, 0.01, 0.02, 0.05)


def _format_band(band: float) -> str:
    return str(band).replace(".", "p")


def load_prediction_frame(
    path: str | Path,
    *,
    probability_column: str = "probability",
    target_column: str = "target",
) -> pd.DataFrame:
    """Load a prediction CSV and coerce probability/target columns to numeric values."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Prediction CSV does not exist: {csv_path}")

    frame = pd.read_csv(csv_path)
    required = {probability_column, target_column}
    missing = required - set(frame.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Prediction CSV is missing required columns: {missing_text}")

    output = frame.copy()
    output[probability_column] = pd.to_numeric(output[probability_column], errors="coerce")
    output[target_column] = pd.to_numeric(output[target_column], errors="coerce")
    output = output.dropna(subset=[probability_column, target_column]).reset_index(drop=True)
    if output.empty:
        raise ValueError("Prediction CSV does not contain any valid numeric target/probability rows.")
    return output


def _series_stats(values: pd.Series) -> dict[str, float | int | None]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p05": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p95": None,
            "max": None,
            "value_range": None,
        }

    return {
        "n": int(len(numeric)),
        "mean": float(numeric.mean()),
        "std": float(numeric.std(ddof=0)),
        "min": float(numeric.min()),
        "p05": float(numeric.quantile(0.05)),
        "p25": float(numeric.quantile(0.25)),
        "median": float(numeric.quantile(0.50)),
        "p75": float(numeric.quantile(0.75)),
        "p95": float(numeric.quantile(0.95)),
        "max": float(numeric.max()),
        "value_range": float(numeric.max() - numeric.min()),
    }


def _band_fractions(values: pd.Series, *, center: float) -> dict[str, float]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {
            f"fraction_within_{_format_band(band)}_of_{str(center).replace('.', 'p')}": 0.0
            for band in PROBABILITY_BANDS
        }

    fractions = {}
    for band in PROBABILITY_BANDS:
        key = f"fraction_within_{_format_band(band)}_of_{str(center).replace('.', 'p')}"
        fractions[key] = float((numeric.sub(center).abs() <= band).mean())
    return fractions


def detect_effectively_constant(
    values: pd.Series,
    *,
    center: float = 0.5,
    std_threshold: float = 0.01,
    range_threshold: float = 0.05,
    near_center_band: float = 0.01,
    near_center_fraction_threshold: float = 0.95,
) -> dict[str, Any]:
    """Detect whether prediction scores are nearly constant or collapsed near one value."""
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    stats = _series_stats(numeric)
    if numeric.empty:
        return {
            "is_effectively_constant": True,
            "reasons": ["No valid prediction probabilities were available."],
        }

    near_center_fraction = float((numeric.sub(center).abs() <= near_center_band).mean())
    reasons: list[str] = []
    if stats["std"] is not None and stats["std"] <= std_threshold:
        reasons.append(f"std <= {std_threshold:.4f}")
    if stats["value_range"] is not None and stats["value_range"] <= range_threshold:
        reasons.append(f"range <= {range_threshold:.4f}")
    if near_center_fraction >= near_center_fraction_threshold:
        reasons.append(
            f"fraction within +/-{near_center_band:.4f} of {center:.3f} >= {near_center_fraction_threshold:.2f}"
        )

    return {
        "is_effectively_constant": bool(reasons),
        "reasons": reasons,
        "std_threshold": float(std_threshold),
        "range_threshold": float(range_threshold),
        "near_center_band": float(near_center_band),
        "near_center_fraction_threshold": float(near_center_fraction_threshold),
        "observed_near_center_fraction": near_center_fraction,
    }


def summarize_probability_frame(
    frame: pd.DataFrame,
    *,
    probability_column: str = "probability",
    target_column: str = "target",
) -> dict[str, Any]:
    """Summarize probability distributions overall and by class."""
    overall_stats = _series_stats(frame[probability_column])
    overall_stats.update(_band_fractions(frame[probability_column], center=0.5))
    overall_stats["fraction_predicted_positive_at_0p5"] = float(
        (frame[probability_column].astype(float) >= 0.5).mean()
    )

    by_class: dict[str, dict[str, float | int | None]] = {}
    for class_value, class_frame in frame.groupby(target_column, sort=True):
        class_key = str(int(class_value)) if float(class_value).is_integer() else str(class_value)
        class_stats = _series_stats(class_frame[probability_column])
        class_stats.update(_band_fractions(class_frame[probability_column], center=0.5))
        class_stats["fraction_predicted_positive_at_0p5"] = float(
            (class_frame[probability_column].astype(float) >= 0.5).mean()
        )
        by_class[class_key] = class_stats

    constant_check = detect_effectively_constant(frame[probability_column])
    return {
        "overall": overall_stats,
        "by_class": by_class,
        "constant_check": constant_check,
    }


def plot_probability_histograms(
    frame: pd.DataFrame,
    *,
    probability_column: str = "probability",
    target_column: str = "target",
    histogram_path: str | Path,
    density_path: str | Path,
) -> None:
    """Save histogram and density-style line plots for probabilities by class."""
    histogram_output = Path(histogram_path)
    density_output = Path(density_path)
    histogram_output.parent.mkdir(parents=True, exist_ok=True)
    density_output.parent.mkdir(parents=True, exist_ok=True)

    bins = np.linspace(0.0, 1.0, 41)
    class_groups = list(frame.groupby(target_column, sort=True))
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for index, (class_value, class_frame) in enumerate(class_groups):
        ax.hist(
            class_frame[probability_column].astype(float).to_numpy(),
            bins=bins,
            alpha=0.45,
            density=False,
            color=colors[index % len(colors)],
            label=f"class={int(class_value) if float(class_value).is_integer() else class_value}",
        )
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_title("Predicted Probability Histogram by Class")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(histogram_output, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    centers = (bins[:-1] + bins[1:]) / 2.0
    for index, (class_value, class_frame) in enumerate(class_groups):
        counts, _ = np.histogram(
            class_frame[probability_column].astype(float).to_numpy(),
            bins=bins,
            density=True,
        )
        ax.plot(
            centers,
            counts,
            linewidth=2,
            color=colors[index % len(colors)],
            label=f"class={int(class_value) if float(class_value).is_integer() else class_value}",
        )
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_title("Predicted Probability Density by Class")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(density_output, dpi=160)
    plt.close(fig)


def build_threshold_sweep(
    frame: pd.DataFrame,
    *,
    probability_column: str = "patient_probability",
    target_column: str = "target",
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compute threshold-dependent metrics for a prediction frame."""
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 201)

    labels = frame[target_column].astype(int).to_numpy()
    scores = frame[probability_column].astype(float).to_numpy()
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        metrics = compute_binary_classification_metrics(labels, scores, threshold=float(threshold))
        sensitivity = metrics["sensitivity"]
        specificity = metrics["specificity"]
        balanced_accuracy = None
        youden_j = None
        if sensitivity is not None and specificity is not None:
            balanced_accuracy = float((sensitivity + specificity) / 2.0)
            youden_j = float(sensitivity + specificity - 1.0)
        rows.append(
            {
                "threshold": float(threshold),
                "accuracy": metrics["accuracy"],
                "sensitivity": sensitivity,
                "specificity": specificity,
                "precision": metrics["precision"],
                "f1": metrics["f1"],
                "balanced_accuracy": balanced_accuracy,
                "youden_j": youden_j,
                "positive_prediction_rate": float((scores >= threshold).mean()),
            }
        )
    return pd.DataFrame(rows)


def select_best_threshold(sweep_frame: pd.DataFrame, metric_column: str) -> dict[str, Any]:
    """Select the best threshold for one metric, preferring thresholds near 0.5 on ties."""
    valid = sweep_frame.dropna(subset=[metric_column]).copy()
    if valid.empty:
        return {
            "metric": metric_column,
            "threshold": None,
            "value": None,
        }

    best_value = valid[metric_column].max()
    tied = valid[valid[metric_column] == best_value].copy()
    tied["distance_from_0p5"] = (tied["threshold"] - 0.5).abs()
    best_row = tied.sort_values(["distance_from_0p5", "threshold"]).iloc[0]
    return {
        "metric": metric_column,
        "threshold": float(best_row["threshold"]),
        "value": float(best_row[metric_column]),
    }


def plot_threshold_sweep(
    sweep_frame: pd.DataFrame,
    *,
    output_path: str | Path,
) -> None:
    """Save a compact plot of threshold-dependent diagnostic metrics."""
    figure_path = Path(output_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for metric_name, color in [
        ("balanced_accuracy", "#1f77b4"),
        ("youden_j", "#2ca02c"),
        ("f1", "#d62728"),
    ]:
        series = pd.to_numeric(sweep_frame[metric_name], errors="coerce")
        ax.plot(sweep_frame["threshold"], series, label=metric_name, linewidth=2, color=color)

    ax.axvline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_title("Threshold Sweep")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric value")
    ax.set_xlim(0.0, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
