"""Evaluation helpers for image- and group-level metrics."""

from schisto_mobile_ai.eval.diagnostics import (
    build_threshold_sweep,
    detect_effectively_constant,
    load_prediction_frame,
    plot_probability_histograms,
    plot_threshold_sweep,
    select_best_threshold,
    summarize_probability_frame,
)
from schisto_mobile_ai.eval.metrics import (
    aggregate_group_scores,
    compute_binary_classification_metrics,
)
from schisto_mobile_ai.eval.patient_level import (
    compare_metrics_payloads,
    evaluate_patient_level,
    format_patient_eval_summary,
)

__all__ = [
    "aggregate_group_scores",
    "build_threshold_sweep",
    "compare_metrics_payloads",
    "compute_binary_classification_metrics",
    "detect_effectively_constant",
    "evaluate_patient_level",
    "format_patient_eval_summary",
    "load_prediction_frame",
    "plot_probability_histograms",
    "plot_threshold_sweep",
    "select_best_threshold",
    "summarize_probability_frame",
]
