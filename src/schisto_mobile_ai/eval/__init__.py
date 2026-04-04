"""Evaluation helpers for image- and group-level metrics."""

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
    "compare_metrics_payloads",
    "compute_binary_classification_metrics",
    "evaluate_patient_level",
    "format_patient_eval_summary",
]
