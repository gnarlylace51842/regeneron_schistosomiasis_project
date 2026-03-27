"""Dataset utilities that stay schema-agnostic until the real audit is complete."""

from schisto_mobile_ai.data.manifest import load_table, maybe_limit_rows, validate_required_columns
from schisto_mobile_ai.data.splits import assign_folds

__all__ = [
    "assign_folds",
    "load_table",
    "maybe_limit_rows",
    "validate_required_columns",
]

