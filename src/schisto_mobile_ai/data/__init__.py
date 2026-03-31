"""Dataset utilities for discovery, auditing, metadata indexing, and split helpers."""

from schisto_mobile_ai.data.discovery import (
    IMAGE_SUFFIXES,
    METADATA_SUFFIXES,
    classify_file,
    infer_contrast_from_text,
    infer_pair_key,
    iter_dataset_files,
)
from schisto_mobile_ai.data.manifest import (
    TABULAR_METADATA_SUFFIXES,
    first_non_empty,
    is_tabular_metadata_file,
    load_table,
    maybe_limit_rows,
    normalize_optional_string,
    validate_required_columns,
)
from schisto_mobile_ai.data.metadata_builder import (
    DatasetAuditResult,
    analyze_dataset,
    format_audit_summary,
    format_pair_examples,
    write_audit_outputs,
    write_metadata_outputs,
)
from schisto_mobile_ai.data.schisto_dataset import (
    detect_schisto_table_kind,
    make_pair_key,
    make_patient_key,
    map_schisto_contrast_code,
    parse_schisto_image_name,
)


def assign_folds(*args, **kwargs):
    """Lazily import split helpers so audit scripts do not require sklearn at import time."""
    from schisto_mobile_ai.data.splits import assign_folds as _assign_folds

    return _assign_folds(*args, **kwargs)

__all__ = [
    "DatasetAuditResult",
    "IMAGE_SUFFIXES",
    "METADATA_SUFFIXES",
    "TABULAR_METADATA_SUFFIXES",
    "analyze_dataset",
    "assign_folds",
    "classify_file",
    "first_non_empty",
    "format_audit_summary",
    "format_pair_examples",
    "infer_contrast_from_text",
    "infer_pair_key",
    "is_tabular_metadata_file",
    "iter_dataset_files",
    "load_table",
    "maybe_limit_rows",
    "normalize_optional_string",
    "detect_schisto_table_kind",
    "make_pair_key",
    "make_patient_key",
    "map_schisto_contrast_code",
    "parse_schisto_image_name",
    "validate_required_columns",
    "write_audit_outputs",
    "write_metadata_outputs",
]
