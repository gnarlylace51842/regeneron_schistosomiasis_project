#!/usr/bin/env python3
"""Stub CLI for dataset auditing."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from schisto_mobile_ai.utils.script_base import run_placeholder_task


if __name__ == "__main__":
    raise SystemExit(
        run_placeholder_task(
            task_name="audit_dataset",
            description="Audit raw data files and metadata before any modeling.",
            default_output_subdir="results/audits",
            extra_todos=[
                "TODO: define what one unit of analysis means for this dataset.",
                "TODO: check file integrity, missing values, duplicates, and label coverage.",
            ],
        )
    )

