#!/usr/bin/env python3
"""Stub CLI for image-quality analysis."""

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
            task_name="compute_quality_metrics",
            description="Compute image-quality metrics for exploratory data analysis.",
            default_output_subdir="results/quality_metrics",
            extra_todos=[
                "TODO: define which quality metrics matter for mobile microscopy images.",
                "TODO: implement dataset-specific loading once file organization is known.",
            ],
        )
    )

