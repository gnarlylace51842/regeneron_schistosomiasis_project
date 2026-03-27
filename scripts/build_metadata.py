#!/usr/bin/env python3
"""Stub CLI for building a canonical metadata table."""

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
            task_name="build_metadata",
            description="Build a canonical metadata table from raw dataset files.",
            default_output_subdir="results/metadata",
            extra_todos=[
                "TODO: define the real metadata schema after dataset auditing.",
                "TODO: parse file paths, labels, groups, and contrast identifiers from the real source data.",
            ],
        )
    )

