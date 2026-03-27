#!/usr/bin/env python3
"""Stub CLI for patient-level or group-level evaluation."""

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
            task_name="eval_patient_level",
            description="Aggregate predictions to the evaluation group level and report metrics.",
            default_output_subdir="results/evaluations",
            extra_todos=[
                "TODO: define the real grouping column used for patient-level evaluation.",
                "TODO: choose aggregation rules that match the study design.",
            ],
        )
    )

