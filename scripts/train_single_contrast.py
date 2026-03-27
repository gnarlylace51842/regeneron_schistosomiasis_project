#!/usr/bin/env python3
"""Stub CLI for single-contrast model training."""

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
            task_name="train_single_contrast",
            description="Train a baseline single-contrast classifier.",
            default_output_subdir="runs/experiments/train_single_contrast",
            extra_todos=[
                "TODO: implement dataset objects, training loop, validation logic, and checkpointing.",
                "TODO: define how a single contrast is represented in the canonical metadata.",
            ],
        )
    )

