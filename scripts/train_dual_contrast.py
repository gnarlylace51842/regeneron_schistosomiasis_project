#!/usr/bin/env python3
"""Stub CLI for dual-contrast model training."""

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
            task_name="train_dual_contrast",
            description="Train a dual-contrast classifier with lightweight fusion.",
            default_output_subdir="runs/experiments/train_dual_contrast",
            extra_todos=[
                "TODO: define how paired contrasts are matched and validated.",
                "TODO: implement paired batching, fusion training, and checkpointing.",
            ],
        )
    )

