"""Central repository paths used across scripts and modules."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"
DATA_DIR = REPO_ROOT / "data"
RUNS_DIR = REPO_ROOT / "runs"
RESULTS_DIR = REPO_ROOT / "results"


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not already exist and return it."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory

