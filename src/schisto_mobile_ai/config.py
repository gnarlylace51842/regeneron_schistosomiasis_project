"""Configuration helpers for YAML or JSON experiment files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML or JSON configuration file into a dictionary."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    suffix = config_path.suffix.lower()
    if suffix not in {".yaml", ".yml", ".json"}:
        raise ValueError(
            f"Unsupported config format for {config_path}. Use .yaml, .yml, or .json."
        )

    with config_path.open("r", encoding="utf-8") as handle:
        if suffix in {".yaml", ".yml"}:
            config = yaml.safe_load(handle) or {}
        else:
            config = json.load(handle)

    if not isinstance(config, dict):
        raise ValueError("Top-level config value must be a mapping/dictionary.")

    return config


def save_config_snapshot(config: dict[str, Any], path: str | Path) -> None:
    """Save a normalized JSON snapshot of the currently resolved config."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")

