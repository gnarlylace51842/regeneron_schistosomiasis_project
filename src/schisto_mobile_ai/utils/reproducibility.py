"""Helpers for deterministic behavior and device selection."""

from __future__ import annotations

import random

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - used only if torch is unavailable
    torch = None


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch when available."""
    random.seed(seed)
    np.random.seed(seed)

    if torch is None:
        return

    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def resolve_device(requested_device: str = "auto") -> str:
    """Return 'mps' when available or fall back to 'cpu'."""
    allowed = {"auto", "cpu", "mps"}
    if requested_device not in allowed:
        allowed_text = ", ".join(sorted(allowed))
        raise ValueError(f"requested_device must be one of: {allowed_text}")

    if requested_device == "cpu":
        return "cpu"

    if torch is None:
        if requested_device == "mps":
            raise RuntimeError("Torch is not installed, so MPS cannot be used.")
        return "cpu"

    mps_available = torch.backends.mps.is_available()
    if requested_device == "mps":
        if not mps_available:
            raise RuntimeError("MPS was requested but is not available on this machine.")
        return "mps"

    return "mps" if mps_available else "cpu"

