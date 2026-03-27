"""Consistent logging setup for scripts and modules."""

from __future__ import annotations

import logging


def configure_logging(quiet: bool = False) -> logging.Logger:
    """Set up basic logging and return a project logger."""
    level = logging.WARNING if quiet else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger("schisto_mobile_ai")

