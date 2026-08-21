"""Logging configuration shared across the platform."""

from __future__ import annotations

import logging
import sys

from app.config import settings

_CONFIGURED = False


def configure_logging() -> None:
    """Configure the root logger once for the whole application."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    root = logging.getLogger("app")
    root.setLevel(settings.LOG_LEVEL)
    root.addHandler(handler)
    root.propagate = False

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger under the ``app`` namespace."""
    configure_logging()
    return logging.getLogger(f"app.{name}")
