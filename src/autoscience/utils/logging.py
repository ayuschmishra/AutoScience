"""Structured logging setup shared by CLI and library code."""

from __future__ import annotations

import logging

from rich.logging import RichHandler

_FORMAT = "%(message)s"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging with rich output. Idempotent."""
    root = logging.getLogger()
    if any(isinstance(h, RichHandler) for h in root.handlers):
        root.setLevel(level)
        return
    logging.basicConfig(
        level=level,
        format=_FORMAT,
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )
    # Third-party chatter we never want at INFO.
    for noisy in ("urllib3", "matplotlib", "openml"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
