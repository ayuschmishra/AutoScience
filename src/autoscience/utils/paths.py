"""Canonical project paths, overridable via environment for cloud runs."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("AUTOSCIENCE_ROOT", Path.cwd()))
DATA_DIR = Path(os.environ.get("AUTOSCIENCE_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DATA_DIR = DATA_DIR / "raw"
SPLITS_DIR = DATA_DIR / "splits"
REPORTS_DIR = PROJECT_ROOT / "reports"
# MLflow >= 3.14 deprecates the filesystem store; SQLite is the recommended
# local backend and is a single portable file (easy to sync back from cloud).
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
)
