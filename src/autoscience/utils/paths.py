"""Canonical project paths, overridable via environment for cloud runs."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("AUTOSCIENCE_ROOT", Path.cwd()))
DATA_DIR = Path(os.environ.get("AUTOSCIENCE_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DATA_DIR = DATA_DIR / "raw"
SPLITS_DIR = DATA_DIR / "splits"
REPORTS_DIR = PROJECT_ROOT / "reports"
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", (PROJECT_ROOT / "mlruns").as_uri())
