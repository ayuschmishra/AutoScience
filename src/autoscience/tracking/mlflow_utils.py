"""MLflow helpers: tracking setup, code-version capture, safe param logging."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

import mlflow

from autoscience.utils import paths

DEFAULT_EXPERIMENT = "autoscience"


def setup_mlflow(experiment: str = DEFAULT_EXPERIMENT, tracking_uri: str | None = None) -> None:
    mlflow.set_tracking_uri(tracking_uri or paths.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment)


def code_version() -> dict[str, str]:
    """Git SHA + uv.lock hash: enough to reconstruct the exact code+env."""
    version: dict[str, str] = {}
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=paths.PROJECT_ROOT,
        ).stdout.strip()
        version["git_sha"] = sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        version["git_sha"] = "unknown"

    lock = Path(paths.PROJECT_ROOT) / "uv.lock"
    if lock.exists():
        version["uv_lock_sha256"] = hashlib.sha256(lock.read_bytes()).hexdigest()[:16]
    return version


def log_params_safe(params: dict[str, Any]) -> None:
    """Log params with MLflow's 500-char value limit respected."""
    mlflow.log_params({k: str(v)[:500] for k, v in params.items()})
