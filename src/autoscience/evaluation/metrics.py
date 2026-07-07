"""Metric computation shared by HPO objectives and final evaluation.

``primary_score`` is the single scalar optimized by HPO (higher is better).
``metric_panel`` is the full per-fold panel logged to MLflow (expanded with
calibration/uncertainty in the evaluation layer).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from autoscience.data.registry import Task

PRIMARY_METRIC = {Task.CLASSIFICATION: "roc_auc", Task.REGRESSION: "neg_rmse"}


def _roc_auc(y_true: np.ndarray, proba: np.ndarray) -> float:
    if proba.shape[1] == 2:
        return float(roc_auc_score(y_true, proba[:, 1]))
    return float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro"))


def primary_score(
    task: Task,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray | None = None,
) -> float:
    """Scalar objective, higher is better for both tasks."""
    if task is Task.CLASSIFICATION:
        assert proba is not None, "classification primary score needs predict_proba"
        return _roc_auc(y_true, proba)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return -rmse


def metric_panel(
    task: Task,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray | None = None,
) -> dict[str, float]:
    """Full point-prediction metric panel for one evaluation fold."""
    if task is Task.CLASSIFICATION:
        panel = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        }
        if proba is not None:
            panel["roc_auc"] = _roc_auc(y_true, proba)
            panel["log_loss"] = float(log_loss(y_true, proba))
        return panel
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "rmse": rmse,
        "neg_rmse": -rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
