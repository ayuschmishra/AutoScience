"""Calibration and predictive-uncertainty metrics.

Classification: top-label expected calibration error (ECE) and multiclass
Brier score. Regression: prediction-interval coverage probability (PICP) and
normalized mean interval width (MPIW) for 80% intervals produced by quantile
models or MC-dropout.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 15) -> float:
    """Top-label ECE: |confidence - accuracy| averaged over confidence bins."""
    confidence = proba.max(axis=1)
    predicted = proba.argmax(axis=1)
    correct = (predicted == np.asarray(y_true)).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in pairwise(bins):
        mask = (confidence > lo) & (confidence <= hi)
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def brier_score(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Multiclass Brier score: mean squared error against one-hot targets."""
    y_true = np.asarray(y_true, dtype=int)
    one_hot = np.zeros_like(proba)
    one_hot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))


def interval_metrics(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> dict[str, float]:
    """PICP and normalized MPIW for a prediction interval.

    MPIW is normalized by the target's standard deviation so it is comparable
    across datasets with different target scales.
    """
    y_true = np.asarray(y_true, dtype=float)
    covered = (y_true >= lower) & (y_true <= upper)
    y_std = float(y_true.std()) or 1.0
    return {
        "picp_80": float(covered.mean()),
        "mpiw_80_norm": float(np.mean(upper - lower) / y_std),
    }


def gaussian_interval(
    mean: np.ndarray, std: np.ndarray, coverage: float = 0.8
) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric interval from a Gaussian predictive distribution (MC dropout)."""
    from scipy.stats import norm

    z = float(norm.ppf(0.5 + coverage / 2))
    return mean - z * std, mean + z * std
