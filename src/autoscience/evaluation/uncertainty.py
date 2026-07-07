"""Regression predictive-uncertainty evaluation for supported models.

- ``hist_gb``: two extra quantile-loss models (q=0.1, 0.9) that reuse the
  tuned pipeline's preprocessing/selection and mirrored hyperparameters.
- ``torch_mlp``: MC-dropout mean/std converted to a Gaussian 80% interval.

Other regressors get no interval metrics (logged as absent, not zero).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone

from autoscience.evaluation.calibration import gaussian_interval, interval_metrics
from autoscience.models.zoo import ModelName, build_quantile_regressor

logger = logging.getLogger(__name__)

QUANTILES = (0.1, 0.9)  # central 80% interval


def regression_interval_metrics(
    fitted_pipeline: Any,
    model: ModelName,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    *,
    best_params: dict[str, Any],
    seed: int,
) -> dict[str, float]:
    """80% prediction-interval PICP/MPIW where the model family supports it."""
    if model is ModelName.HIST_GB:
        model_params = {
            k.removeprefix("model."): v for k, v in best_params.items() if k.startswith("model.")
        }
        bounds = []
        for q in QUANTILES:
            # Same preprocessing/selection, quantile-loss head.
            pipe = clone(fitted_pipeline)
            pipe.set_params(model=build_quantile_regressor(q, seed=seed, params=model_params))
            pipe.fit(x_train, y_train)
            bounds.append(pipe.predict(x_test))
        lower, upper = np.minimum(*bounds), np.maximum(*bounds)
        return interval_metrics(y_test, lower, upper)

    if model is ModelName.TORCH_MLP:
        transformed = fitted_pipeline[:-1].transform(x_test)
        unc = fitted_pipeline[-1].predict_uncertainty(np.asarray(transformed))
        lower, upper = gaussian_interval(unc["mean"], unc["std"], coverage=0.8)
        return interval_metrics(y_test, lower, upper)

    return {}
