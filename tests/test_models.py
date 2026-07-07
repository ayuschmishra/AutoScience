"""Model zoo: every model trains/predicts via the common interface; tier
gating works; the torch MLP is bit-reproducible."""

from __future__ import annotations

import numpy as np
import pytest

from autoscience.data.registry import SizeTier, Task
from autoscience.models.torch_mlp import TorchMLPClassifier, TorchMLPRegressor
from autoscience.models.zoo import (
    ModelName,
    available_models,
    build_model,
    build_quantile_regressor,
    is_allowed,
)

RNG = np.random.default_rng(2)
N = 240

X_CLS = RNG.normal(size=(N, 8)).astype(np.float32)
Y_CLS = (X_CLS[:, :3].sum(axis=1) > 0).astype(np.int64)
X_REG = X_CLS.copy()
Y_REG = (X_REG[:, :3].sum(axis=1) + RNG.normal(size=N) * 0.1).astype(np.float64)

# Small but sufficient to learn the easy synthetic problem (~60 gradient steps).
FAST_MLP = {"max_epochs": 60, "hidden_dim": 32, "patience": 20, "lr": 3e-3}


def _fast_params(name: ModelName) -> dict[str, object]:
    if name is ModelName.TORCH_MLP:
        return dict(FAST_MLP)
    if name in (ModelName.RANDOM_FOREST, ModelName.EXTRA_TREES, ModelName.XGBOOST):
        return {"n_estimators": 20}
    return {}


@pytest.mark.parametrize("name", available_models(Task.CLASSIFICATION, SizeTier.SMALL))
def test_classifiers_fit_predict_proba(name: ModelName) -> None:
    model = build_model(name, Task.CLASSIFICATION, seed=0, params=_fast_params(name))
    model.fit(X_CLS, Y_CLS)
    pred = model.predict(X_CLS)
    proba = model.predict_proba(X_CLS)
    assert pred.shape == (N,)
    assert proba.shape == (N, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert (pred == Y_CLS).mean() > 0.7  # learns an easy problem


@pytest.mark.parametrize("name", available_models(Task.REGRESSION, SizeTier.SMALL))
def test_regressors_fit_predict(name: ModelName) -> None:
    model = build_model(name, Task.REGRESSION, seed=0, params=_fast_params(name))
    model.fit(X_REG, Y_REG)
    pred = model.predict(X_REG)
    assert pred.shape == (N,)
    assert np.isfinite(pred).all()


def test_tier_gating() -> None:
    assert is_allowed(ModelName.SVM, SizeTier.SMALL)
    assert not is_allowed(ModelName.SVM, SizeTier.MEDIUM)
    assert not is_allowed(ModelName.KNN, SizeTier.LARGE)
    assert is_allowed(ModelName.XGBOOST, SizeTier.LARGE)

    large_cls = available_models(Task.CLASSIFICATION, SizeTier.LARGE)
    assert ModelName.SVM not in large_cls
    assert ModelName.TORCH_MLP in large_cls
    # lasso is regression-only.
    assert ModelName.LASSO not in available_models(Task.CLASSIFICATION, SizeTier.SMALL)
    assert ModelName.LASSO in available_models(Task.REGRESSION, SizeTier.SMALL)


def test_lasso_rejected_for_classification() -> None:
    with pytest.raises(ValueError, match="regression-only"):
        build_model(ModelName.LASSO, Task.CLASSIFICATION)


def test_torch_mlp_deterministic() -> None:
    """Same seed -> identical predictions; different seed -> different weights."""
    a = TorchMLPClassifier(random_state=7, **FAST_MLP).fit(X_CLS, Y_CLS)
    b = TorchMLPClassifier(random_state=7, **FAST_MLP).fit(X_CLS, Y_CLS)
    np.testing.assert_array_equal(a.predict_proba(X_CLS), b.predict_proba(X_CLS))

    c = TorchMLPClassifier(random_state=8, **FAST_MLP).fit(X_CLS, Y_CLS)
    assert not np.array_equal(a.predict_proba(X_CLS), c.predict_proba(X_CLS))


def test_torch_mlp_regressor_mc_dropout_uncertainty() -> None:
    model = TorchMLPRegressor(random_state=0, dropout=0.3, **FAST_MLP).fit(X_REG, Y_REG)
    unc = model.predict_uncertainty(X_REG[:32], n_samples=10)
    assert unc["mean"].shape == (32,)
    assert unc["std"].shape == (32,)
    assert (unc["std"] > 0).all()  # dropout active -> nonzero spread
    # Uncertainty is itself reproducible.
    unc2 = model.predict_uncertainty(X_REG[:32], n_samples=10)
    np.testing.assert_allclose(unc["mean"], unc2["mean"])


def test_quantile_regressor_produces_ordered_intervals() -> None:
    lo = build_quantile_regressor(0.1, seed=0).fit(X_REG, Y_REG).predict(X_REG)
    hi = build_quantile_regressor(0.9, seed=0).fit(X_REG, Y_REG).predict(X_REG)
    assert (hi >= lo).mean() > 0.95
