"""Unified model factory with size-tier gating.

Every model is exposed behind the sklearn estimator API so a single
evaluation harness serves them all. ``available_models`` gates models that
don't scale (SVM/KNN kernel matrices, deep trees on tiny CPU budgets) to the
tiers where they are practical — gated models are simply absent from the
benchmark grid for bigger datasets, never silently subsampled.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    Lasso,
    LogisticRegression,
    Ridge,
    SGDClassifier,
    SGDRegressor,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

from autoscience.data.registry import SizeTier, Task
from autoscience.models.torch_mlp import TorchMLPClassifier, TorchMLPRegressor


class ModelName(StrEnum):
    LINEAR = "linear"  # logistic / ridge
    LASSO = "lasso"  # L1 linear (regression only)
    SGD_LINEAR = "sgd_linear"  # out-of-core capable linear baseline
    RANDOM_FOREST = "random_forest"
    EXTRA_TREES = "extra_trees"
    HIST_GB = "hist_gb"
    SVM = "svm"
    KNN = "knn"
    XGBOOST = "xgboost"
    TORCH_MLP = "torch_mlp"


_TIER_ORDER = [SizeTier.SMALL, SizeTier.MEDIUM, SizeTier.LARGE]

# Highest tier each model is allowed to run on.
MAX_TIER: dict[ModelName, SizeTier] = {
    ModelName.LINEAR: SizeTier.LARGE,
    ModelName.LASSO: SizeTier.LARGE,
    ModelName.SGD_LINEAR: SizeTier.LARGE,
    ModelName.RANDOM_FOREST: SizeTier.LARGE,
    ModelName.EXTRA_TREES: SizeTier.LARGE,
    ModelName.HIST_GB: SizeTier.LARGE,
    ModelName.SVM: SizeTier.SMALL,  # O(n^2) kernel matrix
    ModelName.KNN: SizeTier.MEDIUM,  # distance matrix at predict time
    ModelName.XGBOOST: SizeTier.LARGE,
    ModelName.TORCH_MLP: SizeTier.LARGE,
}

# Models that support incremental partial_fit (out-of-core path).
PARTIAL_FIT_CAPABLE = frozenset({ModelName.SGD_LINEAR})


def is_allowed(name: ModelName | str, tier: SizeTier) -> bool:
    return _TIER_ORDER.index(MAX_TIER[ModelName(name)]) >= _TIER_ORDER.index(tier)


def available_models(task: Task, tier: SizeTier) -> list[ModelName]:
    models = [m for m in ModelName if is_allowed(m, tier)]
    if task is Task.CLASSIFICATION:
        models = [m for m in models if m is not ModelName.LASSO]
    return models


def build_model(
    name: ModelName | str,
    task: Task,
    *,
    seed: int = 0,
    n_jobs: int = -1,
    params: dict[str, Any] | None = None,
) -> Any:
    """Instantiate an unfitted estimator with sensible defaults + overrides."""
    name = ModelName(name)
    params = params or {}
    classification = task is Task.CLASSIFICATION

    if name is ModelName.LINEAR:
        est = (
            LogisticRegression(max_iter=2000, random_state=seed)
            if classification
            else Ridge(random_state=seed)
        )
    elif name is ModelName.LASSO:
        if classification:
            raise ValueError("lasso is regression-only; use linear for classification")
        est = Lasso(max_iter=5000, random_state=seed)
    elif name is ModelName.SGD_LINEAR:
        est = (
            SGDClassifier(loss="log_loss", random_state=seed)  # log_loss -> predict_proba
            if classification
            else SGDRegressor(random_state=seed)
        )
    elif name is ModelName.RANDOM_FOREST:
        est = (
            RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=n_jobs)
            if classification
            else RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=n_jobs)
        )
    elif name is ModelName.EXTRA_TREES:
        est = (
            ExtraTreesClassifier(n_estimators=300, random_state=seed, n_jobs=n_jobs)
            if classification
            else ExtraTreesRegressor(n_estimators=300, random_state=seed, n_jobs=n_jobs)
        )
    elif name is ModelName.HIST_GB:
        est = (
            HistGradientBoostingClassifier(random_state=seed)
            if classification
            else HistGradientBoostingRegressor(random_state=seed)
        )
    elif name is ModelName.SVM:
        est = SVC(probability=True, random_state=seed) if classification else SVR()
    elif name is ModelName.KNN:
        est = (
            KNeighborsClassifier(n_jobs=n_jobs)
            if classification
            else KNeighborsRegressor(n_jobs=n_jobs)
        )
    elif name is ModelName.XGBOOST:
        common = {
            "tree_method": "hist",
            "random_state": seed,
            "n_jobs": n_jobs,
            "n_estimators": 300,
            "verbosity": 0,
        }
        est = XGBClassifier(**common) if classification else XGBRegressor(**common)
    else:  # TORCH_MLP
        est = (
            TorchMLPClassifier(random_state=seed)
            if classification
            else TorchMLPRegressor(random_state=seed)
        )

    if params:
        est.set_params(**params)
    return est


def build_quantile_regressor(
    quantile: float, *, seed: int = 0, params: dict[str, Any] | None = None
) -> HistGradientBoostingRegressor:
    """Quantile-loss gradient boosting for prediction intervals (Phase 5)."""
    est = HistGradientBoostingRegressor(loss="quantile", quantile=quantile, random_state=seed)
    if params:
        # Interval models mirror the tuned point model's hyperparameters.
        valid = {k: v for k, v in params.items() if k in est.get_params()}
        est.set_params(**valid)
    return est
