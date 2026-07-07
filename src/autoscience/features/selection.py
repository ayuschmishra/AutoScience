"""Feature selection strategies as sklearn-compatible pipeline steps.

Each strategy is a factory returning an (unfitted) transformer, so selection
is fit inside CV folds like everything else. The strategy itself is an
HPO-searchable categorical; expensive wrapper methods (RFE) are only offered
on the small tier (gating happens in the search-space layer).
"""

from __future__ import annotations

from enum import StrEnum

from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.feature_selection import (
    RFE,
    SelectFromModel,
    SelectPercentile,
    VarianceThreshold,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.pipeline import Pipeline

from autoscience.data.registry import Task


class SelectionStrategy(StrEnum):
    NONE = "none"
    FILTER = "filter"  # variance threshold + mutual-information percentile
    L1 = "l1"  # embedded: L1-regularized linear model
    TREE = "tree"  # embedded: extra-trees importance
    RFE = "rfe"  # wrapper: recursive feature elimination (small tier only)


def build_selector(
    strategy: SelectionStrategy | str,
    task: Task,
    *,
    percentile: int = 50,
    n_features_fraction: float = 0.5,
    seed: int = 0,
) -> object:
    """Return an unfitted selector transformer (or ``"passthrough"``).

    Args:
        strategy: Which selection family to use.
        task: Classification or regression (picks the scoring/estimator).
        percentile: For FILTER — percentage of features kept by mutual info.
        n_features_fraction: For RFE — fraction of features to keep.
        seed: Random state for embedded estimators.
    """
    strategy = SelectionStrategy(strategy)
    classification = task is Task.CLASSIFICATION

    if strategy is SelectionStrategy.NONE:
        return "passthrough"

    if strategy is SelectionStrategy.FILTER:
        mi = mutual_info_classif if classification else mutual_info_regression
        return Pipeline(
            [
                ("variance", VarianceThreshold()),
                ("mi", SelectPercentile(mi, percentile=percentile)),
            ]
        )

    if strategy is SelectionStrategy.L1:
        estimator = (
            # saga: the only L1-capable solver that also handles multiclass.
            # l1_ratio=1.0 is pure L1 (the `penalty` param is deprecated in 1.9).
            LogisticRegression(l1_ratio=1.0, solver="saga", C=0.1, max_iter=2000, random_state=seed)
            if classification
            else Lasso(alpha=0.01, max_iter=5000, random_state=seed)
        )
        return SelectFromModel(estimator)

    if strategy is SelectionStrategy.TREE:
        estimator = (
            ExtraTreesClassifier(n_estimators=50, random_state=seed, n_jobs=-1)
            if classification
            else ExtraTreesRegressor(n_estimators=50, random_state=seed, n_jobs=-1)
        )
        return SelectFromModel(estimator, threshold="median")

    # RFE: expensive wrapper method, gated to the small tier upstream.
    estimator = (
        ExtraTreesClassifier(n_estimators=25, random_state=seed, n_jobs=-1)
        if classification
        else ExtraTreesRegressor(n_estimators=25, random_state=seed, n_jobs=-1)
    )
    return RFE(estimator, n_features_to_select=n_features_fraction, step=0.2)
