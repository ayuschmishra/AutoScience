"""Feature selection strategies fit inside pipelines and actually select."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from autoscience.data.registry import Task
from autoscience.features.selection import SelectionStrategy, build_selector

RNG = np.random.default_rng(1)


def _data(task: Task) -> tuple[pd.DataFrame, pd.Series]:
    n = 200
    informative = RNG.normal(size=(n, 3))
    noise = RNG.normal(size=(n, 7)) * 0.1
    x = pd.DataFrame(np.hstack([informative, noise]), columns=[f"f{i}" for i in range(10)])
    signal = informative.sum(axis=1)
    if task is Task.CLASSIFICATION:
        y = pd.Series((signal > 0).astype(int))
    else:
        y = pd.Series(signal + RNG.normal(size=n) * 0.1)
    return x, y


@pytest.mark.parametrize("task", [Task.CLASSIFICATION, Task.REGRESSION])
@pytest.mark.parametrize("strategy", list(SelectionStrategy))
def test_every_strategy_fits_and_transforms(strategy: SelectionStrategy, task: Task) -> None:
    x, y = _data(task)
    selector = build_selector(strategy, task, seed=0)
    if strategy is SelectionStrategy.NONE:
        assert selector == "passthrough"
        return
    pipe = Pipeline([("select", selector)])
    out = pipe.fit_transform(x, y)
    assert out.shape[0] == len(x)
    assert 1 <= out.shape[1] <= x.shape[1]


def test_filter_keeps_informative_features() -> None:
    x, y = _data(Task.CLASSIFICATION)
    selector = build_selector(SelectionStrategy.FILTER, Task.CLASSIFICATION, percentile=30)
    pipe = Pipeline([("select", selector)])
    pipe.fit(x, y)
    kept = pipe.named_steps["select"].named_steps["mi"].get_support(indices=True)
    # The three informative features are f0..f2.
    assert set(kept) & {0, 1, 2}


def test_selection_is_seeded() -> None:
    x, y = _data(Task.CLASSIFICATION)
    a = build_selector(SelectionStrategy.TREE, Task.CLASSIFICATION, seed=5)
    b = build_selector(SelectionStrategy.TREE, Task.CLASSIFICATION, seed=5)
    mask_a = a.fit(x, y).get_support()
    mask_b = b.fit(x, y).get_support()
    np.testing.assert_array_equal(mask_a, mask_b)
