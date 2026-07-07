"""Auto-preprocessor: correct auto decisions, leakage safety, all-dataset build."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autoscience.data.registry import Task
from autoscience.preprocessing.auto import (
    MAX_CARDINALITY_FOR_ONEHOT,
    PreprocessorConfig,
    build_preprocessor,
)

RNG = np.random.default_rng(0)


def _mixed_frame(n: int = 300) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "num_clean": RNG.normal(size=n),
            "num_skewed": np.exp(RNG.normal(size=n) * 3),
            "num_missing": np.where(RNG.random(n) < 0.2, np.nan, RNG.normal(size=n)),
            "cat_low": pd.Categorical(RNG.choice(list("abc"), size=n)),
            "cat_high": pd.Categorical(RNG.choice([f"c{i}" for i in range(40)], size=n)),
        }
    )


def test_auto_decisions_reflect_data_traits() -> None:
    x = _mixed_frame()
    _, decisions = build_preprocessor(x, Task.CLASSIFICATION)
    assert decisions.n_numeric == 3
    assert decisions.n_categorical == 2
    assert decisions.missing_fraction > 0
    assert decisions.scaler == "quantile"  # exp-skewed column
    assert decisions.max_cardinality == 40
    assert decisions.cat_encoder == "target"  # cardinality 40 > threshold
    assert decisions.reasons  # every auto choice explains itself


def test_low_cardinality_gets_onehot() -> None:
    x = _mixed_frame().drop(columns=["cat_high", "num_skewed"])
    _, decisions = build_preprocessor(x, Task.CLASSIFICATION)
    assert decisions.max_cardinality <= MAX_CARDINALITY_FOR_ONEHOT
    assert decisions.cat_encoder == "onehot"


def test_pinned_config_overrides_auto() -> None:
    x = _mixed_frame()
    _, decisions = build_preprocessor(
        x, Task.REGRESSION, PreprocessorConfig(scaler="robust", cat_encoder="ordinal")
    )
    assert decisions.scaler == "robust"
    assert decisions.cat_encoder == "ordinal"


def test_fit_transform_produces_finite_matrix() -> None:
    x = _mixed_frame()
    y = pd.Series(RNG.integers(0, 2, size=len(x)))
    ct, _ = build_preprocessor(x, Task.CLASSIFICATION)
    out = ct.fit_transform(x, y)
    assert out.shape[0] == len(x)
    assert np.isfinite(np.asarray(out, dtype=float)).all()


def test_no_leakage_fold_statistics_differ() -> None:
    """Fitting on different folds must give different learned statistics."""
    x = _mixed_frame(400)
    y = pd.Series(RNG.integers(0, 2, size=400))
    ct1, _ = build_preprocessor(x, Task.CLASSIFICATION)
    ct2, _ = build_preprocessor(x, Task.CLASSIFICATION)
    ct1.fit(x.iloc[:200], y.iloc[:200])
    ct2.fit(x.iloc[200:], y.iloc[200:])
    m1 = ct1.named_transformers_["numeric"].named_steps["impute"].statistics_
    m2 = ct2.named_transformers_["numeric"].named_steps["impute"].statistics_
    assert not np.allclose(m1, m2)


def test_numeric_only_frame_has_no_categorical_branch() -> None:
    x = pd.DataFrame({"a": RNG.normal(size=50), "b": RNG.normal(size=50)})
    ct, decisions = build_preprocessor(x, Task.REGRESSION)
    assert decisions.n_categorical == 0
    assert [name for name, *_ in ct.transformers] == ["numeric"]


@pytest.mark.parametrize("dataset", ["wine", "breast_cancer", "diabetes"])
def test_builds_for_cached_sklearn_datasets(dataset: str, tmp_path) -> None:
    from autoscience.data import loaders

    ds = loaders.load_dataset(dataset, data_dir=tmp_path)
    ct, _ = build_preprocessor(ds.x, ds.spec.task)
    out = ct.fit_transform(ds.x, ds.y)
    assert out.shape[0] == ds.n_rows
