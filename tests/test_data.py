"""Data layer tests: registry integrity, cache round-trip, subsetting, splits.

Network-dependent OpenML/UCI downloads are exercised by
`autoscience data validate`, not unit tests; here we use sklearn built-ins.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autoscience.data import loaders, registry, splits
from autoscience.data.registry import (
    REGISTRY,
    DatasetSpec,
    SizeTier,
    SklearnSource,
    Task,
    get_spec,
)


def test_registry_integrity() -> None:
    assert len(REGISTRY) >= 14
    for spec in REGISTRY.values():
        assert spec.expected_rows > 0 and spec.expected_features > 0
        assert spec.description
    assert get_spec("higgs").tier is SizeTier.LARGE
    assert get_spec("wine").tier is SizeTier.SMALL
    assert get_spec("adult").tier is SizeTier.MEDIUM


def test_get_spec_unknown_name() -> None:
    with pytest.raises(KeyError, match="Unknown dataset"):
        get_spec("does_not_exist")


def test_download_and_load_round_trip(tmp_path: Path) -> None:
    cache = loaders.download("wine", data_dir=tmp_path)
    assert cache.exists()
    assert loaders.verify_checksum("wine", data_dir=tmp_path)

    ds = loaders.load_dataset("wine", data_dir=tmp_path)
    assert ds.n_rows == 178
    assert ds.x.shape[1] == 13
    assert not ds.is_subset
    assert isinstance(ds.y.dtype, pd.CategoricalDtype)
    # Downcast happened.
    assert all(dt == np.float32 for dt in ds.x.dtypes)

    # Second load hits the cache and is byte-identical.
    ds2 = loaders.load_dataset("wine", data_dir=tmp_path)
    pd.testing.assert_frame_equal(ds.x, ds2.x)


def test_local_subset_is_deterministic_and_stratified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = DatasetSpec(
        name="wine_subset_test",
        task=Task.CLASSIFICATION,
        source=SklearnSource("load_wine"),
        expected_rows=178,
        expected_features=13,
        description="test",
        local_subset_rows=90,
    )
    monkeypatch.setitem(registry.REGISTRY, spec.name, spec)

    ds1 = loaders.load_dataset(spec.name, data_dir=tmp_path)
    ds2 = loaders.load_dataset(spec.name, data_dir=tmp_path)
    assert ds1.is_subset and ds1.n_rows == pytest.approx(90, abs=2)
    pd.testing.assert_frame_equal(ds1.x, ds2.x)

    full = loaders.load_dataset(spec.name, full=True, data_dir=tmp_path)
    assert full.n_rows == 178 and not full.is_subset

    # Class balance preserved within a few percent.
    orig = full.y.value_counts(normalize=True)
    sub = ds1.y.value_counts(normalize=True)
    assert max(abs(orig - sub)) < 0.05


def test_iter_batches_streams_everything(tmp_path: Path) -> None:
    loaders.download("wine", data_dir=tmp_path)
    total = 0
    for x, y in loaders.iter_batches("wine", batch_size=64, data_dir=tmp_path):
        assert len(x) == len(y) <= 64
        total += len(x)
    assert total == 178


class TestSplits:
    def _spec(self, tier_rows: int) -> DatasetSpec:
        return DatasetSpec(
            name="split_test",
            task=Task.CLASSIFICATION,
            source=SklearnSource("load_wine"),
            expected_rows=tier_rows,
            expected_features=5,
            description="test",
        )

    def test_kfold_disjoint_and_complete(self, tmp_path: Path) -> None:
        y = pd.Series(np.repeat([0, 1], 100))
        folds = splits.get_splits(self._spec(200), y, seed=7, splits_dir=tmp_path)
        assert len(folds) == splits.N_FOLDS
        all_test = np.concatenate([f.test_idx for f in folds])
        assert sorted(all_test) == list(range(200))
        for f in folds:
            assert set(f.train_idx).isdisjoint(f.test_idx)
            assert len(f.train_idx) + len(f.test_idx) == 200

    def test_persisted_and_reloaded_identical(self, tmp_path: Path) -> None:
        y = pd.Series(np.repeat([0, 1], 100))
        first = splits.get_splits(self._spec(200), y, seed=7, splits_dir=tmp_path)
        second = splits.get_splits(self._spec(200), y, seed=7, splits_dir=tmp_path)
        for a, b in zip(first, second, strict=True):
            np.testing.assert_array_equal(a.train_idx, b.train_idx)
            np.testing.assert_array_equal(a.test_idx, b.test_idx)

    def test_different_seeds_differ(self, tmp_path: Path) -> None:
        y = pd.Series(np.repeat([0, 1], 100))
        a = splits.get_splits(self._spec(200), y, seed=1, splits_dir=tmp_path)
        b = splits.get_splits(self._spec(200), y, seed=2, splits_dir=tmp_path)
        assert not np.array_equal(a[0].test_idx, b[0].test_idx)

    def test_large_tier_uses_single_stratified_holdout(self, tmp_path: Path) -> None:
        y = pd.Series(np.repeat([0, 1], 60_000))
        folds = splits.get_splits(self._spec(120_000), y, seed=3, splits_dir=tmp_path)
        assert len(folds) == 1
        fold = folds[0]
        assert len(fold.test_idx) == pytest.approx(24_000, abs=2)
        test_balance = y.iloc[fold.test_idx].mean()
        assert abs(test_balance - 0.5) < 0.01
