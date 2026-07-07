"""Determinism utilities are the foundation of every reproducibility claim."""

from __future__ import annotations

import numpy as np

from autoscience.utils.seed import set_global_seed, spawn_seeds


def test_numpy_reproducible() -> None:
    # Deliberately exercises the legacy *global* RNG: that is what
    # set_global_seed promises to control.
    set_global_seed(123)
    a = np.random.rand(16)  # noqa: NPY002
    set_global_seed(123)
    b = np.random.rand(16)  # noqa: NPY002
    np.testing.assert_array_equal(a, b)


def test_torch_reproducible() -> None:
    import torch

    set_global_seed(123)
    a = torch.rand(16)
    set_global_seed(123)
    b = torch.rand(16)
    assert torch.equal(a, b)


def test_spawn_seeds_deterministic_and_distinct() -> None:
    s1 = spawn_seeds(42, 10)
    s2 = spawn_seeds(42, 10)
    assert s1 == s2
    assert len(set(s1)) == 10
    assert spawn_seeds(43, 10) != s1
