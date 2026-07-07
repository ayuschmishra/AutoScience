"""Statistical machinery: Wilcoxon, Friedman/Nemenyi, win-tie-loss."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autoscience.evaluation.stats import friedman_nemenyi, paired_wilcoxon, win_tie_loss

RNG = np.random.default_rng(4)


class TestPairedWilcoxon:
    def test_clear_improvement_is_significant(self) -> None:
        b = RNG.normal(size=20)
        a = b + 0.5  # uniformly better
        cmp = paired_wilcoxon(a, b)
        assert cmp.significant
        assert cmp.median_delta == pytest.approx(0.5)
        assert cmp.effect_size == pytest.approx(1.0)

    def test_identical_scores_are_not_significant(self) -> None:
        a = RNG.normal(size=10)
        cmp = paired_wilcoxon(a, a.copy())
        assert cmp.p_value == 1.0
        assert cmp.effect_size == 0.0

    def test_noise_is_not_significant(self) -> None:
        a = RNG.normal(size=30)
        b = a + RNG.normal(size=30) * 0.001 * np.sign(RNG.normal(size=30))
        cmp = paired_wilcoxon(a, b)
        assert not cmp.significant

    def test_too_few_observations_raises(self) -> None:
        with pytest.raises(ValueError, match=">= 3"):
            paired_wilcoxon(np.array([1.0, 2.0]), np.array([0.0, 1.0]))


class TestFriedman:
    def test_dominant_treatment_detected(self) -> None:
        base = RNG.normal(size=(12, 1))
        scores = pd.DataFrame(
            {
                "a": (base + 1.0).ravel(),  # always best
                "b": base.ravel(),
                "c": (base - 1.0).ravel(),  # always worst
            }
        )
        result = friedman_nemenyi(scores)
        assert result.significant
        assert result.avg_ranks["a"] == 1.0
        assert result.avg_ranks["c"] == 3.0
        assert result.nemenyi_p.loc["a", "c"] < 0.05

    def test_needs_minimum_shape(self) -> None:
        with pytest.raises(ValueError, match=">= 3"):
            friedman_nemenyi(pd.DataFrame({"a": [1, 2], "b": [2, 3]}))


def test_win_tie_loss() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([0.5, 2.0, 3.5, 3.0])
    assert win_tie_loss(a, b) == (2, 1, 1)
