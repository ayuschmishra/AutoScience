"""Calibration/uncertainty metrics, profiler, and the repro audit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autoscience.evaluation.audit import repro_audit
from autoscience.evaluation.calibration import (
    brier_score,
    expected_calibration_error,
    gaussian_interval,
    interval_metrics,
)
from autoscience.profiling.profiler import (
    inference_latency_ms_per_1k,
    model_size_mb,
    profile,
)
from autoscience.utils import paths

RNG = np.random.default_rng(3)


class TestCalibration:
    def test_perfectly_confident_and_correct_has_zero_ece(self) -> None:
        y = np.array([0, 1, 0, 1])
        proba = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        assert expected_calibration_error(y, proba) == pytest.approx(0.0)
        assert brier_score(y, proba) == pytest.approx(0.0)

    def test_overconfident_wrong_predictions_have_high_ece(self) -> None:
        y = np.array([0, 0, 0, 0])
        proba = np.array([[0.05, 0.95]] * 4)  # confidently wrong
        assert expected_calibration_error(y, proba) > 0.9
        assert brier_score(y, proba) > 1.5

    def test_interval_metrics(self) -> None:
        y = np.array([0.0, 1.0, 2.0, 3.0])
        m = interval_metrics(y, lower=y - 1, upper=y + 1)
        assert m["picp_80"] == 1.0
        assert m["mpiw_80_norm"] > 0

        m_miss = interval_metrics(y, lower=y + 10, upper=y + 11)
        assert m_miss["picp_80"] == 0.0

    def test_gaussian_interval_covers_at_nominal_rate(self) -> None:
        mean = np.zeros(20_000)
        std = np.ones(20_000)
        samples = RNG.normal(size=20_000)
        lo, hi = gaussian_interval(mean, std, coverage=0.8)
        assert ((samples >= lo) & (samples <= hi)).mean() == pytest.approx(0.8, abs=0.01)


class TestProfiler:
    def test_profile_measures_time_and_memory(self) -> None:
        with profile() as prof:
            _ = np.zeros((2000, 2000))  # ~32 MB
            import time

            time.sleep(0.05)
        assert prof.seconds >= 0.05
        assert prof.peak_rss_mb > 0

    def test_model_size_and_latency(self) -> None:
        from sklearn.linear_model import LogisticRegression

        x = pd.DataFrame(RNG.normal(size=(500, 5)))
        y = (x.iloc[:, 0] > 0).astype(int)
        est = LogisticRegression().fit(x, y)
        assert model_size_mb(est) > 0
        assert inference_latency_ms_per_1k(est, x) > 0


@pytest.mark.integration
def test_repro_audit_passes_for_automated_hpo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The flagship reproducibility claim: identical config -> identical metrics,
    including through a full Optuna search."""
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(paths, "SPLITS_DIR", tmp_path / "data" / "splits")

    from autoscience.hpo.budgets import PROFILES, Budget

    tiny = Budget(n_trials=2, timeout_s=120, inner_folds=2)
    monkeypatch.setitem(PROFILES, "tiny", dict.fromkeys(PROFILES["smoke"], tiny))

    report = repro_audit(
        "wine",
        "linear",
        seed=11,
        budget_profile="tiny",
        tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
    )
    assert report.deltas  # compared something real
    assert report.ok, f"non-reproducible metrics: {report.deltas}"
