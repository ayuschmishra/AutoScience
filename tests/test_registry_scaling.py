"""Model registry round-trip and scaling study integration."""

from __future__ import annotations

from pathlib import Path

import pytest

from autoscience.hpo.budgets import PROFILES, Budget
from autoscience.utils import paths


@pytest.fixture()
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(paths, "SPLITS_DIR", tmp_path / "data" / "splits")
    tiny = Budget(n_trials=2, timeout_s=120, inner_folds=2)
    monkeypatch.setitem(PROFILES, "tiny", dict.fromkeys(PROFILES["smoke"], tiny))
    return f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"


@pytest.mark.integration
def test_register_and_predict_round_trip(env: str) -> None:
    from autoscience.data import loaders
    from autoscience.hpo.runner import run_experiment
    from autoscience.tracking.registry import load_registered, register_best

    run_experiment("wine", "linear", seed=3, budget_profile="tiny", tracking_uri=env)
    name = register_best("wine", tracking_uri=env)
    assert name == "autoscience-wine"

    model = load_registered("wine", tracking_uri=env)
    ds = loaders.load_dataset("wine")
    pred = model.predict(ds.x.head(10))
    assert len(pred) == 10
    proba = model.predict_proba(ds.x.head(10))
    assert proba.shape == (10, 3)


@pytest.mark.integration
def test_scaling_study_curves(env: str) -> None:
    from autoscience.scaling import run_scaling_study

    df = run_scaling_study("phoneme", ["hist_gb"], fractions=(0.1, 1.0), seed=3, tracking_uri=env)
    assert len(df) == 2
    assert set(df["fraction"]) == {0.1, 1.0}
    # More data should not make training faster in a meaningful way, and the
    # points must be finite/positive.
    assert (df["fit_seconds"] > 0).all()
    assert (df["n_train_rows"].diff().dropna() > 0).all()

    # Collected back from MLflow for the report.
    from autoscience.reporting.report import collect_scaling_runs

    collected = collect_scaling_runs(tracking_uri=env)
    assert len(collected) == 2
    assert set(collected.columns) >= {"dataset", "model", "fraction", "primary", "fit_seconds"}


@pytest.mark.integration
def test_register_without_runs_raises(env: str) -> None:
    from autoscience.tracking.registry import register_best

    with pytest.raises(LookupError, match="No finished automated runs"):
        register_best("diabetes", tracking_uri=env)
