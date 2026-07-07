"""Benchmark orchestration: config loading, expert params, resume, isolation."""

from __future__ import annotations

from pathlib import Path

import pytest

from autoscience.benchmark import BenchmarkConfig, load_expert_params, run_benchmark
from autoscience.data.registry import Task
from autoscience.hpo.budgets import PROFILES, Budget
from autoscience.utils import paths


def test_config_load(tmp_path: Path) -> None:
    cfg_file = tmp_path / "b.yaml"
    cfg_file.write_text(
        "name: t\nseeds: [1, 2]\ndatasets: [wine]\nmodels: [linear]\n"
        "modes: [automated]\nbudget_profile: smoke\n"
    )
    cfg = BenchmarkConfig.load(cfg_file)
    assert cfg.seeds == [1, 2]
    assert cfg.modes == ["automated"]
    assert not cfg.full_data


def test_expert_params_task_and_dataset_resolution() -> None:
    cls = load_expert_params("adult", "xgboost", Task.CLASSIFICATION)
    assert cls["model.n_estimators"] == 500
    # Per-dataset override wins over the model default.
    higgs = load_expert_params("higgs", "xgboost", Task.CLASSIFICATION)
    assert higgs["model.max_depth"] == 8
    # Regression linear uses alpha, not C.
    reg = load_expert_params("diabetes", "linear", Task.REGRESSION)
    assert "model.alpha" in reg and "model.C" not in reg


@pytest.mark.integration
def test_sweep_runs_skips_and_resumes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(paths, "SPLITS_DIR", tmp_path / "data" / "splits")
    tiny = Budget(n_trials=2, timeout_s=120, inner_folds=2)
    monkeypatch.setitem(PROFILES, "tiny", dict.fromkeys(PROFILES["smoke"], tiny))
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"

    cfg = BenchmarkConfig(
        name="t",
        seeds=[5],
        datasets=["wine", "covertype"],  # covertype x svm must be tier-gated
        models=["linear", "svm"],
        modes=["automated", "baseline_expert"],
        budget_profile="tiny",
    )
    # Keep it fast: only wine actually runs (covertype would download 100MB+).
    cfg.datasets = ["wine"]
    first = run_benchmark(cfg, tracking_uri=uri)
    assert first.count("completed") == 4  # 2 models x 2 modes
    assert first.count("failed") == 0

    # Second invocation resumes: everything is skipped.
    second = run_benchmark(cfg, tracking_uri=uri)
    assert second.count("skipped_existing") == 4
    assert second.count("completed") == 0
