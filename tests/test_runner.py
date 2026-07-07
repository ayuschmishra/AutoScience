"""End-to-end runner integration: HPO run, baseline run, determinism."""

from __future__ import annotations

from pathlib import Path

import pytest

from autoscience.hpo.budgets import PROFILES, Budget
from autoscience.hpo.runner import run_experiment
from autoscience.utils import paths


@pytest.fixture()
def isolated_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(paths, "SPLITS_DIR", tmp_path / "data" / "splits")
    return tmp_path


def _tracking_uri(tmp_path: Path) -> str:
    return f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"


@pytest.fixture()
def tiny_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    tiny = Budget(n_trials=3, timeout_s=120, inner_folds=2)
    monkeypatch.setitem(PROFILES, "tiny", dict.fromkeys(PROFILES["smoke"], tiny))


@pytest.mark.integration
def test_automated_run_end_to_end(isolated_dirs: Path, tiny_budget: None) -> None:
    result = run_experiment(
        "wine",
        "linear",
        seed=42,
        budget_profile="tiny",
        tracking_uri=_tracking_uri(isolated_dirs),
    )
    assert len(result.fold_results) == 5
    assert result.aggregated["roc_auc_mean"] > 0.9  # wine is easy
    assert result.aggregated["hpo_seconds_total"] > 0
    assert result.run_id
    # Every fold searched and found pipeline-level params too.
    for fold in result.fold_results:
        assert any(k.startswith("prep.") for k in fold.best_params)
        assert fold.n_trials == 3


@pytest.mark.integration
def test_baseline_mode_is_deterministic(isolated_dirs: Path) -> None:
    kwargs = dict(
        seed=7,
        budget_profile="smoke",
        mode="baseline",
        fixed_params={"model.C": 1.0},
        tracking_uri=_tracking_uri(isolated_dirs),
    )
    a = run_experiment("wine", "linear", **kwargs)
    b = run_experiment("wine", "linear", **kwargs)
    # Bit-identical reproduction of every model-quality metric (wall-clock
    # timings are the only metrics allowed to differ).
    quality = {k: v for k, v in a.aggregated.items() if "seconds" not in k}
    assert quality == {k: v for k, v in b.aggregated.items() if "seconds" not in k}
    assert a.aggregated["hpo_seconds_total"] == 0.0


@pytest.mark.integration
def test_tier_gating_enforced(isolated_dirs: Path) -> None:
    with pytest.raises(ValueError, match="gated"):
        run_experiment("covertype", "svm", tracking_uri=_tracking_uri(isolated_dirs))
