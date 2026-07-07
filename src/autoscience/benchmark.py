"""Benchmark orchestration: sweep (datasets x models x seeds x modes).

The sweep is declared in a YAML experiment config, resumes by skipping
(dataset, model, seed, mode) combinations that already have a FINISHED MLflow
run with the same budget profile, and never lets one failed combination kill
the sweep — failures are recorded and reported at the end.

Modes:
- ``automated``       : full nested-CV HPO pipeline.
- ``baseline_default``: library-default hyperparameters, auto preprocessing.
- ``baseline_expert`` : hand-tuned, literature-typical hyperparameters from
  ``experiments/baselines/expert.yaml`` — the "sensible expert" a scientist
  without an AutoML system would be.
All modes are evaluated on the same persisted outer folds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlflow
from omegaconf import DictConfig, OmegaConf

from autoscience.data.registry import Task, get_spec
from autoscience.hpo.runner import run_experiment
from autoscience.models.zoo import ModelName, available_models, is_allowed
from autoscience.tracking.mlflow_utils import setup_mlflow
from autoscience.utils import paths

logger = logging.getLogger(__name__)

MODES = ("automated", "baseline_default", "baseline_expert")
EXPERT_BASELINES_PATH = Path("experiments") / "baselines" / "expert.yaml"


@dataclass
class BenchmarkConfig:
    name: str
    seeds: list[int]
    datasets: list[str]
    models: list[str]
    modes: list[str]
    budget_profile: str = "smoke"
    full_data: bool = False

    @staticmethod
    def load(path: Path) -> BenchmarkConfig:
        raw = OmegaConf.load(path)
        assert isinstance(raw, DictConfig), f"{path} must be a mapping"
        return BenchmarkConfig(
            name=str(raw.name),
            seeds=[int(s) for s in raw.seeds],
            datasets=[str(d) for d in raw.datasets],
            models=[str(m) for m in raw.models],
            modes=[str(m) for m in raw.get("modes", list(MODES))],
            budget_profile=str(raw.get("budget_profile", "smoke")),
            full_data=bool(raw.get("full_data", False)),
        )


@dataclass
class RunOutcome:
    dataset: str
    model: str
    seed: int
    mode: str
    status: str  # completed | skipped_existing | skipped_gated | failed
    detail: str = ""


@dataclass
class BenchmarkSummary:
    outcomes: list[RunOutcome] = field(default_factory=list)

    def count(self, status: str) -> int:
        return sum(1 for o in self.outcomes if o.status == status)


def load_expert_params(
    dataset: str, model: str, task: Task, base_dir: Path | None = None
) -> dict[str, Any]:
    """Expert baseline params: per-task model defaults + per-dataset override."""
    path = (base_dir or paths.PROJECT_ROOT) / EXPERT_BASELINES_PATH
    if not path.exists():
        raise FileNotFoundError(f"Expert baseline config missing: {path}")
    raw = OmegaConf.load(path)
    params: dict[str, Any] = {}
    for node in (f"{task.value}.{model}", f"datasets.{dataset}.{model}"):
        selected = OmegaConf.select(raw, node)
        if selected is not None:
            params.update(OmegaConf.to_container(selected))  # type: ignore[arg-type]
    return params


def _existing_finished_run(
    dataset: str, model: str, seed: int, mode: str, budget_profile: str
) -> bool:
    runs = mlflow.search_runs(
        filter_string=(
            f"tags.dataset = '{dataset}' and tags.model = '{model}' "
            f"and tags.benchmark_mode = '{mode}' and params.seed = '{seed}' "
            f"and params.budget_profile = '{budget_profile}' "
            "and attributes.status = 'FINISHED'"
        ),
        max_results=1,
    )
    return len(runs) > 0


def run_benchmark(
    config: BenchmarkConfig,
    *,
    tracking_uri: str | None = None,
    experiment_name: str = "autoscience",
) -> BenchmarkSummary:
    """Execute the sweep; resumable and failure-isolated."""
    setup_mlflow(experiment_name, tracking_uri)
    summary = BenchmarkSummary()

    for dataset in config.datasets:
        spec = get_spec(dataset)
        allowed = set(available_models(spec.task, spec.tier))
        for model in config.models:
            model_name = ModelName(model)
            if model_name not in allowed:
                reason = "tier-gated" if not is_allowed(model_name, spec.tier) else "task-mismatch"
                summary.outcomes.append(
                    RunOutcome(dataset, model, -1, "-", "skipped_gated", reason)
                )
                continue
            for seed in config.seeds:
                for mode in config.modes:
                    outcome = _run_one(
                        dataset,
                        model_name,
                        seed,
                        mode,
                        config,
                        tracking_uri=tracking_uri,
                        experiment_name=experiment_name,
                    )
                    summary.outcomes.append(outcome)
                    logger.info(
                        "[%s] %s x %s seed=%d %s: %s",
                        outcome.status,
                        dataset,
                        model,
                        seed,
                        mode,
                        outcome.detail or "ok",
                    )
    return summary


def _run_one(
    dataset: str,
    model: ModelName,
    seed: int,
    mode: str,
    config: BenchmarkConfig,
    *,
    tracking_uri: str | None,
    experiment_name: str,
) -> RunOutcome:
    if mode not in MODES:
        return RunOutcome(dataset, model.value, seed, mode, "failed", f"unknown mode {mode!r}")
    if _existing_finished_run(dataset, model.value, seed, mode, config.budget_profile):
        return RunOutcome(dataset, model.value, seed, mode, "skipped_existing")

    runner_mode = "automated" if mode == "automated" else "baseline"
    fixed_params: dict[str, Any] | None = None
    if mode == "baseline_default":
        fixed_params = {}
    elif mode == "baseline_expert":
        fixed_params = load_expert_params(dataset, model.value, get_spec(dataset).task)

    try:
        run_experiment(
            dataset,
            model,
            seed=seed,
            budget_profile=config.budget_profile,
            full_data=config.full_data,
            mode=runner_mode,
            fixed_params=fixed_params,
            experiment_name=experiment_name,
            tracking_uri=tracking_uri,
            extra_tags={"benchmark_mode": mode, "benchmark_name": config.name},
        )
        return RunOutcome(dataset, model.value, seed, mode, "completed")
    except Exception as exc:  # isolate failures, keep sweeping
        logger.exception("Run failed: %s x %s seed=%d %s", dataset, model, seed, mode)
        return RunOutcome(dataset, model.value, seed, mode, "failed", str(exc)[:300])
