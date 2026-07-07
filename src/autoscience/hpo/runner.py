"""Experiment runner: nested-CV HPO for one (dataset, model, seed).

Protocol
--------
- Outer folds come from the persisted split store (identical for every model
  and baseline).
- ``mode="automated"``: per outer fold, an Optuna TPE study optimizes the
  whole pipeline on inner CV (K-fold on small/medium; single validation split
  with successive data-fraction pruning on large). The best pipeline is refit
  on the outer-train split and scored on the outer-test split.
- ``mode="baseline"``: ``fixed_params`` is used directly — same folds, same
  refit-and-score path, no search.

Everything (params, per-fold metrics, HPO cost, trial history, preprocessing
decisions, code version) is logged to one MLflow run.
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

from autoscience.data import loaders, splits
from autoscience.data.registry import SizeTier, Task, get_spec
from autoscience.evaluation.calibration import brier_score, expected_calibration_error
from autoscience.evaluation.metrics import PRIMARY_METRIC, metric_panel, primary_score
from autoscience.evaluation.uncertainty import regression_interval_metrics
from autoscience.hpo.budgets import Budget, get_budget
from autoscience.hpo.pipeline import build_pipeline
from autoscience.hpo.spaces import suggest_params
from autoscience.models.zoo import ModelName, is_allowed
from autoscience.profiling.profiler import inference_latency_ms_per_1k, model_size_mb, profile
from autoscience.tracking.mlflow_utils import code_version, log_params_safe, setup_mlflow
from autoscience.utils.seed import set_global_seed

logger = logging.getLogger(__name__)

# Data fractions for successive-halving style pruning on the large tier.
LARGE_TIER_FRACTIONS = (0.25, 1.0)
INNER_SEED_OFFSET = 1000  # decorrelates inner CV from outer splits


@dataclass
class FoldResult:
    fold: int
    best_params: dict[str, Any]
    metrics: dict[str, float]
    hpo_seconds: float
    fit_seconds: float
    n_trials: int


@dataclass
class ExperimentResult:
    dataset: str
    model: str
    seed: int
    mode: str
    fold_results: list[FoldResult] = field(default_factory=list)
    aggregated: dict[str, float] = field(default_factory=dict)
    run_id: str = ""

    def aggregate(self) -> None:
        keys = self.fold_results[0].metrics
        self.aggregated = {}
        for key in keys:
            values = [f.metrics[key] for f in self.fold_results]
            self.aggregated[f"{key}_mean"] = float(np.mean(values))
            self.aggregated[f"{key}_std"] = float(np.std(values))
        self.aggregated["hpo_seconds_total"] = float(sum(f.hpo_seconds for f in self.fold_results))
        self.aggregated["fit_seconds_mean"] = float(
            np.mean([f.fit_seconds for f in self.fold_results])
        )


def _predict_with_proba(
    pipeline: Any, x: pd.DataFrame, task: Task
) -> tuple[np.ndarray, np.ndarray | None]:
    pred = pipeline.predict(x)
    proba = None
    if task is Task.CLASSIFICATION and hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(x)
    return pred, proba


def _inner_splitter(task: Task, n_folds: int, seed: int) -> StratifiedKFold | KFold:
    cls = StratifiedKFold if task is Task.CLASSIFICATION else KFold
    return cls(n_splits=n_folds, shuffle=True, random_state=seed + INNER_SEED_OFFSET)


def _make_objective(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    *,
    task: Task,
    tier: SizeTier,
    model: ModelName,
    seed: int,
    budget: Budget,
    has_categorical: bool,
) -> Any:
    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, model, task, tier, has_categorical=has_categorical)
        pipeline, _ = build_pipeline(x_train, task, model, params, seed=seed)

        if budget.inner_folds >= 2:
            splitter = _inner_splitter(task, budget.inner_folds, seed)
            scores: list[float] = []
            for step, (tr, va) in enumerate(splitter.split(x_train, y_train)):
                est = clone(pipeline)
                est.fit(x_train.iloc[tr], y_train[tr])
                pred, proba = _predict_with_proba(est, x_train.iloc[va], task)
                scores.append(primary_score(task, y_train[va], pred, proba))
                trial.report(float(np.mean(scores)), step)
                if trial.should_prune():
                    raise optuna.TrialPruned
            return float(np.mean(scores))

        # Large tier: one validation split, successive data fractions so bad
        # configs die after seeing 25% of the data.
        stratify = y_train if task is Task.CLASSIFICATION else None
        tr_idx, va_idx = train_test_split(
            np.arange(len(x_train)),
            test_size=0.2,
            random_state=seed + INNER_SEED_OFFSET,
            stratify=stratify,
        )
        rng = np.random.default_rng(seed + INNER_SEED_OFFSET)
        score = float("-inf")
        for step, fraction in enumerate(LARGE_TIER_FRACTIONS):
            if fraction < 1.0:
                take = rng.choice(tr_idx, size=int(len(tr_idx) * fraction), replace=False)
            else:
                take = tr_idx
            est = clone(pipeline)
            est.fit(x_train.iloc[take], y_train[take])
            pred, proba = _predict_with_proba(est, x_train.iloc[va_idx], task)
            score = primary_score(task, y_train[va_idx], pred, proba)
            trial.report(score, step)
            if trial.should_prune():
                raise optuna.TrialPruned
        return score

    return objective


def _optimize_fold(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    *,
    task: Task,
    tier: SizeTier,
    model: ModelName,
    seed: int,
    budget: Budget,
    has_categorical: bool,
) -> tuple[dict[str, Any], optuna.Study]:
    pruner: optuna.pruners.BasePruner
    if budget.inner_folds >= 2:
        pruner = optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=1)
    else:
        pruner = optuna.pruners.SuccessiveHalvingPruner()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=pruner,
    )
    objective = _make_objective(
        x_train,
        y_train,
        task=task,
        tier=tier,
        model=model,
        seed=seed,
        budget=budget,
        has_categorical=has_categorical,
    )
    study.optimize(
        objective,
        n_trials=budget.n_trials,
        timeout=budget.timeout_s,
        show_progress_bar=False,
    )
    return dict(study.best_params), study


def run_experiment(
    dataset: str,
    model: ModelName | str,
    *,
    seed: int = 42,
    budget_profile: str = "smoke",
    full_data: bool = False,
    mode: str = "automated",
    fixed_params: dict[str, Any] | None = None,
    experiment_name: str = "autoscience",
    tracking_uri: str | None = None,
    extra_tags: dict[str, str] | None = None,
) -> ExperimentResult:
    """Run one (dataset, model, seed) experiment and log it to MLflow."""
    model = ModelName(model)
    spec = get_spec(dataset)
    if not is_allowed(model, spec.tier):
        raise ValueError(f"{model} is gated off the {spec.tier} tier ({dataset})")
    if mode not in ("automated", "baseline"):
        raise ValueError(f"mode must be 'automated' or 'baseline', got {mode!r}")
    if mode == "baseline" and fixed_params is None:
        fixed_params = {}

    set_global_seed(seed)
    ds = loaders.load_dataset(dataset, full=full_data)
    task = spec.task
    budget = get_budget(budget_profile, spec.tier)

    if task is Task.CLASSIFICATION:
        y = LabelEncoder().fit_transform(ds.y)
    else:
        y = ds.y.to_numpy(dtype=np.float64)
    has_categorical = ds.x.select_dtypes(exclude="number").shape[1] > 0

    folds = splits.get_splits(spec, pd.Series(y), seed)
    result = ExperimentResult(dataset=dataset, model=model.value, seed=seed, mode=mode)

    setup_mlflow(experiment_name, tracking_uri)
    run_name = f"{dataset}__{model.value}__seed{seed}__{mode}"
    with mlflow.start_run(run_name=run_name) as run:
        result.run_id = run.info.run_id
        mlflow.set_tags(
            {
                "dataset": dataset,
                "model": model.value,
                "mode": mode,
                "tier": spec.tier.value,
                "task": task.value,
                **code_version(),
                **(extra_tags or {}),
            }
        )
        log_params_safe(
            {
                "seed": seed,
                "budget_profile": budget_profile,
                "n_trials_budget": budget.n_trials,
                "timeout_s": budget.timeout_s,
                "inner_folds": budget.inner_folds,
                "data_variant": "full" if not ds.is_subset else "local_subset",
                "n_rows": ds.n_rows,
                "n_features": ds.x.shape[1],
                "primary_metric": PRIMARY_METRIC[task],
            }
        )

        with tempfile.TemporaryDirectory() as artifact_dir:
            for fold_i, fold in enumerate(folds):
                fold_result = _run_fold(
                    fold_i,
                    fold,
                    ds.x,
                    y,
                    task=task,
                    tier=spec.tier,
                    model=model,
                    seed=seed,
                    budget=budget,
                    has_categorical=has_categorical,
                    mode=mode,
                    fixed_params=fixed_params,
                    artifact_dir=Path(artifact_dir),
                )
                result.fold_results.append(fold_result)
                for name, value in fold_result.metrics.items():
                    mlflow.log_metric(name, value, step=fold_i)
                mlflow.log_metric("hpo_seconds", fold_result.hpo_seconds, step=fold_i)
                mlflow.log_metric("fit_seconds", fold_result.fit_seconds, step=fold_i)
            mlflow.log_artifacts(artifact_dir)

        result.aggregate()
        mlflow.log_metrics(result.aggregated)

    logger.info(
        "%s: %s",
        run_name,
        {k: round(v, 4) for k, v in result.aggregated.items() if k.endswith("_mean")},
    )
    return result


def _run_fold(
    fold_i: int,
    fold: splits.Fold,
    x: pd.DataFrame,
    y: np.ndarray,
    *,
    task: Task,
    tier: SizeTier,
    model: ModelName,
    seed: int,
    budget: Budget,
    has_categorical: bool,
    mode: str,
    fixed_params: dict[str, Any] | None,
    artifact_dir: Path,
) -> FoldResult:
    x_train, y_train = x.iloc[fold.train_idx], y[fold.train_idx]
    x_test, y_test = x.iloc[fold.test_idx], y[fold.test_idx]

    if mode == "automated":
        hpo_start = time.perf_counter()
        best_params, study = _optimize_fold(
            x_train,
            y_train,
            task=task,
            tier=tier,
            model=model,
            seed=seed,
            budget=budget,
            has_categorical=has_categorical,
        )
        hpo_seconds = time.perf_counter() - hpo_start
        n_trials = len(study.trials)
        study.trials_dataframe().to_csv(artifact_dir / f"trials_fold{fold_i}.csv", index=False)
    else:
        best_params = dict(fixed_params or {})
        hpo_seconds = 0.0
        n_trials = 0

    pipeline, decisions = build_pipeline(x_train, task, model, best_params, seed=seed)
    with profile() as fit_profile:
        pipeline.fit(x_train, y_train)
    fit_seconds = fit_profile.seconds

    pred, proba = _predict_with_proba(pipeline, x_test, task)
    metrics = metric_panel(task, y_test, pred, proba)

    # Calibration / uncertainty panel.
    if task is Task.CLASSIFICATION and proba is not None:
        metrics["ece"] = expected_calibration_error(y_test, proba)
        metrics["brier"] = brier_score(y_test, proba)
    elif task is Task.REGRESSION:
        metrics.update(
            regression_interval_metrics(
                pipeline,
                model,
                x_train,
                y_train,
                x_test,
                y_test,
                best_params=best_params,
                seed=seed,
            )
        )

    # Efficiency panel.
    metrics["fit_peak_mb"] = fit_profile.peak_increase_mb
    metrics["model_size_mb"] = model_size_mb(pipeline)
    metrics["latency_ms_per_1k"] = inference_latency_ms_per_1k(pipeline, x_test)

    (artifact_dir / f"best_params_fold{fold_i}.json").write_text(
        json.dumps(best_params, indent=2, default=str)
    )
    if fold_i == 0:
        (artifact_dir / "preprocessor_decisions.json").write_text(
            json.dumps(decisions.as_params(), indent=2, default=str)
        )

    return FoldResult(
        fold=fold_i,
        best_params=best_params,
        metrics=metrics,
        hpo_seconds=hpo_seconds,
        fit_seconds=fit_seconds,
        n_trials=n_trials,
    )
