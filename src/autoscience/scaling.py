"""Scaling study: train on nested data fractions, measure score/time/memory.

Direct evidence that the pipeline scales smoothly, and a quantification of the
accuracy/compute trade-off of subsampling. Each (model, fraction) point is one
MLflow run tagged ``scaling=true`` — the report's scaling section collects and
plots them.
"""

from __future__ import annotations

import logging

import mlflow
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from autoscience.benchmark import load_expert_params
from autoscience.data import loaders, splits
from autoscience.data.registry import Task, get_spec
from autoscience.evaluation.metrics import primary_score
from autoscience.hpo.pipeline import build_pipeline
from autoscience.models.zoo import ModelName, is_allowed
from autoscience.profiling.profiler import profile
from autoscience.tracking.mlflow_utils import code_version, log_params_safe, setup_mlflow
from autoscience.utils.seed import set_global_seed

logger = logging.getLogger(__name__)

DEFAULT_FRACTIONS = (0.01, 0.05, 0.1, 0.33, 1.0)


def run_scaling_study(
    dataset: str,
    models: list[str],
    *,
    fractions: tuple[float, ...] = DEFAULT_FRACTIONS,
    seed: int = 42,
    full_data: bool = False,
    experiment_name: str = "autoscience",
    tracking_uri: str | None = None,
) -> pd.DataFrame:
    """Train each model on nested fractions of the train split; log each point."""
    spec = get_spec(dataset)
    set_global_seed(seed)
    ds = loaders.load_dataset(dataset, full=full_data)
    task = spec.task
    y = (
        LabelEncoder().fit_transform(ds.y)
        if task is Task.CLASSIFICATION
        else ds.y.to_numpy(dtype=np.float64)
    )
    fold = splits.get_splits(spec, pd.Series(y), seed)[0]
    x_train, y_train = ds.x.iloc[fold.train_idx], y[fold.train_idx]
    x_test, y_test = ds.x.iloc[fold.test_idx], y[fold.test_idx]

    # Nested subsets: every smaller fraction is contained in the larger one,
    # so the curve reflects data volume only, not sample luck.
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x_train))

    setup_mlflow(experiment_name, tracking_uri)
    records = []
    for model_name in models:
        model = ModelName(model_name)
        if not is_allowed(model, spec.tier):
            logger.info("Skipping %s: gated off %s tier", model, spec.tier)
            continue
        params = load_expert_params(dataset, model.value, task)
        for fraction in sorted(fractions):
            n_take = max(int(len(order) * fraction), 50)
            take = np.sort(order[:n_take])
            record = _run_point(
                dataset,
                model,
                fraction,
                x_train.iloc[take],
                y_train[take],
                x_test,
                y_test,
                task=task,
                params=params,
                seed=seed,
            )
            records.append(record)
            logger.info(
                "%s x %s @ %.0f%%: score=%.4f fit=%.1fs",
                dataset,
                model.value,
                fraction * 100,
                record["primary"],
                record["fit_seconds"],
            )
    return pd.DataFrame(records)


def _run_point(
    dataset: str,
    model: ModelName,
    fraction: float,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
    *,
    task: Task,
    params: dict[str, object],
    seed: int,
) -> dict[str, float | str]:
    run_name = f"scaling__{dataset}__{model.value}__f{fraction:g}"
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags(
            {
                "scaling": "true",
                "scaling_fraction": str(fraction),
                "dataset": dataset,
                "model": model.value,
                **code_version(),
            }
        )
        log_params_safe({"seed": seed, "n_train_rows": len(x_train), **params})

        pipeline, _ = build_pipeline(x_train, task, model, dict(params), seed=seed)
        with profile() as prof:
            pipeline.fit(x_train, y_train)
        pred = pipeline.predict(x_test)
        proba = (
            pipeline.predict_proba(x_test)
            if task is Task.CLASSIFICATION and hasattr(pipeline, "predict_proba")
            else None
        )
        score = primary_score(task, y_test, pred, proba)

        metrics = {
            "primary": score,
            "fit_seconds": prof.seconds,
            "fit_peak_mb": prof.peak_increase_mb,
            "n_train_rows": float(len(x_train)),
        }
        mlflow.log_metrics(metrics)
    return {"dataset": dataset, "model": model.value, "fraction": fraction, **metrics}
