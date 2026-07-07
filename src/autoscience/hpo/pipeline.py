"""Assemble the full sklearn Pipeline from a flat param dict.

The pipeline is preprocess -> select -> model; every stage is fit inside CV
folds only, so there is no leakage anywhere in the search or evaluation.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from autoscience.data.registry import Task
from autoscience.features.selection import build_selector
from autoscience.models.zoo import ModelName, build_model
from autoscience.preprocessing.auto import (
    PreprocessorConfig,
    PreprocessorDecisions,
    build_preprocessor,
)


def build_pipeline(
    x: pd.DataFrame,
    task: Task,
    model: ModelName,
    params: dict[str, Any],
    *,
    seed: int,
    n_jobs: int = -1,
) -> tuple[Pipeline, PreprocessorDecisions]:
    """Build the unfitted pipeline described by ``params``.

    ``x`` is used only to *choose* preprocessing structure (column types and,
    for "auto" fields, distribution stats) — callers pass the training frame
    of the current outer fold, never test data.
    """
    prep_config = PreprocessorConfig(
        numeric_imputer=params.get("prep.numeric_imputer", "auto"),
        scaler=params.get("prep.scaler", "auto"),
        cat_encoder=params.get("prep.cat_encoder", "auto"),
    )
    preprocessor, decisions = build_preprocessor(x, task, prep_config)

    selector = build_selector(
        params.get("select.strategy", "none"),
        task,
        percentile=int(params.get("select.percentile", 50)),
        n_features_fraction=float(params.get("select.n_features_fraction", 0.5)),
        seed=seed,
    )

    model_params = {
        key.removeprefix("model."): value
        for key, value in params.items()
        if key.startswith("model.")
    }
    estimator = build_model(model, task, seed=seed, n_jobs=n_jobs, params=model_params)

    pipeline = Pipeline([("prep", preprocessor), ("select", selector), ("model", estimator)])
    return pipeline, decisions
