"""Optuna search spaces over the ENTIRE pipeline.

Preprocessing and feature-selection choices are searched alongside model
hyperparameters — the automated pipeline optimizes its own preprocessing,
which is exactly the "automated decision" the study evaluates. Params are
flat dicts with ``prep.`` / ``select.`` / ``model.`` prefixes.
"""

from __future__ import annotations

from typing import Any

import optuna

from autoscience.data.registry import SizeTier, Task
from autoscience.models.zoo import ModelName


def suggest_params(
    trial: optuna.Trial,
    model: ModelName,
    task: Task,
    tier: SizeTier,
    *,
    has_categorical: bool,
) -> dict[str, Any]:
    params: dict[str, Any] = {}
    _suggest_preprocessing(trial, params, has_categorical)
    _suggest_selection(trial, params, tier)
    _suggest_model(trial, params, model, task, tier)
    return params


def _suggest_preprocessing(
    trial: optuna.Trial, params: dict[str, Any], has_categorical: bool
) -> None:
    params["prep.numeric_imputer"] = trial.suggest_categorical(
        "prep.numeric_imputer", ["median", "mean"]
    )
    params["prep.scaler"] = trial.suggest_categorical(
        "prep.scaler", ["standard", "robust", "quantile", "none"]
    )
    if has_categorical:
        params["prep.cat_encoder"] = trial.suggest_categorical(
            "prep.cat_encoder", ["onehot", "ordinal", "target"]
        )


def _suggest_selection(trial: optuna.Trial, params: dict[str, Any], tier: SizeTier) -> None:
    strategies = ["none", "filter", "l1", "tree"]
    if tier is SizeTier.SMALL:
        strategies.append("rfe")  # wrapper method only affordable on small data
    strategy = trial.suggest_categorical("select.strategy", strategies)
    params["select.strategy"] = strategy
    if strategy == "filter":
        params["select.percentile"] = trial.suggest_int("select.percentile", 25, 100, step=25)
    elif strategy == "rfe":
        params["select.n_features_fraction"] = trial.suggest_float(
            "select.n_features_fraction", 0.3, 0.9
        )


def _suggest_model(
    trial: optuna.Trial,
    params: dict[str, Any],
    model: ModelName,
    task: Task,
    tier: SizeTier,
) -> None:
    t = trial
    classification = task is Task.CLASSIFICATION

    if model is ModelName.LINEAR:
        if classification:
            params["model.C"] = t.suggest_float("model.C", 1e-3, 100, log=True)
        else:
            params["model.alpha"] = t.suggest_float("model.alpha", 1e-3, 100, log=True)
    elif model is ModelName.LASSO:
        params["model.alpha"] = t.suggest_float("model.alpha", 1e-4, 10, log=True)
    elif model is ModelName.SGD_LINEAR:
        params["model.alpha"] = t.suggest_float("model.alpha", 1e-6, 1e-2, log=True)
        params["model.penalty"] = t.suggest_categorical("model.penalty", ["l2", "l1", "elasticnet"])
    elif model in (ModelName.RANDOM_FOREST, ModelName.EXTRA_TREES):
        params["model.n_estimators"] = t.suggest_int("model.n_estimators", 100, 500, step=100)
        params["model.max_depth"] = t.suggest_int("model.max_depth", 4, 24)
        params["model.min_samples_leaf"] = t.suggest_int("model.min_samples_leaf", 1, 10)
        params["model.max_features"] = t.suggest_float("model.max_features", 0.3, 1.0)
    elif model is ModelName.HIST_GB:
        params["model.learning_rate"] = t.suggest_float("model.learning_rate", 0.01, 0.3, log=True)
        params["model.max_iter"] = t.suggest_int("model.max_iter", 100, 500, step=50)
        params["model.max_leaf_nodes"] = t.suggest_int("model.max_leaf_nodes", 15, 127)
        params["model.l2_regularization"] = t.suggest_float(
            "model.l2_regularization", 1e-6, 1.0, log=True
        )
    elif model is ModelName.XGBOOST:
        params["model.n_estimators"] = t.suggest_int("model.n_estimators", 100, 600, step=50)
        params["model.learning_rate"] = t.suggest_float("model.learning_rate", 0.01, 0.3, log=True)
        params["model.max_depth"] = t.suggest_int("model.max_depth", 3, 10)
        params["model.subsample"] = t.suggest_float("model.subsample", 0.6, 1.0)
        params["model.colsample_bytree"] = t.suggest_float("model.colsample_bytree", 0.6, 1.0)
        params["model.min_child_weight"] = t.suggest_float(
            "model.min_child_weight", 0.1, 10, log=True
        )
        params["model.reg_lambda"] = t.suggest_float("model.reg_lambda", 1e-3, 10, log=True)
    elif model is ModelName.SVM:
        params["model.C"] = t.suggest_float("model.C", 0.01, 100, log=True)
        params["model.gamma"] = t.suggest_float("model.gamma", 1e-4, 1.0, log=True)
    elif model is ModelName.KNN:
        params["model.n_neighbors"] = t.suggest_int("model.n_neighbors", 3, 50)
        params["model.weights"] = t.suggest_categorical("model.weights", ["uniform", "distance"])
    elif model is ModelName.TORCH_MLP:
        params["model.hidden_dim"] = t.suggest_categorical("model.hidden_dim", [64, 128, 256])
        params["model.n_layers"] = t.suggest_int("model.n_layers", 1, 4)
        params["model.dropout"] = t.suggest_float("model.dropout", 0.0, 0.4)
        params["model.lr"] = t.suggest_float("model.lr", 1e-4, 1e-2, log=True)
        params["model.weight_decay"] = t.suggest_float("model.weight_decay", 1e-6, 1e-3, log=True)
        # Epoch/batch budget scales with dataset size.
        epochs = {SizeTier.SMALL: 100, SizeTier.MEDIUM: 50, SizeTier.LARGE: 20}[tier]
        params["model.max_epochs"] = epochs
        params["model.batch_size"] = 1024 if tier is SizeTier.LARGE else 256
