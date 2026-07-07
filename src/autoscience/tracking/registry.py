"""MLflow Model Registry integration.

``register_best`` finds the best automated benchmark run for a dataset,
rebuilds that run's winning pipeline from its logged params, refits it on the
full dataset, and registers it (with signature and input example) as
``autoscience-<dataset>``. ``load_registered`` fetches it back for serving.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from autoscience.data import loaders
from autoscience.data.registry import Task, get_spec
from autoscience.hpo.pipeline import build_pipeline
from autoscience.models.zoo import ModelName
from autoscience.tracking.mlflow_utils import setup_mlflow

logger = logging.getLogger(__name__)


def registered_name(dataset: str) -> str:
    return f"autoscience-{dataset}"


def find_best_automated_run(
    dataset: str, experiment_name: str = "autoscience", tracking_uri: str | None = None
) -> pd.Series:
    """Best finished automated run for a dataset by the primary metric."""
    setup_mlflow(experiment_name, tracking_uri)
    spec = get_spec(dataset)
    metric = "roc_auc_mean" if spec.task is Task.CLASSIFICATION else "neg_rmse_mean"
    runs = pd.DataFrame(
        mlflow.search_runs(
            filter_string=(
                f"tags.dataset = '{dataset}' and tags.mode = 'automated' "
                "and attributes.status = 'FINISHED'"
            ),
            order_by=[f"metrics.{metric} DESC"],
            max_results=1,
        )
    )
    if runs.empty:
        raise LookupError(f"No finished automated runs for {dataset!r}; run a benchmark first.")
    return runs.iloc[0]


def register_best(
    dataset: str,
    *,
    experiment_name: str = "autoscience",
    tracking_uri: str | None = None,
    full_data: bool = False,
) -> str:
    """Refit the winning pipeline on the full dataset and register it."""
    best = find_best_automated_run(dataset, experiment_name, tracking_uri)
    run_id = str(best["run_id"])
    model_name = ModelName(best["tags.model"])
    params_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="best_params_fold0.json"
    )
    best_params: dict[str, Any] = json.loads(Path(params_path).read_text())

    spec = get_spec(dataset)
    ds = loaders.load_dataset(dataset, full=full_data)
    if spec.task is Task.CLASSIFICATION:
        encoder = LabelEncoder().fit(ds.y)
        y = encoder.transform(ds.y)
    else:
        y = ds.y.to_numpy(dtype=np.float64)

    pipeline, _ = build_pipeline(ds.x, spec.task, model_name, best_params, seed=42)
    pipeline.fit(ds.x, y)

    from mlflow.models import infer_signature

    example = ds.x.head(5)
    signature = infer_signature(example, pipeline.predict(example))
    name = registered_name(dataset)
    with mlflow.start_run(run_name=f"register__{dataset}"):
        mlflow.set_tags({"dataset": dataset, "source_run": run_id, "registered": "true"})
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            signature=signature,
            input_example=example,
            registered_model_name=name,
            # skops (the 3.x default) rejects the custom torch wrapper and
            # selector classes as untrusted; cloudpickle handles them.
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
    logger.info("Registered %s (from run %s, model=%s)", name, run_id, model_name.value)
    return name


def load_registered(dataset: str, tracking_uri: str | None = None) -> Any:
    setup_mlflow(tracking_uri=tracking_uri)
    return mlflow.sklearn.load_model(f"models:/{registered_name(dataset)}/latest")
