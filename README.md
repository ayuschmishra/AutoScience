# AutoScience

**Automated Machine Learning Pipeline for Scientific Datasets**

An end-to-end ML pipeline that automates preprocessing, feature selection, model training, and
hyperparameter optimization, benchmarking scikit-learn, XGBoost, and PyTorch models across public
scientific datasets with full experiment tracking via MLflow. It evaluates model reproducibility,
performance, uncertainty estimation, and computational efficiency across varying dataset
characteristics to assess automated pipeline decisions against manual baselines.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) (Python 3.12–3.13).

```bash
git clone <repo-url> && cd AutoScience
uv sync --all-groups
uv run autoscience --help
```

Run a single automated pipeline (dataset × model) with a smoke HPO budget:

```bash
uv run autoscience run --dataset wine --model xgboost --budget smoke
```

Run the benchmark sweep and inspect results:

```bash
uv run autoscience benchmark --profile smoke
uv run mlflow ui          # browse runs at http://127.0.0.1:5000
uv run autoscience report # regenerate reports/benchmark_report.md
```

## What it does

1. **Data layer** — a registry of ~14 public scientific datasets (classification + regression) in
   three size tiers, from `breast_cancer` (569 rows) to `HIGGS` (11M rows), cached as parquet with
   checksums and deterministic, persisted CV splits.
2. **Automated preprocessing** — dataset-aware imputation, encoding, and scaling built as
   leakage-safe scikit-learn `ColumnTransformer`s that are fit inside every CV fold.
3. **Feature selection** — filter, embedded, and model-based strategies exposed as searchable
   pipeline choices.
4. **Model zoo** — linear models, tree ensembles, XGBoost, and a PyTorch tabular MLP behind one
   sklearn-compatible interface, with per-tier gating so every model only runs where it scales.
5. **HPO** — Optuna optimizes the *entire pipeline* (preprocessing + features + model) under
   nested cross-validation with pruning and hard wall-clock budgets.
6. **Evaluation study** — multi-seed benchmarks against manual baselines with statistical
   significance testing, calibration/uncertainty metrics, efficiency profiling, and a generated
   report.

## Reproducibility

- `uv.lock` pins the exact environment; CI verifies a clean bootstrap.
- Every run is fully described by config + seed; MLflow logs the resolved config, git SHA, and
  seeds.
- `autoscience audit repro` re-runs a config twice and diffs the metrics.

## Development

```bash
uv run pytest            # tests
uv run ruff check .      # lint
uv run mypy              # type check
uv run pre-commit install
```

## License

MIT
