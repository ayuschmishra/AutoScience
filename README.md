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

## Commands

| Command | What it does |
|---|---|
| `autoscience data list` | Show the 14-dataset registry (3 size tiers, 569 rows to 11M) |
| `autoscience data validate --all` | Download, cache (parquet + sha256), and schema-validate datasets |
| `autoscience run -d wine -m xgboost --budget smoke` | One nested-CV HPO experiment, logged to MLflow |
| `autoscience benchmark -c experiments/benchmark_smoke.yaml` | Resumable sweep: datasets x models x seeds x modes |
| `autoscience report` | Regenerate `reports/benchmark_report.md` + figures from MLflow |
| `autoscience scaling -d covertype` | Scaling study: score/time/memory vs training-set size |
| `autoscience audit repro -d wine -m linear` | Run the identical config twice; verify bit-identical metrics |
| `autoscience register -d wine` / `predict` | Register the best pipeline in the Model Registry; serve it |
| `uv run mlflow ui` | Browse all runs at http://127.0.0.1:5000 |

## How it works

1. **Data layer** — a declarative registry of public scientific datasets in three size tiers
   (small < 10k, medium < 100k, large up to 11M rows: covertype, YearPredictionMSD, HIGGS, SUSY).
   Every dataset is downloaded once, schema-validated against pinned expectations, downcast to
   memory-efficient dtypes, and cached as checksummed parquet. Outer CV splits are computed once
   per (dataset, seed) and persisted, so every model — automated or baseline — sees identical folds.
2. **Automated preprocessing** — a dataset-aware `ColumnTransformer` (imputation, scaling,
   encoding with rare-category grouping) whose every decision is recorded with the dataset trait
   that drove it. Fit strictly inside CV folds: zero leakage by construction.
3. **Feature selection** — filter (variance + mutual information), embedded (L1, tree
   importance), and wrapper (RFE, small tier only) strategies as searchable pipeline steps.
4. **Model zoo** — linear models, SGD (out-of-core capable), random forest, extra trees,
   HistGradientBoosting, SVM, KNN, XGBoost (`hist`), and a fully seeded PyTorch tabular MLP with
   mini-batch training and MC-dropout uncertainty — all behind one sklearn-compatible interface,
   with per-tier gating so models only run where they scale.
5. **HPO** — Optuna TPE optimizes the *entire pipeline* (preprocessing + selection + model) under
   nested cross-validation. Small/medium tiers use inner K-fold with median pruning; the large
   tier uses a single validation split with successive data-fraction pruning so bad configs die
   after seeing 25% of the data. Hard trial + wall-clock budgets per tier (`smoke`/`local`/`full`).
6. **Evaluation** — per-fold panels: quality (ROC-AUC, F1, log-loss / RMSE, MAE, R2), calibration
   (ECE, Brier), uncertainty (80% prediction-interval PICP/MPIW via quantile GBM or MC-dropout),
   and efficiency (fit time, peak RSS, model size, inference latency).
7. **Study** — resumable benchmark sweeps against two manual baselines (library defaults and a
   hand-tuned "sensible expert"), Wilcoxon signed-rank + Friedman/Nemenyi significance testing
   with a critical-difference diagram, seed-variance reproducibility analysis, a
   pipeline-decisions-vs-dataset-characteristics analysis, and a scaling study — all rendered into
   one regenerable markdown report.

## Reproducibility

- `uv.lock` pins the exact environment; CI verifies a clean bootstrap on 3.12 and 3.13.
- Every run records config, seed, git SHA, and `uv.lock` hash in MLflow.
- Global seeding covers `random`, NumPy, and torch (deterministic algorithms requested); the
  PyTorch MLP is bit-reproducible (verified by tests).
- `autoscience audit repro` re-runs an identical config — including the full Optuna search — and
  asserts every quality metric reproduces exactly (timing/memory metrics excluded).

## Cloud benchmarks

`notebooks/run_benchmark.ipynb` bootstraps the repo on Colab/any Linux GPU box, swaps in CUDA
torch, downloads the full datasets (incl. 11M-row HIGGS), runs `experiments/benchmark_full.yaml`
(5 seeds, full budgets), and packages `mlflow.db` + artifacts for download. Drop them into the
repo locally and run `autoscience report`.

## Development

```bash
uv run pytest                    # full suite (unit + integration)
uv run pytest -m "not slow"      # what CI runs
uv run ruff check . && uv run ruff format .
uv run mypy                      # strict on src/
uv run pre-commit install
```

Project layout: `src/autoscience/{data,preprocessing,features,models,hpo,evaluation,profiling,tracking,reporting}`,
experiment configs in `experiments/`, generated report in `reports/`. Design notes in
[docs/design.md](docs/design.md).

## License

MIT
