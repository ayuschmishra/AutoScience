# AutoScience design notes

Decisions that shape the system, and why.

## Evaluation protocol

- **Persisted outer splits.** Folds are computed once per (dataset, seed, n_rows) and stored in
  `data/splits/`. Every mode (automated, both baselines) and every model reads the same file, so
  comparisons are paired by construction and Wilcoxon signed-rank tests are valid.
- **Nested CV, tier-aware.** Small/medium: 5 outer folds, 3 inner folds for HPO. Large
  (n >= 100k): one stratified 80/20 outer holdout and one inner validation split — K-fold variance
  reduction is negligible at that scale and the cost multiplier is not. The inner protocol is a
  budget field, not a code path fork.
- **The whole pipeline is the search space.** Optuna suggests preprocessing (imputer, scaler,
  encoder), feature-selection strategy, and model hyperparameters together. "Automated pipeline
  decisions" in the study are therefore real optimized decisions, not fixed heuristics — the
  heuristic ("auto") path exists for baselines and default runs and logs its reasons.

## Leakage policy

Everything that learns from data (imputers, scalers, target encoders, selectors, models) is a
step in one sklearn `Pipeline`, cloned and fit inside each fold. The only pre-fold computation is
*structural*: choosing transformer types from training-frame statistics (column dtypes, skew,
cardinality) on the outer-train split. Tested: fold-wise fitted statistics must differ.

## Scalability

- Parquet cache with float32/category downcasting (HIGGS ~1.2 GB in memory).
- Loaders enforce a memory guard (estimated footprint <= 60% of available RAM) and expose a
  chunked `iter_batches` API for `partial_fit` models (SGD).
- SVM/KNN/RFE are tier-gated via registry metadata rather than subsampled — a model that can't
  scale is absent from the grid, never silently handicapped.
- Large-tier HPO trials train on successive data fractions (25% -> 100%) under a
  successive-halving pruner; the torch MLP always streams mini-batches, so its memory is bounded
  by batch size at any n.
- Local development uses fixed-seed subsets of large datasets (a module-level constant seed,
  deliberately independent of experiment seeds); cloud `full` runs use complete datasets. Which
  variant produced a run is recorded in its MLflow params.

## Determinism

`set_global_seed` seeds `random`, NumPy's global RNG (sklearn draws from it when
`random_state=None`), and torch (+ deterministic algorithms, warn-only). Optuna's TPE sampler,
all splitters, and all estimators receive explicit seeds derived from the experiment seed.
`autoscience audit repro` is the enforcement mechanism: identical config -> identical quality
metrics, verified through the full HPO path in CI.

## Tracking

- SQLite MLflow backend (`mlflow.db`) — the 3.x-recommended local store and a single portable
  file for cloud-to-local result syncing.
- One MLflow run per (dataset, model, seed, mode); Optuna trial history attached as a CSV
  artifact instead of thousands of nested runs.
- Tags carry identity (dataset/model/mode/tier/task/git SHA/uv.lock hash); metrics carry the
  per-fold panel (step = fold index) plus aggregates.

## Known trade-offs

- XGBoost early stopping is not used inside HPO (it needs a held-out eval set per fit);
  `n_estimators` is searched instead. Simpler, fair across models.
- Expert baselines are honest but conventional (documented in
  `experiments/baselines/expert.yaml`); they are meant to represent a competent practitioner
  without search budget, not an adversarial expert.
- Regression prediction intervals are only produced where the model family supports them
  (quantile HistGB, MC-dropout MLP); other models report no interval rather than a fake one.
