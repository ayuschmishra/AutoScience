# AutoScience Benchmark Report

Generated 2026-07-07 22:06 | git `7b93c3cb75` | 12 runs | 2 datasets | seeds: [42]

## Leaderboard (primary metric per dataset)

Primary metric: ROC-AUC (classification) / -RMSE (regression); mean over seeds, std over seeds in parentheses. Best row per dataset in bold.

### diabetes (regression)

| model | mode | primary (std) | HPO cost (s) |
|---|---|---|---|
| linear | Automated (HPO) | **-54.8089 (0.0000)** | 5.5 |
| linear | Manual: defaults | -54.8271 (0.0000) | 0.0 |
| linear | Manual: expert | -54.8271 (0.0000) | 0.0 |
| hist_gb | Automated (HPO) | -55.9848 (0.0000) | 37.5 |
| hist_gb | Manual: defaults | -58.5248 (0.0000) | 0.0 |
| hist_gb | Manual: expert | -60.2895 (0.0000) | 0.0 |

### wine (classification)

| model | mode | primary (std) | HPO cost (s) |
|---|---|---|---|
| linear | Manual: defaults | **1.0000 (0.0000)** | 0.0 |
| linear | Manual: expert | **1.0000 (0.0000)** | 0.0 |
| hist_gb | Manual: expert | 0.9982 (0.0000) | 0.0 |
| hist_gb | Automated (HPO) | 0.9982 (0.0000) | 51.0 |
| hist_gb | Manual: defaults | 0.9981 (0.0000) | 0.0 |
| linear | Automated (HPO) | 0.9974 (0.0000) | 13.0 |

## Automated vs manual baselines (paired across (dataset, model) blocks)

| comparison | blocks | win/tie/loss | median delta | Wilcoxon p | effect (r) |
|---|---|---|---|---|---|
| automated vs Manual: defaults | 4 | 3/0/1 | +0.0091 | 0.3750 | +0.60 |
| automated vs Manual: expert | 4 | 2/1/1 | +0.0091 | 0.5000 | +0.67 |

`*` p < 0.05. Positive delta/effect favors the automated pipeline.

## Across-block mode ranking (Friedman + Nemenyi)

Friedman p = 0.5836

| mode | average rank (1 = best) |
|---|---|
| Automated (HPO) | 1.62 |
| Manual: expert | 2.12 |
| Manual: defaults | 2.25 |

## Calibration & uncertainty

Classification (mean over datasets/models/seeds):

| mode | ECE | Brier |
|---|---|---|
| Automated (HPO) | 0.0408 | 0.0565 |
| Manual: defaults | 0.0376 | 0.0400 |
| Manual: expert | 0.0373 | 0.0406 |

Regression 80% prediction intervals (nominal PICP = 0.80):

| mode | PICP | MPIW (normalized) |
|---|---|---|
| Automated (HPO) | 0.7284 | 1.8231 |
| Manual: defaults | 0.5564 | 1.2860 |
| Manual: expert | 0.5383 | 1.1966 |

## Computational efficiency

![Score vs compute](figures/efficiency_pareto.png)

| mode | median compute (s) | median model size (MB) | median latency (ms/1k) |
|---|---|---|---|
| Automated (HPO) | 25.4 | 0.173 | 109.0 |
| Manual: defaults | 0.1 | 0.088 | 53.4 |
| Manual: expert | 0.1 | 0.204 | 88.1 |

## Reproducibility (variance across seeds)

_Single seed (42) — rerun with multiple seeds for seed-variance estimates. Bit-level reproducibility is verified separately by `autoscience audit repro`._

## Automated pipeline decisions vs dataset characteristics

Most frequently selected options (outer fold 0, across runs):

| dataset | tier | top scaler | top selection strategy |
|---|---|---|---|
| diabetes | small | robust | filter |
| wine | small | standard | filter |

## Scaling study

![Scaling on covertype](figures/scaling_covertype.png)
