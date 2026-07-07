"""Report generation from a synthetic tidy runs frame (no MLflow needed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from autoscience.reporting.report import generate_report

RNG = np.random.default_rng(5)

DATASETS = [("wine", "classification"), ("phoneme", "classification"), ("diabetes", "regression")]
MODELS = ["linear", "hist_gb"]
MODES = ["automated", "baseline_default", "baseline_expert"]
SEEDS = [42, 43, 44]


def _synthetic_runs() -> pd.DataFrame:
    rows = []
    for dataset, task in DATASETS:
        for model in MODELS:
            for mode in MODES:
                for seed in SEEDS:
                    base = 0.9 if task == "classification" else -50.0
                    bonus = {"automated": 0.03, "baseline_expert": 0.015}.get(mode, 0.0)
                    noise = RNG.normal() * 0.005
                    primary = base + bonus + noise
                    row = {
                        "run_id": f"{dataset}-{model}-{mode}-{seed}",
                        "dataset": dataset,
                        "model": model,
                        "mode": mode,
                        "seed": seed,
                        "tier": "small",
                        "task": task,
                        "hpo_seconds_total": 120.0 if mode == "automated" else 0.0,
                        "fit_seconds_mean": 1.0,
                        "model_size_mb_mean": 0.5,
                        "latency_ms_per_1k_mean": 10.0,
                    }
                    if task == "classification":
                        row |= {
                            "roc_auc_mean": primary,
                            "ece_mean": 0.05,
                            "brier_mean": 0.1,
                        }
                    else:
                        row |= {
                            "neg_rmse_mean": primary,
                            "picp_80_mean": 0.78,
                            "mpiw_80_norm_mean": 1.1,
                        }
                    rows.append(row)
    df = pd.DataFrame(rows)
    df["primary"] = df["roc_auc_mean"].fillna(df["neg_rmse_mean"])
    return df


def _synthetic_scaling() -> pd.DataFrame:
    rows = []
    for model in ("hist_gb", "xgboost"):
        for frac, n in [(0.01, 4600), (0.1, 46000), (1.0, 460000)]:
            rows.append(
                {
                    "dataset": "covertype",
                    "model": model,
                    "fraction": frac,
                    "n_rows": n,
                    "primary": 0.8 + 0.05 * np.log10(n / 4600),
                    "fit_seconds": n / 10000,
                    "fit_peak_mb": n / 1000,
                }
            )
    return pd.DataFrame(rows)


def test_generate_report_full_sections(tmp_path: Path) -> None:
    path = generate_report(_synthetic_runs(), out_dir=tmp_path, scaling_df=_synthetic_scaling())
    text = path.read_text(encoding="utf-8")

    for heading in (
        "# AutoScience Benchmark Report",
        "## Leaderboard",
        "## Automated vs manual baselines",
        "## Across-block mode ranking",
        "## Calibration & uncertainty",
        "## Computational efficiency",
        "## Reproducibility",
        "## Scaling study",
    ):
        assert heading in text, f"missing section: {heading}"

    # The synthetic data was constructed so automation wins significantly.
    assert "Friedman p = 0.0" in text
    # Figures rendered.
    assert (tmp_path / "figures" / "efficiency_pareto.png").exists()
    assert (tmp_path / "figures" / "scaling_covertype.png").exists()


def test_generate_report_handles_single_seed_and_few_blocks(tmp_path: Path) -> None:
    df = _synthetic_runs()
    df = df[(df["seed"] == 42) & (df["dataset"] == "wine")]
    path = generate_report(df, out_dir=tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "Single seed" in text
    assert "needs >= 3 blocks" in text or "Needs >= 3 blocks" in text


def test_generate_report_empty(tmp_path: Path) -> None:
    path = generate_report(pd.DataFrame(), out_dir=tmp_path)
    assert "No benchmark runs found" in path.read_text(encoding="utf-8")
