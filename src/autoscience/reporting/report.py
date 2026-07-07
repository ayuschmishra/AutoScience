"""Generate the benchmark report from MLflow runs.

``collect_runs`` pulls every benchmark run into a tidy frame; ``generate_report``
renders ``reports/benchmark_report.md`` plus figures. Every section degrades
gracefully when the data is thin (single seed, few datasets) — it states what
is missing instead of failing, so the report works for smoke runs and the full
study alike.
"""

from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from autoscience.data.registry import Task
from autoscience.evaluation.stats import friedman_nemenyi, paired_wilcoxon, win_tie_loss
from autoscience.reporting.style import (
    MODE_COLORS,
    MODE_LABELS,
    MODE_ORDER,
    apply_style,
    model_colors,
)
from autoscience.tracking.mlflow_utils import code_version, setup_mlflow
from autoscience.utils import paths

logger = logging.getLogger(__name__)

MIN_BLOCKS_FOR_TESTS = 3


# --------------------------------------------------------------------------
# Data collection
# --------------------------------------------------------------------------


def collect_runs(
    experiment_name: str = "autoscience", tracking_uri: str | None = None
) -> pd.DataFrame:
    """Tidy frame of finished benchmark runs (one row per run)."""
    setup_mlflow(experiment_name, tracking_uri)
    raw = pd.DataFrame(
        mlflow.search_runs(filter_string="attributes.status = 'FINISHED'", max_results=50000)
    )
    if raw.empty or "tags.benchmark_mode" not in raw.columns:
        return pd.DataFrame()
    return _tidy(raw[raw["tags.benchmark_mode"].notna()])


def collect_scaling_runs(
    experiment_name: str = "autoscience", tracking_uri: str | None = None
) -> pd.DataFrame:
    setup_mlflow(experiment_name, tracking_uri)
    raw = pd.DataFrame(
        mlflow.search_runs(
            filter_string="tags.scaling = 'true' and attributes.status = 'FINISHED'",
            max_results=50000,
        )
    )
    if raw.empty:
        return pd.DataFrame()
    df = pd.DataFrame(
        {
            "dataset": raw["tags.dataset"],
            "model": raw["tags.model"],
            "fraction": raw["tags.scaling_fraction"].astype(float),
            "n_rows": raw["metrics.n_train_rows"],
            "primary": raw["metrics.primary"],
            "fit_seconds": raw["metrics.fit_seconds"],
            "fit_peak_mb": raw["metrics.fit_peak_mb"],
        }
    )
    return df.sort_values(["dataset", "model", "fraction"]).reset_index(drop=True)


def _tidy(raw: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "run_id": raw["run_id"],
            "dataset": raw["tags.dataset"],
            "model": raw["tags.model"],
            "mode": raw["tags.benchmark_mode"],
            "seed": raw["params.seed"].astype(int),
            "tier": raw["tags.tier"],
            "task": raw["tags.task"],
        }
    )
    for col in raw.columns:
        if col.startswith("metrics."):
            df[col.removeprefix("metrics.")] = raw[col]
    primary = pd.Series(np.nan, index=df.index)
    classification = df["task"] == Task.CLASSIFICATION.value
    if "roc_auc_mean" in df.columns:
        primary[classification] = df.loc[classification, "roc_auc_mean"]
    if "neg_rmse_mean" in df.columns:
        primary[~classification] = df.loc[~classification, "neg_rmse_mean"]
    df["primary"] = primary
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# Report generation
# --------------------------------------------------------------------------


def generate_report(
    df: pd.DataFrame,
    out_dir: Path | None = None,
    scaling_df: pd.DataFrame | None = None,
    tracking_uri: str | None = None,
) -> Path:
    out_dir = out_dir or paths.REPORTS_DIR
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    apply_style()

    version = code_version()
    lines = [
        "# AutoScience Benchmark Report",
        "",
        f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M} | "
        f"git `{version.get('git_sha', 'unknown')[:10]}` | "
        f"{df['run_id'].nunique() if not df.empty else 0} runs | "
        f"{df['dataset'].nunique() if not df.empty else 0} datasets | "
        f"seeds: {sorted(df['seed'].unique().tolist()) if not df.empty else []}",
        "",
    ]
    if df.empty:
        lines.append("No benchmark runs found. Run `autoscience benchmark` first.")
        report_path = out_dir / "benchmark_report.md"
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return report_path

    lines += _leaderboard_section(df)
    lines += _significance_section(df)
    lines += _ranking_section(df, figures)
    lines += _calibration_section(df)
    lines += _efficiency_section(df, figures)
    lines += _reproducibility_section(df)
    lines += _decisions_section(df, tracking_uri)
    if scaling_df is not None and not scaling_df.empty:
        lines += _scaling_section(scaling_df, figures)

    report_path = out_dir / "benchmark_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report written to %s", report_path)
    return report_path


def _fmt(value: float, digits: int = 4) -> str:
    return "-" if pd.isna(value) else f"{value:.{digits}f}"


def _seed_means(df: pd.DataFrame) -> pd.DataFrame:
    """Per (dataset, model, mode): mean primary across seeds (the block value)."""
    return (
        df.groupby(["dataset", "model", "mode"], as_index=False)
        .agg(primary=("primary", "mean"), primary_seed_std=("primary", "std"))
        .fillna({"primary_seed_std": 0.0})
    )


def _leaderboard_section(df: pd.DataFrame) -> list[str]:
    lines = ["## Leaderboard (primary metric per dataset)", ""]
    lines.append(
        "Primary metric: ROC-AUC (classification) / -RMSE (regression); "
        "mean over seeds, std over seeds in parentheses. Best row per dataset in bold."
    )
    lines.append("")
    seed_means = _seed_means(df)
    for dataset, group in seed_means.groupby("dataset"):
        task = df.loc[df["dataset"] == dataset, "task"].iloc[0]
        lines.append(f"### {dataset} ({task})")
        lines.append("")
        lines.append("| model | mode | primary (std) | HPO cost (s) |")
        lines.append("|---|---|---|---|")
        group = group.sort_values("primary", ascending=False)
        best = group["primary"].max()
        for _, row in group.iterrows():
            cost = df.loc[
                (df["dataset"] == dataset)
                & (df["model"] == row["model"])
                & (df["mode"] == row["mode"]),
                "hpo_seconds_total",
            ].mean()
            cell = f"{_fmt(row['primary'])} ({_fmt(row['primary_seed_std'])})"
            if row["primary"] == best:
                cell = f"**{cell}**"
            lines.append(
                f"| {row['model']} | {MODE_LABELS.get(row['mode'], row['mode'])} "
                f"| {cell} | {_fmt(cost, 1)} |"
            )
        lines.append("")
    return lines


def _significance_section(df: pd.DataFrame) -> list[str]:
    lines = ["## Automated vs manual baselines (paired across (dataset, model) blocks)", ""]
    seed_means = _seed_means(df)
    pivot = seed_means.pivot_table(index=["dataset", "model"], columns="mode", values="primary")
    if "automated" not in pivot.columns:
        return [*lines, "_No automated runs found._", ""]

    lines.append("| comparison | blocks | win/tie/loss | median delta | Wilcoxon p | effect (r) |")
    lines.append("|---|---|---|---|---|---|")
    for baseline in ("baseline_default", "baseline_expert"):
        if baseline not in pivot.columns:
            continue
        paired = pivot[["automated", baseline]].dropna()
        n = len(paired)
        if n < MIN_BLOCKS_FOR_TESTS:
            lines.append(
                f"| automated vs {MODE_LABELS[baseline]} | {n} | - | - | "
                f"_needs >= {MIN_BLOCKS_FOR_TESTS} blocks_ | - |"
            )
            continue
        a, b = paired["automated"].to_numpy(), paired[baseline].to_numpy()
        w, t, loss = win_tie_loss(a, b)
        cmp = paired_wilcoxon(a, b)
        sig = " *" if cmp.significant else ""
        lines.append(
            f"| automated vs {MODE_LABELS[baseline]} | {n} | {w}/{t}/{loss} "
            f"| {cmp.median_delta:+.4f} | {cmp.p_value:.4f}{sig} | {cmp.effect_size:+.2f} |"
        )
    lines.append("")
    lines.append("`*` p < 0.05. Positive delta/effect favors the automated pipeline.")
    lines.append("")
    return lines


def _ranking_section(df: pd.DataFrame, figures: Path) -> list[str]:
    lines = ["## Across-block mode ranking (Friedman + Nemenyi)", ""]
    pivot = _seed_means(df).pivot_table(
        index=["dataset", "model"], columns="mode", values="primary"
    )
    pivot = pivot[[m for m in MODE_ORDER if m in pivot.columns]].dropna()
    if pivot.shape[0] < MIN_BLOCKS_FOR_TESTS or pivot.shape[1] < 3:
        return [
            *lines,
            f"_Needs >= {MIN_BLOCKS_FOR_TESTS} blocks and 3 modes "
            f"(have {pivot.shape[0]} blocks, {pivot.shape[1]} modes)._",
            "",
        ]

    result = friedman_nemenyi(pivot)
    lines.append(
        f"Friedman p = {result.p_value:.4f}"
        + (" (significant at 0.05)" if result.significant else "")
    )
    lines.append("")
    lines.append("| mode | average rank (1 = best) |")
    lines.append("|---|---|")
    for mode, rank in result.avg_ranks.sort_values().items():
        lines.append(f"| {MODE_LABELS.get(str(mode), str(mode))} | {rank:.2f} |")
    lines.append("")

    try:
        import scikit_posthocs as sp

        fig, ax = plt.subplots(figsize=(7, 2.2))
        sp.critical_difference_diagram(
            ranks=result.avg_ranks.rename(index=MODE_LABELS),
            sig_matrix=result.nemenyi_p.rename(index=MODE_LABELS, columns=MODE_LABELS),
            ax=ax,
            label_props={"color": "#1a1a19"},
        )
        ax.set_title("Critical-difference diagram (lower rank = better)")
        fig.tight_layout()
        fig.savefig(figures / "critical_difference.png", bbox_inches="tight")
        plt.close(fig)
        lines.append("![Critical difference diagram](figures/critical_difference.png)")
        lines.append("")
    except Exception as exc:  # diagram is a bonus; the table above carries the result
        logger.warning("CD diagram skipped: %s", exc)
    return lines


def _calibration_section(df: pd.DataFrame) -> list[str]:
    lines = ["## Calibration & uncertainty", ""]
    cls = df[df["task"] == Task.CLASSIFICATION.value]
    reg = df[df["task"] == Task.REGRESSION.value]

    if not cls.empty and "ece_mean" in cls.columns:
        lines.append("Classification (mean over datasets/models/seeds):")
        lines.append("")
        lines.append("| mode | ECE | Brier |")
        lines.append("|---|---|---|")
        for mode, grp in cls.groupby("mode"):
            lines.append(
                f"| {MODE_LABELS.get(str(mode), str(mode))} | "
                f"{_fmt(grp['ece_mean'].mean())} | {_fmt(grp['brier_mean'].mean())} |"
            )
        lines.append("")
    if not reg.empty and "picp_80_mean" in reg.columns and reg["picp_80_mean"].notna().any():
        lines.append("Regression 80% prediction intervals (nominal PICP = 0.80):")
        lines.append("")
        lines.append("| mode | PICP | MPIW (normalized) |")
        lines.append("|---|---|---|")
        has_intervals = reg[reg["picp_80_mean"].notna()]
        for mode, grp in has_intervals.groupby("mode"):
            lines.append(
                f"| {MODE_LABELS.get(str(mode), str(mode))} | "
                f"{_fmt(grp['picp_80_mean'].mean())} | {_fmt(grp['mpiw_80_norm_mean'].mean())} |"
            )
        lines.append("")
    if len(lines) == 2:
        lines += ["_No calibration metrics available._", ""]
    return lines


def _efficiency_section(df: pd.DataFrame, figures: Path) -> list[str]:
    lines = ["## Computational efficiency", ""]

    # Pareto scatter: within-dataset normalized score vs total compute.
    plot_df = df.copy()
    plot_df["compute_s"] = plot_df["hpo_seconds_total"].fillna(0) + plot_df[
        "fit_seconds_mean"
    ].fillna(0)
    normalized = []
    for _, grp in plot_df.groupby("dataset"):
        lo, hi = grp["primary"].min(), grp["primary"].max()
        span = (hi - lo) or 1.0
        normalized.append((grp["primary"] - lo) / span)
    plot_df["score_norm"] = pd.concat(normalized)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mode in MODE_ORDER:  # fixed slot order, never cycled
        sub = plot_df[plot_df["mode"] == mode]
        if sub.empty:
            continue
        ax.scatter(
            sub["compute_s"].clip(lower=1e-2),
            sub["score_norm"],
            s=55,
            color=MODE_COLORS[mode],
            edgecolors="#1a1a19",
            linewidths=0.5,
            label=MODE_LABELS[mode],
            alpha=0.85,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Training + HPO compute per run (s, log scale)")
    ax.set_ylabel("Within-dataset normalized score (1 = best)")
    ax.set_title("Score vs compute cost")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(figures / "efficiency_pareto.png", bbox_inches="tight")
    plt.close(fig)
    lines.append("![Score vs compute](figures/efficiency_pareto.png)")
    lines.append("")

    lines.append("| mode | median compute (s) | median model size (MB) | median latency (ms/1k) |")
    lines.append("|---|---|---|---|")
    for group_mode, grp in plot_df.groupby("mode"):
        size = grp["model_size_mb_mean"].median() if "model_size_mb_mean" in grp else float("nan")
        latency = (
            grp["latency_ms_per_1k_mean"].median()
            if "latency_ms_per_1k_mean" in grp
            else float("nan")
        )
        lines.append(
            f"| {MODE_LABELS.get(str(group_mode), str(group_mode))} "
            f"| {_fmt(grp['compute_s'].median(), 1)} "
            f"| {_fmt(size, 3)} | {_fmt(latency, 1)} |"
        )
    lines.append("")
    return lines


def _reproducibility_section(df: pd.DataFrame) -> list[str]:
    lines = ["## Reproducibility (variance across seeds)", ""]
    n_seeds = df["seed"].nunique()
    if n_seeds < 2:
        return [
            *lines,
            f"_Single seed ({df['seed'].iloc[0]}) — rerun with multiple seeds for "
            "seed-variance estimates. Bit-level reproducibility is verified separately "
            "by `autoscience audit repro`._",
            "",
        ]
    spread = df.groupby(["dataset", "model", "mode"])["primary"].std().dropna().rename("seed_std")
    lines.append(f"{n_seeds} seeds. Std of the primary metric across seeds:")
    lines.append("")
    lines.append("| statistic | value |")
    lines.append("|---|---|")
    lines.append(f"| median seed-std | {_fmt(spread.median())} |")
    lines.append(f"| worst seed-std | {_fmt(spread.max())} |")
    worst: object = spread.idxmax()
    worst_label = " / ".join(map(str, worst)) if isinstance(worst, tuple) else str(worst)
    lines.append(f"| worst combination | {worst_label} |")
    lines.append("")
    return lines


def _decisions_section(df: pd.DataFrame, tracking_uri: str | None) -> list[str]:
    """What preprocessing/selection the HPO chose, vs dataset characteristics."""
    lines = ["## Automated pipeline decisions vs dataset characteristics", ""]
    automated = df[df["mode"] == "automated"]
    if automated.empty:
        return [*lines, "_No automated runs._", ""]

    rows = []
    for _, run in automated.iterrows():
        try:
            path = mlflow.artifacts.download_artifacts(
                run_id=run["run_id"], artifact_path="best_params_fold0.json"
            )
            best = json.loads(Path(path).read_text())
        except Exception:  # artifact may be missing on failed/foreign runs
            continue
        rows.append(
            {
                "dataset": run["dataset"],
                "tier": run["tier"],
                "model": run["model"],
                "seed": run["seed"],
                "scaler": best.get("prep.scaler", "-"),
                "imputer": best.get("prep.numeric_imputer", "-"),
                "selection": best.get("select.strategy", "-"),
            }
        )
    if not rows:
        return [*lines, "_Best-params artifacts not accessible._", ""]

    chosen = pd.DataFrame(rows)
    lines.append("Most frequently selected options (outer fold 0, across runs):")
    lines.append("")
    lines.append("| dataset | tier | top scaler | top selection strategy |")
    lines.append("|---|---|---|---|")
    for (dataset, tier), grp in chosen.groupby(["dataset", "tier"]):
        lines.append(
            f"| {dataset} | {tier} | {grp['scaler'].mode().iloc[0]} "
            f"| {grp['selection'].mode().iloc[0]} |"
        )
    lines.append("")
    return lines


def _scaling_section(scaling_df: pd.DataFrame, figures: Path) -> list[str]:
    lines = ["## Scaling study", ""]
    colors = model_colors(sorted(scaling_df["model"].unique()))
    for dataset, grp in scaling_df.groupby("dataset"):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for model, mg in grp.groupby("model"):
            mg = mg.sort_values("fraction")
            axes[0].plot(
                mg["n_rows"], mg["primary"], marker="o", color=colors[str(model)], label=model
            )
            axes[1].plot(
                mg["n_rows"], mg["fit_seconds"], marker="o", color=colors[str(model)], label=model
            )
        axes[0].set_xscale("log")
        axes[0].set_xlabel("Training rows (log)")
        axes[0].set_ylabel("Primary score")
        axes[0].set_title(f"{dataset}: score vs data size")
        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
        axes[1].set_xlabel("Training rows (log)")
        axes[1].set_ylabel("Fit time (s, log)")
        axes[1].set_title(f"{dataset}: cost vs data size")
        axes[0].legend()
        fig.tight_layout()
        name = f"scaling_{dataset}.png"
        fig.savefig(figures / name, bbox_inches="tight")
        plt.close(fig)
        lines.append(f"![Scaling on {dataset}](figures/{name})")
        lines.append("")
    return lines
