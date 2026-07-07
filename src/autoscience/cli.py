"""AutoScience command-line interface.

Subcommand groups are registered by their phases:
  data       - dataset registry operations (list, validate, download)
  run        - single (dataset, model) HPO pipeline run
  benchmark  - full sweep orchestration
  report     - regenerate the benchmark report from MLflow
  audit      - reproducibility audits
"""

from __future__ import annotations

import logging

import typer

from autoscience import __version__
from autoscience.data.commands import data_app
from autoscience.utils.logging import setup_logging

app = typer.Typer(
    name="autoscience",
    help="Automated ML pipeline for scientific datasets.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
app.add_typer(data_app, name="data")


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging.")) -> None:
    setup_logging(logging.DEBUG if verbose else logging.INFO)


@app.command()
def version() -> None:
    """Print the AutoScience version."""
    typer.echo(__version__)


@app.command()
def run(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Registry dataset name."),
    model: str = typer.Option(..., "--model", "-m", help="Model zoo name."),
    seed: int = typer.Option(42, help="Experiment seed."),
    budget: str = typer.Option("smoke", help="HPO budget profile: smoke|local|full."),
    full: bool = typer.Option(False, "--full", help="Use the full dataset (no local subset)."),
    mode: str = typer.Option("automated", help="'automated' (HPO) or 'baseline' (defaults)."),
) -> None:
    """Run one automated-pipeline experiment (nested-CV HPO) and log to MLflow."""
    # Heavy imports stay out of `--help` / `data list`.
    from rich.console import Console
    from rich.table import Table

    from autoscience.hpo.runner import run_experiment

    result = run_experiment(
        dataset, model, seed=seed, budget_profile=budget, full_data=full, mode=mode
    )
    table = Table(title=f"{dataset} x {model} (seed={seed}, {mode})")
    table.add_column("metric")
    table.add_column("mean +/- std")
    seen = sorted({k.removesuffix("_mean") for k in result.aggregated if k.endswith("_mean")})
    for name in seen:
        mean = result.aggregated[f"{name}_mean"]
        std = result.aggregated.get(f"{name}_std", 0.0)
        table.add_row(name, f"{mean:.4f} +/- {std:.4f}")
    Console().print(table)
    typer.echo(f"MLflow run: {result.run_id}")


@app.command()
def benchmark(
    config: str = typer.Option(
        "experiments/benchmark_smoke.yaml", "--config", "-c", help="Experiment YAML."
    ),
) -> None:
    """Run the benchmark sweep declared in an experiment config (resumable)."""
    from pathlib import Path

    from rich.console import Console
    from rich.table import Table

    from autoscience.benchmark import BenchmarkConfig, run_benchmark

    cfg = BenchmarkConfig.load(Path(config))
    summary = run_benchmark(cfg)

    table = Table(title=f"Benchmark '{cfg.name}' summary")
    for col in ("dataset", "model", "seed", "mode", "status", "detail"):
        table.add_column(col)
    for o in summary.outcomes:
        style = {"completed": "green", "failed": "red"}.get(o.status, "yellow")
        table.add_row(
            o.dataset, o.model, str(o.seed), o.mode, f"[{style}]{o.status}[/{style}]", o.detail
        )
    Console().print(table)
    typer.echo(
        f"completed={summary.count('completed')} "
        f"skipped_existing={summary.count('skipped_existing')} "
        f"skipped_gated={summary.count('skipped_gated')} failed={summary.count('failed')}"
    )
    if summary.count("failed"):
        raise typer.Exit(code=1)


@app.command()
def report() -> None:
    """Regenerate reports/benchmark_report.md from MLflow benchmark runs."""
    from autoscience.reporting.report import collect_runs, collect_scaling_runs, generate_report

    df = collect_runs()
    scaling_df = collect_scaling_runs()
    path = generate_report(df, scaling_df=scaling_df)
    typer.echo(f"Report: {path} ({len(df)} benchmark runs, {len(scaling_df)} scaling points)")


@app.command()
def scaling(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Large-tier dataset name."),
    models: str = typer.Option(
        "sgd_linear,hist_gb,xgboost,torch_mlp", "--models", "-m", help="Comma-separated models."
    ),
    fractions: str = typer.Option("0.01,0.05,0.1,0.33,1.0", help="Comma-separated fractions."),
    seed: int = typer.Option(42),
    full: bool = typer.Option(False, "--full", help="Use the full dataset (cloud)."),
) -> None:
    """Scaling study: train on nested data fractions, log score/time/memory."""
    from autoscience.scaling import run_scaling_study

    result = run_scaling_study(
        dataset,
        [m.strip() for m in models.split(",")],
        fractions=tuple(float(f) for f in fractions.split(",")),
        seed=seed,
        full_data=full,
    )
    typer.echo(result.to_string(index=False))


@app.command()
def register(
    dataset: str = typer.Option(..., "--dataset", "-d"),
    full: bool = typer.Option(False, "--full", help="Refit on the full dataset."),
) -> None:
    """Register the best automated pipeline for a dataset in the Model Registry."""
    from autoscience.tracking.registry import register_best

    name = register_best(dataset, full_data=full)
    typer.echo(f"Registered: {name}")


@app.command()
def predict(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Registered dataset name."),
    input_csv: str = typer.Option(..., "--input", "-i", help="CSV with feature columns."),
    output_csv: str = typer.Option("predictions.csv", "--output", "-o"),
) -> None:
    """Predict with the registered pipeline for a dataset (demo serving path)."""
    import pandas as pd

    from autoscience.tracking.registry import load_registered

    model = load_registered(dataset)
    x = pd.read_csv(input_csv)
    out = pd.DataFrame({"prediction": model.predict(x)})
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        for i in range(proba.shape[1]):
            out[f"proba_{i}"] = proba[:, i]
    out.to_csv(output_csv, index=False)
    typer.echo(f"{len(out)} predictions -> {output_csv}")


audit_app = typer.Typer(help="Reproducibility audits.", no_args_is_help=True)
app.add_typer(audit_app, name="audit")


@audit_app.command("repro")
def audit_repro(
    dataset: str = typer.Option(..., "--dataset", "-d"),
    model: str = typer.Option(..., "--model", "-m"),
    seed: int = typer.Option(42),
    budget: str = typer.Option("smoke"),
    mode: str = typer.Option("automated"),
    tolerance: float = typer.Option(0.0, help="Max allowed |delta| per quality metric."),
) -> None:
    """Run the identical experiment twice and verify metrics reproduce."""
    from rich.console import Console
    from rich.table import Table

    from autoscience.evaluation.audit import repro_audit

    report = repro_audit(
        dataset, model, seed=seed, budget_profile=budget, mode=mode, tolerance=tolerance
    )
    table = Table(title=f"Reproducibility audit: {dataset} x {model} (seed={seed})")
    table.add_column("metric")
    table.add_column("|delta|")
    for name, delta in sorted(report.deltas.items()):
        table.add_row(name, f"{delta:.3e}")
    Console().print(table)
    if report.ok:
        typer.echo(f"OK: max |delta| = {report.max_abs_delta:.3e} <= {tolerance:.1e}")
    else:
        typer.echo(f"FAIL: max |delta| = {report.max_abs_delta:.3e} > {tolerance:.1e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
