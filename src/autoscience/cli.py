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


if __name__ == "__main__":
    app()
