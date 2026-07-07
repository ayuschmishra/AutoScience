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
from autoscience.utils.logging import setup_logging

app = typer.Typer(
    name="autoscience",
    help="Automated ML pipeline for scientific datasets.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging.")) -> None:
    setup_logging(logging.DEBUG if verbose else logging.INFO)


@app.command()
def version() -> None:
    """Print the AutoScience version."""
    typer.echo(__version__)


if __name__ == "__main__":
    app()
