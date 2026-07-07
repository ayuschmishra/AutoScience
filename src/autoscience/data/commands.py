"""`autoscience data` subcommands."""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from autoscience.data import loaders, schema
from autoscience.data.registry import REGISTRY, SizeTier, get_spec

data_app = typer.Typer(help="Dataset registry operations.", no_args_is_help=True)
console = Console()


@data_app.command("list")
def list_datasets() -> None:
    """Show all registered datasets."""
    table = Table(title="AutoScience dataset registry")
    for col in ("name", "task", "tier", "rows", "features", "local subset", "description"):
        table.add_column(col)
    for spec in REGISTRY.values():
        table.add_row(
            spec.name,
            spec.task.value,
            spec.tier.value,
            f"{spec.expected_rows:,}",
            str(spec.expected_features),
            f"{spec.local_subset_rows:,}" if spec.local_subset_rows else "-",
            spec.description,
        )
    console.print(table)


def _resolve_names(name: str | None, all_: bool, include_large: bool) -> list[str]:
    if name is not None:
        return [name]
    if not all_:
        raise typer.BadParameter("Provide a dataset name or --all.")
    return [s.name for s in REGISTRY.values() if include_large or s.tier is not SizeTier.LARGE]


@data_app.command("download")
def download_cmd(
    name: str | None = typer.Argument(None),
    all_: bool = typer.Option(False, "--all", help="Download all datasets."),
    include_large: bool = typer.Option(
        False, "--include-large", help="With --all, also download large-tier datasets."
    ),
    force: bool = typer.Option(False, "--force", help="Re-download even if cached."),
) -> None:
    """Download and cache dataset(s) as validated parquet."""
    for n in _resolve_names(name, all_, include_large):
        path = loaders.download(n, force=force)
        console.print(f"[green]ok[/green] {n} -> {path}")


@data_app.command("validate")
def validate_cmd(
    name: str | None = typer.Argument(None),
    all_: bool = typer.Option(False, "--all", help="Validate all datasets."),
    include_large: bool = typer.Option(
        False, "--include-large", help="With --all, also validate large-tier datasets."
    ),
) -> None:
    """Load dataset(s) from cache (downloading if needed) and validate schemas."""
    table = Table(title="Validation report")
    for col in ("dataset", "rows", "features", "missing %", "classes", "checksum", "status"):
        table.add_column(col)

    failed = False
    for n in _resolve_names(name, all_, include_large):
        spec = get_spec(n)
        try:
            loaders.download(n)
            ds = loaders.load_dataset(n, full=True)
            report = schema.validate(spec, ds.x, ds.y)
            checksum_ok = loaders.verify_checksum(n)
            status = "[green]ok[/green]" if report.ok and checksum_ok else "[red]FAIL[/red]"
            failed |= not (report.ok and checksum_ok)
            table.add_row(
                n,
                f"{report.rows:,}",
                str(report.features),
                f"{report.missing_fraction * 100:.2f}",
                str(report.n_classes) if report.n_classes is not None else "-",
                "ok" if checksum_ok else "MISMATCH",
                status if report.ok else f"[red]{'; '.join(report.errors)}[/red]",
            )
        except Exception as exc:  # present all failures, then exit non-zero
            failed = True
            table.add_row(n, "-", "-", "-", "-", "-", f"[red]{exc}[/red]")
    console.print(table)
    if failed:
        raise typer.Exit(code=1)
