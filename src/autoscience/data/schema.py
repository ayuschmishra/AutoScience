"""Schema validation: fail loudly when a downloaded dataset drifts from its spec."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from autoscience.data.registry import DatasetSpec, Task


@dataclass
class ValidationReport:
    name: str
    rows: int
    features: int
    missing_fraction: float
    n_classes: int | None
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def validate(spec: DatasetSpec, x: pd.DataFrame, y: pd.Series) -> ValidationReport:
    """Validate a freshly downloaded (full) dataset against its spec."""
    errors: list[str] = []
    rows, features = x.shape

    if rows != spec.expected_rows:
        errors.append(f"row count {rows} != expected {spec.expected_rows}")
    if features != spec.expected_features:
        errors.append(f"feature count {features} != expected {spec.expected_features}")
    if len(y) != rows:
        errors.append(f"target length {len(y)} != row count {rows}")
    if y.isna().any():
        errors.append(f"target contains {int(y.isna().sum())} missing values")

    n_classes: int | None = None
    if spec.task is Task.CLASSIFICATION:
        n_classes = int(y.nunique())
        if n_classes < 2:
            errors.append(f"classification target has {n_classes} classes")
    elif not pd.api.types.is_numeric_dtype(y):
        errors.append(f"regression target dtype {y.dtype} is not numeric")

    total = rows * features
    missing_fraction = float(x.isna().to_numpy().sum() / total) if total else 0.0

    return ValidationReport(
        name=spec.name,
        rows=rows,
        features=features,
        missing_fraction=missing_fraction,
        n_classes=n_classes,
        errors=errors,
    )
