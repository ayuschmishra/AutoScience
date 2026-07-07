"""Automated, leakage-safe preprocessing.

:func:`build_preprocessor` inspects the feature frame and emits a
``ColumnTransformer`` plus a *decisions* dict describing every automated
choice and why it was made. The decisions are logged to MLflow so the final
study can analyze pipeline decisions against dataset characteristics.

Any config field can be pinned by the HPO search instead of ``"auto"`` — the
whole preprocessor is part of the searched pipeline and is always fit inside
CV folds only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
    TargetEncoder,
)

from autoscience.data.registry import Task

Scaler = Literal["auto", "standard", "robust", "quantile", "none"]
NumericImputer = Literal["auto", "median", "mean"]
CatEncoder = Literal["auto", "onehot", "ordinal", "target"]

# Rules of thumb behind the "auto" decisions.
OUTLIER_FRACTION_FOR_ROBUST = 0.05
SKEW_FOR_QUANTILE = 4.0
MAX_CARDINALITY_FOR_ONEHOT = 15
RARE_CATEGORY_MIN_FREQUENCY = 0.01


@dataclass
class PreprocessorConfig:
    numeric_imputer: NumericImputer = "auto"
    scaler: Scaler = "auto"
    cat_encoder: CatEncoder = "auto"


@dataclass
class PreprocessorDecisions:
    """What the auto-preprocessor chose, and the dataset traits that drove it."""

    n_numeric: int
    n_categorical: int
    missing_fraction: float
    outlier_fraction: float
    max_abs_skew: float
    max_cardinality: int
    numeric_imputer: str
    scaler: str
    cat_encoder: str
    reasons: dict[str, str] = field(default_factory=dict)

    def as_params(self) -> dict[str, str | int | float]:
        """Flat dict for MLflow param logging."""
        out: dict[str, str | int | float] = {
            "prep.n_numeric": self.n_numeric,
            "prep.n_categorical": self.n_categorical,
            "prep.missing_fraction": round(self.missing_fraction, 5),
            "prep.outlier_fraction": round(self.outlier_fraction, 5),
            "prep.max_abs_skew": round(self.max_abs_skew, 3),
            "prep.max_cardinality": self.max_cardinality,
            "prep.numeric_imputer": self.numeric_imputer,
            "prep.scaler": self.scaler,
            "prep.cat_encoder": self.cat_encoder,
        }
        out.update({f"prep.reason.{k}": v for k, v in self.reasons.items()})
        return out


def _numeric_and_categorical(x: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = x.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in x.columns if c not in numeric]
    return numeric, categorical


def _outlier_fraction(x_num: pd.DataFrame) -> float:
    """Fraction of numeric cells outside 1.5*IQR — drives robust scaling."""
    if x_num.empty:
        return 0.0
    q1, q3 = x_num.quantile(0.25), x_num.quantile(0.75)
    iqr = q3 - q1
    mask = (x_num < q1 - 1.5 * iqr) | (x_num > q3 + 1.5 * iqr)
    return float(mask.to_numpy().mean())


def _resolve_auto(
    config: PreprocessorConfig,
    *,
    outlier_fraction: float,
    max_abs_skew: float,
    max_cardinality: int,
) -> tuple[str, str, str, dict[str, str]]:
    reasons: dict[str, str] = {}

    imputer = config.numeric_imputer
    if imputer == "auto":
        imputer = "median"
        reasons["numeric_imputer"] = "median is robust to skew/outliers"

    scaler = config.scaler
    if scaler == "auto":
        if max_abs_skew > SKEW_FOR_QUANTILE:
            scaler = "quantile"
            reasons["scaler"] = f"max |skew| {max_abs_skew:.1f} > {SKEW_FOR_QUANTILE}"
        elif outlier_fraction > OUTLIER_FRACTION_FOR_ROBUST:
            scaler = "robust"
            reasons["scaler"] = (
                f"outlier fraction {outlier_fraction:.3f} > {OUTLIER_FRACTION_FOR_ROBUST}"
            )
        else:
            scaler = "standard"
            reasons["scaler"] = "well-behaved distributions"

    encoder = config.cat_encoder
    if encoder == "auto":
        if max_cardinality > MAX_CARDINALITY_FOR_ONEHOT:
            encoder = "target"
            reasons["cat_encoder"] = (
                f"max cardinality {max_cardinality} > {MAX_CARDINALITY_FOR_ONEHOT}"
            )
        else:
            encoder = "onehot"
            reasons["cat_encoder"] = "low cardinality"

    return imputer, scaler, encoder, reasons


def _make_scaler(name: str) -> object:
    if name == "standard":
        return StandardScaler()
    if name == "robust":
        return RobustScaler()
    if name == "quantile":
        return QuantileTransformer(output_distribution="normal", subsample=100_000)
    return "passthrough"


def _make_cat_encoder(name: str, task: Task) -> object:
    if name == "onehot":
        return OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=RARE_CATEGORY_MIN_FREQUENCY,  # groups rare categories
            sparse_output=False,
            dtype=np.float32,
        )
    if name == "ordinal":
        return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    target_type = "continuous" if task is Task.REGRESSION else "auto"
    return TargetEncoder(target_type=target_type)


def build_preprocessor(
    x: pd.DataFrame,
    task: Task,
    config: PreprocessorConfig | None = None,
) -> tuple[ColumnTransformer, PreprocessorDecisions]:
    """Build the dataset-aware preprocessing transformer (unfitted)."""
    config = config or PreprocessorConfig()
    numeric, categorical = _numeric_and_categorical(x)
    x_num = x[numeric]

    missing_fraction = float(x.isna().to_numpy().mean()) if x.size else 0.0
    outlier_fraction = _outlier_fraction(x_num)
    skews = x_num.skew() if not x_num.empty else pd.Series(dtype=float)
    max_abs_skew = float(skews.abs().max()) if len(skews) else 0.0
    max_cardinality = int(max(x[c].nunique() for c in categorical)) if categorical else 0

    imputer, scaler, encoder, reasons = _resolve_auto(
        config,
        outlier_fraction=outlier_fraction,
        max_abs_skew=max_abs_skew,
        max_cardinality=max_cardinality,
    )

    transformers: list[tuple[str, object, list[str]]] = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy=imputer)),
                        ("scale", _make_scaler(scaler)),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", _make_cat_encoder(encoder, task)),
                    ]
                ),
                categorical,
            )
        )

    column_transformer = ColumnTransformer(
        transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    decisions = PreprocessorDecisions(
        n_numeric=len(numeric),
        n_categorical=len(categorical),
        missing_fraction=missing_fraction,
        outlier_fraction=outlier_fraction,
        max_abs_skew=max_abs_skew,
        max_cardinality=max_cardinality,
        numeric_imputer=imputer,
        scaler=scaler,
        cat_encoder=encoder,
        reasons=reasons,
    )
    return column_transformer, decisions
