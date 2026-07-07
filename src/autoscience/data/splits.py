"""Deterministic, persisted outer-split protocol.

Splits are computed once per (dataset, seed, n_rows) and persisted to disk,
so every model — automated pipeline and manual baselines alike — is evaluated
on byte-identical folds. The protocol is tier-aware:

- small/medium: (stratified) shuffled K-fold cross-validation
- large: a single stratified 80/20 holdout — at n >= 100k the variance
  reduction from K folds is negligible and the cost multiplier is not.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from autoscience.data.registry import DatasetSpec, SizeTier, Task
from autoscience.utils import paths

logger = logging.getLogger(__name__)

N_FOLDS = 5
HOLDOUT_TEST_FRACTION = 0.2


@dataclass(frozen=True)
class Fold:
    train_idx: np.ndarray
    test_idx: np.ndarray


def _splits_path(spec: DatasetSpec, seed: int, n_rows: int, splits_dir: Path | None) -> Path:
    base = splits_dir or paths.SPLITS_DIR
    # n_rows in the key: the local subset and the full dataset must never
    # silently share split files.
    return base / f"{spec.name}_seed{seed}_n{n_rows}.npz"


def _compute_folds(spec: DatasetSpec, y: pd.Series, seed: int) -> list[Fold]:
    n = len(y)
    indices = np.arange(n)
    stratify = spec.task is Task.CLASSIFICATION

    if spec.tier is SizeTier.LARGE:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=HOLDOUT_TEST_FRACTION,
            random_state=seed,
            stratify=y if stratify else None,
        )
        return [Fold(np.sort(train_idx), np.sort(test_idx))]

    splitter = (
        StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        if stratify
        else KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    )
    return [Fold(np.sort(train), np.sort(test)) for train, test in splitter.split(indices, y=y)]


def get_splits(
    spec: DatasetSpec,
    y: pd.Series,
    seed: int,
    splits_dir: Path | None = None,
) -> list[Fold]:
    """Return the persisted outer folds for (dataset, seed), computing them once."""
    path = _splits_path(spec, seed, len(y), splits_dir)
    if path.exists():
        with np.load(path) as data:
            n_folds = int(data["n_folds"])
            return [Fold(data[f"fold{i}_train"], data[f"fold{i}_test"]) for i in range(n_folds)]

    folds = _compute_folds(spec, y, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {"n_folds": np.array(len(folds))}
    for i, fold in enumerate(folds):
        arrays[f"fold{i}_train"] = fold.train_idx
        arrays[f"fold{i}_test"] = fold.test_idx
    np.savez_compressed(path, **arrays)  # type: ignore[arg-type]
    logger.info("Persisted %d fold(s) for %s (seed=%d) -> %s", len(folds), spec.name, seed, path)
    return folds
