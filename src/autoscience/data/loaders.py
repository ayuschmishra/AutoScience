"""Dataset download, caching, and loading.

Every dataset is downloaded once, validated against its spec, downcast to
memory-efficient dtypes, and cached as a single parquet file whose sha256 is
recorded. All later loads read only the parquet cache, so experiments never
depend on upstream availability and always see byte-identical data.

Large-tier datasets can be loaded as a fixed-seed subset for local
development (``full=False``); the subset seed is a module constant that is
independent of experiment seeds, so the local subset is one deterministic
dataset, not one per experiment.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import shutil
import tempfile
import urllib.request
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import IO, cast

import numpy as np
import pandas as pd
import psutil
import pyarrow.parquet as pq

from autoscience.data import schema
from autoscience.data.registry import (
    DatasetSpec,
    OpenMLSource,
    SklearnSource,
    Task,
    UrlCsvSource,
    get_spec,
)
from autoscience.utils import paths

logger = logging.getLogger(__name__)

TARGET_COL = "__target__"
# Fixed seed for local development subsets — deliberately NOT an experiment
# seed: the local subset must be the same dataset across all experiments.
SUBSET_SEED = 20260707
# Refuse to load a dataset whose estimated footprint exceeds this share of
# currently available RAM; callers should use iter_batches() instead.
MEMORY_FRACTION_LIMIT = 0.6


@dataclass
class LoadedDataset:
    spec: DatasetSpec
    x: pd.DataFrame
    y: pd.Series
    is_subset: bool

    @property
    def n_rows(self) -> int:
        return len(self.x)


def _raw_dir(data_dir: Path | None) -> Path:
    return (data_dir or paths.DATA_DIR) / "raw"


def _cache_path(name: str, data_dir: Path | None) -> Path:
    return _raw_dir(data_dir) / f"{name}.parquet"


def _checksum_file(data_dir: Path | None) -> Path:
    return _raw_dir(data_dir) / "checksums.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_checksum(name: str, path: Path, data_dir: Path | None) -> None:
    checksum_path = _checksum_file(data_dir)
    checksums: dict[str, str] = {}
    if checksum_path.exists():
        checksums = json.loads(checksum_path.read_text())
    checksums[name] = _sha256(path)
    checksum_path.write_text(json.dumps(checksums, indent=2, sort_keys=True))


def verify_checksum(name: str, data_dir: Path | None = None) -> bool:
    """True if the cached parquet matches its recorded sha256."""
    checksum_path = _checksum_file(data_dir)
    cache = _cache_path(name, data_dir)
    if not (checksum_path.exists() and cache.exists()):
        return False
    recorded = json.loads(checksum_path.read_text()).get(name)
    return recorded is not None and recorded == _sha256(cache)


def _downcast(x: pd.DataFrame) -> pd.DataFrame:
    """Shrink to float32/int32/category — halves memory on large datasets."""
    out = x.copy()
    for col in out.columns:
        dtype = out[col].dtype
        if pd.api.types.is_float_dtype(dtype):
            out[col] = out[col].astype(np.float32)
        elif pd.api.types.is_integer_dtype(dtype):
            out[col] = pd.to_numeric(out[col], downcast="integer")
        elif isinstance(dtype, pd.CategoricalDtype):
            pass
        elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
            out[col] = out[col].astype("category")
    return out


# --------------------------------------------------------------------------
# Source-specific fetchers: return (features, target) for the FULL dataset.
# --------------------------------------------------------------------------


def _fetch_sklearn(source: SklearnSource) -> tuple[pd.DataFrame, pd.Series]:
    import sklearn.datasets

    fetcher = getattr(sklearn.datasets, source.fetcher)
    bunch = fetcher(as_frame=True)
    return bunch.data, bunch.target


def _fetch_openml(source: OpenMLSource) -> tuple[pd.DataFrame, pd.Series]:
    from sklearn.datasets import fetch_openml

    bunch = fetch_openml(
        data_id=source.data_id,
        as_frame=True,
        target_column=source.target,
        parser="auto",
    )
    return bunch.data, bunch.target


def _fetch_url_csv(source: UrlCsvSource) -> tuple[pd.DataFrame, pd.Series]:
    with tempfile.TemporaryDirectory() as tmp:
        download_path = Path(tmp) / "download.bin"
        logger.info("Downloading %s ...", source.url)
        with urllib.request.urlopen(source.url) as resp, download_path.open("wb") as out:
            shutil.copyfileobj(resp, out, length=1 << 20)
        logger.info("Downloaded %.1f MB", download_path.stat().st_size / 1e6)

        if source.zip_member is not None:
            with zipfile.ZipFile(download_path) as zf, zf.open(source.zip_member) as raw:
                if source.zip_member.endswith(".gz"):
                    with gzip.open(raw) as stream:
                        df = _read_headerless_csv(stream, source)
                else:
                    df = _read_headerless_csv(raw, source)
        else:
            df = _read_headerless_csv(download_path, source)

    y = df.iloc[:, source.target_col]
    x = df.drop(columns=df.columns[source.target_col])
    x.columns = [f"f{i}" for i in range(x.shape[1])]
    return x, y.rename("target")


def _read_headerless_csv(
    source_file: IO[bytes] | gzip.GzipFile | Path, source: UrlCsvSource
) -> pd.DataFrame:
    # GzipFile/ZipExtFile satisfy pandas' buffer protocol at runtime but not
    # in the stubs, hence the cast.
    return pd.read_csv(
        cast("IO[bytes]", source_file),
        header=0 if source.has_header else None,
        dtype="float32",
    )


def download(name: str, data_dir: Path | None = None, force: bool = False) -> Path:
    """Ensure the dataset is cached as validated parquet; return the cache path."""
    spec = get_spec(name)
    cache = _cache_path(name, data_dir)
    if cache.exists() and not force:
        return cache

    if isinstance(spec.source, SklearnSource):
        x, y = _fetch_sklearn(spec.source)
    elif isinstance(spec.source, OpenMLSource):
        x, y = _fetch_openml(spec.source)
    else:
        x, y = _fetch_url_csv(spec.source)

    report = schema.validate(spec, x, y)
    if not report.ok:
        raise ValueError(f"Dataset {name!r} failed validation: {'; '.join(report.errors)}")

    x = _downcast(x)
    y = y.astype("category") if spec.task is Task.CLASSIFICATION else y.astype(np.float32)

    df = x.copy()
    df[TARGET_COL] = y.to_numpy()
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    _record_checksum(name, cache, data_dir)
    logger.info("Cached %s: %d rows x %d features -> %s", name, len(df), x.shape[1], cache)
    return cache


def _estimated_memory_bytes(cache: Path) -> int:
    meta = pq.read_metadata(cache)  # type: ignore[no-untyped-call]
    # float32-dominated frames: ~4 bytes/cell plus pandas overhead.
    return int(meta.num_rows * meta.num_columns * 4 * 1.7)


def _assert_fits_in_memory(name: str, cache: Path) -> None:
    estimated = _estimated_memory_bytes(cache)
    available = psutil.virtual_memory().available
    if estimated > available * MEMORY_FRACTION_LIMIT:
        raise MemoryError(
            f"Loading {name!r} needs ~{estimated / 1e9:.1f} GB but only "
            f"{available / 1e9:.1f} GB RAM is available. Use iter_batches() "
            f"(partial_fit path) or the local subset (full=False)."
        )


def _stratified_subset_idx(y: pd.Series, n: int, task: Task) -> np.ndarray:
    rng = np.random.default_rng(SUBSET_SEED)
    if task is Task.CLASSIFICATION:
        # Proportional allocation per class keeps the class balance intact.
        parts = []
        frac = n / len(y)
        for _, grp in y.groupby(y, observed=True):
            take = max(1, round(len(grp) * frac))
            parts.append(rng.choice(grp.index.to_numpy(), size=take, replace=False))
        idx = np.concatenate(parts)[:n]
    else:
        idx = rng.choice(len(y), size=n, replace=False)
    return np.sort(idx)


def load_dataset(name: str, *, full: bool = False, data_dir: Path | None = None) -> LoadedDataset:
    """Load a dataset from the parquet cache (downloading it on first use).

    Args:
        name: Registry name.
        full: If True, never subsample — used by cloud benchmark runs.
        data_dir: Override the data directory (tests).
    """
    spec = get_spec(name)
    cache = download(name, data_dir=data_dir)
    _assert_fits_in_memory(name, cache)

    df = pd.read_parquet(cache)
    y = df[TARGET_COL]
    x = df.drop(columns=[TARGET_COL])

    is_subset = False
    subset_rows = spec.local_subset_rows
    if not full and subset_rows is not None and len(df) > subset_rows:
        idx = _stratified_subset_idx(y, subset_rows, spec.task)
        x = x.iloc[idx].reset_index(drop=True)
        y = y.iloc[idx].reset_index(drop=True)
        is_subset = True
        logger.info("Loaded %s local subset: %d of %d rows", name, len(x), len(df))

    if spec.task is Task.CLASSIFICATION:
        y = y.astype("category")
    return LoadedDataset(spec=spec, x=x, y=y, is_subset=is_subset)


def iter_batches(
    name: str, batch_size: int = 65536, data_dir: Path | None = None
) -> Iterator[tuple[pd.DataFrame, pd.Series]]:
    """Stream (features, target) batches from the parquet cache.

    Out-of-core path for ``partial_fit``-capable models on datasets that fail
    the in-memory guard.
    """
    cache = download(name, data_dir=data_dir)
    parquet = pq.ParquetFile(cache)  # type: ignore[no-untyped-call]
    for batch in parquet.iter_batches(batch_size=batch_size):  # type: ignore[no-untyped-call]
        df = batch.to_pandas()
        yield df.drop(columns=[TARGET_COL]), df[TARGET_COL]
