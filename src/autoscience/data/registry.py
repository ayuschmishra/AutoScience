"""Declarative dataset registry.

Every benchmark dataset is described by a :class:`DatasetSpec` with enough
metadata to download it, validate it, pick a size-tier-appropriate evaluation
protocol, and later analyze pipeline decisions against dataset
characteristics.

OpenML ids are pinned (never resolved by name at runtime) and the expected
shape is verified after download, so a silently changed upstream dataset
fails loudly instead of corrupting the benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Task(StrEnum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class SizeTier(StrEnum):
    """Drives split protocol, model gating, and HPO budgets."""

    SMALL = "small"  # n < 10_000
    MEDIUM = "medium"  # 10_000 <= n < 100_000
    LARGE = "large"  # n >= 100_000


@dataclass(frozen=True)
class SklearnSource:
    """Built-in or fetchable scikit-learn dataset; ``fetcher`` is the function name."""

    fetcher: str


@dataclass(frozen=True)
class OpenMLSource:
    data_id: int
    target: str


@dataclass(frozen=True)
class UrlCsvSource:
    """CSV (optionally inside a zip and/or gzipped) downloaded from a stable URL.

    Used for large UCI datasets that have no maintained OpenML mirror.
    ``target_col`` is a positional index for header-less files.
    """

    url: str
    target_col: int
    n_features: int
    zip_member: str | None = None
    has_header: bool = False


Source = SklearnSource | OpenMLSource | UrlCsvSource


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    task: Task
    source: Source
    expected_rows: int
    expected_features: int
    description: str
    license: str = "public"
    # Rows used for local development runs; None means use the full dataset
    # locally. Cloud `full` runs always use the complete dataset.
    local_subset_rows: int | None = None

    @property
    def tier(self) -> SizeTier:
        if self.expected_rows < 10_000:
            return SizeTier.SMALL
        if self.expected_rows < 100_000:
            return SizeTier.MEDIUM
        return SizeTier.LARGE


_UCI = "https://archive.ics.uci.edu/static/public"

REGISTRY: dict[str, DatasetSpec] = {
    spec.name: spec
    for spec in [
        # ------------------------------- small -------------------------------
        DatasetSpec(
            name="breast_cancer",
            task=Task.CLASSIFICATION,
            source=SklearnSource("load_breast_cancer"),
            expected_rows=569,
            expected_features=30,
            description="Wisconsin diagnostic breast cancer (binary).",
        ),
        DatasetSpec(
            name="wine",
            task=Task.CLASSIFICATION,
            source=SklearnSource("load_wine"),
            expected_rows=178,
            expected_features=13,
            description="Wine cultivar identification from chemical analysis (3 classes).",
        ),
        DatasetSpec(
            name="ionosphere",
            task=Task.CLASSIFICATION,
            source=OpenMLSource(data_id=59, target="class"),
            expected_rows=351,
            expected_features=34,
            description="Radar returns from the ionosphere (binary).",
        ),
        DatasetSpec(
            name="diabetes",
            task=Task.REGRESSION,
            source=SklearnSource("load_diabetes"),
            expected_rows=442,
            expected_features=10,
            description="Disease progression one year after baseline.",
        ),
        DatasetSpec(
            name="concrete_strength",
            task=Task.REGRESSION,
            source=OpenMLSource(data_id=44959, target="strength"),
            expected_rows=1030,
            expected_features=8,
            description="Concrete compressive strength from mixture composition.",
        ),
        DatasetSpec(
            name="energy_efficiency",
            task=Task.REGRESSION,
            source=OpenMLSource(data_id=44960, target="Y1"),
            expected_rows=768,
            expected_features=8,
            description="Building heating load from simulated geometry parameters.",
        ),
        # ------------------------------ medium -------------------------------
        DatasetSpec(
            name="phoneme",
            task=Task.CLASSIFICATION,
            source=OpenMLSource(data_id=1489, target="Class"),
            expected_rows=5404,
            expected_features=5,
            description="Nasal vs oral vowel discrimination from harmonic amplitudes.",
        ),
        DatasetSpec(
            name="spambase",
            task=Task.CLASSIFICATION,
            source=OpenMLSource(data_id=44, target="class"),
            expected_rows=4601,
            expected_features=57,
            description="Spam email detection from word/char frequencies (binary).",
        ),
        DatasetSpec(
            name="adult",
            task=Task.CLASSIFICATION,
            source=OpenMLSource(data_id=1590, target="class"),
            expected_rows=48842,
            expected_features=14,
            description="Census income prediction (binary, mixed numeric/categorical).",
        ),
        DatasetSpec(
            name="california_housing",
            task=Task.REGRESSION,
            source=SklearnSource("fetch_california_housing"),
            expected_rows=20640,
            expected_features=8,
            description="Median house value per California census block group.",
        ),
        DatasetSpec(
            name="superconductivity",
            task=Task.REGRESSION,
            source=OpenMLSource(data_id=44964, target="criticaltemp"),
            expected_rows=21263,
            expected_features=81,
            description="Superconductor critical temperature from material features.",
        ),
        # ------------------------------- large -------------------------------
        DatasetSpec(
            name="covertype",
            task=Task.CLASSIFICATION,
            source=OpenMLSource(data_id=1596, target="class"),
            expected_rows=581012,
            expected_features=54,
            description="Forest cover type from cartographic variables (7 classes).",
        ),
        DatasetSpec(
            name="year_prediction_msd",
            task=Task.REGRESSION,
            source=UrlCsvSource(
                url=f"{_UCI}/203/yearpredictionmsd.zip",
                zip_member="YearPredictionMSD.txt",
                target_col=0,
                n_features=90,
            ),
            expected_rows=515345,
            expected_features=90,
            description="Song release year from audio timbre features (Million Song Dataset).",
            local_subset_rows=200_000,
        ),
        DatasetSpec(
            name="higgs",
            task=Task.CLASSIFICATION,
            source=UrlCsvSource(
                url=f"{_UCI}/280/higgs.zip",
                zip_member="HIGGS.csv.gz",
                target_col=0,
                n_features=28,
            ),
            expected_rows=11000000,
            expected_features=28,
            description="Higgs boson signal vs background from kinematic features.",
            local_subset_rows=1_000_000,
        ),
        DatasetSpec(
            name="susy",
            task=Task.CLASSIFICATION,
            source=UrlCsvSource(
                url=f"{_UCI}/279/susy.zip",
                zip_member="SUSY.csv.gz",
                target_col=0,
                n_features=18,
            ),
            expected_rows=5000000,
            expected_features=18,
            description="Supersymmetric particle detection from kinematic features.",
            local_subset_rows=1_000_000,
        ),
    ]
}


def get_spec(name: str) -> DatasetSpec:
    try:
        return REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(REGISTRY))
        raise KeyError(f"Unknown dataset {name!r}. Available: {available}") from None


def names(tier: SizeTier | None = None, task: Task | None = None) -> list[str]:
    """Registry names, optionally filtered by tier and/or task."""
    return [
        s.name
        for s in REGISTRY.values()
        if (tier is None or s.tier == tier) and (task is None or s.task == task)
    ]
