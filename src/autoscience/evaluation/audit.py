"""Reproducibility audit: run the identical config twice, diff the metrics.

Wall-clock and memory metrics legitimately vary between runs; every
model-quality metric (scores, calibration, uncertainty) must reproduce to
within ``tolerance`` (0.0 by default — sklearn/XGBoost/torch on CPU are
bit-deterministic under our seeding).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from autoscience.hpo.runner import run_experiment

_NON_DETERMINISTIC_SUBSTRINGS = ("seconds", "peak_mb", "latency")


def _quality_metrics(aggregated: dict[str, float]) -> dict[str, float]:
    return {
        k: v
        for k, v in aggregated.items()
        if not any(s in k for s in _NON_DETERMINISTIC_SUBSTRINGS)
    }


@dataclass
class ReproReport:
    dataset: str
    model: str
    seed: int
    max_abs_delta: float
    deltas: dict[str, float]
    tolerance: float

    @property
    def ok(self) -> bool:
        return self.max_abs_delta <= self.tolerance


def repro_audit(
    dataset: str,
    model: str,
    *,
    seed: int = 42,
    budget_profile: str = "smoke",
    mode: str = "automated",
    tolerance: float = 0.0,
    tracking_uri: str | None = None,
    **run_kwargs: Any,
) -> ReproReport:
    """Run (dataset, model, seed) twice and report metric deltas."""
    results = [
        run_experiment(
            dataset,
            model,
            seed=seed,
            budget_profile=budget_profile,
            mode=mode,
            tracking_uri=tracking_uri,
            extra_tags={"audit": "repro", "audit_rep": str(i)},
            **run_kwargs,
        )
        for i in range(2)
    ]
    a, b = (_quality_metrics(r.aggregated) for r in results)
    deltas = {k: abs(a[k] - b[k]) for k in a if k in b}
    max_delta = max(deltas.values()) if deltas else float("inf")
    return ReproReport(
        dataset=dataset,
        model=model,
        seed=seed,
        max_abs_delta=max_delta,
        deltas=deltas,
        tolerance=tolerance,
    )
