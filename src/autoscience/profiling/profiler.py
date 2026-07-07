"""Computational-efficiency instrumentation.

``profile()`` measures wall time and true peak process memory (RSS sampled by
a background thread, so C/C++ allocations from XGBoost/torch are captured too,
unlike tracemalloc). ``model_size_mb`` and ``inference_latency_ms_per_1k``
complete the efficiency panel logged for every run.
"""

from __future__ import annotations

import pickle
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import psutil

_SAMPLE_INTERVAL_S = 0.02


@dataclass
class ProfileResult:
    seconds: float = 0.0
    peak_rss_mb: float = 0.0
    baseline_rss_mb: float = 0.0

    @property
    def peak_increase_mb(self) -> float:
        return max(self.peak_rss_mb - self.baseline_rss_mb, 0.0)


class _RssSampler(threading.Thread):
    def __init__(self, process: psutil.Process) -> None:
        super().__init__(daemon=True)
        self.process = process
        self.peak = 0.0
        self._stop_event = threading.Event()

    def run(self) -> None:
        while not self._stop_event.is_set():
            self.peak = max(self.peak, self.process.memory_info().rss)
            self._stop_event.wait(_SAMPLE_INTERVAL_S)

    def stop(self) -> None:
        self._stop_event.set()


@contextmanager
def profile() -> Iterator[ProfileResult]:
    """Measure wall time and peak RSS of the enclosed block."""
    result = ProfileResult()
    process = psutil.Process()
    result.baseline_rss_mb = process.memory_info().rss / 1e6
    sampler = _RssSampler(process)
    sampler.start()
    start = time.perf_counter()
    try:
        yield result
    finally:
        result.seconds = time.perf_counter() - start
        sampler.stop()
        sampler.join(timeout=1.0)
        result.peak_rss_mb = max(sampler.peak / 1e6, result.baseline_rss_mb)


def model_size_mb(estimator: Any) -> float:
    """Serialized size of the fitted estimator (the deployment footprint)."""
    return len(pickle.dumps(estimator)) / 1e6


def inference_latency_ms_per_1k(estimator: Any, x: pd.DataFrame, n_repeats: int = 3) -> float:
    """Median predict latency, normalized to milliseconds per 1000 rows."""
    timings = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        estimator.predict(x)
        timings.append(time.perf_counter() - start)
    per_row = float(np.median(timings)) / len(x)
    return per_row * 1000 * 1000  # -> ms per 1k rows
