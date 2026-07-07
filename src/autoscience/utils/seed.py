"""Global determinism control.

Every entry point (CLI run, HPO trial, benchmark worker) must call
:func:`set_global_seed` before touching any stochastic library. Torch is
imported lazily so that light-weight commands (``autoscience data list``)
don't pay its import cost.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np

logger = logging.getLogger(__name__)


def set_global_seed(seed: int, *, deterministic_torch: bool = True) -> None:
    """Seed ``random``, ``numpy`` and (if already imported or importable) ``torch``.

    Args:
        seed: The seed applied to all libraries.
        deterministic_torch: Additionally request deterministic torch kernels.
            Uses ``warn_only=True`` because a few ops have no deterministic
            implementation; those emit a warning instead of crashing, and the
            reproducibility audit (Phase 5) verifies actual run-to-run variance.
    """
    random.seed(seed)
    # Libraries like sklearn draw from NumPy's *global* state when
    # random_state=None, so the legacy global seed must be set too.
    np.random.seed(seed)  # noqa: NPY002
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard dep, but stay robust
        logger.debug("torch not installed; skipping torch seeding")
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Required for deterministic cuBLAS matmuls (no-op without CUDA).
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if deterministic_torch:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def spawn_seeds(base_seed: int, n: int) -> list[int]:
    """Derive ``n`` independent child seeds from ``base_seed``.

    Used by multi-seed protocols so that per-seed runs are decorrelated but
    fully determined by the experiment's base seed.
    """
    rng = np.random.default_rng(base_seed)
    return [int(s) for s in rng.integers(0, 2**31 - 1, size=n)]
