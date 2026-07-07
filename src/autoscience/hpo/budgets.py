"""HPO budget profiles: trial counts and wall-clock limits per size tier.

Local vs cloud is a config switch: ``smoke`` proves the plumbing in minutes,
``local`` is a laptop overnighter, ``full`` is the cloud benchmark budget.
Fairness note for the study: automated-vs-baseline comparisons always report
the compute cost alongside the score.
"""

from __future__ import annotations

from dataclasses import dataclass

from autoscience.data.registry import SizeTier


@dataclass(frozen=True)
class Budget:
    n_trials: int
    timeout_s: int
    inner_folds: int  # 1 means a single validation split


PROFILES: dict[str, dict[SizeTier, Budget]] = {
    "smoke": {
        SizeTier.SMALL: Budget(n_trials=8, timeout_s=300, inner_folds=3),
        SizeTier.MEDIUM: Budget(n_trials=6, timeout_s=600, inner_folds=3),
        SizeTier.LARGE: Budget(n_trials=4, timeout_s=900, inner_folds=1),
    },
    "local": {
        SizeTier.SMALL: Budget(n_trials=50, timeout_s=1800, inner_folds=3),
        SizeTier.MEDIUM: Budget(n_trials=30, timeout_s=3600, inner_folds=3),
        SizeTier.LARGE: Budget(n_trials=15, timeout_s=7200, inner_folds=1),
    },
    "full": {
        SizeTier.SMALL: Budget(n_trials=150, timeout_s=7200, inner_folds=3),
        SizeTier.MEDIUM: Budget(n_trials=100, timeout_s=14400, inner_folds=3),
        SizeTier.LARGE: Budget(n_trials=40, timeout_s=28800, inner_folds=1),
    },
}


def get_budget(profile: str, tier: SizeTier) -> Budget:
    try:
        return PROFILES[profile][tier]
    except KeyError:
        raise KeyError(f"Unknown budget profile {profile!r}; use one of {list(PROFILES)}") from None
