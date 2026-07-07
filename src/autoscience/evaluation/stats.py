"""Statistical comparison machinery for the benchmark study.

- Per-dataset paired comparison (automated vs a baseline) across seeds:
  Wilcoxon signed-rank + rank-biserial effect size.
- Across-dataset comparison of modes: Friedman test on mean ranks with a
  Nemenyi post-hoc / critical-difference diagram (scikit-posthocs).

Conventions: scores are "higher is better" (use ``neg_rmse`` for regression).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sps


@dataclass
class PairedComparison:
    n: int
    median_delta: float  # a - b; positive means `a` better
    p_value: float
    effect_size: float  # rank-biserial correlation in [-1, 1]

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05


def paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> PairedComparison:
    """Wilcoxon signed-rank test on paired score vectors (a vs b)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) != len(b) or len(a) < 3:
        raise ValueError("paired_wilcoxon needs >= 3 paired observations")
    delta = a - b
    if np.allclose(delta, 0):
        return PairedComparison(len(a), 0.0, 1.0, 0.0)
    res = sps.wilcoxon(a, b, zero_method="wilcox", method="auto")
    nonzero = delta[~np.isclose(delta, 0)]
    ranks = sps.rankdata(np.abs(nonzero))
    r_plus = ranks[nonzero > 0].sum()
    total = ranks.sum()
    effect = float(2 * r_plus / total - 1)  # rank-biserial
    return PairedComparison(
        n=len(a),
        median_delta=float(np.median(delta)),
        p_value=float(res.pvalue),
        effect_size=effect,
    )


@dataclass
class FriedmanResult:
    p_value: float
    avg_ranks: pd.Series  # treatment -> average rank (1 = best)
    nemenyi_p: pd.DataFrame  # pairwise post-hoc p-values

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05


def friedman_nemenyi(scores: pd.DataFrame) -> FriedmanResult:
    """Friedman test + Nemenyi post-hoc.

    Args:
        scores: blocks x treatments (e.g. datasets x modes), higher is better.
    """
    import scikit_posthocs as sp

    if scores.shape[0] < 3 or scores.shape[1] < 3:
        raise ValueError("friedman_nemenyi needs >= 3 blocks and >= 3 treatments")
    _, p = sps.friedmanchisquare(*[scores[c] for c in scores.columns])
    ranks = scores.rank(axis=1, ascending=False).mean(axis=0)
    nemenyi = sp.posthoc_nemenyi_friedman(scores.to_numpy())
    nemenyi.index = scores.columns
    nemenyi.columns = scores.columns
    return FriedmanResult(p_value=float(p), avg_ranks=ranks, nemenyi_p=nemenyi)


def win_tie_loss(a: np.ndarray, b: np.ndarray, atol: float = 1e-9) -> tuple[int, int, int]:
    """Count of blocks where `a` beats / ties / loses to `b`."""
    delta = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return (
        int((delta > atol).sum()),
        int((np.abs(delta) <= atol).sum()),
        int((delta < -atol).sum()),
    )
