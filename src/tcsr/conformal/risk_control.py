"""Conformal Risk Control / Learn-then-Test (optional calibration mode)."""

from __future__ import annotations

import numpy as np


def ltt_calibrate(
    scores: np.ndarray,
    labels: np.ndarray,
    epsilon: float = 0.1,
    delta: float = 0.1,
    candidate_thresholds: np.ndarray | None = None,
) -> dict:
    """
    Learn-then-Test: find the largest threshold such that the risk
    (missed failure rate) is controlled at epsilon with confidence 1-delta.

    Risk: R(lambda) = P(score < lambda | y=1)
    We want the largest lambda s.t. we can certify R(lambda) <= epsilon.
    Uses Hoeffding-style bound.
    """
    from scipy.stats import binom

    failure_idx = labels == 1
    n = failure_idx.sum()
    if n == 0:
        raise ValueError("No failure frames for risk control calibration.")

    failure_scores = scores[failure_idx]

    if candidate_thresholds is None:
        candidate_thresholds = np.linspace(0, 1, 200)

    best_threshold = 0.0
    for lam in sorted(candidate_thresholds):
        k = (failure_scores < lam).sum()
        # p-value: P(Binomial(n, epsilon) >= k)
        p_val = 1 - binom.cdf(k - 1, n, epsilon)
        if p_val > delta:
            best_threshold = lam
        else:
            break  # thresholds are sorted ascending; once rejected, stop

    missed_rate = float((failure_scores < best_threshold).mean())
    return {
        "threshold": float(best_threshold),
        "epsilon": epsilon,
        "delta": delta,
        "n_cal_failures": int(n),
        "realized_missed_rate": missed_rate,
        "method": "ltt",
    }
