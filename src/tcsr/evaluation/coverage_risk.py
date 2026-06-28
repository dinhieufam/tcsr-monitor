"""Selective-prediction coverage–risk curve."""

from __future__ import annotations

import numpy as np


def coverage_risk_curve(
    scores: np.ndarray,
    labels: np.ndarray,
    n_points: int = 100,
) -> dict[str, np.ndarray]:
    """
    Sweep threshold from high to low. At each threshold:
      - coverage = fraction of frames where score >= threshold (we abstain below)
      - risk     = missed-failure rate among alarmed frames

    Returns arrays suitable for plotting.
    """
    thresholds = np.linspace(scores.min(), scores.max(), n_points)[::-1]
    coverages, risks = [], []

    for thr in thresholds:
        selected = scores >= thr
        coverage = selected.mean()
        if selected.sum() == 0 or labels.sum() == 0:
            risk = float("nan")
        else:
            # risk = fraction of true failures NOT alarmed (missed)
            missed = ((labels == 1) & ~selected).sum()
            risk = missed / labels.sum()
        coverages.append(coverage)
        risks.append(risk)

    return {
        "thresholds": thresholds,
        "coverage": np.array(coverages),
        "risk": np.array(risks),
    }
