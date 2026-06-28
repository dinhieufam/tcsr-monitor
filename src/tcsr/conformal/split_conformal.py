"""Split conformal calibration: bound P(r_t < tau | y_t=1) <= alpha."""

from __future__ import annotations

import numpy as np


def calibrate_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.1,
    score_type: str = "one_minus_prob",
) -> dict:
    """
    Calibrate a detection threshold on the calibration split.

    The nonconformity score for a failure frame is s_t = 1 - r_t
    (where r_t is the monitor's risk score). We want to set threshold
    lambda such that P(s_t > lambda | y_t = 1) <= alpha, i.e. we bound
    missed failures.

    Exchangeability assumption: cal frames are exchangeable with test frames
    (holds within-dataset; does NOT hold under cross-dataset shift).

    Returns dict with threshold and realized coverage on calibration data.
    """
    failure_idx = labels == 1
    if failure_idx.sum() == 0:
        raise ValueError("No failure frames in calibration split.")

    if score_type == "one_minus_prob":
        nonconf = 1.0 - scores  # lower risk score → higher nonconformity
    elif score_type == "prob":
        nonconf = scores
    else:
        raise ValueError(f"Unknown score_type: {score_type}")

    failure_nonconf = nonconf[failure_idx]
    n = len(failure_nonconf)
    # Conformal quantile: ceil((n+1)(1-alpha))/n adjusted
    level = np.ceil((n + 1) * (1 - alpha)) / n
    level = min(level, 1.0)
    threshold_nonconf = np.quantile(failure_nonconf, level)

    # Convert back to risk-score threshold
    if score_type == "one_minus_prob":
        threshold = 1.0 - threshold_nonconf
    else:
        threshold = threshold_nonconf

    # Realized missed-failure rate on cal
    cal_missed = float((scores[failure_idx] < threshold).mean())

    return {
        "threshold": float(threshold),
        "alpha": alpha,
        "n_cal_failures": int(failure_idx.sum()),
        "realized_missed_rate": cal_missed,
        "score_type": score_type,
    }


def apply_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Return binary alarm predictions (1 = alarm, 0 = no alarm)."""
    return (scores >= threshold).astype(int)
