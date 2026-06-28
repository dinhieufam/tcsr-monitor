"""Confidence features: entropy, max prob, margin, fg-confidence."""

from __future__ import annotations

import numpy as np


_EPS = 1e-7


def _entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def compute_confidence_features(prob_map: np.ndarray) -> dict[str, float]:
    """
    prob_map: (H, W) float in [0, 1] — sigmoid output.
    Returns dict of scalar confidence features.
    """
    h = _entropy(prob_map)
    fg_mask = prob_map >= 0.5

    return {
        "conf_mean_fg_prob": float(prob_map[fg_mask].mean()) if fg_mask.any() else 0.0,
        "conf_max_prob": float(prob_map.max()),
        "conf_mean_entropy": float(h.mean()),
        "conf_max_entropy": float(h.max()),
        "conf_frac_uncertain": float((h > np.log(2) * 0.9).mean()),  # near-max entropy
        "conf_margin": float(np.abs(prob_map - 0.5).mean()),
        "conf_frac_fg": float(fg_mask.mean()),
    }
