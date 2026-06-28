"""Assign failure-mode labels to frames for per-stratum analysis.

Strata (only meaningful for failure frames, i.e. labels == 1):
  low_confidence  – model is uncertain: entropy score is high
  tracking_loss   – temporal discontinuity: frame-to-frame mask IoU is low
  ghost_mask      – fragmented prediction: many disconnected components
  other_fail      – silent, confident, coherent failure (hardest for entropy)
  non_failure     – labels == 0
"""

from __future__ import annotations

import numpy as np
import pandas as pd


STRATA = ["non_failure", "low_confidence", "tracking_loss", "ghost_mask", "other_fail"]


def assign_failure_strata(
    labels: np.ndarray,
    scores_entropy: np.ndarray,
    features_df: pd.DataFrame,
    entropy_thresh: float = 0.25,
    temporal_iou_thresh: float = 0.4,
    component_thresh: int = 3,
) -> np.ndarray:
    """
    Assign a stratum label string to every frame.

    Parameters
    ----------
    labels : (N,) binary failure labels
    scores_entropy : (N,) entropy baseline scores (higher = more uncertain)
    features_df : DataFrame with at least columns:
                  'temp_prev_iou' (temporal IoU to previous frame)
                  'shape_n_components' (number of connected components)
    entropy_thresh : entropy score above which = 'low_confidence'
    temporal_iou_thresh : prev-frame IoU below which = 'tracking_loss'
    component_thresh : component count above which = 'ghost_mask'

    Returns
    -------
    stratum : (N,) array of strings
    """
    n = len(labels)
    stratum = np.full(n, "non_failure", dtype=object)
    fail_idx = labels == 1

    if fail_idx.sum() == 0:
        return stratum

    ent = scores_entropy[fail_idx]
    temp_iou = features_df["temp_prev_iou"].values[fail_idx] if "temp_prev_iou" in features_df.columns else np.ones(fail_idx.sum())
    n_comp = features_df["shape_n_components"].values[fail_idx] if "shape_n_components" in features_df.columns else np.zeros(fail_idx.sum())

    sub = np.full(fail_idx.sum(), "other_fail", dtype=object)
    # Priority order: ghost_mask > tracking_loss > low_confidence > other_fail
    sub[ent > entropy_thresh] = "low_confidence"
    sub[temp_iou < temporal_iou_thresh] = "tracking_loss"
    sub[n_comp > component_thresh] = "ghost_mask"

    stratum[fail_idx] = sub
    return stratum


def per_stratum_auroc(
    labels: np.ndarray,
    scores_monitor: np.ndarray,
    scores_entropy: np.ndarray,
    strata: np.ndarray,
) -> dict:
    """
    Compute AUROC for monitor and entropy within each failure stratum.

    A stratum comparison only makes sense when it contains both positives and
    negatives (here: failure frames vs non-failure frames in the same subset).
    For per-stratum evaluation we compare failed frames in that stratum vs ALL
    non-failure frames, which gives a realistic operating condition.
    """
    from sklearn.metrics import roc_auc_score

    results: dict = {}
    non_fail_idx = labels == 0

    for s in STRATA[1:]:  # skip non_failure
        stratum_pos = strata == s
        if stratum_pos.sum() < 2:
            results[s] = {"n": int(stratum_pos.sum()), "monitor_auroc": float("nan"), "entropy_auroc": float("nan")}
            continue

        # Positive: frames in this stratum (failures). Negative: all non-failures.
        mask = stratum_pos | non_fail_idx
        y = labels[mask]
        sm = scores_monitor[mask]
        se = scores_entropy[mask]

        if y.sum() == 0 or y.sum() == mask.sum():
            results[s] = {"n": int(stratum_pos.sum()), "monitor_auroc": float("nan"), "entropy_auroc": float("nan")}
            continue

        results[s] = {
            "n": int(stratum_pos.sum()),
            "monitor_auroc": float(roc_auc_score(y, sm)),
            "entropy_auroc": float(roc_auc_score(y, se)),
        }

    return results
