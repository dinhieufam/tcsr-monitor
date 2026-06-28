"""Monitor detection metrics: AUROC, AUPRC, recall@cov, FA/min, ECE, Brier."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


def _ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        acc = labels[mask].mean()
        conf = probs[mask].mean()
        ece += mask.sum() * abs(acc - conf)
    return ece / len(probs)


def compute_monitor_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    fps: float = 25.0,
    coverage_target: float = 0.9,
    threshold: float | None = None,
) -> dict:
    """
    scores: per-frame risk score in [0,1] (higher = more likely failure).
    labels: binary failure labels.
    fps: frames per second (for FA/min).
    """
    metrics: dict = {}

    if labels.sum() == 0 or labels.sum() == len(labels):
        metrics["auroc"] = float("nan")
        metrics["auprc"] = float("nan")
    else:
        metrics["auroc"] = float(roc_auc_score(labels, scores))
        metrics["auprc"] = float(average_precision_score(labels, scores))

    metrics["brier"] = float(brier_score_loss(labels, scores))
    metrics["ece"] = _ece(scores, labels)

    # Recall @ coverage_target (keep top coverage_target fraction of frames)
    n = len(scores)
    top_k = int(np.ceil(n * coverage_target))
    top_idx = np.argsort(scores)[-top_k:]
    alarm_mask = np.zeros(n, dtype=bool)
    alarm_mask[top_idx] = True
    failure_idx = labels == 1
    metrics[f"recall_at_cov{int(coverage_target*100)}"] = float(
        alarm_mask[failure_idx].mean() if failure_idx.any() else float("nan")
    )

    # False alarms per minute (at given threshold)
    if threshold is not None:
        alarms = (scores >= threshold).astype(int)
        n_false_alarms = int(((alarms == 1) & (labels == 0)).sum())
        duration_minutes = n / (fps * 60)
        metrics["fa_per_min"] = n_false_alarms / max(duration_minutes, 1e-6)
        metrics["recall"] = float(alarms[failure_idx].mean()) if failure_idx.any() else float("nan")
        metrics["precision"] = float(
            labels[alarms == 1].mean() if alarms.sum() > 0 else float("nan")
        )

    return metrics


def bootstrap_ci(
    scores: np.ndarray,
    labels: np.ndarray,
    n_bootstraps: int = 1000,
    ci_level: float = 0.95,
    rng_seed: int = 0,
) -> dict:
    """Bootstrap CI on AUROC and AUPRC (resample frames with replacement)."""
    rng = np.random.default_rng(rng_seed)
    aurocs: list[float] = []
    auprcs: list[float] = []
    n = len(labels)
    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        s, lb = scores[idx], labels[idx]
        if lb.sum() == 0 or lb.sum() == n:
            continue
        aurocs.append(float(roc_auc_score(lb, s)))
        auprcs.append(float(average_precision_score(lb, s)))

    if len(aurocs) < 10:
        return {
            "auroc_ci_lo": float("nan"),
            "auroc_ci_hi": float("nan"),
            "auprc_ci_lo": float("nan"),
            "auprc_ci_hi": float("nan"),
            "n_bootstraps": len(aurocs),
        }

    alpha = 1.0 - ci_level
    lo, hi = float(np.percentile(aurocs, 100 * alpha / 2)), float(np.percentile(aurocs, 100 * (1 - alpha / 2)))
    lo_p, hi_p = float(np.percentile(auprcs, 100 * alpha / 2)), float(np.percentile(auprcs, 100 * (1 - alpha / 2)))
    return {
        "auroc_ci_lo": lo,
        "auroc_ci_hi": hi,
        "auprc_ci_lo": lo_p,
        "auprc_ci_hi": hi_p,
        "n_bootstraps": len(aurocs),
    }
