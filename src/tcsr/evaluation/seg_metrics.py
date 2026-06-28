"""Segmentation quality metrics: mIoU, Dice, precision, recall, failure rate."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_seg_metrics(labels_df: pd.DataFrame, taus: list[float] = (0.5, 0.6, 0.75)) -> dict:
    metrics: dict = {"mean_iou": float(labels_df["iou"].mean())}

    for tau in taus:
        col = f"failure_tau{tau:.2f}".replace(".", "_")
        if col not in labels_df.columns:
            continue
        failure_rate = float(labels_df[col].mean())
        metrics[f"failure_rate_tau{tau}"] = failure_rate
        metrics[f"n_failures_tau{tau}"] = int(labels_df[col].sum())

    # Dice from IoU: Dice = 2*IoU / (1 + IoU)
    metrics["mean_dice"] = float(((2 * labels_df["iou"]) / (1 + labels_df["iou"])).mean())

    return metrics
