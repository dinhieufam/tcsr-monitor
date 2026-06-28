"""Temporal features: frame-to-frame IoU, centroid jump, area delta."""

from __future__ import annotations

import numpy as np
from skimage import measure


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = (a & b).sum()
    union = (a | b).sum()
    return float(inter) / float(union) if union > 0 else 1.0


def mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    regions = measure.regionprops(measure.label(mask))
    if not regions:
        return 0.0, 0.0
    largest = max(regions, key=lambda r: r.area)
    cy, cx = largest.centroid
    return float(cy), float(cx)


def compute_temporal_features(
    curr_mask: np.ndarray,
    prev_mask: np.ndarray | None,
    curr_area: float,
    prev_area: float | None,
    history_ious: list[float],
) -> dict[str, float]:
    """
    curr_mask / prev_mask: (H, W) uint8 binary.
    history_ious: list of recent frame-to-frame IoUs (within-video).
    """
    if prev_mask is None:
        return {
            "temp_prev_iou": 1.0,
            "temp_centroid_jump": 0.0,
            "temp_area_delta": 0.0,
            "temp_rolling_iou_mean": 1.0,
            "temp_rolling_iou_std": 0.0,
        }

    prev_iou = mask_iou(curr_mask.astype(bool), prev_mask.astype(bool))
    cy_curr, cx_curr = mask_centroid(curr_mask)
    cy_prev, cx_prev = mask_centroid(prev_mask)
    h, w = curr_mask.shape
    centroid_jump = np.sqrt((cy_curr - cy_prev) ** 2 + (cx_curr - cx_prev) ** 2) / np.sqrt(h * w)
    area_delta = abs(curr_area - prev_area) if prev_area is not None else 0.0

    all_ious = history_ious + [prev_iou]
    return {
        "temp_prev_iou": prev_iou,
        "temp_centroid_jump": centroid_jump,
        "temp_area_delta": area_delta,
        "temp_rolling_iou_mean": float(np.mean(all_ious)),
        "temp_rolling_iou_std": float(np.std(all_ious)),
    }
