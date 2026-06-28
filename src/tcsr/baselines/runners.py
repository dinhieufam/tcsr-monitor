"""Baseline risk-score runners. Each emits a per-frame risk score in [0, 1]."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from tcsr.utils.io import load_npz, save_parquet
from tcsr.utils.logging import get_logger

log = get_logger(__name__)

_EPS = 1e-7


def _entropy_score(prob_map: np.ndarray) -> float:
    p = np.clip(prob_map, _EPS, 1 - _EPS)
    h = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    return float(h.mean())


def _max_softmax_score(prob_map: np.ndarray) -> float:
    return float(1.0 - prob_map.max())


def _tta_variance_score(npz: dict) -> float | None:
    if "tta_variance" in npz:
        return float(npz["tta_variance"].mean())
    return None


def _temporal_heuristic_score(curr_mask: np.ndarray, prev_mask: np.ndarray | None) -> float:
    if prev_mask is None:
        return 0.0
    inter = (curr_mask.astype(bool) & prev_mask.astype(bool)).sum()
    union = (curr_mask.astype(bool) | prev_mask.astype(bool)).sum()
    iou = inter / union if union > 0 else 1.0
    return 1.0 - iou


class TemperatureScaling:
    """Calibrate temperature on cal split, then rescale entropy for test."""

    def __init__(self):
        self.temperature: float = 1.0

    def fit(self, logits_list: list[np.ndarray], labels: np.ndarray) -> "TemperatureScaling":
        from scipy.optimize import minimize_scalar
        from scipy.special import expit

        all_logits = np.concatenate([l.ravel() for l in logits_list])
        all_labels = np.repeat(labels, [l.size for l in logits_list])

        def nll(t):
            p = expit(all_logits / t)
            p = np.clip(p, _EPS, 1 - _EPS)
            return -np.mean(all_labels * np.log(p) + (1 - all_labels) * np.log(1 - p))

        result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
        self.temperature = float(result.x)
        log.info("Temperature scaling: T=%.4f", self.temperature)
        return self

    def score(self, logit: float) -> float:
        from scipy.special import expit
        p = expit(logit / self.temperature)
        p = np.clip(p, _EPS, 1 - _EPS)
        return float(-(p * np.log(p) + (1 - p) * np.log(1 - p)))


def run_all_baselines(
    manifest_df: pd.DataFrame,
    pred_dir: Path,
    output_path: Path,
    temperature_scaler: "TemperatureScaling | None" = None,
    shift_detector=None,
) -> pd.DataFrame:
    rows = []
    prev_mask = None
    prev_video_id = None

    for _, row in tqdm(manifest_df.iterrows(), total=len(manifest_df), desc="Baselines"):
        if row.video_id != prev_video_id:
            prev_mask = None
            prev_video_id = row.video_id

        frame_dir = pred_dir / row.frame_id
        npz = load_npz(frame_dir / "probs.npz")
        prob_map = npz["probs"].astype(np.float32)
        curr_mask = (prob_map >= 0.5).astype(np.uint8)

        entry: dict = {"frame_id": row.frame_id}
        entry["bl_max_softmax"] = _max_softmax_score(prob_map)
        entry["bl_entropy"] = _entropy_score(prob_map)
        entry["bl_temporal_heuristic"] = _temporal_heuristic_score(curr_mask, prev_mask)

        tta_score = _tta_variance_score(npz)
        if tta_score is not None:
            entry["bl_tta_variance"] = tta_score

        if temperature_scaler is not None:
            logit_val = float(np.log(prob_map.mean() / (1 - prob_map.mean() + _EPS) + _EPS))
            entry["bl_temperature_scaling"] = temperature_scaler.score(logit_val)

        if shift_detector is not None:
            entry["bl_feature_distance_ood"] = float(shift_detector.score(prob_map))

        rows.append(entry)
        prev_mask = curr_mask

    df = pd.DataFrame(rows)
    save_parquet(df, output_path)
    log.info("Baseline scores saved to %s", output_path)
    return df
