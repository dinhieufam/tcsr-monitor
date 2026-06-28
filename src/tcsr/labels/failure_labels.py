"""Compute per-frame failure labels y_t = 1[IoU(pred, gt) < tau]."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from tcsr.utils.io import save_parquet
from tcsr.utils.logging import get_logger

log = get_logger(__name__)


def mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b = pred > 0
    gt_b = gt > 0
    inter = (pred_b & gt_b).sum()
    union = (pred_b | gt_b).sum()
    return float(inter) / float(union) if union > 0 else 1.0


def compute_failure_labels(
    manifest_df: pd.DataFrame,
    pred_dir: Path,
    output_path: Path,
    taus: list[float] = (0.5, 0.6, 0.75),
) -> pd.DataFrame:
    rows = []
    for _, row in tqdm(manifest_df.iterrows(), total=len(manifest_df), desc="Labels"):
        pred_mask_path = pred_dir / row.frame_id / "mask.png"
        pred = cv2.imread(str(pred_mask_path), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(row.mask_path, cv2.IMREAD_GRAYSCALE)

        if pred is None or gt is None:
            log.warning("Missing pred or gt for %s", row.frame_id)
            iou_val = 0.0
        else:
            if pred.shape != gt.shape:
                gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
            iou_val = mask_iou(pred, gt)

        entry: dict = {"frame_id": row.frame_id, "iou": iou_val}
        for tau in taus:
            entry[f"failure_tau{tau:.2f}".replace(".", "_")] = int(iou_val < tau)
        rows.append(entry)

    df = pd.DataFrame(rows)
    save_parquet(df, output_path)
    log.info(
        "Labels saved: %s | failure rates: %s",
        output_path,
        {f"tau={t}": f"{df[f'failure_tau{t:.2f}'.replace('.','_')].mean():.3f}" for t in taus},
    )
    return df
