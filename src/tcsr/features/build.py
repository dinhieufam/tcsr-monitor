"""Assemble per-frame feature table (parquet) from all feature groups."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from tcsr.features.confidence import compute_confidence_features
from tcsr.features.quality import compute_quality_features
from tcsr.features.shape import compute_shape_features
from tcsr.features.temporal import compute_temporal_features
from tcsr.utils.io import load_npz, save_parquet
from tcsr.utils.logging import get_logger

log = get_logger(__name__)


def build_feature_table(
    manifest_df: pd.DataFrame,
    pred_dir: Path,
    output_path: Path,
    feature_groups: dict[str, bool],
    shift_detector=None,
    temporal_window: int = 5,
) -> pd.DataFrame:
    rows = []
    # Process per video to maintain temporal ordering and avoid cross-video leakage
    for video_id, vdf in manifest_df.groupby("video_id"):
        vdf = vdf.sort_values("frame_idx_in_video").reset_index(drop=True)

        prev_mask = None
        prev_area = None
        iou_history: list[float] = []

        for _, row in tqdm(vdf.iterrows(), total=len(vdf), desc=f"Features {video_id}", leave=False):
            feat: dict = {"frame_id": row.frame_id}

            npz = load_npz(pred_dir / row.frame_id / "probs.npz")
            prob_map = npz["probs"].astype(np.float32)
            curr_mask = (prob_map >= 0.5).astype(np.uint8)
            curr_area = float(curr_mask.mean())

            if feature_groups.get("confidence", False):
                feat.update(compute_confidence_features(prob_map))

            if feature_groups.get("shape", False):
                feat.update(compute_shape_features(curr_mask))

            if feature_groups.get("temporal", False):
                window_ious = iou_history[-temporal_window:]
                feat.update(compute_temporal_features(
                    curr_mask, prev_mask, curr_area, prev_area, window_ious
                ))
                if prev_mask is not None:
                    from tcsr.features.temporal import mask_iou
                    iou_val = mask_iou(curr_mask.astype(bool), prev_mask.astype(bool))
                    iou_history.append(iou_val)

            if feature_groups.get("quality", False):
                bgr = cv2.imread(row.frame_path)
                if bgr is not None:
                    feat.update(compute_quality_features(bgr))

            if feature_groups.get("shift", False) and shift_detector is not None:
                # shift scores are added via a separate pass with the model
                pass

            prev_mask = curr_mask
            prev_area = curr_area
            rows.append(feat)

    df = pd.DataFrame(rows)
    save_parquet(df, output_path)
    log.info("Feature table saved: %s (%d rows, %d cols)", output_path, len(df), len(df.columns))
    return df
