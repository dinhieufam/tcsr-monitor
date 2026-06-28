#!/usr/bin/env python
"""
Stage 4c: Compute failure labels for corrupted training predictions.

Reads corrupted_train_manifest.parquet (produced by 03d), computes
IoU(corrupted_pred, original_gt) for each (frame, corruption_type, severity),
and writes corrupted_train_labels.parquet.

columns: frame_id, video_id, corruption_type, severity, iou,
         failure_tau0_50, failure_tau0_60, failure_tau0_75
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tcsr.labels.failure_labels import mask_iou
from tcsr.utils.logging import get_logger

log = get_logger(__name__)

PROCESSED_DIR = Path("data/processed/endovis2017")
CORRUPTED_TRAIN_MANIFEST_PATH = PROCESSED_DIR / "corrupted_train_manifest.parquet"
CORRUPTED_TRAIN_LABELS_PATH = PROCESSED_DIR / "corrupted_train_labels.parquet"
TAUS = [0.5, 0.6, 0.75]


def main() -> None:
    if not CORRUPTED_TRAIN_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Run 03d_run_corrupted_oof_predictions.py first. Missing: {CORRUPTED_TRAIN_MANIFEST_PATH}"
        )

    c_manifest = pd.read_parquet(CORRUPTED_TRAIN_MANIFEST_PATH)
    log.info("Computing failure labels for %d corrupted-train rows ...", len(c_manifest))

    rows = []
    for _, row in tqdm(c_manifest.iterrows(), total=len(c_manifest), desc="Corrupted train labels"):
        pred_dir = Path(row.pred_dir)
        pred_mask_path = pred_dir / row.frame_id / "mask.png"
        pred = cv2.imread(str(pred_mask_path), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(row.mask_path, cv2.IMREAD_GRAYSCALE)

        if pred is None or gt is None:
            iou_val = 0.0
        else:
            if pred.shape != gt.shape:
                gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
            iou_val = mask_iou(pred, gt)

        entry: dict = {
            "frame_id": row.frame_id,
            "video_id": row.video_id,
            "corruption_type": row.corruption_type,
            "severity": row.severity,
            "iou": iou_val,
        }
        for tau in TAUS:
            entry[f"failure_tau{tau:.2f}".replace(".", "_")] = int(iou_val < tau)
        rows.append(entry)

    df = pd.DataFrame(rows)
    df.to_parquet(CORRUPTED_TRAIN_LABELS_PATH, index=False)

    # Summary stats
    for tau in TAUS:
        col = f"failure_tau{tau:.2f}".replace(".", "_")
        log.info("Overall failure_rate@τ=%.2f = %.3f (n=%d)", tau, df[col].mean(), len(df))
    for sev in sorted(df.severity.unique()):
        sub = df[df.severity == sev]
        fr = sub["failure_tau0_75"].mean()
        log.info("  Severity %d: failure_rate@τ=0.75 = %.3f (n=%d)", sev, fr, len(sub))

    log.info("Corrupted train labels saved → %s", CORRUPTED_TRAIN_LABELS_PATH)
    log.info("Run scripts/06b_train_monitor_combined.py next.")


if __name__ == "__main__":
    main()
