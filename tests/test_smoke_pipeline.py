"""Smoke test: 20 synthetic frames through predict→labels→features→monitor→conformal→eval."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from tcsr.conformal.split_conformal import calibrate_threshold
from tcsr.evaluation.monitor_metrics import compute_monitor_metrics
from tcsr.features.build import build_feature_table
from tcsr.labels.failure_labels import compute_failure_labels
from tcsr.monitor.classifiers import LogRegMonitor
from tcsr.monitor.train import get_feature_columns, train_monitor
from tcsr.utils.io import save_npz


@pytest.mark.slow
def test_full_smoke_pipeline(tmp_path: Path):
    """20 synthetic frames: assert every artifact exists and is non-empty."""
    n_frames = 20
    H, W = 64, 64

    # ---- 1. Create synthetic data ----------------------------------------
    raw_dir = tmp_path / "raw"
    pred_dir = tmp_path / "predictions"
    raw_dir.mkdir()

    rng = np.random.default_rng(42)
    rows = []

    for i in range(n_frames):
        fid = f"frame_{i:03d}"
        video_id = "v0" if i < 15 else "v1"
        split = "train" if i < 10 else ("cal" if i < 15 else "test")

        # Synthetic frame
        img = (rng.integers(30, 220, (H, W, 3), dtype=np.uint8))
        frame_path = raw_dir / f"{fid}.png"
        cv2.imwrite(str(frame_path), img)

        # Synthetic GT mask
        gt_mask = np.zeros((H, W), dtype=np.uint8)
        gt_mask[10:30, 10:30] = 255
        mask_path = raw_dir / f"{fid}_mask.png"
        cv2.imwrite(str(mask_path), gt_mask)

        # Synthetic prediction (slightly perturbed)
        pred_mask = gt_mask.copy()
        if i % 4 == 0:  # introduce some failures
            pred_mask[:] = 0
            pred_mask[5:15, 5:15] = 255

        pred_dir_i = pred_dir / fid
        pred_dir_i.mkdir(parents=True)
        cv2.imwrite(str(pred_dir_i / "mask.png"), pred_mask)
        prob_map = (pred_mask.astype(np.float32) / 255.0 * 0.9 + 0.05)
        save_npz({"probs": prob_map.astype(np.float16)}, pred_dir_i / "probs.npz")

        rows.append({
            "frame_id": fid,
            "video_id": video_id,
            "frame_idx_in_video": i % 15,
            "frame_path": str(frame_path),
            "mask_path": str(mask_path),
            "split": split,
            "dataset": "synthetic",
        })

    manifest = pd.DataFrame(rows)

    # ---- 2. Labels ----------------------------------------------------------
    labels_path = tmp_path / "labels.parquet"
    labels_df = compute_failure_labels(manifest, pred_dir, labels_path, taus=[0.5])
    assert labels_path.exists()
    assert len(labels_df) == n_frames

    # ---- 3. Features --------------------------------------------------------
    feature_groups = {
        "confidence": True, "shape": True, "temporal": True,
        "quality": True, "shift": False,
    }
    features_path = tmp_path / "features.parquet"
    features_df = build_feature_table(
        manifest, pred_dir, features_path, feature_groups=feature_groups
    )
    assert features_path.exists()
    assert len(features_df) == n_frames
    feat_cols = get_feature_columns(features_df)
    assert len(feat_cols) > 0

    # ---- 4. Monitor training -----------------------------------------------
    monitor_dir = tmp_path / "monitor"
    monitor_dir.mkdir()
    monitor = LogRegMonitor(C=1.0)
    features_df = features_df.merge(manifest[["frame_id", "split"]], on="frame_id")
    train_monitor(
        monitor, features_df, labels_df,
        tau_col="failure_tau0_50",
        output_dir=monitor_dir,
    )
    assert (monitor_dir / "monitor.pkl").exists()
    assert (monitor_dir / "feature_columns.json").exists()

    # ---- 5. Conformal calibration ------------------------------------------
    cal_df = features_df[features_df.split == "cal"]
    cal_labels = labels_df[labels_df.frame_id.isin(cal_df.frame_id)]
    X_cal = cal_df[feat_cols].values
    y_cal = cal_labels.set_index("frame_id").loc[cal_df.frame_id, "failure_tau0_50"].values

    if y_cal.sum() > 0:
        result = calibrate_threshold(X_cal @ np.zeros(len(feat_cols)), y_cal, alpha=0.1)
        assert "threshold" in result

    # ---- 6. Evaluation -----------------------------------------------------
    test_df = features_df[features_df.split == "test"]
    test_labels = labels_df[labels_df.frame_id.isin(test_df.frame_id)]
    X_test = test_df[feat_cols].values
    y_test = test_labels.set_index("frame_id").loc[test_df.frame_id, "failure_tau0_50"].values

    from tcsr.monitor.classifiers import load_monitor
    loaded_monitor = load_monitor(monitor_dir / "monitor.pkl")
    scores = loaded_monitor.predict_proba(X_test)
    assert len(scores) == len(y_test)
    assert scores.min() >= 0.0 and scores.max() <= 1.0

    metrics = compute_monitor_metrics(scores, y_test, fps=25.0)
    assert "brier" in metrics


def test_smoke_pipeline_runs(tmp_path: Path):
    """Non-slow version: just check imports and basic API work."""
    from tcsr.conformal.split_conformal import calibrate_threshold
    from tcsr.evaluation.monitor_metrics import compute_monitor_metrics
    from tcsr.labels.failure_labels import mask_iou
    from tcsr.monitor.classifiers import LogRegMonitor

    iou = mask_iou(np.ones((10, 10), dtype=np.uint8), np.ones((10, 10), dtype=np.uint8))
    assert iou == pytest.approx(1.0)

    monitor = LogRegMonitor()
    X = np.random.rand(50, 5).astype(np.float32)
    y = (np.random.rand(50) > 0.5).astype(int)
    monitor.fit(X, y)
    scores = monitor.predict_proba(X)
    assert len(scores) == 50
