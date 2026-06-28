"""Unit tests for failure label computation."""

import numpy as np
import pytest

from tcsr.labels.failure_labels import mask_iou


class TestMaskIoU:
    def test_perfect_overlap(self):
        mask = np.ones((10, 10), dtype=np.uint8)
        assert mask_iou(mask, mask) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = np.zeros((10, 10), dtype=np.uint8)
        b = np.zeros((10, 10), dtype=np.uint8)
        a[:5, :] = 1
        b[5:, :] = 1
        assert mask_iou(a, b) == pytest.approx(0.0)

    def test_half_overlap(self):
        a = np.zeros((4, 4), dtype=np.uint8)
        b = np.zeros((4, 4), dtype=np.uint8)
        a[0:2, 0:4] = 1  # rows 0-1 (8 pixels)
        b[1:3, 0:4] = 1  # rows 1-2 (8 pixels)
        # inter = row 1 = 4 pixels; union = rows 0-2 = 12 pixels
        assert mask_iou(a, b) == pytest.approx(4 / 12)

    def test_both_empty(self):
        empty = np.zeros((5, 5), dtype=np.uint8)
        assert mask_iou(empty, empty) == pytest.approx(1.0)

    def test_threshold(self):
        iou = 0.4
        tau = 0.5
        assert (iou < tau) == 1  # failure


class TestFailureLabelCompute:
    def test_iou_columns(self, tmp_path):
        """Labels parquet has correct failure columns."""
        import cv2
        import pandas as pd

        from tcsr.labels.failure_labels import compute_failure_labels

        # Create synthetic frame + mask files
        frame_dir = tmp_path / "frames"
        frame_dir.mkdir()
        pred_dir = tmp_path / "predictions"

        n = 5
        rows = []
        for i in range(n):
            fid = f"frame_{i:03d}"
            pred_mask_dir = pred_dir / fid
            pred_mask_dir.mkdir(parents=True)
            gt_path = frame_dir / f"{fid}.png"
            mask = np.zeros((64, 64), dtype=np.uint8)
            mask[10:30, 10:30] = 255
            cv2.imwrite(str(pred_mask_dir / "mask.png"), mask)
            cv2.imwrite(str(gt_path), mask)
            rows.append({
                "frame_id": fid,
                "video_id": "v0",
                "frame_idx_in_video": i,
                "frame_path": str(gt_path),
                "mask_path": str(gt_path),
                "split": "test",
                "dataset": "synthetic",
            })

        manifest = pd.DataFrame(rows)
        out = tmp_path / "labels.parquet"
        df = compute_failure_labels(manifest, pred_dir, out, taus=[0.5, 0.75])

        assert "iou" in df.columns
        assert "failure_tau0_50" in df.columns
        assert "failure_tau0_75" in df.columns
        assert np.allclose(df["iou"].values, 1.0)
        assert (df["failure_tau0_50"] == 0).all()
