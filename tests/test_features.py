"""Unit tests for each feature family."""

import numpy as np
import pytest

from tcsr.features.confidence import compute_confidence_features
from tcsr.features.quality import compute_quality_features
from tcsr.features.shape import compute_shape_features
from tcsr.features.temporal import compute_temporal_features


class TestConfidenceFeatures:
    def test_uniform_half(self):
        prob = np.full((32, 32), 0.5, dtype=np.float32)
        feats = compute_confidence_features(prob)
        assert feats["conf_mean_entropy"] == pytest.approx(np.log(2), abs=1e-4)
        assert feats["conf_margin"] == pytest.approx(0.0, abs=1e-4)

    def test_certain_fg(self):
        prob = np.ones((32, 32), dtype=np.float32) * 0.99
        feats = compute_confidence_features(prob)
        assert feats["conf_frac_uncertain"] == pytest.approx(0.0)
        assert feats["conf_max_prob"] == pytest.approx(0.99)

    def test_returns_all_keys(self):
        prob = np.random.rand(16, 16).astype(np.float32)
        feats = compute_confidence_features(prob)
        expected = ["conf_mean_fg_prob", "conf_max_prob", "conf_mean_entropy",
                    "conf_max_entropy", "conf_frac_uncertain", "conf_margin", "conf_frac_fg"]
        for k in expected:
            assert k in feats


class TestShapeFeatures:
    def test_empty_mask(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        feats = compute_shape_features(mask)
        assert feats["shape_n_components"] == 0.0
        assert feats["shape_area_frac"] == pytest.approx(0.0)

    def test_full_mask(self):
        mask = np.ones((64, 64), dtype=np.uint8)
        feats = compute_shape_features(mask)
        assert feats["shape_area_frac"] == pytest.approx(1.0)
        assert feats["shape_n_components"] == pytest.approx(1.0)

    def test_two_components(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[5:10, 5:10] = 1
        mask[50:55, 50:55] = 1
        feats = compute_shape_features(mask)
        assert feats["shape_n_components"] == pytest.approx(2.0)


class TestTemporalFeatures:
    def test_first_frame(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[10:20, 10:20] = 1
        feats = compute_temporal_features(mask, None, 0.1, None, [])
        assert feats["temp_prev_iou"] == pytest.approx(1.0)
        assert feats["temp_centroid_jump"] == pytest.approx(0.0)

    def test_identical_frames(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[10:20, 10:20] = 1
        feats = compute_temporal_features(mask, mask.copy(), 0.1, 0.1, [])
        assert feats["temp_prev_iou"] == pytest.approx(1.0)
        assert feats["temp_centroid_jump"] == pytest.approx(0.0)

    def test_disjoint_frames(self):
        a = np.zeros((32, 32), dtype=np.uint8)
        b = np.zeros((32, 32), dtype=np.uint8)
        a[0:5, 0:5] = 1
        b[25:30, 25:30] = 1
        feats = compute_temporal_features(b, a, float(b.mean()), float(a.mean()), [])
        assert feats["temp_prev_iou"] == pytest.approx(0.0)


class TestQualityFeatures:
    def test_black_image(self):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        feats = compute_quality_features(img)
        assert feats["qual_brightness"] == pytest.approx(0.0)
        assert feats["qual_specular_ratio"] == pytest.approx(0.0)

    def test_white_image(self):
        img = np.ones((64, 64, 3), dtype=np.uint8) * 255
        feats = compute_quality_features(img)
        assert feats["qual_brightness"] == pytest.approx(1.0)
        assert feats["qual_specular_ratio"] == pytest.approx(1.0)

    def test_returns_all_keys(self):
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        feats = compute_quality_features(img)
        for k in ["qual_brightness", "qual_blur_score", "qual_contrast", "qual_specular_ratio"]:
            assert k in feats
