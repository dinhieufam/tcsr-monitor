"""Unit tests for conformal calibration."""

import numpy as np
import pytest

from tcsr.conformal.split_conformal import apply_threshold, calibrate_threshold


class TestSplitConformal:
    def test_threshold_bounds_missed_failures(self):
        np.random.seed(0)
        n = 200
        scores = np.random.beta(2, 5, n)  # model tends to underestimate risk
        labels = (np.random.rand(n) < 0.3).astype(int)
        alpha = 0.1

        result = calibrate_threshold(scores, labels, alpha=alpha)
        assert "threshold" in result
        assert result["realized_missed_rate"] <= alpha + 2 / (labels.sum() + 1)

    def test_no_failures_raises(self):
        scores = np.random.rand(100)
        labels = np.zeros(100, dtype=int)
        with pytest.raises(ValueError, match="No failure frames"):
            calibrate_threshold(scores, labels)

    def test_threshold_in_range(self):
        scores = np.linspace(0, 1, 100)
        labels = (scores > 0.5).astype(int)
        result = calibrate_threshold(scores, labels, alpha=0.1)
        assert 0.0 <= result["threshold"] <= 1.0

    def test_apply_threshold(self):
        scores = np.array([0.2, 0.5, 0.8, 0.9])
        alarms = apply_threshold(scores, threshold=0.6)
        np.testing.assert_array_equal(alarms, [0, 0, 1, 1])


class TestCoverageGuarantee:
    def test_coverage_holds_large_n(self):
        """On large synthetic data, realized missed rate stays near alpha."""
        rng = np.random.default_rng(42)
        n = 1000
        scores = rng.uniform(0, 1, n)
        labels = (rng.uniform(0, 1, n) < 0.4).astype(int)
        alpha = 0.1

        result = calibrate_threshold(scores, labels, alpha=alpha)
        assert result["realized_missed_rate"] <= alpha + 0.05
