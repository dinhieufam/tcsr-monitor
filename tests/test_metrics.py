"""Unit tests for evaluation metric implementations."""

import numpy as np
import pytest

from tcsr.evaluation.coverage_risk import coverage_risk_curve
from tcsr.evaluation.lead_time import compute_lead_times
from tcsr.evaluation.monitor_metrics import compute_monitor_metrics


class TestMonitorMetrics:
    def test_perfect_detector(self):
        labels = np.array([0, 0, 1, 1, 0, 1])
        scores = np.array([0.1, 0.1, 0.9, 0.9, 0.1, 0.9])
        m = compute_monitor_metrics(scores, labels, fps=25.0, threshold=0.5)
        assert m["auroc"] == pytest.approx(1.0)
        assert m["auprc"] == pytest.approx(1.0)
        assert m["recall"] == pytest.approx(1.0)
        assert m["fa_per_min"] == pytest.approx(0.0)

    def test_random_detector(self):
        rng = np.random.default_rng(0)
        n = 500
        labels = rng.integers(0, 2, n)
        scores = rng.uniform(0, 1, n)
        m = compute_monitor_metrics(scores, labels, fps=25.0, threshold=0.5)
        assert 0.3 < m["auroc"] < 0.7  # roughly random

    def test_all_failures_no_auroc(self):
        labels = np.ones(10, dtype=int)
        scores = np.random.rand(10)
        m = compute_monitor_metrics(scores, labels)
        assert np.isnan(m["auroc"])

    def test_brier_score_range(self):
        labels = np.array([0, 1, 0, 1])
        scores = np.array([0.1, 0.9, 0.2, 0.8])
        m = compute_monitor_metrics(scores, labels)
        assert 0.0 <= m["brier"] <= 1.0


class TestCoverageRiskCurve:
    def test_output_shapes(self):
        scores = np.linspace(0, 1, 50)
        labels = (scores > 0.5).astype(int)
        curve = coverage_risk_curve(scores, labels, n_points=20)
        assert len(curve["coverage"]) == 20
        assert len(curve["risk"]) == 20

    def test_coverage_monotone(self):
        scores = np.random.rand(100)
        labels = (np.random.rand(100) > 0.7).astype(int)
        curve = coverage_risk_curve(scores, labels, n_points=50)
        # Coverage should be non-decreasing as threshold drops
        diffs = np.diff(curve["coverage"])
        assert (diffs >= -1e-9).all()


class TestLeadTime:
    def test_basic_lead(self):
        alarms = np.array([1, 0, 0, 0, 0, 0])
        labels = np.array([0, 0, 0, 1, 1, 1])
        video_ids = np.array(["v0"] * 6)
        result = compute_lead_times(alarms, labels, video_ids, fps=1.0, min_failure_run=3)
        assert result["n_events"] == 1
        assert result["mean_lead_frames"] == pytest.approx(3.0)

    def test_no_alarms(self):
        alarms = np.zeros(10, dtype=int)
        labels = np.array([0, 0, 1, 1, 1, 0, 0, 0, 0, 0])
        video_ids = np.array(["v0"] * 10)
        result = compute_lead_times(alarms, labels, video_ids, fps=25.0, min_failure_run=3)
        assert result["n_events"] == 0
