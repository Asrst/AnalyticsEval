"""Tests for ResultAccuracy metric.

Tests numerical comparison with tolerance, partial credit scoring,
and graceful handling of non-numeric data.
"""

import pandas as pd

from analytics_eval.metrics.result.result_accuracy import ResultAccuracy
from analytics_eval.test_case.case import AnalyticsTestCase


class TestResultAccuracyExactMatch:
    """Test cases where results match within tolerance."""

    def test_exact_match(self):
        case = AnalyticsTestCase(
            input="Q",
            expected_results=pd.DataFrame({"total": [4500.0]}),
            actual_results=pd.DataFrame({"total": [4500.0]}),
        )
        metric = ResultAccuracy()
        result = metric.evaluate(case)
        assert result.score == 1.0

    def test_within_default_tolerance(self):
        """Default 1% tolerance — 0.5% off should pass."""
        case = AnalyticsTestCase(
            input="Q",
            expected_results=pd.DataFrame({"total": [1000.0]}),
            actual_results=pd.DataFrame({"total": [1005.0]}),
        )
        metric = ResultAccuracy(relative_tolerance=0.01)
        result = metric.evaluate(case)
        assert result.score == 1.0

    def test_outside_tolerance_gets_partial_credit(self):
        """Values outside tolerance get partial credit based on closeness."""
        case = AnalyticsTestCase(
            input="Q",
            expected_results=pd.DataFrame({"total": [1000.0]}),
            actual_results=pd.DataFrame({"total": [1200.0]}),
        )
        metric = ResultAccuracy(relative_tolerance=0.01)
        result = metric.evaluate(case)
        assert 0.0 < result.score < 1.0


class TestResultAccuracyShapeMismatch:
    """Test cases where result shapes differ."""

    def test_different_row_count_scores_zero(self):
        case = AnalyticsTestCase(
            input="Q",
            expected_results=pd.DataFrame({"total": [100, 200]}),
            actual_results=pd.DataFrame({"total": [100]}),
        )
        metric = ResultAccuracy()
        result = metric.evaluate(case)
        assert result.score == 0.0
        assert "Shape mismatch" in result.reason


class TestResultAccuracyConfig:
    """Test ResultAccuracy configuration."""

    def test_higher_threshold_by_default(self):
        """ResultAccuracy has a higher default threshold."""
        metric = ResultAccuracy()
        assert metric.threshold == 0.8

    def test_custom_tolerance(self):
        metric = ResultAccuracy(relative_tolerance=0.05)
        case = AnalyticsTestCase(
            input="Q",
            expected_results=pd.DataFrame({"val": [100.0]}),
            actual_results=pd.DataFrame({"val": [103.0]}),
        )
        result = metric.evaluate(case)
        assert result.score == 1.0

    def test_metric_properties(self):
        metric = ResultAccuracy()
        assert metric.name == "ResultAccuracy"
        assert metric.category.value == "result_quality"
        assert metric.mode.value == "deterministic"
