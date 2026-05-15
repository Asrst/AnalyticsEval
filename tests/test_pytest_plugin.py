"""Tests for the pytest plugin (assert_test, assert_analytics, hooks, markers).

TDD: These tests verify the DeepEval-style pytest integration:
- assert_test() is the primary assertion function
- assert_analytics() is a backward-compatible alias
- Results are captured in the global test run
- Markers are applied to analytics tests
"""

import pandas as pd
import pytest

from analytics_eval.metrics.result.result_accuracy import ResultAccuracy
from analytics_eval.metrics.sql.execution_accuracy import ExecutionAccuracy
from analytics_eval.pytest_plugin import (
    assert_analytics,
    assert_test,
    get_test_run,
    reset_test_run,
)
from analytics_eval.test_case.case import AnalyticsTestCase


class TestAssertTest:
    """Test the assert_test() function — the primary DeepEval-style API."""

    def test_passing_assertion(self):
        """assert_test should not raise when all metrics pass."""
        case = AnalyticsTestCase(
            input="What is revenue?",
            expected_results=pd.DataFrame({"total": [4500.0]}),
            actual_results=pd.DataFrame({"total": [4500.0]}),
        )
        results = assert_test(case, [ExecutionAccuracy()])
        assert len(results) == 1
        assert results[0].score == 1.0

    def test_failing_assertion(self):
        """assert_test should raise AssertionError with details on failure."""
        case = AnalyticsTestCase(
            input="What is revenue?",
            expected_results=pd.DataFrame({"total": [4500.0]}),
            actual_results=pd.DataFrame({"total": [3000.0]}),
        )
        with pytest.raises(
            AssertionError, match="Analytics evaluation failed"
        ):
            assert_test(case, [ExecutionAccuracy(threshold=0.5)])

    def test_multiple_metrics_one_fails(self):
        """If any metric fails, the assertion fails."""
        case = AnalyticsTestCase(
            input="Q",
            expected_results=pd.DataFrame({"v": [100.0]}),
            actual_results=pd.DataFrame({"v": [200.0]}),
        )
        with pytest.raises(AssertionError):
            assert_test(case, [
                ExecutionAccuracy(threshold=0.5),
                ResultAccuracy(threshold=0.5),
            ])

    def test_multiple_metrics_all_pass(self):
        """All metrics passing should succeed."""
        case = AnalyticsTestCase(
            input="Q",
            expected_results=pd.DataFrame({"v": [100.0]}),
            actual_results=pd.DataFrame({"v": [100.0]}),
        )
        results = assert_test(case, [
            ExecutionAccuracy(),
            ResultAccuracy(),
        ])
        assert len(results) == 2
        assert all(r.passed for r in results)

    def test_skipped_metric_fails_assertion(self):
        """A skipped metric (missing fields) should cause assertion failure."""
        case = AnalyticsTestCase(input="Q")
        with pytest.raises(AssertionError):
            assert_test(case, [ExecutionAccuracy(threshold=0.5)])

    def test_returns_metric_results(self):
        """assert_test should return the list of MetricResult objects."""
        case = AnalyticsTestCase(
            input="Q",
            expected_results=pd.DataFrame({"v": [1.0]}),
            actual_results=pd.DataFrame({"v": [1.0]}),
        )
        results = assert_test(case, [ExecutionAccuracy()])
        assert len(results) == 1
        assert results[0].metric_name == "ExecutionAccuracy"
        assert results[0].score == 1.0


class TestAssertAnalyticsCompat:
    """Test that assert_analytics() works as a backward-compatible alias."""

    def test_assert_analytics_works_like_assert_test(self):
        """assert_analytics should have the same behavior as assert_test."""
        case = AnalyticsTestCase(
            input="Q",
            expected_results=pd.DataFrame({"v": [1.0]}),
            actual_results=pd.DataFrame({"v": [1.0]}),
        )
        results = assert_analytics(case, [ExecutionAccuracy()])
        assert len(results) == 1
        assert results[0].score == 1.0

    def test_assert_analytics_raises_on_failure(self):
        """assert_analytics should raise on failure just like assert_test."""
        case = AnalyticsTestCase(
            input="Q",
            expected_results=pd.DataFrame({"v": [100.0]}),
            actual_results=pd.DataFrame({"v": [200.0]}),
        )
        with pytest.raises(AssertionError):
            assert_analytics(case, [ExecutionAccuracy(threshold=0.5)])


class TestResultCapture:
    """Test that results are captured in the global test run."""

    def setup_method(self):
        """Reset the test run before each test."""
        reset_test_run()

    def test_results_recorded_in_test_run(self):
        """assert_test should record results in the global test run."""
        case = AnalyticsTestCase(
            input="Q",
            expected_results=pd.DataFrame({"v": [1.0]}),
            actual_results=pd.DataFrame({"v": [1.0]}),
        )
        assert_test(case, [ExecutionAccuracy()])

        test_run = get_test_run()
        assert len(test_run.cases) >= 1

    def test_test_run_summary(self):
        """Test run summary should compute aggregate statistics."""
        case = AnalyticsTestCase(
            input="Q",
            expected_results=pd.DataFrame({"v": [1.0]}),
            actual_results=pd.DataFrame({"v": [1.0]}),
        )
        assert_test(case, [ExecutionAccuracy()])

        test_run = get_test_run()
        summary = test_run.summary()
        assert summary["total_cases"] >= 1
        assert summary["overall_score"] > 0

    def test_test_run_to_json(self):
        """Test run should export to JSON."""
        case = AnalyticsTestCase(
            input="Q",
            expected_results=pd.DataFrame({"v": [1.0]}),
            actual_results=pd.DataFrame({"v": [1.0]}),
        )
        assert_test(case, [ExecutionAccuracy()])

        test_run = get_test_run()
        json_str = test_run.to_json()
        assert "overall_score" in json_str
        assert "ExecutionAccuracy" in json_str

    def test_reset_test_run(self):
        """Resetting the test run should clear all results."""
        case = AnalyticsTestCase(
            input="Q",
            expected_results=pd.DataFrame({"v": [1.0]}),
            actual_results=pd.DataFrame({"v": [1.0]}),
        )
        assert_test(case, [ExecutionAccuracy()])
        reset_test_run()
        test_run = get_test_run()
        assert len(test_run.cases) == 0


class TestAnalyticsMarker:
    """Test the @pytest.mark.analytics marker."""

    @pytest.mark.analytics
    def test_explicitly_marked_analytics(self):
        """Tests marked with @pytest.mark.analytics should be recognized."""
        case = AnalyticsTestCase(
            input="Q",
            expected_results=pd.DataFrame({"v": [1.0]}),
            actual_results=pd.DataFrame({"v": [1.0]}),
        )
        assert_test(case, [ExecutionAccuracy()])

    def test_auto_marked_by_assert_test(self):
        """Tests using assert_test should be auto-marked by the plugin."""
        case = AnalyticsTestCase(
            input="Q",
            expected_results=pd.DataFrame({"v": [1.0]}),
            actual_results=pd.DataFrame({"v": [1.0]}),
        )
        assert_test(case, [ExecutionAccuracy()])
