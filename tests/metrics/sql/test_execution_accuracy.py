"""Tests for ExecutionAccuracy metric.

This is the first concrete metric — the foundation of SQL evaluation.
It compares result sets deterministically, following BIRD's execution
accuracy approach but with richer partial credit and detail reporting.
"""

import pandas as pd

from analytics_eval.metrics.sql.execution_accuracy import ExecutionAccuracy
from analytics_eval.test_case.case import AnalyticsTestCase


class TestExecutionAccuracyExactMatch:
    """Test cases where expected and actual results match exactly."""

    def test_exact_match_single_row(self):
        case = AnalyticsTestCase(
            input="What is total revenue?",
            expected_results=pd.DataFrame({"total": [4500.0]}),
            actual_results=pd.DataFrame({"total": [4500.0]}),
        )
        metric = ExecutionAccuracy()
        result = metric.evaluate(case)
        assert result.score == 1.0
        assert result.passed is True

    def test_exact_match_multiple_rows(self, full_case):
        metric = ExecutionAccuracy()
        result = metric.evaluate(full_case)
        assert result.score == 1.0

    def test_both_empty_results(self):
        case = AnalyticsTestCase(
            input="Q",
            expected_results=pd.DataFrame(),
            actual_results=pd.DataFrame(),
        )
        metric = ExecutionAccuracy()
        result = metric.evaluate(case)
        assert result.details.get("skipped") is True

    def test_match_with_integer_values(self):
        case = AnalyticsTestCase(
            input="Q",
            expected_results=pd.DataFrame({"count": [100]}),
            actual_results=pd.DataFrame({"count": [100]}),
        )
        metric = ExecutionAccuracy()
        result = metric.evaluate(case)
        assert result.score == 1.0


class TestExecutionAccuracyMismatch:
    """Test cases where results don't match."""

    def test_value_mismatch(self, mismatched_case):
        metric = ExecutionAccuracy()
        result = metric.evaluate(mismatched_case)
        assert result.score == 0.0
        assert "No matching rows" in result.reason

    def test_column_count_mismatch(self):
        case = AnalyticsTestCase(
            input="Q",
            expected_results=pd.DataFrame({"a": [1], "b": [2]}),
            actual_results=pd.DataFrame({"a": [1]}),
        )
        metric = ExecutionAccuracy()
        result = metric.evaluate(case)
        assert result.score == 0.0
        assert "Column count mismatch" in result.reason

    def test_row_count_mismatch_partial_credit(self, shape_mismatch_case):
        """When row counts differ, score = min_rows / max_rows."""
        metric = ExecutionAccuracy()
        result = metric.evaluate(shape_mismatch_case)
        assert 0.0 < result.score < 1.0
        assert "Row count mismatch" in result.reason


class TestExecutionAccuracyConfig:
    """Test ExecutionAccuracy configuration options."""

    def test_custom_threshold(self):
        metric = ExecutionAccuracy(threshold=0.9)
        assert metric.threshold == 0.9

    def test_custom_float_tolerance(self):
        metric = ExecutionAccuracy(float_tolerance=0.01)
        case = AnalyticsTestCase(
            input="Q",
            expected_results=pd.DataFrame({"val": [100.0]}),
            actual_results=pd.DataFrame({"val": [100.005]}),
        )
        result = metric.evaluate(case)
        assert result.score == 1.0

    def test_order_independence_default(self):
        """By default, row order is ignored."""
        metric = ExecutionAccuracy()
        case = AnalyticsTestCase(
            input="Q",
            expected_results=pd.DataFrame(
                {
                    "region": ["East", "West"],
                    "total": [100, 200],
                }
            ),
            actual_results=pd.DataFrame(
                {
                    "region": ["West", "East"],
                    "total": [200, 100],
                }
            ),
        )
        result = metric.evaluate(case)
        assert result.score == 1.0


class TestExecutionAccuracyRequiredFields:
    """Test that ExecutionAccuracy properly declares required fields."""

    def test_required_fields(self):
        metric = ExecutionAccuracy()
        assert "expected_results" in metric.required_fields
        assert "actual_results" in metric.required_fields

    def test_skip_when_no_results(self):
        case = AnalyticsTestCase(input="Q")
        metric = ExecutionAccuracy()
        result = metric.evaluate(case)
        assert result.details.get("skipped") is True

    def test_skip_when_only_expected(self):
        case = AnalyticsTestCase(
            input="Q",
            expected_results=pd.DataFrame({"a": [1]}),
        )
        metric = ExecutionAccuracy()
        result = metric.evaluate(case)
        assert result.details.get("skipped") is True

    def test_metric_properties(self):
        metric = ExecutionAccuracy()
        assert metric.name == "ExecutionAccuracy"
        assert metric.category.value == "sql_correctness"
        assert metric.mode.value == "deterministic"
