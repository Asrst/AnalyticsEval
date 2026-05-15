"""Tests for AnalyticsMetric base class and MetricResult.

These tests define the contract for the metric system:
- Every metric produces a MetricResult with score, reason, and details
- Metrics declare required_fields for graceful degradation
- can_evaluate checks if required data is available
- evaluate() wraps measure() with timing and error handling
"""

import pytest

from analytics_eval.metrics.base import (
    AnalyticsMetric,
    EvaluationMode,
    MetricCategory,
    MetricResult,
)
from analytics_eval.test_case.case import AnalyticsTestCase

# ── Concrete test metric for testing the abstract base class ──

class StubMetric(AnalyticsMetric):
    """A minimal concrete metric for testing the base class contract."""

    @property
    def name(self) -> str:
        return "StubMetric"

    @property
    def category(self) -> MetricCategory:
        return MetricCategory.SQL_CORRECTNESS

    @property
    def mode(self) -> EvaluationMode:
        return EvaluationMode.DETERMINISTIC

    @property
    def required_fields(self) -> list[str]:
        return ["expected_sql", "actual_sql"]

    def measure(self, test_case: AnalyticsTestCase) -> MetricResult:
        if test_case.expected_sql == test_case.actual_sql:
            score = 1.0
            reason = "SQL matches exactly."
        else:
            score = 0.0
            reason = "SQL does not match."
        return MetricResult(
            metric_name=self.name,
            score=score,
            reason=reason,
            category=self.category,
            mode=self.mode,
            threshold=self.threshold,
        )


class ErrorMetric(AnalyticsMetric):
    """A metric that always throws — tests error handling."""

    @property
    def name(self) -> str:
        return "ErrorMetric"

    @property
    def category(self) -> MetricCategory:
        return MetricCategory.RESULT_QUALITY

    @property
    def mode(self) -> EvaluationMode:
        return EvaluationMode.DETERMINISTIC

    @property
    def required_fields(self) -> list[str]:
        return ["expected_sql"]

    def measure(self, test_case: AnalyticsTestCase) -> MetricResult:
        raise RuntimeError("Intentional error for testing")


# ── MetricResult tests ──

class TestMetricResult:
    """Test MetricResult model and its derived properties."""

    def test_passing_result(self):
        result = MetricResult(
            metric_name="Test",
            score=0.85,
            reason="Good enough",
            category=MetricCategory.SQL_CORRECTNESS,
            mode=EvaluationMode.DETERMINISTIC,
            threshold=0.5,
        )
        assert result.passed is True
        assert result.score == 0.85

    def test_failing_result(self):
        result = MetricResult(
            metric_name="Test",
            score=0.3,
            reason="Not good enough",
            category=MetricCategory.SQL_CORRECTNESS,
            mode=EvaluationMode.DETERMINISTIC,
            threshold=0.5,
        )
        assert result.passed is False

    def test_score_at_threshold_passes(self):
        result = MetricResult(
            metric_name="Test",
            score=0.5,
            reason="Exactly at threshold",
            category=MetricCategory.SQL_CORRECTNESS,
            mode=EvaluationMode.DETERMINISTIC,
            threshold=0.5,
        )
        assert result.passed is True

    def test_score_out_of_range_raises_error(self):
        with pytest.raises(ValueError):
            MetricResult(
                metric_name="Test",
                score=1.5,
                reason="Invalid",
                category=MetricCategory.SQL_CORRECTNESS,
                mode=EvaluationMode.DETERMINISTIC,
            )

    def test_negative_score_raises_error(self):
        with pytest.raises(ValueError):
            MetricResult(
                metric_name="Test",
                score=-0.1,
                reason="Invalid",
                category=MetricCategory.SQL_CORRECTNESS,
                mode=EvaluationMode.DETERMINISTIC,
            )

    def test_empty_reason_raises_error(self):
        with pytest.raises(ValueError):
            MetricResult(
                metric_name="Test",
                score=0.5,
                reason="",
                category=MetricCategory.SQL_CORRECTNESS,
                mode=EvaluationMode.DETERMINISTIC,
            )

    def test_str_representation(self):
        result = MetricResult(
            metric_name="ExecutionAccuracy",
            score=0.92,
            reason="Results match.",
            category=MetricCategory.SQL_CORRECTNESS,
            mode=EvaluationMode.DETERMINISTIC,
            threshold=0.5,
        )
        text = str(result)
        assert "PASS" in text
        assert "0.920" in text
        assert "ExecutionAccuracy" in text

    def test_details_dict(self):
        result = MetricResult(
            metric_name="Test",
            score=1.0,
            reason="Match",
            category=MetricCategory.SQL_CORRECTNESS,
            mode=EvaluationMode.DETERMINISTIC,
            details={"matching_rows": 10, "total_rows": 10},
        )
        assert result.details["matching_rows"] == 10


# ── AnalyticsMetric base class tests ──

class TestAnalyticsMetric:
    """Test the AnalyticsMetric abstract base class contract."""

    def test_stub_metric_matching_sql(self):
        metric = StubMetric()
        case = AnalyticsTestCase(
            input="Q",
            expected_sql="SELECT 1",
            actual_sql="SELECT 1",
        )
        result = metric.evaluate(case)
        assert result.score == 1.0
        assert result.passed is True

    def test_stub_metric_mismatched_sql(self):
        metric = StubMetric()
        case = AnalyticsTestCase(
            input="Q",
            expected_sql="SELECT 1",
            actual_sql="SELECT 2",
        )
        result = metric.evaluate(case)
        assert result.score == 0.0

    def test_can_evaluate_with_all_required_fields(self):
        metric = StubMetric()
        case = AnalyticsTestCase(
            input="Q",
            expected_sql="SELECT 1",
            actual_sql="SELECT 1",
        )
        assert metric.can_evaluate(case) is True

    def test_can_evaluate_missing_required_field(self):
        metric = StubMetric()
        case = AnalyticsTestCase(input="Q", expected_sql="SELECT 1")
        assert metric.can_evaluate(case) is False

    def test_can_evaluate_blank_string_field(self):
        metric = StubMetric()
        case = AnalyticsTestCase(
            input="Q",
            expected_sql="SELECT 1",
            actual_sql="   ",
        )
        assert metric.can_evaluate(case) is False

    def test_evaluate_skips_when_fields_missing(self):
        metric = StubMetric(threshold=0.5)
        case = AnalyticsTestCase(input="Q")
        result = metric.evaluate(case)
        assert result.score == 0.0
        assert "missing required fields" in result.reason.lower()
        assert result.details.get("skipped") is True

    def test_evaluate_catches_exception_in_measure(self):
        metric = ErrorMetric()
        case = AnalyticsTestCase(
            input="Q",
            expected_sql="SELECT 1",
        )
        result = metric.evaluate(case)
        assert result.score == 0.0
        assert "Error" in result.reason
        assert result.details.get("error_type") == "RuntimeError"

    def test_evaluate_measures_timing(self):
        metric = StubMetric()
        case = AnalyticsTestCase(
            input="Q",
            expected_sql="SELECT 1",
            actual_sql="SELECT 1",
        )
        result = metric.evaluate(case)
        assert result.evaluation_time_ms is not None
        assert result.evaluation_time_ms >= 0

    def test_custom_threshold(self):
        metric = StubMetric(threshold=0.9)
        assert metric.threshold == 0.9

    def test_metric_repr(self):
        metric = StubMetric(threshold=0.7)
        repr_str = repr(metric)
        assert "StubMetric" in repr_str
        assert "0.7" in repr_str

    def test_required_fields_empty_for_no_requirement_metrics(self):
        """Metrics that don't require any specific fields should work with any case."""
        class NoRequirementMetric(AnalyticsMetric):
            @property
            def name(self): return "NoReq"
            @property
            def category(self): return MetricCategory.SQL_CORRECTNESS
            @property
            def mode(self): return EvaluationMode.DETERMINISTIC
            @property
            def required_fields(self): return []
            def measure(self, test_case):
                return MetricResult(
                    metric_name=self.name, score=1.0, reason="OK",
                    category=self.category, mode=self.mode,
                )

        metric = NoRequirementMetric()
        case = AnalyticsTestCase(input="Q")
        assert metric.can_evaluate(case) is True
        result = metric.evaluate(case)
        assert result.score == 1.0
