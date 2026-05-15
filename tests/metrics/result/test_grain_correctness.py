"""Tests for GrainCorrectness metric — checks aggregation level of results.

TDD: These tests define the contract for grain validation. The metric
should detect wrong GROUP BY levels and cardinality mismatches.
"""

import pandas as pd

from analytics_eval.metrics.result.grain_correctness import GrainCorrectness
from analytics_eval.test_case.case import AnalyticsTestCase


class TestGrainCorrectnessMatching:
    """Test cases where grain matches expected results."""

    def test_matching_grain_single_dimension(self):
        """Same dimension and same row count should score 1.0."""
        case = AnalyticsTestCase(
            input="Revenue by region",
            expected_results=pd.DataFrame(
                {
                    "region": ["East", "West", "North"],
                    "total": [1000.0, 2000.0, 1500.0],
                }
            ),
            actual_results=pd.DataFrame(
                {
                    "region": ["East", "West", "North"],
                    "total": [1000.0, 2000.0, 1500.0],
                }
            ),
        )
        metric = GrainCorrectness()
        result = metric.evaluate(case)
        assert result.score == 1.0
        assert result.passed is True

    def test_matching_grain_two_dimensions(self):
        """Two dimensions with matching rows."""
        case = AnalyticsTestCase(
            input="Revenue by region and year",
            expected_results=pd.DataFrame(
                {
                    "region": ["East", "East", "West"],
                    "year": [2023, 2024, 2023],
                    "revenue": [500.0, 600.0, 700.0],
                }
            ),
            actual_results=pd.DataFrame(
                {
                    "region": ["East", "East", "West"],
                    "year": [2023, 2024, 2023],
                    "revenue": [500.0, 600.0, 700.0],
                }
            ),
        )
        metric = GrainCorrectness()
        result = metric.evaluate(case)
        assert result.score == 1.0


class TestGrainCorrectnessMismatch:
    """Test cases where grain doesn't match."""

    def test_missing_dimension(self):
        """Missing a dimension column should reduce score."""
        case = AnalyticsTestCase(
            input="Revenue by region and year",
            expected_results=pd.DataFrame(
                {
                    "region": ["East", "West"],
                    "year": [2023, 2023],
                    "revenue": [1000.0, 2000.0],
                }
            ),
            actual_results=pd.DataFrame(
                {
                    "region": ["East", "West"],
                    "revenue": [1000.0, 2000.0],
                }
            ),
        )
        metric = GrainCorrectness()
        result = metric.evaluate(case)
        assert result.score < 1.0
        assert "year" in result.details.get("missing_dimensions", [])

    def test_extra_dimension(self):
        """Extra dimension column should reduce score but less than missing."""
        case = AnalyticsTestCase(
            input="Revenue by region",
            expected_results=pd.DataFrame(
                {
                    "region": ["East", "West"],
                    "revenue": [1000.0, 2000.0],
                }
            ),
            actual_results=pd.DataFrame(
                {
                    "region": ["East", "West"],
                    "year": [2023, 2024],
                    "revenue": [1000.0, 2000.0],
                }
            ),
        )
        metric = GrainCorrectness()
        result = metric.evaluate(case)
        assert result.score < 1.0
        assert "year" in result.details.get("extra_dimensions", [])

    def test_cardinality_mismatch(self):
        """Different number of rows with same dimensions."""
        case = AnalyticsTestCase(
            input="Revenue by region",
            expected_results=pd.DataFrame(
                {
                    "region": ["East", "West", "North", "South"],
                    "revenue": [1000.0, 2000.0, 1500.0, 1200.0],
                }
            ),
            actual_results=pd.DataFrame(
                {
                    "region": ["East", "West"],
                    "revenue": [1000.0, 2000.0],
                }
            ),
        )
        metric = GrainCorrectness()
        result = metric.evaluate(case)
        assert result.score < 1.0
        assert result.details["cardinality_ratio"] == 0.5


class TestGrainCorrectnessEdgeCases:
    """Edge cases for grain correctness."""

    def test_both_empty_results(self):
        """Empty result sets with matching structure should score 1.0."""
        case = AnalyticsTestCase(
            input="Q",
            expected_results=pd.DataFrame(
                {
                    "region": pd.Series([], dtype=str),
                }
            ),
            actual_results=pd.DataFrame(
                {
                    "region": pd.Series([], dtype=str),
                }
            ),
        )
        metric = GrainCorrectness()
        result = metric.evaluate(case)
        assert result.score == 1.0

    def test_one_empty_one_nonempty(self):
        """One empty and one non-empty should score 0.0."""
        case = AnalyticsTestCase(
            input="Q",
            expected_results=pd.DataFrame(
                {
                    "region": ["East"],
                    "total": [100.0],
                }
            ),
            actual_results=pd.DataFrame(),
        )
        metric = GrainCorrectness()
        result = metric.evaluate(case)
        assert result.score == 0.0

    def test_no_dimension_columns(self):
        """Results with only numeric columns (no dimensions)."""
        case = AnalyticsTestCase(
            input="What is total revenue?",
            expected_results=pd.DataFrame({"total": [4500.0]}),
            actual_results=pd.DataFrame({"total": [4500.0]}),
        )
        metric = GrainCorrectness()
        result = metric.evaluate(case)
        assert result.score == 1.0

    def test_cardinality_within_tolerance(self):
        """Row counts within 20% tolerance should still get high score."""
        metric = GrainCorrectness(cardinality_tolerance=0.2)
        case = AnalyticsTestCase(
            input="Revenue by region",
            expected_results=pd.DataFrame(
                {
                    "region": [f"Region_{i}" for i in range(10)],
                    "revenue": [float(i * 100) for i in range(10)],
                }
            ),
            actual_results=pd.DataFrame(
                {
                    "region": [f"Region_{i}" for i in range(9)],
                    "revenue": [float(i * 100) for i in range(9)],
                }
            ),
        )
        result = metric.evaluate(case)
        assert result.score >= 0.9

    def test_missing_required_fields_skips(self):
        """Without results, the metric should skip."""
        case = AnalyticsTestCase(input="Q")
        metric = GrainCorrectness()
        assert metric.can_evaluate(case) is False
