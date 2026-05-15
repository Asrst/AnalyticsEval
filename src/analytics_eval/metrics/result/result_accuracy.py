"""Result Accuracy metric — numerical comparison of query results.

Similar to ExecutionAccuracy but with configurable numerical tolerance,
enabling "soft" matching where small differences in floating-point values
are acceptable. This is useful for analytics queries involving aggregations,
percentages, and statistical calculations where exact bit-level equality
is too strict.
"""

from __future__ import annotations

from typing import Any

from analytics_eval.metrics.base import (
    AnalyticsMetric,
    EvaluationMode,
    MetricCategory,
    MetricResult,
)
from analytics_eval.test_case.case import AnalyticsTestCase


class ResultAccuracy(AnalyticsMetric):
    """Measures numerical accuracy of query results against ground truth.

    Unlike ExecutionAccuracy which requires exact (or near-exact) match,
    ResultAccuracy provides graded scoring based on how close the actual
    values are to expected values. This is essential for:
    - Aggregation queries where floating-point arithmetic may differ
    - Statistical calculations with inherent precision limits
    - Queries where "close enough" is still valuable

    Scoring:
    - Each cell is scored: 1.0 if within tolerance, else based on relative error
    - Score = mean of all cell scores
    - This provides fine-grained partial credit
    """

    def __init__(
        self,
        threshold: float = 0.8,
        relative_tolerance: float = 0.01,
        **kwargs: Any,
    ) -> None:
        super().__init__(threshold=threshold, **kwargs)
        self.relative_tolerance = relative_tolerance

    @property
    def name(self) -> str:
        return "ResultAccuracy"

    @property
    def category(self) -> MetricCategory:
        return MetricCategory.RESULT_QUALITY

    @property
    def mode(self) -> EvaluationMode:
        return EvaluationMode.DETERMINISTIC

    @property
    def required_fields(self) -> list[str]:
        return ["expected_results", "actual_results"]

    def measure(self, test_case: AnalyticsTestCase) -> MetricResult:
        """Compare expected vs actual results with numerical tolerance."""
        expected = test_case.expected_results
        actual = test_case.actual_results

        assert expected is not None and actual is not None

        if expected.shape != actual.shape:
            score = 0.0
            reason = f"Shape mismatch: expected {expected.shape}, got {actual.shape}"
            return MetricResult(
                metric_name=self.name,
                score=score,
                reason=reason,
                category=self.category,
                mode=self.mode,
                threshold=self.threshold,
            )

        if expected.shape[0] == 0:
            return MetricResult(
                metric_name=self.name,
                score=1.0,
                reason="Both result sets are empty.",
                category=self.category,
                mode=self.mode,
                threshold=self.threshold,
            )

        total_cells = expected.shape[0] * expected.shape[1]
        cell_scores = []

        for idx in range(expected.shape[0]):
            for col in expected.columns:
                exp_val = expected.iloc[idx][col]
                act_val = actual.iloc[idx][col]
                cell_score = self._cell_score(exp_val, act_val)
                cell_scores.append(cell_score)

        score = sum(cell_scores) / total_cells if total_cells > 0 else 1.0

        if score >= 0.99:
            reason = "Results match within tolerance."
        elif score >= 0.9:
            reason = f"Results mostly match (score={score:.3f}). Some numerical differences."
        else:
            reason = (
                f"Significant differences between expected and actual results (score={score:.3f})."
            )

        return MetricResult(
            metric_name=self.name,
            score=score,
            reason=reason,
            category=self.category,
            mode=self.mode,
            threshold=self.threshold,
            details={
                "total_cells": total_cells,
                "avg_cell_score": score,
                "relative_tolerance": self.relative_tolerance,
            },
        )

    def _cell_score(self, expected: Any, actual: Any) -> float:
        """Score a single cell: 1.0 if within tolerance, else based on error."""
        if expected is None and actual is None:
            return 1.0
        if expected is None or actual is None:
            return 0.0

        import pandas as pd

        try:
            if pd.isna(expected) and pd.isna(actual):
                return 1.0
            if pd.isna(expected) or pd.isna(actual):
                return 0.0
        except (TypeError, ValueError):
            pass

        if isinstance(expected, str) and isinstance(actual, str):
            return 1.0 if expected.strip() == actual.strip() else 0.0

        try:
            exp_num = float(expected)
            act_num = float(actual)
            if exp_num == 0.0 and act_num == 0.0:
                return 1.0
            if exp_num == 0.0:
                return 0.0 if abs(act_num) > self.relative_tolerance else 1.0
            relative_error = abs(exp_num - act_num) / abs(exp_num)
            if relative_error <= self.relative_tolerance:
                return 1.0
            return max(0.0, 1.0 - relative_error)
        except (TypeError, ValueError):
            return 1.0 if expected == actual else 0.0
