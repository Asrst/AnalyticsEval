"""Execution Accuracy metric — the foundational SQL evaluation metric.

Compares the result set of the agent's SQL against the ground truth SQL
by executing both and comparing outputs. This is the same core metric
used by BIRD (EX) and Spider 2.0 (execution match).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from analytics_eval.metrics.base import (
    AnalyticsMetric,
    EvaluationMode,
    MetricCategory,
    MetricResult,
)
from analytics_eval.test_case.case import AnalyticsTestCase


class ExecutionAccuracy(AnalyticsMetric):
    """Measures whether the agent's SQL produces the same results as ground truth.

    This is the most fundamental SQL evaluation metric. It executes both the
    expected SQL and the actual SQL against the database and compares result
    sets. Unlike string matching, this correctly handles semantically equivalent
    SQL that uses different syntax.

    Comparison logic:
    1. If both DataFrames are provided directly, compare them
    2. Compare shapes (row/column count)
    3. Sort both by all columns to handle ordering differences
    4. Compare values with configurable float tolerance
    5. Score = fraction of matching rows (or 1.0 for exact match)

    Configuration:
    - threshold: Default 0.5 (pass if score >= threshold)
    - float_tolerance: Relative tolerance for floating-point comparison
    - ignore_order: Whether to ignore row ordering (default True)
    - ignore_column_order: Whether to ignore column ordering (default True)
    """

    def __init__(
        self,
        threshold: float = 0.5,
        float_tolerance: float = 1e-6,
        ignore_order: bool = True,
        ignore_column_order: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(threshold=threshold, **kwargs)
        self.float_tolerance = float_tolerance
        self.ignore_order = ignore_order
        self.ignore_column_order = ignore_column_order

    @property
    def name(self) -> str:
        return "ExecutionAccuracy"

    @property
    def category(self) -> MetricCategory:
        return MetricCategory.SQL_CORRECTNESS

    @property
    def mode(self) -> EvaluationMode:
        return EvaluationMode.DETERMINISTIC

    @property
    def required_fields(self) -> list[str]:
        return ["expected_results", "actual_results"]

    def measure(self, test_case: AnalyticsTestCase) -> MetricResult:
        """Compare expected vs actual result sets."""
        expected = test_case.expected_results
        actual = test_case.actual_results

        assert expected is not None and actual is not None

        if expected.shape[0] == 0 and actual.shape[0] == 0:
            return MetricResult(
                metric_name=self.name,
                score=1.0,
                reason="Both result sets are empty — match by vacuity.",
                category=self.category,
                mode=self.mode,
                threshold=self.threshold,
            )

        if expected.shape[1] != actual.shape[1]:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                reason=(
                    f"Column count mismatch: expected {expected.shape[1]}, got {actual.shape[1]}"
                ),
                category=self.category,
                mode=self.mode,
                threshold=self.threshold,
                details={
                    "expected_cols": expected.shape[1],
                    "actual_cols": actual.shape[1],
                },
            )

        if self.ignore_column_order:
            expected = expected.reindex(sorted(expected.columns), axis=1)
            actual = actual.reindex(sorted(actual.columns), axis=1)

        if list(expected.columns) != list(actual.columns):
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                reason=(
                    f"Column name mismatch: expected {list(expected.columns)}, "
                    f"got {list(actual.columns)}"
                ),
                category=self.category,
                mode=self.mode,
                threshold=self.threshold,
                details={
                    "expected_columns": list(expected.columns),
                    "actual_columns": list(actual.columns),
                },
            )

        if self.ignore_order:
            expected = expected.sort_values(by=list(expected.columns)).reset_index(drop=True)
            actual = actual.sort_values(by=list(actual.columns)).reset_index(drop=True)

        if expected.shape[0] != actual.shape[0]:
            min_rows = min(expected.shape[0], actual.shape[0])
            score = min_rows / max(expected.shape[0], actual.shape[0])
            return MetricResult(
                metric_name=self.name,
                score=score,
                reason=(f"Row count mismatch: expected {expected.shape[0]}, got {actual.shape[0]}"),
                category=self.category,
                mode=self.mode,
                threshold=self.threshold,
                details={
                    "expected_rows": expected.shape[0],
                    "actual_rows": actual.shape[0],
                },
            )

        matching_rows = 0
        mismatched_cells = []
        for idx in range(expected.shape[0]):
            row_match = True
            for col in expected.columns:
                exp_val = expected.iloc[idx][col]
                act_val = actual.iloc[idx][col]
                if not self._values_match(exp_val, act_val):
                    row_match = False
                    if len(mismatched_cells) < 5:
                        mismatched_cells.append(
                            f"Row {idx}, Col {col}: expected={exp_val!r}, actual={act_val!r}"
                        )
            if row_match:
                matching_rows += 1

        score = matching_rows / expected.shape[0]

        if score == 1.0:
            reason = "Result sets match exactly."
        elif score == 0.0:
            reason = "No matching rows between expected and actual results."
        else:
            reason = (
                f"{matching_rows}/{expected.shape[0]} rows match. "
                f"Mismatches: {'; '.join(mismatched_cells[:3])}"
            )

        return MetricResult(
            metric_name=self.name,
            score=score,
            reason=reason,
            category=self.category,
            mode=self.mode,
            threshold=self.threshold,
            details={
                "matching_rows": matching_rows,
                "total_rows": expected.shape[0],
                "mismatched_cells": mismatched_cells,
            },
        )

    def _values_match(self, expected: Any, actual: Any) -> bool:
        """Compare two values with float tolerance."""
        if expected is None and actual is None:
            return True
        if expected is None or actual is None:
            return False

        if pd.isna(expected) and pd.isna(actual):
            return True
        if pd.isna(expected) or pd.isna(actual):
            return False

        if isinstance(expected, float) and isinstance(actual, float):
            if expected == 0.0 and actual == 0.0:
                return True
            return abs(expected - actual) <= self.float_tolerance * max(abs(expected), abs(actual))

        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            return abs(float(expected) - float(actual)) <= (
                self.float_tolerance * max(abs(float(expected)), abs(float(actual)), 1e-10)
            )

        if isinstance(expected, str) and isinstance(actual, str):
            return expected.strip() == actual.strip()

        return expected == actual
