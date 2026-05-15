"""Grain Correctness metric — checks the aggregation level of query results.

Verifies that the agent's SQL produces results at the correct grain
(level of detail). A common failure mode is generating SQL that aggregates
at the wrong level (e.g., daily instead of monthly, or missing a GROUP BY
dimension). This metric compares the structure of expected vs actual results
to detect grain mismatches.

Grain is determined by:
1. Number of rows (cardinality)
2. Column names and types (dimension vs measure columns)
3. Uniqueness of dimension columns (what defines each row)
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


class GrainCorrectness(AnalyticsMetric):
    """Measures whether the agent's results are at the correct grain.

    Grain correctness checks that the result set has the right structure:
    - Same number of dimension (grouping) columns
    - Same uniqueness pattern in dimension columns
    - Appropriate number of rows for the intended grain

    Scoring:
    - 1.0: Grain matches exactly (same columns, same cardinality)
    - 0.5+: Right columns but wrong cardinality (partial credit)
    - 0.0: Completely wrong grain (missing/extra dimension columns)

    Configuration:
    - threshold: Default 0.7
    - cardinality_tolerance: How much row count can differ (default 0.2)
    """

    def __init__(
        self,
        threshold: float = 0.7,
        cardinality_tolerance: float = 0.2,
        **kwargs: Any,
    ) -> None:
        super().__init__(threshold=threshold, **kwargs)
        self.cardinality_tolerance = cardinality_tolerance

    @property
    def name(self) -> str:
        return "GrainCorrectness"

    @property
    def category(self) -> MetricCategory:
        return MetricCategory.RESULT_QUALITY

    @property
    def mode(self) -> EvaluationMode:
        return EvaluationMode.DETERMINISTIC

    @property
    def required_fields(self) -> list[str]:
        return ["expected_results", "actual_results"]

    def can_evaluate(self, test_case: AnalyticsTestCase) -> bool:
        """Check if grain evaluation is possible.

        Override base class to allow empty DataFrames — grain comparison
        can still check column structure even with no rows.
        """
        for field_name in self.required_fields:
            value = getattr(test_case, field_name, None)
            if value is None:
                return False
        return True

    def measure(self, test_case: AnalyticsTestCase) -> MetricResult:
        """Compare the grain of expected vs actual results."""
        expected = test_case.expected_results
        actual = test_case.actual_results

        assert expected is not None and actual is not None

        if expected.shape[0] == 0 and actual.shape[0] == 0:
            expected_cols = set(expected.columns)
            actual_cols = set(actual.columns)
            if expected_cols == actual_cols:
                return MetricResult(
                    metric_name=self.name,
                    score=1.0,
                    reason=("Both result sets are empty with matching column structure."),
                    category=self.category,
                    mode=self.mode,
                    threshold=self.threshold,
                )
            else:
                return MetricResult(
                    metric_name=self.name,
                    score=0.5,
                    reason="Both empty but column structures differ.",
                    category=self.category,
                    mode=self.mode,
                    threshold=self.threshold,
                    details={
                        "expected_columns": sorted(expected_cols),
                        "actual_columns": sorted(actual_cols),
                    },
                )

        if expected.shape[0] == 0 or actual.shape[0] == 0:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                reason="One result set is empty while the other is not.",
                category=self.category,
                mode=self.mode,
                threshold=self.threshold,
            )

        expected_cols = set(expected.columns)
        actual_cols = set(actual.columns)

        expected_dims = self._get_dimension_columns(expected)
        actual_dims = self._get_dimension_columns(actual)

        dim_match = expected_dims == actual_dims
        missing_dims = expected_dims - actual_dims
        extra_dims = actual_dims - expected_dims

        expected_rows = expected.shape[0]
        actual_rows = actual.shape[0]

        if expected_rows > 0:
            cardinality_ratio = min(expected_rows, actual_rows) / max(expected_rows, actual_rows)
        else:
            cardinality_ratio = 1.0 if actual_rows == 0 else 0.0

        cardinality_ok = cardinality_ratio >= (1.0 - self.cardinality_tolerance)

        expected_unique = self._get_unique_count(expected, expected_dims)
        actual_unique = self._get_unique_count(actual, actual_dims)

        score = 0.0

        if dim_match:
            col_score = 1.0
        elif len(missing_dims) == 0 and len(extra_dims) > 0:
            col_score = 0.6
        elif len(missing_dims) > 0 and len(extra_dims) == 0:
            col_score = 0.4
        else:
            col_score = 0.2

        card_score = cardinality_ratio

        score = 0.6 * col_score + 0.4 * card_score

        if dim_match and cardinality_ok:
            reason = "Grain matches: same dimensions and similar cardinality."
        elif dim_match and not cardinality_ok:
            reason = (
                f"Same dimensions but cardinality differs: "
                f"expected {expected_rows} rows, got {actual_rows} rows."
            )
        elif not dim_match:
            parts = []
            if missing_dims:
                parts.append(f"missing dimensions: {', '.join(sorted(missing_dims))}")
            if extra_dims:
                parts.append(f"extra dimensions: {', '.join(sorted(extra_dims))}")
            reason = f"Dimension mismatch: {'; '.join(parts)}."
        else:
            reason = "Grain does not match expected results."

        return MetricResult(
            metric_name=self.name,
            score=score,
            reason=reason,
            category=self.category,
            mode=self.mode,
            threshold=self.threshold,
            details={
                "expected_dimensions": sorted(expected_dims),
                "actual_dimensions": sorted(actual_dims),
                "missing_dimensions": sorted(missing_dims),
                "extra_dimensions": sorted(extra_dims),
                "expected_rows": expected_rows,
                "actual_rows": actual_rows,
                "cardinality_ratio": cardinality_ratio,
                "expected_unique_in_dims": expected_unique,
                "actual_unique_in_dims": actual_unique,
                "column_score": col_score,
                "cardinality_score": card_score,
            },
        )

    def _get_dimension_columns(self, df: pd.DataFrame) -> set[str]:
        """Identify dimension columns (typically non-numeric, or low-cardinality numeric).

        A column is considered a dimension if:
        - It has a non-numeric dtype (string, category, etc.), OR
        - It has a numeric dtype but low cardinality relative to total rows
          (heuristic: if unique count <= min(10, rows * 0.3))
        """
        dims = set()
        n_rows = df.shape[0]
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                dims.add(col)
            else:
                n_unique = df[col].nunique()
                if n_rows > 0 and n_unique <= min(10, max(2, n_rows * 0.3)):
                    dims.add(col)
        return dims

    def _get_unique_count(self, df: pd.DataFrame, dims: set[str]) -> int:
        """Count unique rows based on dimension columns."""
        if not dims:
            return df.shape[0]
        dim_cols = [c for c in df.columns if c in dims]
        return df[dim_cols].drop_duplicates().shape[0]
