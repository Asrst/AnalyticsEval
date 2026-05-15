"""Metrics package — all evaluation metrics for analytics agents.

Usage::

    from analytics_eval.metrics import ExecutionAccuracy, SchemaAdherence
    from analytics_eval.metrics import AnalyticsMetric, MetricResult

Quick reference:
    SQL Correctness:    ExecutionAccuracy, SchemaAdherence
    Result Quality:     ResultAccuracy, GrainCorrectness
    Insight Quality:    (planned)
    Visualization:      (planned)
    Model Quality:      (planned)
"""

from analytics_eval.metrics.base import (
    AnalyticsMetric,
    EvaluationMode,
    MetricCategory,
    MetricResult,
)
from analytics_eval.metrics.result.grain_correctness import GrainCorrectness
from analytics_eval.metrics.result.result_accuracy import ResultAccuracy
from analytics_eval.metrics.sql.execution_accuracy import ExecutionAccuracy
from analytics_eval.metrics.sql.schema_adherence import SchemaAdherence

__all__ = [
    # Base classes
    "AnalyticsMetric",
    "MetricResult",
    "MetricCategory",
    "EvaluationMode",
    # SQL Correctness metrics
    "ExecutionAccuracy",
    "SchemaAdherence",
    # Result Quality metrics
    "ResultAccuracy",
    "GrainCorrectness",
]
