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

__all__ = [
    # Base classes
    "AnalyticsMetric",
    "MetricResult",
    "MetricCategory",
    "EvaluationMode",
]
