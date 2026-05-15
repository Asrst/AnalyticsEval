"""Base metric class and metric result model.

Design principles:
- Every metric produces a MetricResult (score 0-1 + reason + details)
- Metrics declare required fields so the pipeline can skip when data is missing
- Supports both deterministic and LLM-as-judge evaluation
- Extensible via subclassing — new metrics inherit AnalyticsMetric
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from analytics_eval.test_case.case import AnalyticsTestCase


class MetricCategory(StrEnum):
    """Families of analytics evaluation metrics."""

    SQL_CORRECTNESS = "sql_correctness"
    RESULT_QUALITY = "result_quality"
    INSIGHT_QUALITY = "insight_quality"
    VISUALIZATION_QUALITY = "visualization_quality"
    MODEL_QUALITY = "model_quality"


class EvaluationMode(StrEnum):
    """How a metric evaluates — deterministic or LLM-as-judge."""

    DETERMINISTIC = "deterministic"
    LLM_AS_JUDGE = "llm_as_judge"
    HYBRID = "hybrid"


class MetricResult(BaseModel):
    """The output of evaluating a metric against a test case.

    Design decisions:
    - score is always 0.0-1.0 for consistency across metrics
    - reason is always provided — no silent failures
    - passed is derived from score vs threshold
    - details dict enables metric-specific structured output
    - evaluation_time_ms tracks cost for efficiency scoring
    """

    metric_name: str = Field(
        ...,
        description="Name of the metric that produced this result",
    )
    score: float = Field(
        ...,
        description="Score between 0.0 and 1.0",
        ge=0.0,
        le=1.0,
    )
    reason: str = Field(
        ...,
        description="Human-readable explanation of the score",
        min_length=1,
    )
    category: MetricCategory = Field(
        ...,
        description="Metric family this result belongs to",
    )
    mode: EvaluationMode = Field(
        ...,
        description="How this metric was evaluated",
    )
    threshold: float = Field(
        default=0.5,
        description="Pass threshold (0-1). score >= threshold means passed.",
        ge=0.0,
        le=1.0,
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Metric-specific structured output (e.g., diff, claims)",
    )
    evaluation_time_ms: float | None = Field(
        default=None,
        description="Time taken to evaluate this metric in milliseconds",
        ge=0,
    )

    @field_validator("reason")
    @classmethod
    def reason_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("reason must not be empty")
        return v

    @property
    def passed(self) -> bool:
        """Whether the metric score meets the threshold."""
        return self.score >= self.threshold

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.metric_name}: {self.score:.3f} "
            f"(threshold={self.threshold}) — {self.reason}"
        )


class AnalyticsMetric(ABC):
    """Abstract base class for all analytics evaluation metrics.

    Subclasses must implement:
    - measure(test_case) -> MetricResult: The core evaluation logic
    - required_fields: Which AnalyticsTestCase fields must be populated

    Subclasses should set:
    - name: Human-readable metric name
    - category: Which metric family this belongs to
    - mode: Deterministic, LLM-as-judge, or Hybrid
    - threshold: Default pass threshold

    Design decisions:
    - Class-based (not function-based) to enable configuration per metric
    - threshold is per-metric and configurable at construction time
    - required_fields enables the pipeline to skip metrics when data is missing
      instead of failing silently or throwing errors
    - measure() is synchronous; LLM-backed metrics handle async internally
    """

    def __init__(
        self,
        threshold: float = 0.5,
        **kwargs: Any,
    ) -> None:
        self.threshold = threshold

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable metric name (e.g., 'ExecutionAccuracy')."""
        ...

    @property
    @abstractmethod
    def category(self) -> MetricCategory:
        """Which metric family this belongs to."""
        ...

    @property
    @abstractmethod
    def mode(self) -> EvaluationMode:
        """How this metric evaluates."""
        ...

    @property
    @abstractmethod
    def required_fields(self) -> list[str]:
        """AnalyticsTestCase fields that must be non-None for this metric to run.

        The pipeline checks these before calling measure(). If any are missing,
        the metric is skipped and a skip result is produced instead.
        Field names correspond to AnalyticsTestCase attribute names.
        """
        ...

    @abstractmethod
    def measure(self, test_case: AnalyticsTestCase) -> MetricResult:
        """Evaluate the test case and produce a score.

        This is the core method that every metric must implement.
        It receives a populated AnalyticsTestCase and returns a MetricResult
        with a score (0-1), a reason, and optional details.

        Args:
            test_case: The analytics interaction to evaluate.

        Returns:
            MetricResult with score, reason, and metric-specific details.
        """
        ...

    def can_evaluate(self, test_case: AnalyticsTestCase) -> bool:
        """Check whether this metric has all required data to evaluate.

        Uses the required_fields property to check if the test case
        has the necessary data populated. This is called by the pipeline
        before measure() to enable graceful degradation.
        """
        for field_name in self.required_fields:
            value = getattr(test_case, field_name, None)
            if value is None:
                return False
            # For string fields, check not just whitespace
            if isinstance(value, str) and not value.strip():
                return False
            # For DataFrame fields, check not empty
            if hasattr(value, "empty") and value.empty:
                return False
            # For dict fields, check not empty
            if isinstance(value, dict) and not value:
                return False
        return True

    def evaluate(self, test_case: AnalyticsTestCase) -> MetricResult:
        """Evaluate with timing, validation, and error handling.

        This is the public API called by the pipeline. It wraps measure()
        with:
        1. Pre-validation: check required_fields
        2. Timing: measure evaluation latency
        3. Error handling: catch exceptions and return score=0 with reason

        Subclasses should override measure(), not evaluate().
        """
        if not self.can_evaluate(test_case):
            missing = [
                f for f in self.required_fields
                if getattr(test_case, f, None) is None
            ]
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                reason=f"Skipped: missing required fields: {', '.join(missing)}",
                category=self.category,
                mode=self.mode,
                threshold=self.threshold,
                details={"skipped": True, "missing_fields": missing},
            )

        start_time = time.monotonic()
        try:
            result = self.measure(test_case)
        except Exception as e:
            elapsed = (time.monotonic() - start_time) * 1000
            result = MetricResult(
                metric_name=self.name,
                score=0.0,
                reason=f"Error during evaluation: {type(e).__name__}: {e}",
                category=self.category,
                mode=self.mode,
                threshold=self.threshold,
                details={"error": str(e), "error_type": type(e).__name__},
                evaluation_time_ms=elapsed,
            )
        else:
            elapsed = (time.monotonic() - start_time) * 1000
            result.evaluation_time_ms = elapsed

        return result

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, threshold={self.threshold})"
