"""Evaluation pipeline — orchestrates the Plan-Execute-Evaluate cycle.

Design principles:
- Pipeline is composable: swap metrics, agents, datasets independently
- EvalConfig is a Pydantic model for reproducibility
- EvalResult aggregates metric results with per-metric and aggregate scores
- Supports parallel execution and timeout handling
- SQL execution: when an executor is provided, the pipeline executes both
  expected_sql and actual_sql to populate DataFrames for metrics
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

from pydantic import BaseModel, Field

from analytics_eval.agents.base import AgentInterface
from analytics_eval.execution.base import SQLExecutionError, SQLExecutor
from analytics_eval.metrics.base import (
    AnalyticsMetric,
    MetricResult,
)
from analytics_eval.test_case.case import AnalyticsTestCase


class EvalConfig(BaseModel):
    """Configuration for an evaluation pipeline run.

    This is a Pydantic model that captures all configuration needed for a
    reproducible evaluation run. It can be serialized to YAML/JSON for
    version control and sharing.
    """

    metrics: list[str] = Field(
        default_factory=lambda: ["ExecutionAccuracy", "ResultAccuracy"],
        description="Names of metrics to apply",
    )
    thresholds: dict[str, float] = Field(
        default_factory=dict,
        description="Per-metric pass thresholds (overrides metric defaults)",
    )
    timeout_seconds: float = Field(
        default=30.0,
        description="Per-query timeout in seconds",
        ge=1.0,
    )
    parallel: int = Field(
        default=1,
        description="Number of parallel agent evaluations",
        ge=1,
    )
    fail_on_error: bool = Field(
        default=False,
        description="Whether to stop the pipeline on agent errors",
    )
    skip_missing_fields: bool = Field(
        default=True,
        description="Skip metrics when required test case fields are missing",
    )
    execute_sql: bool = Field(
        default=True,
        description="Whether to execute SQL to populate result DataFrames",
    )


class AggregatedScores(BaseModel):
    """Aggregated scores across all test cases, grouped by metric."""

    by_metric: dict[str, float] = Field(
        default_factory=dict,
        description="Mean score per metric name",
    )
    by_category: dict[str, float] = Field(
        default_factory=dict,
        description="Mean score per metric category",
    )
    overall: float = Field(
        default=0.0,
        description="Overall mean score across all metrics and cases",
        ge=0.0,
        le=1.0,
    )
    pass_rate: float = Field(
        default=0.0,
        description="Fraction of metric-case pairs that passed threshold",
        ge=0.0,
        le=1.0,
    )


class EvalResult(BaseModel):
    """Complete result of an evaluation pipeline run.

    Contains both per-case results and aggregated scores, along with
    metadata about the evaluation run (timing, configuration, etc.).
    """

    case_results: list[dict[str, MetricResult]] = Field(
        default_factory=list,
        description="Per-test-case metric results, keyed by metric name",
    )
    aggregated: AggregatedScores = Field(
        default_factory=AggregatedScores,
        description="Aggregated scores across all cases",
    )
    total_cases: int = Field(
        default=0,
        description="Total number of test cases evaluated",
    )
    total_time_ms: float = Field(
        default=0.0,
        description="Total evaluation time in milliseconds",
        ge=0,
    )
    errors: list[str] = Field(
        default_factory=list,
        description="Errors encountered during evaluation",
    )
    config: EvalConfig | None = Field(
        default=None,
        description="Configuration used for this evaluation run",
    )
    sql_execution_errors: int = Field(
        default=0,
        description="Number of SQL execution failures",
    )


class EvalPipeline:
    """Orchestrates the end-to-end evaluation: Plan-Execute-Evaluate.

    The pipeline takes an agent, a list of metrics, and a dataset (list of
    test cases), then:
    1. Sends each question to the agent (to get actual_sql)
    2. Executes both expected_sql and actual_sql against the database
    3. Applies each metric to produce scores
    4. Aggregates results across all cases and metrics
    5. Produces an EvalResult with scores, reasons, and suggestions

    SQL Execution:
        When an executor is provided (via executor_factory or executor):
        - expected_sql is executed to populate expected_results
        - actual_sql (from agent) is executed to populate actual_results
        - SQL execution errors are captured, not fatal

    Usage (with SQL execution):
        from analytics_eval.datasets.adapters import BirdAdapter
        from analytics_eval.agents import LLMSQLAgent

        adapter = BirdAdapter(data_dir="/data/bird")
        agent = LLMSQLAgent(llm_call=my_llm)

        pipeline = EvalPipeline(
            agent=agent,
            metrics=[ExecutionAccuracy(), SchemaAdherence()],
            executor_factory=adapter.get_executor_for_db,
        )
        result = pipeline.evaluate_sync(adapter.load(split="mini_dev"))

    Usage (without SQL execution — pre-populated DataFrames):
        pipeline = EvalPipeline(
            agent=my_agent,
            metrics=[ExecutionAccuracy(), ResultAccuracy()],
        )
        result = pipeline.evaluate_sync(test_cases)
    """

    def __init__(
        self,
        agent: AgentInterface,
        metrics: Sequence[AnalyticsMetric],
        config: EvalConfig | None = None,
        executor: SQLExecutor | None = None,
        executor_factory: (
            Callable[[str], SQLExecutor | None] | None
        ) = None,
    ) -> None:
        """Initialize the evaluation pipeline.

        Args:
            agent: The analytics agent to evaluate.
            metrics: Sequence of metrics to apply.
            config: Optional pipeline configuration.
            executor: A single SQLExecutor for all databases.
            executor_factory: A callable that takes db_id and returns
                            a SQLExecutor. Used when different test cases
                            target different databases (e.g., BIRD).
        """
        self.agent = agent
        self.metrics = list(metrics)
        self.config = config or EvalConfig()
        self.executor = executor
        self.executor_factory = executor_factory

        # Apply config thresholds to metrics
        for metric in self.metrics:
            if metric.name in self.config.thresholds:
                metric.threshold = self.config.thresholds[metric.name]

    def _get_executor(self, db_id: str | None) -> SQLExecutor | None:
        """Get the SQL executor for a given database.

        Priority:
        1. executor (if set — single database mode)
        2. executor_factory(db_id) (if set — multi-database mode)
        3. None (no SQL execution)
        """
        if self.executor is not None:
            return self.executor
        if self.executor_factory is not None and db_id is not None:
            return self.executor_factory(db_id)
        return None

    async def evaluate(
        self,
        test_cases: Sequence[AnalyticsTestCase],
    ) -> EvalResult:
        """Run evaluation across all test cases and metrics.

        Args:
            test_cases: The analytics interactions to evaluate against.

        Returns:
            EvalResult with per-case results and aggregated scores.
        """
        start_time = time.monotonic()
        all_case_results: list[dict[str, MetricResult]] = []
        errors: list[str] = []
        sql_errors = 0

        for i, case in enumerate(test_cases):
            populated_case = await self._execute_agent_if_needed(case)

            if self.config.execute_sql:
                populated_case, sql_err = self._execute_sql_if_needed(
                    populated_case
                )
                if sql_err:
                    sql_errors += 1
                    errors.append(
                        f"Case {i} ({case.input[:50]}): {sql_err}"
                    )

            case_results: dict[str, MetricResult] = {}
            for metric in self.metrics:
                result = metric.evaluate(populated_case)
                case_results[metric.name] = result

            all_case_results.append(case_results)

        total_time_ms = (time.monotonic() - start_time) * 1000

        aggregated = self._aggregate(all_case_results)

        return EvalResult(
            case_results=all_case_results,
            aggregated=aggregated,
            total_cases=len(test_cases),
            total_time_ms=total_time_ms,
            errors=errors,
            config=self.config,
            sql_execution_errors=sql_errors,
        )

    def evaluate_sync(
        self,
        test_cases: Sequence[AnalyticsTestCase],
    ) -> EvalResult:
        """Synchronous wrapper for evaluate()."""
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run, self.evaluate(test_cases)
                )
                return future.result()
        else:
            return asyncio.run(self.evaluate(test_cases))

    async def _execute_agent_if_needed(
        self, case: AnalyticsTestCase
    ) -> AnalyticsTestCase:
        """Call the agent if the case doesn't already have actual SQL."""
        if case.has_actual_sql():
            return case

        if case.db_id is None:
            return case

        try:
            agent_context = case.semantic_context or {}
            if case.evidence:
                agent_context["evidence"] = case.evidence

            executor = self._get_executor(case.db_id)
            if executor:
                schema = executor.get_schema()
                if schema:
                    agent_context["schema"] = schema

            response = await self.agent.query(
                question=case.input,
                db_id=case.db_id,
                context=agent_context,
            )
            if response.actual_sql is not None:
                case.actual_sql = response.actual_sql
            if response.actual_results is not None:
                case.actual_results = response.actual_results
            if response.insight_text is not None:
                case.insight_text = response.insight_text
            if response.chart_spec is not None:
                case.chart_spec = response.chart_spec
            if response.latency_ms is not None:
                case.latency_ms = response.latency_ms
            if response.token_count is not None:
                case.token_count = response.token_count
            if response.cost_usd is not None:
                case.cost_usd = response.cost_usd
            if response.error is not None:
                case.metadata["agent_error"] = response.error
        except Exception as e:
            case.metadata["agent_error"] = f"{type(e).__name__}: {e}"

        return case

    def _execute_sql_if_needed(
        self, case: AnalyticsTestCase
    ) -> tuple[AnalyticsTestCase, str | None]:
        """Execute expected_sql and actual_sql to populate result DataFrames.

        Returns:
            Tuple of (populated case, error message or None)
        """
        executor = self._get_executor(case.db_id)
        if executor is None:
            return case, None

        error_msg = None

        if case.has_expected_sql() and not case.has_expected_results():
            try:
                case.expected_results = executor.execute(
                    case.expected_sql,
                    timeout=self.config.timeout_seconds,
                )
            except SQLExecutionError as e:
                error_msg = f"Expected SQL execution failed: {e}"
                case.metadata["expected_sql_error"] = str(e)

        if case.has_actual_sql() and not case.has_actual_results():
            try:
                case.actual_results = executor.execute(
                    case.actual_sql,
                    timeout=self.config.timeout_seconds,
                )
            except SQLExecutionError as e:
                error_msg = f"Actual SQL execution failed: {e}"
                case.metadata["actual_sql_error"] = str(e)

        return case, error_msg

    def _aggregate(
        self, case_results: list[dict[str, MetricResult]]
    ) -> AggregatedScores:
        """Compute aggregated scores from per-case results."""
        if not case_results:
            return AggregatedScores()

        metric_scores: dict[str, list[float]] = {}
        category_scores: dict[str, list[float]] = {}
        all_scores: list[float] = []
        pass_count = 0
        total_count = 0

        for case_result in case_results:
            for metric_name, result in case_result.items():
                metric_scores.setdefault(
                    metric_name, []
                ).append(result.score)
                cat = result.category.value
                category_scores.setdefault(cat, []).append(result.score)
                all_scores.append(result.score)
                total_count += 1
                if result.passed:
                    pass_count += 1

        by_metric = {
            k: sum(v) / len(v) for k, v in metric_scores.items()
        }
        by_category = {
            k: sum(v) / len(v) for k, v in category_scores.items()
        }
        overall = sum(all_scores) / len(all_scores) if all_scores else 0.0
        pass_rate = pass_count / total_count if total_count > 0 else 0.0

        return AggregatedScores(
            by_metric=by_metric,
            by_category=by_category,
            overall=overall,
            pass_rate=pass_rate,
        )
