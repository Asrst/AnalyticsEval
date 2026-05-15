"""Tests for EvalPipeline — the evaluation orchestrator."""

import pandas as pd
import pytest

from analytics_eval.agents.base import AgentInterface, AgentResponse
from analytics_eval.metrics.result.result_accuracy import ResultAccuracy
from analytics_eval.metrics.sql.execution_accuracy import ExecutionAccuracy
from analytics_eval.pipeline.base import EvalConfig, EvalPipeline
from analytics_eval.test_case.case import AnalyticsTestCase


class MatchingAgent(AgentInterface):
    """Returns results that match ground truth."""

    async def query(self, question, db_id, context=None):
        return AgentResponse(
            actual_sql="SELECT SUM(amount) FROM orders",
            actual_results=pd.DataFrame({"total": [4500.0]}),
            latency_ms=50.0,
        )


class MismatchAgent(AgentInterface):
    """Returns results that don't match ground truth."""

    async def query(self, question, db_id, context=None):
        return AgentResponse(
            actual_sql="SELECT COUNT(*) FROM orders",
            actual_results=pd.DataFrame({"count": [100]}),
            latency_ms=30.0,
        )


class TestEvalPipeline:
    """Test the EvalPipeline orchestration."""

    @pytest.mark.asyncio
    async def test_evaluate_with_prepopulated_results(self):
        """Pipeline should work with test cases that already have results."""
        cases = [
            AnalyticsTestCase(
                input="What is revenue?",
                expected_results=pd.DataFrame({"total": [4500.0]}),
                actual_results=pd.DataFrame({"total": [4500.0]}),
            ),
        ]
        pipeline = EvalPipeline(
            agent=MatchingAgent(),
            metrics=[ExecutionAccuracy()],
        )
        result = await pipeline.evaluate(cases)
        assert result.total_cases == 1
        assert result.aggregated.overall == 1.0

    @pytest.mark.asyncio
    async def test_evaluate_with_agent_execution(self):
        """Pipeline should call the agent when actual results are missing."""
        cases = [
            AnalyticsTestCase(
                input="What is revenue?",
                db_id="sales_db",
                expected_results=pd.DataFrame({"total": [4500.0]}),
            ),
        ]
        pipeline = EvalPipeline(
            agent=MatchingAgent(),
            metrics=[ExecutionAccuracy()],
        )
        result = await pipeline.evaluate(cases)
        assert result.total_cases == 1

    @pytest.mark.asyncio
    async def test_evaluate_multiple_cases(self):
        """Pipeline should handle multiple test cases."""
        cases = [
            AnalyticsTestCase(
                input="What is revenue?",
                expected_results=pd.DataFrame({"total": [4500.0]}),
                actual_results=pd.DataFrame({"total": [4500.0]}),
            ),
            AnalyticsTestCase(
                input="What is order count?",
                expected_results=pd.DataFrame({"count": [100]}),
                actual_results=pd.DataFrame({"count": [100]}),
            ),
        ]
        pipeline = EvalPipeline(
            agent=MatchingAgent(),
            metrics=[ExecutionAccuracy()],
        )
        result = await pipeline.evaluate(cases)
        assert result.total_cases == 2
        assert result.aggregated.overall == 1.0

    @pytest.mark.asyncio
    async def test_aggregated_scores_by_metric(self):
        """Aggregated scores should be computed per metric."""
        cases = [
            AnalyticsTestCase(
                input="Q",
                expected_results=pd.DataFrame({"v": [100.0]}),
                actual_results=pd.DataFrame({"v": [100.0]}),
            ),
        ]
        pipeline = EvalPipeline(
            agent=MatchingAgent(),
            metrics=[ExecutionAccuracy(), ResultAccuracy()],
        )
        result = await pipeline.evaluate(cases)
        assert "ExecutionAccuracy" in result.aggregated.by_metric
        assert "ResultAccuracy" in result.aggregated.by_metric

    @pytest.mark.asyncio
    async def test_pass_rate(self):
        """Pass rate should reflect fraction of metric-case pairs that passed."""
        cases = [
            AnalyticsTestCase(
                input="Q",
                expected_results=pd.DataFrame({"v": [100.0]}),
                actual_results=pd.DataFrame({"v": [200.0]}),
            ),
        ]
        pipeline = EvalPipeline(
            agent=MatchingAgent(),
            metrics=[ExecutionAccuracy(threshold=0.5)],
        )
        result = await pipeline.evaluate(cases)
        assert result.aggregated.pass_rate == 0.0

    @pytest.mark.asyncio
    async def test_timing_is_recorded(self):
        """Total evaluation time should be recorded."""
        cases = [
            AnalyticsTestCase(
                input="Q",
                expected_results=pd.DataFrame({"v": [1]}),
                actual_results=pd.DataFrame({"v": [1]}),
            ),
        ]
        pipeline = EvalPipeline(
            agent=MatchingAgent(),
            metrics=[ExecutionAccuracy()],
        )
        result = await pipeline.evaluate(cases)
        assert result.total_time_ms > 0


class TestEvalPipelineSync:
    """Test synchronous pipeline wrapper."""

    def test_evaluate_sync(self):
        cases = [
            AnalyticsTestCase(
                input="Q",
                expected_results=pd.DataFrame({"v": [1]}),
                actual_results=pd.DataFrame({"v": [1]}),
            ),
        ]
        pipeline = EvalPipeline(
            agent=MatchingAgent(),
            metrics=[ExecutionAccuracy()],
        )
        result = pipeline.evaluate_sync(cases)
        assert result.total_cases == 1
        assert result.aggregated.overall == 1.0


class TestEvalConfig:
    """Test pipeline configuration."""

    def test_default_config(self):
        config = EvalConfig()
        assert config.timeout_seconds == 30.0
        assert config.parallel == 1
        assert config.skip_missing_fields is True

    def test_custom_thresholds_applied(self):
        """Config thresholds should override metric defaults."""
        config = EvalConfig(thresholds={"ExecutionAccuracy": 0.9})
        metric = ExecutionAccuracy()
        EvalPipeline(
            agent=MatchingAgent(),
            metrics=[metric],
            config=config,
        )
        assert metric.threshold == 0.9
