"""Tests for EvalPipeline SQL execution integration.

Tests that the pipeline can execute SQL to populate DataFrames
when a SQLExecutor is provided, bridging the gap between SQL text
and the DataFrames that metrics need.
"""

import sqlite3
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from analytics_eval.agents.base import AgentInterface, AgentResponse
from analytics_eval.execution.sqlite_executor import SQLiteExecutor
from analytics_eval.metrics.result.grain_correctness import GrainCorrectness
from analytics_eval.metrics.sql.execution_accuracy import ExecutionAccuracy
from analytics_eval.pipeline.base import EvalPipeline
from analytics_eval.test_case.case import AnalyticsTestCase


@pytest.fixture
def test_sqlite_db():
    """Create a test SQLite database."""
    tmp = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    db_path = tmp.name
    tmp.close()

    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            region TEXT,
            amount REAL
        )
    """)
    conn.execute("INSERT INTO orders VALUES (1, 'East', 100.0)")
    conn.execute("INSERT INTO orders VALUES (2, 'West', 200.0)")
    conn.execute("INSERT INTO orders VALUES (3, 'East', 150.0)")
    conn.commit()
    conn.close()

    yield db_path

    Path(db_path).unlink(missing_ok=True)


class PerfectSQLAgent(AgentInterface):
    """Agent that returns the expected SQL."""

    async def query(self, question, db_id, context=None):
        return AgentResponse(
            actual_sql=(
                "SELECT region, SUM(amount) as total "
                "FROM orders GROUP BY region"
            ),
            latency_ms=50.0,
        )


class BrokenSQLAgent(AgentInterface):
    """Agent that returns incorrect SQL."""

    async def query(self, question, db_id, context=None):
        return AgentResponse(
            actual_sql="SELECT region FROM orders",
            latency_ms=30.0,
        )


class SyntaxErrorAgent(AgentInterface):
    """Agent that returns SQL with syntax errors."""

    async def query(self, question, db_id, context=None):
        return AgentResponse(
            actual_sql="SELEC region FROM orders",
            latency_ms=10.0,
        )


class TestPipelineSQLExecution:
    """Test that the pipeline executes SQL to populate DataFrames."""

    @pytest.mark.asyncio
    async def test_pipeline_executes_expected_and_actual_sql(
        self, test_sqlite_db
    ):
        """Pipeline should execute both expected_sql and actual_sql."""
        executor = SQLiteExecutor(db_path=test_sqlite_db)
        case = AnalyticsTestCase(
            input="Revenue by region",
            db_id="test",
            expected_sql=(
                "SELECT region, SUM(amount) as total "
                "FROM orders GROUP BY region"
            ),
        )

        pipeline = EvalPipeline(
            agent=PerfectSQLAgent(),
            metrics=[ExecutionAccuracy()],
            executor=executor,
        )
        result = await pipeline.evaluate([case])

        assert result.total_cases == 1
        assert result.aggregated.overall == 1.0

    @pytest.mark.asyncio
    async def test_pipeline_with_incorrect_sql(self, test_sqlite_db):
        """Pipeline should detect incorrect SQL via result comparison."""
        executor = SQLiteExecutor(db_path=test_sqlite_db)
        case = AnalyticsTestCase(
            input="Revenue by region",
            db_id="test",
            expected_sql=(
                "SELECT region, SUM(amount) as total "
                "FROM orders GROUP BY region"
            ),
        )

        pipeline = EvalPipeline(
            agent=BrokenSQLAgent(),
            metrics=[ExecutionAccuracy()],
            executor=executor,
        )
        result = await pipeline.evaluate([case])

        assert result.total_cases == 1
        assert result.aggregated.overall < 1.0

    @pytest.mark.asyncio
    async def test_pipeline_handles_sql_execution_error(self, test_sqlite_db):
        """Pipeline should handle SQL execution errors gracefully."""
        executor = SQLiteExecutor(db_path=test_sqlite_db)
        case = AnalyticsTestCase(
            input="Revenue by region",
            db_id="test",
            expected_sql=(
                "SELECT region, SUM(amount) as total "
                "FROM orders GROUP BY region"
            ),
        )

        pipeline = EvalPipeline(
            agent=SyntaxErrorAgent(),
            metrics=[ExecutionAccuracy()],
            executor=executor,
        )
        result = await pipeline.evaluate([case])

        assert result.total_cases == 1
        assert result.sql_execution_errors > 0

    @pytest.mark.asyncio
    async def test_pipeline_with_executor_factory(self, test_sqlite_db):
        """Pipeline should use executor_factory for multi-database scenarios."""
        executor = SQLiteExecutor(db_path=test_sqlite_db)

        def factory(db_id: str):
            if db_id == "test":
                return executor
            return None

        case = AnalyticsTestCase(
            input="Revenue by region",
            db_id="test",
            expected_sql=(
                "SELECT region, SUM(amount) as total "
                "FROM orders GROUP BY region"
            ),
        )

        pipeline = EvalPipeline(
            agent=PerfectSQLAgent(),
            metrics=[ExecutionAccuracy()],
            executor_factory=factory,
        )
        result = await pipeline.evaluate([case])
        assert result.aggregated.overall == 1.0

    @pytest.mark.asyncio
    async def test_pipeline_without_executor_skips_sql(self, test_sqlite_db):
        """Pipeline without executor should work with pre-populated DataFrames."""
        case = AnalyticsTestCase(
            input="Revenue by region",
            expected_results=pd.DataFrame({
                "region": ["East", "West"],
                "total": [250.0, 200.0],
            }),
            actual_results=pd.DataFrame({
                "region": ["East", "West"],
                "total": [250.0, 200.0],
            }),
        )

        pipeline = EvalPipeline(
            agent=PerfectSQLAgent(),
            metrics=[ExecutionAccuracy()],
            executor=None,
        )
        result = await pipeline.evaluate([case])
        assert result.aggregated.overall == 1.0

    @pytest.mark.asyncio
    async def test_pipeline_provides_schema_to_agent(self, test_sqlite_db):
        """Pipeline should provide schema from executor to agent."""
        executor = SQLiteExecutor(db_path=test_sqlite_db)

        class SchemaCapturingAgent(AgentInterface):
            def __init__(self):
                self.received_context = None

            async def query(self, question, db_id, context=None):
                self.received_context = context
                return AgentResponse(
                    actual_sql="SELECT COUNT(*) FROM orders",
                    latency_ms=10.0,
                )

        agent = SchemaCapturingAgent()
        case = AnalyticsTestCase(
            input="How many orders?",
            db_id="test",
            expected_sql="SELECT COUNT(*) FROM orders",
        )

        pipeline = EvalPipeline(
            agent=agent,
            metrics=[ExecutionAccuracy()],
            executor=executor,
        )
        await pipeline.evaluate([case])

        assert agent.received_context is not None
        assert "schema" in agent.received_context

    @pytest.mark.asyncio
    async def test_pipeline_multiple_metrics_with_sql_execution(
        self, test_sqlite_db
    ):
        """Pipeline should work with multiple metrics when executing SQL."""
        executor = SQLiteExecutor(db_path=test_sqlite_db)
        case = AnalyticsTestCase(
            input="Revenue by region",
            db_id="test",
            expected_sql=(
                "SELECT region, SUM(amount) as total "
                "FROM orders GROUP BY region"
            ),
            semantic_context={
                "schema": {
                    "tables": ["orders"],
                    "columns": {"orders": ["id", "region", "amount"]},
                },
            },
        )

        pipeline = EvalPipeline(
            agent=PerfectSQLAgent(),
            metrics=[
                ExecutionAccuracy(threshold=0.5),
                GrainCorrectness(threshold=0.5),
            ],
            executor=executor,
        )
        result = await pipeline.evaluate([case])
        assert result.total_cases == 1
        assert "ExecutionAccuracy" in result.aggregated.by_metric
        assert "GrainCorrectness" in result.aggregated.by_metric


class TestPipelineSyncWithSQL:
    """Test synchronous pipeline with SQL execution."""

    def test_evaluate_sync_with_sql(self, test_sqlite_db):
        """Synchronous pipeline should work with SQL execution."""
        executor = SQLiteExecutor(db_path=test_sqlite_db)
        case = AnalyticsTestCase(
            input="How many orders?",
            db_id="test",
            expected_sql="SELECT COUNT(*) as cnt FROM orders",
        )

        class CountAgent(AgentInterface):
            async def query(self, question, db_id, context=None):
                return AgentResponse(
                    actual_sql="SELECT COUNT(*) as cnt FROM orders",
                    latency_ms=10.0,
                )

        pipeline = EvalPipeline(
            agent=CountAgent(),
            metrics=[ExecutionAccuracy()],
            executor=executor,
        )
        result = pipeline.evaluate_sync([case])
        assert result.aggregated.overall == 1.0
