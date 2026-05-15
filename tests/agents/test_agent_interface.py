"""Tests for AgentInterface base class."""

import pytest

from analytics_eval.agents.base import AgentInterface, AgentResponse


class StubAgent(AgentInterface):
    """A minimal agent for testing the interface contract."""

    def __init__(self, sql: str = "SELECT 1", results=None):
        self._sql = sql
        self._results = results

    async def query(self, question, db_id, context=None):
        return AgentResponse(
            actual_sql=self._sql,
            actual_results=self._results,
            insight_text="Here are the results.",
            latency_ms=42.0,
        )


class ErrorAgent(AgentInterface):
    """An agent that always fails."""

    async def query(self, question, db_id, context=None):
        raise ConnectionError("Database unavailable")


class TestAgentInterface:
    """Test the AgentInterface contract."""

    @pytest.mark.asyncio
    async def test_stub_agent_returns_response(self):
        agent = StubAgent()
        response = await agent.query("What is revenue?", "db1")
        assert response.actual_sql == "SELECT 1"
        assert response.insight_text == "Here are the results."
        assert response.latency_ms == 42.0

    @pytest.mark.asyncio
    async def test_agent_name(self):
        agent = StubAgent()
        assert agent.name == "StubAgent"

    @pytest.mark.asyncio
    async def test_error_agent_raises(self):
        agent = ErrorAgent()
        with pytest.raises(ConnectionError):
            await agent.query("Q", "db1")


class TestAgentResponse:
    """Test the AgentResponse model."""

    def test_minimal_response(self):
        response = AgentResponse()
        assert response.actual_sql is None
        assert response.error is None
        assert response.metadata == {}

    def test_response_with_all_fields(self):
        response = AgentResponse(
            actual_sql="SELECT 1",
            actual_results={"data": [1]},
            insight_text="Result is 1",
            chart_spec={"type": "bar"},
            latency_ms=100.0,
            token_count=50,
            cost_usd=0.002,
        )
        assert response.actual_sql == "SELECT 1"
        assert response.insight_text == "Result is 1"
        assert response.chart_spec["type"] == "bar"

    def test_response_with_error(self):
        response = AgentResponse(error="SQL syntax error")
        assert response.error == "SQL syntax error"
