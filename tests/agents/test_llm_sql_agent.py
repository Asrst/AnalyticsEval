"""Tests for LLMSQLAgent — generates SQL using an LLM API."""

import pytest

from analytics_eval.agents.llm_sql_agent import LLMSQLAgent


class MockLLMCall:
    """A mock LLM callable for testing."""

    def __init__(self, response: str = "SELECT COUNT(*) FROM orders"):
        self._response = response
        self.calls: list[dict] = []

    def __call__(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return self._response


class TestLLMSQLAgentBasic:
    """Basic agent functionality tests."""

    @pytest.mark.asyncio
    async def test_agent_returns_sql(self):
        llm = MockLLMCall("SELECT SUM(amount) FROM orders")
        agent = LLMSQLAgent(llm_call=llm)

        response = await agent.query(
            "What is total revenue?", "sales_db"
        )
        assert response.actual_sql == "SELECT SUM(amount) FROM orders"
        assert response.error is None
        assert response.latency_ms is not None

    @pytest.mark.asyncio
    async def test_agent_includes_schema_in_prompt(self):
        llm = MockLLMCall("SELECT 1")
        agent = LLMSQLAgent(llm_call=llm)

        await agent.query(
            "How many orders?",
            "sales_db",
            context={
                "schema": "CREATE TABLE orders (id INT, amount DECIMAL)"
            },
        )

        assert len(llm.calls) == 1
        user_message = llm.calls[0]["messages"][1]["content"]
        assert "CREATE TABLE orders" in user_message

    @pytest.mark.asyncio
    async def test_agent_includes_evidence_in_prompt(self):
        llm = MockLLMCall("SELECT 1")
        agent = LLMSQLAgent(llm_call=llm)

        await agent.query(
            "Revenue?",
            "sales_db",
            context={"evidence": "Revenue = SUM(amount)"},
        )

        user_message = llm.calls[0]["messages"][1]["content"]
        assert "Revenue = SUM(amount)" in user_message

    @pytest.mark.asyncio
    async def test_agent_handles_llm_error(self):
        def failing_llm(messages, **kwargs):
            raise RuntimeError("LLM API unavailable")

        agent = LLMSQLAgent(llm_call=failing_llm)
        response = await agent.query("What is revenue?", "sales_db")

        assert response.actual_sql is None
        assert response.error is not None
        assert "RuntimeError" in response.error

    @pytest.mark.asyncio
    async def test_agent_with_schema_provider(self):
        llm = MockLLMCall("SELECT COUNT(*) FROM schools")
        agent = LLMSQLAgent(
            llm_call=llm,
            schema_provider=(
                lambda db_id: (
                    f"CREATE TABLE {db_id} (id INT)"
                    if db_id == "schools" else None
                )
            ),
        )

        await agent.query("How many?", "schools")
        user_message = llm.calls[0]["messages"][1]["content"]
        assert "CREATE TABLE schools" in user_message


class TestLLMSQLAgentSQLExtraction:
    """Test SQL extraction from LLM responses."""

    @pytest.mark.asyncio
    async def test_plain_sql_response(self):
        llm = MockLLMCall("SELECT * FROM orders WHERE amount > 100")
        agent = LLMSQLAgent(llm_call=llm)
        response = await agent.query("Q", "db")
        assert response.actual_sql == (
            "SELECT * FROM orders WHERE amount > 100"
        )

    @pytest.mark.asyncio
    async def test_markdown_sql_code_block(self):
        llm = MockLLMCall("```sql\nSELECT * FROM orders\n```")
        agent = LLMSQLAgent(llm_call=llm)
        response = await agent.query("Q", "db")
        assert response.actual_sql == "SELECT * FROM orders"

    @pytest.mark.asyncio
    async def test_markdown_code_block_no_language(self):
        llm = MockLLMCall("```\nSELECT COUNT(*) FROM orders\n```")
        agent = LLMSQLAgent(llm_call=llm)
        response = await agent.query("Q", "db")
        assert response.actual_sql == "SELECT COUNT(*) FROM orders"

    @pytest.mark.asyncio
    async def test_sql_with_leading_whitespace(self):
        llm = MockLLMCall("  SELECT 1  \n")
        agent = LLMSQLAgent(llm_call=llm)
        response = await agent.query("Q", "db")
        assert response.actual_sql == "SELECT 1"


class TestLLMSQLAgentConfig:
    """Test agent configuration."""

    def test_agent_name_without_model(self):
        agent = LLMSQLAgent(llm_call=lambda m, **k: "")
        assert agent.name == "LLMSQLAgent"

    def test_agent_name_with_model(self):
        agent = LLMSQLAgent(llm_call=lambda m, **k: "", model="gpt-4o")
        assert "gpt-4o" in agent.name

    @pytest.mark.asyncio
    async def test_custom_system_prompt(self):
        llm = MockLLMCall("SELECT 1")
        agent = LLMSQLAgent(
            llm_call=llm,
            system_prompt="You are a Snowflake SQL expert.",
        )
        await agent.query("Q", "db")
        system_msg = llm.calls[0]["messages"][0]["content"]
        assert "Snowflake SQL expert" in system_msg

    @pytest.mark.asyncio
    async def test_schema_dict_to_text(self):
        llm = MockLLMCall("SELECT 1")
        agent = LLMSQLAgent(llm_call=llm)

        await agent.query(
            "Q", "db",
            context={
                "schema": {
                    "tables": ["orders", "customers"],
                    "columns": {
                        "orders": ["id INT", "amount DECIMAL"],
                    },
                },
            },
        )
        user_msg = llm.calls[0]["messages"][1]["content"]
        assert "CREATE TABLE orders" in user_msg
