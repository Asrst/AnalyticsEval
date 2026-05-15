"""LLM SQL Agent — generates SQL using an LLM API.

This is a concrete AgentInterface implementation that uses an LLM
(OpenAI, Anthropic, local, etc.) to generate SQL from natural
language questions. It's the primary agent for benchmark evaluation.

Design decisions:
- LLM provider is pluggable via a simple callable
- Schema context is injected into the prompt for better SQL generation
- Returns AgentResponse with actual_sql populated
- Handles LLM errors gracefully (returns error in AgentResponse)
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from analytics_eval.agents.base import AgentInterface, AgentResponse

# Default system prompt for SQL generation
_DEFAULT_SYSTEM_PROMPT = (
    "You are an expert SQL generator. Given a natural language question "
    "and database schema, generate a correct SQL query.\n\n"
    "Rules:\n"
    "1. Generate ONLY the SQL query, no explanations or markdown.\n"
    "2. Use the exact table and column names from the schema.\n"
    "3. If the question is ambiguous, make reasonable assumptions.\n"
    "4. Use appropriate JOINs, aggregations, and filters.\n"
    "5. Follow the database's SQL dialect conventions.\n"
)

_DEFAULT_USER_PROMPT_TEMPLATE = (
    "Database: {db_id}\n\n"
    "Schema:\n"
    "{schema}\n\n"
    "Question: {question}\n\n"
    "{evidence_section}Generate the SQL query:"
)


class LLMSQLAgent(AgentInterface):
    """Generates SQL using an LLM API.

    This agent calls an LLM to translate natural language questions
    into SQL queries. It supports any LLM provider through a simple
    callable interface.

    Usage with OpenAI:
        import openai

        def call_openai(messages, **kwargs):
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.0,
            )
            return response.choices[0].message.content

        agent = LLMSQLAgent(llm_call=call_openai)
    """

    def __init__(
        self,
        llm_call: Callable[..., str],
        system_prompt: str | None = None,
        user_prompt_template: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        schema_provider: Callable[[str], str | None] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the LLM SQL Agent.

        Args:
            llm_call: A callable that takes (messages, **kwargs) and returns
                      the LLM's text response (the SQL string).
            system_prompt: Optional system prompt override.
            user_prompt_template: Optional user prompt template with
                placeholders: {db_id}, {schema}, {question}, {evidence_section}
            model: Optional model name for reporting.
            temperature: LLM temperature (default 0.0 for deterministic).
            schema_provider: Optional callable that takes db_id and returns
                            schema DDL text. If None, relies on context arg.
        """
        self._llm_call = llm_call
        self._system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT
        self._user_prompt_template = (
            user_prompt_template or _DEFAULT_USER_PROMPT_TEMPLATE
        )
        self._model = model
        self._temperature = temperature
        self._schema_provider = schema_provider

    async def query(
        self,
        question: str,
        db_id: str,
        context: dict[str, Any] | None = None,
    ) -> AgentResponse:
        """Generate SQL for a natural language question using the LLM.

        Args:
            question: Natural language question.
            db_id: Target database identifier.
            context: Optional context with 'schema', 'evidence', etc.

        Returns:
            AgentResponse with actual_sql populated.
        """
        start_time = time.monotonic()

        try:
            schema_text = self._get_schema(db_id, context)
            evidence_section = ""
            if context and context.get("evidence"):
                evidence_section = (
                    f"Evidence/Hint: {context['evidence']}\n\n"
                )

            user_message = self._user_prompt_template.format(
                db_id=db_id or "unknown",
                schema=schema_text or "No schema provided.",
                question=question,
                evidence_section=evidence_section,
            )

            messages = [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_message},
            ]

            response_text = self._llm_call(
                messages,
                temperature=self._temperature,
                model=self._model,
            )

            sql = self._extract_sql(response_text)

            elapsed_ms = (time.monotonic() - start_time) * 1000

            return AgentResponse(
                actual_sql=sql,
                latency_ms=elapsed_ms,
                metadata={
                    "model": self._model,
                    "prompt_tokens": None,
                    "raw_response": response_text,
                },
            )

        except Exception as e:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            return AgentResponse(
                actual_sql=None,
                error=f"{type(e).__name__}: {e}",
                latency_ms=elapsed_ms,
            )

    def _get_schema(
        self, db_id: str, context: dict[str, Any] | None
    ) -> str | None:
        """Get schema text for a database.

        Priority:
        1. schema_provider callable (if set)
        2. context['schema'] (if provided)
        3. context['ddl'] (if provided)
        4. None (no schema available)
        """
        if self._schema_provider:
            schema = self._schema_provider(db_id)
            if schema:
                return schema

        if context:
            if isinstance(context.get("schema"), str):
                return context["schema"]
            if isinstance(context.get("ddl"), str):
                return context["ddl"]
            if isinstance(context.get("schema"), dict):
                return self._schema_dict_to_text(context["schema"])

        return None

    def _schema_dict_to_text(self, schema: dict) -> str:
        """Convert a schema dict to text for the LLM prompt."""
        lines = []
        tables = schema.get("tables", [])
        columns = schema.get("columns", {})

        for table in tables:
            cols = columns.get(table, [])
            if cols:
                lines.append(
                    f"CREATE TABLE {table} "
                    f"({', '.join(str(c) for c in cols)});"
                )
            else:
                lines.append(f"CREATE TABLE {table};")

        return "\n".join(lines)

    def _extract_sql(self, response: str) -> str:
        """Extract SQL from the LLM response.

        Handles common patterns:
        - Plain SQL: "SELECT * FROM orders"
        - Markdown code block: ```sql\\nSELECT * FROM orders\\n```
        - Code block without language: ```\\nSELECT * FROM orders\\n```
        """
        text = response.strip()

        if text.startswith("```"):
            first_newline = text.find("\n")
            if first_newline != -1:
                text = text[first_newline + 1:]

            if text.endswith("```"):
                text = text[:-3]

            text = text.strip()

        return text

    @property
    def name(self) -> str:
        model_str = f" ({self._model})" if self._model else ""
        return f"LLMSQLAgent{model_str}"
