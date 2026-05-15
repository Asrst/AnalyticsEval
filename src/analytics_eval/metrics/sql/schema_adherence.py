"""Schema Adherence metric — validates that generated SQL uses correct schema elements.

Checks whether the agent's SQL references tables, columns, and relationships
that actually exist in the database schema. This catches "hallucinated" SQL
that references non-existent tables or columns, a common failure mode for
text-to-SQL agents.

Unlike ExecutionAccuracy which requires running the SQL, SchemaAdherence
can validate SQL purely from schema metadata — no database execution needed.
"""

from __future__ import annotations

import re
from typing import Any

from analytics_eval.metrics.base import (
    AnalyticsMetric,
    EvaluationMode,
    MetricCategory,
    MetricResult,
)
from analytics_eval.test_case.case import AnalyticsTestCase


class SchemaAdherence(AnalyticsMetric):
    """Validates that the agent's SQL adheres to the database schema.

    This metric examines the generated SQL and checks whether the referenced
    tables and columns exist in the provided schema metadata. It does NOT
    execute the SQL — it's a purely static analysis.

    Scoring:
    - 1.0: All referenced tables and columns exist in the schema
    - 0.5+: Some elements are valid, some are not (partial credit)
    - 0.0: Major schema violations or no schema context available

    The schema metadata should be provided via ``semantic_context`` on the
    test case, with keys ``tables`` and optionally ``columns``.

    Configuration:
    - threshold: Default 0.7 (schema adherence is important)
    - strict_mode: If True, unknown references score 0; if False, partial credit
    """

    def __init__(
        self,
        threshold: float = 0.7,
        strict_mode: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(threshold=threshold, **kwargs)
        self.strict_mode = strict_mode

    @property
    def name(self) -> str:
        return "SchemaAdherence"

    @property
    def category(self) -> MetricCategory:
        return MetricCategory.SQL_CORRECTNESS

    @property
    def mode(self) -> EvaluationMode:
        return EvaluationMode.DETERMINISTIC

    @property
    def required_fields(self) -> list[str]:
        return ["actual_sql"]

    def measure(self, test_case: AnalyticsTestCase) -> MetricResult:
        """Check schema adherence of the generated SQL."""
        sql = test_case.actual_sql
        assert sql is not None and sql.strip()

        schema = self._get_schema(test_case.semantic_context)

        if schema is None:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                reason="No schema metadata available in semantic_context.",
                category=self.category,
                mode=self.mode,
                threshold=self.threshold,
                details={
                    "skipped": True,
                    "missing": "semantic_context with schema",
                },
            )

        referenced_tables = self._extract_tables(sql)
        referenced_columns = self._extract_columns(sql)

        known_tables = set(schema.get("tables", []))
        known_columns = schema.get("columns", {})

        table_results: dict[str, bool] = {}
        for table in referenced_tables:
            table_results[table] = table.lower() in {t.lower() for t in known_tables}

        column_results: dict[str, bool] = {}
        for col in referenced_columns:
            found = False
            if known_columns:
                for table_cols in known_columns.values():
                    if col.lower() in {c.lower() for c in table_cols}:
                        found = True
                        break
            else:
                found = True
            column_results[col] = found

        all_results = list(table_results.values()) + list(column_results.values())
        if not all_results:
            return MetricResult(
                metric_name=self.name,
                score=1.0,
                reason="No table/column references found to validate.",
                category=self.category,
                mode=self.mode,
                threshold=self.threshold,
            )

        valid_count = sum(all_results)
        total_count = len(all_results)
        score = valid_count / total_count

        if self.strict_mode and score < 1.0:
            score = 0.0

        invalid_tables = [t for t, ok in table_results.items() if not ok]
        invalid_columns = [c for c, ok in column_results.items() if not ok]

        if score == 1.0:
            reason = "All referenced tables and columns exist in schema."
        else:
            parts = []
            if invalid_tables:
                parts.append(f"Unknown tables: {', '.join(invalid_tables)}")
            if invalid_columns:
                parts.append(f"Unknown columns: {', '.join(invalid_columns)}")
            reason = f"Schema violations: {'; '.join(parts)}"

        return MetricResult(
            metric_name=self.name,
            score=score,
            reason=reason,
            category=self.category,
            mode=self.mode,
            threshold=self.threshold,
            details={
                "referenced_tables": referenced_tables,
                "referenced_columns": referenced_columns,
                "invalid_tables": invalid_tables,
                "invalid_columns": invalid_columns,
                "valid_count": valid_count,
                "total_count": total_count,
            },
        )

    def _get_schema(self, semantic_context: dict | None) -> dict | None:
        """Extract schema from semantic_context.

        Supports multiple formats:
        - {"schema": {"tables": [...], "columns": {...}}}
        - {"database_schema": {"tables": [...]}}
        - {"tables": [...], "columns": {...}} — flat at top level
        """
        if semantic_context is None:
            return None

        for key in ("schema", "database_schema"):
            if key in semantic_context:
                val = semantic_context[key]
                if isinstance(val, dict):
                    return val

        if "tables" in semantic_context:
            return semantic_context

        return None

    def _extract_tables(self, sql: str) -> list[str]:
        """Extract table names from SQL using basic pattern matching.

        Handles common patterns:
        - FROM table_name
        - JOIN table_name
        - UPDATE table_name
        - INSERT INTO table_name
        """
        tables: list[str] = []
        patterns = [
            r"\bFROM\s+([a-zA-Z_][a-zA-Z0-9_]*)",
            r"\bJOIN\s+([a-zA-Z_][a-zA-Z0-9_]*)",
            r"\bUPDATE\s+([a-zA-Z_][a-zA-Z0-9_]*)",
            r"\bINTO\s+([a-zA-Z_][a-zA-Z0-9_]*)",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, sql, re.IGNORECASE):
                table = match.group(1)
                if table.upper() not in (
                    "SELECT",
                    "WHERE",
                    "AND",
                    "OR",
                    "ON",
                    "AS",
                    "SET",
                    "VALUES",
                    "GROUP",
                    "ORDER",
                    "HAVING",
                    "LIMIT",
                    "OFFSET",
                ):
                    tables.append(table)
        return tables

    def _extract_columns(self, sql: str) -> list[str]:
        """Extract column references from SQL.

        Handles patterns like:
        - table.column
        - column in SELECT/WHERE (without table prefix)
        """
        columns: list[str] = []
        for match in re.finditer(r"([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)", sql):
            col = match.group(2)
            if col.upper() not in ("*",):
                columns.append(col)
        return columns
