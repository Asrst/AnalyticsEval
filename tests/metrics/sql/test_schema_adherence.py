"""Tests for SchemaAdherence metric — validates SQL against schema metadata.

TDD: These tests define the contract for schema validation. The metric
should detect hallucinated tables/columns and provide partial credit.
"""

from analytics_eval.metrics.sql.schema_adherence import SchemaAdherence
from analytics_eval.test_case.case import AnalyticsTestCase


class TestSchemaAdherenceBasic:
    """Basic schema adherence tests."""

    def test_valid_sql_with_matching_schema(self):
        """SQL referencing known tables should score 1.0."""
        case = AnalyticsTestCase(
            input="What is revenue?",
            actual_sql="SELECT SUM(amount) FROM orders",
            semantic_context={
                "schema": {
                    "tables": ["orders", "customers", "products"],
                    "columns": {
                        "orders": [
                            "id",
                            "amount",
                            "customer_id",
                            "order_date",
                        ],
                    },
                },
            },
        )
        metric = SchemaAdherence()
        result = metric.evaluate(case)
        assert result.score == 1.0
        assert result.passed is True

    def test_sql_with_unknown_table(self):
        """SQL referencing a non-existent table should score < 1.0."""
        case = AnalyticsTestCase(
            input="What is revenue?",
            actual_sql="SELECT SUM(amount) FROM nonexistent_table",
            semantic_context={
                "schema": {
                    "tables": ["orders", "customers"],
                    "columns": {"orders": ["id", "amount"]},
                },
            },
        )
        metric = SchemaAdherence()
        result = metric.evaluate(case)
        assert result.score < 1.0
        assert "nonexistent_table" in result.details.get("invalid_tables", [])

    def test_sql_with_join_referencing_two_tables(self):
        """SQL with JOIN referencing two valid tables."""
        case = AnalyticsTestCase(
            input="Revenue by customer name",
            actual_sql=(
                "SELECT c.name, SUM(o.amount) "
                "FROM orders o JOIN customers c ON o.customer_id = c.id "
                "GROUP BY c.name"
            ),
            semantic_context={
                "schema": {
                    "tables": ["orders", "customers"],
                    "columns": {
                        "orders": ["id", "amount", "customer_id"],
                        "customers": ["id", "name"],
                    },
                },
            },
        )
        metric = SchemaAdherence()
        result = metric.evaluate(case)
        assert result.score == 1.0

    def test_sql_with_unknown_column_in_dotted_notation(self):
        """SQL referencing table.unknown_column should detect it."""
        case = AnalyticsTestCase(
            input="Q",
            actual_sql="SELECT o.unknown_col FROM orders o",
            semantic_context={
                "schema": {
                    "tables": ["orders"],
                    "columns": {
                        "orders": ["id", "amount", "customer_id"],
                    },
                },
            },
        )
        metric = SchemaAdherence()
        result = metric.evaluate(case)
        assert "unknown_col" in result.details.get("invalid_columns", [])

    def test_no_schema_context_skips(self):
        """Without schema context, the metric should skip."""
        case = AnalyticsTestCase(
            input="Q",
            actual_sql="SELECT * FROM orders",
            semantic_context=None,
        )
        metric = SchemaAdherence()
        result = metric.evaluate(case)
        assert result.score == 0.0
        assert result.details.get("skipped") is True

    def test_no_actual_sql_skips(self):
        """Without actual SQL, the metric should skip."""
        case = AnalyticsTestCase(input="Q")
        metric = SchemaAdherence()
        assert metric.can_evaluate(case) is False


class TestSchemaAdherenceStrictMode:
    """Test strict mode where any violation = 0."""

    def test_strict_mode_zeros_on_violation(self):
        """In strict mode, even one unknown table should give 0."""
        case = AnalyticsTestCase(
            input="Q",
            actual_sql=("SELECT * FROM valid_table JOIN invalid_table ON 1=1"),
            semantic_context={
                "schema": {
                    "tables": ["valid_table"],
                    "columns": {"valid_table": ["id"]},
                },
            },
        )
        metric = SchemaAdherence(strict_mode=True)
        result = metric.evaluate(case)
        assert result.score == 0.0

    def test_strict_mode_passes_on_full_match(self):
        """In strict mode, a perfect match still scores 1.0."""
        case = AnalyticsTestCase(
            input="Q",
            actual_sql="SELECT id FROM valid_table",
            semantic_context={
                "schema": {
                    "tables": ["valid_table"],
                    "columns": {"valid_table": ["id"]},
                },
            },
        )
        metric = SchemaAdherence(strict_mode=True)
        result = metric.evaluate(case)
        assert result.score == 1.0


class TestSchemaAdherencePartialCredit:
    """Test partial credit behavior."""

    def test_mixed_valid_and_invalid_tables(self):
        """One valid table and one invalid should give partial credit."""
        case = AnalyticsTestCase(
            input="Q",
            actual_sql=("SELECT * FROM orders o JOIN fake_table f ON o.id = f.id"),
            semantic_context={
                "schema": {
                    "tables": ["orders"],
                    "columns": {"orders": ["id", "amount"]},
                },
            },
        )
        metric = SchemaAdherence()
        result = metric.evaluate(case)
        assert 0.0 < result.score < 1.0
        assert "fake_table" in result.details.get("invalid_tables", [])


class TestSchemaAdherenceSimpleSQL:
    """Test with simple SQL that has no table references."""

    def test_select_literal(self):
        """Simple SQL like 'SELECT 1' should score 1.0."""
        case = AnalyticsTestCase(
            input="Q",
            actual_sql="SELECT 1 AS value",
            semantic_context={
                "schema": {
                    "tables": ["orders"],
                    "columns": {"orders": ["id"]},
                },
            },
        )
        metric = SchemaAdherence()
        result = metric.evaluate(case)
        assert result.score == 1.0

    def test_schema_with_tables_key_at_top_level(self):
        """Schema metadata at top level of semantic_context."""
        case = AnalyticsTestCase(
            input="Q",
            actual_sql="SELECT * FROM orders",
            semantic_context={
                "tables": ["orders", "customers"],
                "columns": {"orders": ["id"]},
            },
        )
        metric = SchemaAdherence()
        result = metric.evaluate(case)
        assert result.score == 1.0
