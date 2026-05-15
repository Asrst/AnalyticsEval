"""SQL correctness metrics — evaluate SQL generation quality.

Available metrics:
- ExecutionAccuracy: Compare result sets of expected vs actual SQL
- SchemaAdherence: Validate SQL against database schema metadata
"""

from analytics_eval.metrics.sql.execution_accuracy import ExecutionAccuracy
from analytics_eval.metrics.sql.schema_adherence import SchemaAdherence

__all__ = ["ExecutionAccuracy", "SchemaAdherence"]
