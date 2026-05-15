"""SQL Execution package — executes SQL against databases and returns DataFrames.

This is the critical bridge between SQL text and the DataFrames that
metrics need for comparison. Without this layer, the framework can only
evaluate pre-populated results, not run actual benchmarks.

Supported backends:
- SQLite: For BIRD benchmark and local testing
- Snowflake: For Spider 2.0-Snow (requires credentials)
"""

from analytics_eval.execution.base import SQLExecutionError, SQLExecutor
from analytics_eval.execution.sqlite_executor import SQLiteExecutor

__all__ = ["SQLExecutor", "SQLExecutionError", "SQLiteExecutor"]
