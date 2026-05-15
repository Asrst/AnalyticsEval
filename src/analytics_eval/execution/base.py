"""SQL Executor base class — abstract interface for executing SQL.

Design principles:
- One method: execute(sql) -> pd.DataFrame
- Backend-agnostic: SQLite, Snowflake, DuckDB, BigQuery all implement this
- Error handling: SQLExecutionError wraps all DB errors
- Connection management: each executor owns its connection lifecycle
- Timeout support: optional per-query timeout
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class SQLExecutionError(Exception):
    """Raised when SQL execution fails.

    Wraps the original database error with context about which SQL
    statement failed and which database was targeted.
    """

    def __init__(
        self,
        message: str,
        sql: str | None = None,
        db_id: str | None = None,
        original_error: Exception | None = None,
    ) -> None:
        self.sql = sql
        self.db_id = db_id
        self.original_error = original_error
        super().__init__(message)


class SQLExecutor(ABC):
    """Abstract interface for executing SQL queries against a database.

    Every SQL backend (SQLite, Snowflake, DuckDB, etc.) implements this
    interface. The framework uses executors to:
    1. Execute expected_sql from benchmarks to produce ground truth DataFrames
    2. Execute actual_sql from agents to produce result DataFrames
    3. Feed those DataFrames into metrics for comparison

    Usage:
        executor = SQLiteExecutor(db_path="/path/to/bird_db/california_schools.sqlite")
        results = executor.execute("SELECT COUNT(*) FROM schools")

    Design decisions:
    - execute() is synchronous (DB calls are blocking at the Python level)
    - Returns pd.DataFrame for direct use in metrics
    - Raises SQLExecutionError for all failure modes
    - Can_execute() checks if the database is accessible
    """

    @abstractmethod
    def execute(
        self,
        sql: str,
        timeout: float | None = None,
    ) -> pd.DataFrame:
        """Execute a SQL query and return results as a DataFrame.

        Args:
            sql: The SQL query to execute.
            timeout: Optional timeout in seconds. If the query takes longer,
                     a SQLExecutionError is raised.

        Returns:
            pd.DataFrame with the query results. Column names match the
            SQL output column names/aliases.

        Raises:
            SQLExecutionError: If execution fails for any reason
                (syntax error, timeout, connection failure, etc.)
        """
        ...

    @abstractmethod
    def can_connect(self) -> bool:
        """Check if the database is accessible.

        Returns True if a test query can be executed successfully.
        Used by the pipeline to validate executors before running
        the full evaluation.
        """
        ...

    @property
    @abstractmethod
    def db_id(self) -> str:
        """Identifier for this database (e.g., 'california_schools')."""
        ...

    @property
    @abstractmethod
    def dialect(self) -> str:
        """SQL dialect (e.g., 'sqlite', 'snowflake', 'duckdb')."""
        ...

    def get_schema(self) -> dict:
        """Get schema metadata for this database.

        Returns a dict with 'tables' (list of table names) and
        'columns' (dict mapping table name to list of column names).

        Default implementation returns empty dict. Subclasses can
        override for richer schema introspection.
        """
        return {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(db_id={self.db_id!r}, dialect={self.dialect!r})"
