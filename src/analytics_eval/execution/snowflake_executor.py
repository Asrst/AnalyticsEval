"""Snowflake SQL Executor — executes SQL against Snowflake databases.

This is the executor for the Spider 2.0-Snow benchmark, which uses
Snowflake as the backend. Requires the `snowflake-connector-python`
package and valid Snowflake credentials.

Usage:
    executor = SnowflakeExecutor(
        account="xy12345.us-east-1",
        warehouse="COMPUTE_WH",
        database="SPIDER2_SNOW",
        schema="PUBLIC",
        credentials={"user": "...", "password": "..."},
    )
    results = executor.execute("SELECT COUNT(*) FROM orders")

Note: Spider 2.0-Snow evaluation requires access to a shared Snowflake
warehouse or a self-hosted instance with the benchmark data loaded.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from analytics_eval.execution.base import SQLExecutionError, SQLExecutor


class SnowflakeExecutor(SQLExecutor):
    """Executes SQL queries against a Snowflake database.

    Requires the `snowflake-connector-python` package and valid
    credentials. Falls back to a clear error message if the
    package is not installed.
    """

    def __init__(
        self,
        account: str,
        warehouse: str,
        database: str,
        schema: str = "PUBLIC",
        db_id_override: str | None = None,
        credentials: dict[str, str] | None = None,
        role: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._account = account
        self._warehouse = warehouse
        self._database = database
        self._schema = schema
        self._db_id = db_id_override or database
        self._credentials = credentials or {}
        self._role = role
        self._extra_kwargs = kwargs
        self._conn = None

    @property
    def db_id(self) -> str:
        return self._db_id

    @property
    def dialect(self) -> str:
        return "snowflake"

    def _get_connection(self):
        """Get or create a Snowflake connection."""
        if self._conn is not None:
            return self._conn

        try:
            import snowflake.connector
        except ImportError:
            raise ImportError(
                "snowflake-connector-python is required for SnowflakeExecutor. "
                "Install it with: pip install analytics-eval[spider2]"
            )

        conn_params = {
            "account": self._account,
            "warehouse": self._warehouse,
            "database": self._database,
            "schema": self._schema,
            **self._credentials,
            **self._extra_kwargs,
        }
        if self._role:
            conn_params["role"] = self._role

        try:
            self._conn = snowflake.connector.connect(**conn_params)
            return self._conn
        except Exception as e:
            raise SQLExecutionError(
                message=f"Failed to connect to Snowflake: {e}",
                db_id=self._db_id,
                original_error=e,
            ) from e

    def execute(
        self,
        sql: str,
        timeout: float | None = None,
    ) -> pd.DataFrame:
        """Execute a SQL query against Snowflake.

        Args:
            sql: SQL query to execute (Snowflake dialect).
            timeout: Optional query timeout in seconds.

        Returns:
            pd.DataFrame with the query results.

        Raises:
            SQLExecutionError: If execution fails.
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()

            if timeout is not None:
                cursor.execute(f"ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = {int(timeout)}")

            cursor.execute(sql)
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchall()

            if not rows and not columns:
                return pd.DataFrame()

            return pd.DataFrame(rows, columns=columns)

        except ImportError:
            raise
        except Exception as e:
            if isinstance(e, SQLExecutionError):
                raise
            raise SQLExecutionError(
                message=f"Snowflake execution failed: {e}",
                sql=sql,
                db_id=self._db_id,
                original_error=e,
            ) from e

    def can_connect(self) -> bool:
        """Check if Snowflake is accessible."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            return True
        except Exception:
            return False

    def close(self) -> None:
        """Close the Snowflake connection."""
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def __del__(self) -> None:
        self.close()
