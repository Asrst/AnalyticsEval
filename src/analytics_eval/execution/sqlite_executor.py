"""SQLite SQL Executor — executes SQL against SQLite database files.

This is the primary executor for the BIRD benchmark, which uses SQLite
databases. SQLite is built into Python (sqlite3 module), so no external
dependencies are required.

Usage:
    executor = SQLiteExecutor(db_path="/data/bird/california_schools.sqlite")
    results = executor.execute("SELECT COUNT(*) FROM schools")
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from analytics_eval.execution.base import SQLExecutionError, SQLExecutor


class SQLiteExecutor(SQLExecutor):
    """Executes SQL queries against a SQLite database file.

    The BIRD benchmark ships with ~95 SQLite databases, one per
    db_id. This executor connects to a single .sqlite file and
    provides both query execution and schema introspection.

    Design decisions:
    - Opens a new connection per execute() call (stateless, thread-safe)
    - Uses sqlite3.Row for dict-like access to columns
    - Returns DataFrames with column names from the SQL cursor
    - Timeout uses sqlite3's built-in interrupt mechanism
    """

    def __init__(
        self,
        db_path: str | Path,
        db_id_override: str | None = None,
    ) -> None:
        self._db_path = Path(db_path)
        self._db_id = db_id_override or self._db_path.stem

        if not self._db_path.exists():
            raise FileNotFoundError(f"SQLite database not found: {self._db_path}")

    @property
    def db_id(self) -> str:
        return self._db_id

    @property
    def dialect(self) -> str:
        return "sqlite"

    @property
    def db_path(self) -> Path:
        return self._db_path

    def execute(
        self,
        sql: str,
        timeout: float | None = None,
    ) -> pd.DataFrame:
        """Execute a SQL query against the SQLite database.

        Args:
            sql: SQL query to execute.
            timeout: Optional timeout in seconds.

        Returns:
            pd.DataFrame with the query results.

        Raises:
            SQLExecutionError: If the query fails.
        """
        try:
            conn = sqlite3.connect(str(self._db_path))
            try:
                if timeout is not None:
                    conn.execute(f"PRAGMA busy_timeout = {int(timeout * 1000)}")

                cursor = conn.execute(sql)

                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                rows = cursor.fetchall()

                if not rows and not columns:
                    return pd.DataFrame()

                df = pd.DataFrame(rows, columns=columns)
                return df

            finally:
                conn.close()

        except sqlite3.Error as e:
            raise SQLExecutionError(
                message=f"SQLite execution failed: {e}",
                sql=sql,
                db_id=self._db_id,
                original_error=e,
            ) from e
        except Exception as e:
            if isinstance(e, SQLExecutionError):
                raise
            raise SQLExecutionError(
                message=f"Unexpected error during SQL execution: {e}",
                sql=sql,
                db_id=self._db_id,
                original_error=e,
            ) from e

    def can_connect(self) -> bool:
        """Check if the SQLite database is accessible."""
        try:
            conn = sqlite3.connect(str(self._db_path))
            conn.execute("SELECT 1")
            conn.close()
            return True
        except Exception:
            return False

    def get_schema(self) -> dict:
        """Introspect the SQLite database schema.

        Returns:
            Dict with 'tables' (list of table names) and 'columns'
            (dict mapping table name to list of column names with types).
        """
        try:
            conn = sqlite3.connect(str(self._db_path))
            cursor = conn.cursor()

            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
            tables = [row[0] for row in cursor.fetchall()]

            columns: dict[str, list[str]] = {}
            for table in tables:
                cursor.execute(f"PRAGMA table_info({table})")
                columns[table] = [f"{row[1]} ({row[2]})" for row in cursor.fetchall()]

            conn.close()
            return {"tables": tables, "columns": columns}

        except Exception:
            return {}
