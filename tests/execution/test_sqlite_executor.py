"""Tests for SQL Execution layer — SQLiteExecutor + SQLExecutionError."""

import sqlite3
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from analytics_eval.execution.base import SQLExecutionError
from analytics_eval.execution.sqlite_executor import SQLiteExecutor


@pytest.fixture
def tmp_sqlite_db():
    """Create a temporary SQLite database with test data."""
    tmp = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    db_path = tmp.name
    tmp.close()

    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer TEXT,
            amount REAL,
            order_date TEXT
        )
    """)
    conn.execute("""
        INSERT INTO orders VALUES
            (1, 'Alice', 100.0, '2024-01-15'),
            (2, 'Bob', 200.0, '2024-02-20'),
            (3, 'Alice', 150.0, '2024-03-10'),
            (4, 'Charlie', 300.0, '2024-04-05')
    """)
    conn.execute("""
        CREATE TABLE customers (
            id INTEGER PRIMARY KEY,
            name TEXT,
            region TEXT
        )
    """)
    conn.execute("""
        INSERT INTO customers VALUES
            (1, 'Alice', 'East'),
            (2, 'Bob', 'West'),
            (3, 'Charlie', 'North')
    """)
    conn.commit()
    conn.close()

    yield db_path

    Path(db_path).unlink(missing_ok=True)


class TestSQLiteExecutorBasic:
    """Basic execution tests."""

    def test_execute_simple_query(self, tmp_sqlite_db):
        executor = SQLiteExecutor(db_path=tmp_sqlite_db)
        result = executor.execute("SELECT COUNT(*) as cnt FROM orders")
        assert isinstance(result, pd.DataFrame)
        assert result.iloc[0]["cnt"] == 4

    def test_execute_returns_correct_columns(self, tmp_sqlite_db):
        executor = SQLiteExecutor(db_path=tmp_sqlite_db)
        result = executor.execute("SELECT customer, amount FROM orders WHERE customer = 'Alice'")
        assert list(result.columns) == ["customer", "amount"]
        assert len(result) == 2

    def test_execute_aggregation(self, tmp_sqlite_db):
        executor = SQLiteExecutor(db_path=tmp_sqlite_db)
        result = executor.execute("SELECT SUM(amount) as total FROM orders")
        assert result.iloc[0]["total"] == 750.0

    def test_execute_join(self, tmp_sqlite_db):
        executor = SQLiteExecutor(db_path=tmp_sqlite_db)
        result = executor.execute(
            "SELECT o.customer, c.region FROM orders o JOIN customers c ON o.customer = c.name"
        )
        assert len(result) == 4
        assert "region" in result.columns

    def test_execute_empty_result(self, tmp_sqlite_db):
        executor = SQLiteExecutor(db_path=tmp_sqlite_db)
        result = executor.execute("SELECT * FROM orders WHERE 1=0")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_db_id_from_filename(self, tmp_sqlite_db):
        executor = SQLiteExecutor(db_path=tmp_sqlite_db)
        assert executor.db_id == Path(tmp_sqlite_db).stem

    def test_db_id_override(self, tmp_sqlite_db):
        executor = SQLiteExecutor(db_path=tmp_sqlite_db, db_id_override="my_db")
        assert executor.db_id == "my_db"

    def test_dialect_is_sqlite(self, tmp_sqlite_db):
        executor = SQLiteExecutor(db_path=tmp_sqlite_db)
        assert executor.dialect == "sqlite"


class TestSQLiteExecutorErrors:
    """Error handling tests."""

    def test_invalid_sql_raises_execution_error(self, tmp_sqlite_db):
        executor = SQLiteExecutor(db_path=tmp_sqlite_db)
        with pytest.raises(SQLExecutionError) as exc_info:
            executor.execute("SELECTT * FROM orders")
        assert exc_info.value.sql == "SELECTT * FROM orders"
        assert exc_info.value.db_id == executor.db_id
        assert exc_info.value.original_error is not None

    def test_nonexistent_table_raises_execution_error(self, tmp_sqlite_db):
        executor = SQLiteExecutor(db_path=tmp_sqlite_db)
        with pytest.raises(SQLExecutionError):
            executor.execute("SELECT * FROM nonexistent_table")

    def test_nonexistent_db_raises_file_error(self):
        with pytest.raises(FileNotFoundError):
            SQLiteExecutor(db_path="/tmp/nonexistent_db_12345.sqlite")


class TestSQLiteExecutorConnectivity:
    """Connection check tests."""

    def test_can_connect_to_valid_db(self, tmp_sqlite_db):
        executor = SQLiteExecutor(db_path=tmp_sqlite_db)
        assert executor.can_connect() is True

    def test_repr(self, tmp_sqlite_db):
        executor = SQLiteExecutor(db_path=tmp_sqlite_db)
        repr_str = repr(executor)
        assert "SQLiteExecutor" in repr_str
        assert "sqlite" in repr_str


class TestSQLiteExecutorSchema:
    """Schema introspection tests."""

    def test_get_schema_returns_tables(self, tmp_sqlite_db):
        executor = SQLiteExecutor(db_path=tmp_sqlite_db)
        schema = executor.get_schema()
        assert "tables" in schema
        assert "orders" in schema["tables"]
        assert "customers" in schema["tables"]

    def test_get_schema_returns_columns(self, tmp_sqlite_db):
        executor = SQLiteExecutor(db_path=tmp_sqlite_db)
        schema = executor.get_schema()
        assert "columns" in schema
        assert "orders" in schema["columns"]
        assert len(schema["columns"]["orders"]) > 0


class TestSQLExecutionError:
    """Test the SQLExecutionError model."""

    def test_error_with_context(self):
        error = SQLExecutionError(
            message="Query failed",
            sql="SELECT * FROM bad_table",
            db_id="test_db",
            original_error=ValueError("table not found"),
        )
        assert error.sql == "SELECT * FROM bad_table"
        assert error.db_id == "test_db"
        assert isinstance(error.original_error, ValueError)
        assert "Query failed" in str(error)
