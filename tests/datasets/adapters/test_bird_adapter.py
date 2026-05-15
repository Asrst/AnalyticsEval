"""Tests for BIRD Dataset Adapter — loads BIRD benchmark data."""

import json
import shutil
import sqlite3
import tempfile
from pathlib import Path

import pytest

from analytics_eval.datasets.adapters.bird import BirdAdapter
from analytics_eval.test_case.case import AnalyticsTestCase, Difficulty


@pytest.fixture
def bird_data_dir():
    """Create a mock BIRD data directory with test data."""
    tmp_dir = tempfile.mkdtemp()
    data_dir = Path(tmp_dir) / "bird_test"
    data_dir.mkdir()

    # Create mini_dev split
    dev_dir = data_dir / "mini_dev"
    dev_dir.mkdir()

    # Create BIRD JSON
    bird_data = [
        {
            "question_id": 1,
            "db_id": "test_schools",
            "question": "How many schools are there?",
            "evidence": "Count all rows in the schools table",
            "SQL": "SELECT COUNT(*) FROM schools",
            "difficulty": "simple",
        },
        {
            "question_id": 2,
            "db_id": "test_schools",
            "question": "What is the average enrollment?",
            "evidence": "Average enrollment across all schools",
            "SQL": "SELECT AVG(enrollment) FROM schools",
            "difficulty": "moderate",
        },
        {
            "question_id": 3,
            "db_id": "test_schools",
            "question": "List schools by county with enrollment over 1000",
            "evidence": "Schools with enrollment > 1000 grouped by county",
            "SQL": ("SELECT county, school_name FROM schools WHERE enrollment > 1000"),
            "difficulty": "challenging",
        },
        {
            "question_id": 4,
            "db_id": "test_finance",
            "question": "Total revenue by quarter",
            "evidence": "Revenue grouped by quarter",
            "SQL": ("SELECT quarter, SUM(revenue) FROM financials GROUP BY quarter"),
            "difficulty": "moderate",
        },
    ]

    with open(dev_dir / "mini_dev.json", "w") as f:
        json.dump(bird_data, f)

    # Create SQLite databases
    db_dir = dev_dir / "mini_dev_databases"
    db_dir.mkdir()

    # test_schools database
    schools_dir = db_dir / "test_schools"
    schools_dir.mkdir()
    conn = sqlite3.connect(str(schools_dir / "test_schools.sqlite"))
    conn.execute(
        "CREATE TABLE schools (id INTEGER, school_name TEXT, county TEXT, enrollment INTEGER)"
    )
    conn.execute("INSERT INTO schools VALUES (1, 'Lincoln High', 'Alameda', 1500)")
    conn.execute("INSERT INTO schools VALUES (2, 'Jefferson Middle', 'Contra Costa', 800)")
    conn.execute("INSERT INTO schools VALUES (3, 'Washington Elem', 'Alameda', 400)")
    conn.commit()
    conn.close()

    # test_finance database
    finance_dir = db_dir / "test_finance"
    finance_dir.mkdir()
    conn = sqlite3.connect(str(finance_dir / "test_finance.sqlite"))
    conn.execute("CREATE TABLE financials (id INTEGER, quarter TEXT, revenue REAL)")
    conn.execute("INSERT INTO financials VALUES (1, 'Q1', 10000.0)")
    conn.execute("INSERT INTO financials VALUES (2, 'Q2', 12000.0)")
    conn.commit()
    conn.close()

    yield data_dir

    # Cleanup
    shutil.rmtree(tmp_dir, ignore_errors=True)


class TestBirdAdapterLoading:
    """Test loading BIRD test cases."""

    def test_load_returns_test_cases(self, bird_data_dir):
        adapter = BirdAdapter(data_dir=bird_data_dir)
        cases = adapter.load(split="mini_dev")
        assert len(cases) == 4
        assert all(isinstance(c, AnalyticsTestCase) for c in cases)

    def test_load_maps_fields_correctly(self, bird_data_dir):
        adapter = BirdAdapter(data_dir=bird_data_dir)
        cases = adapter.load(split="mini_dev")
        case = cases[0]

        assert case.input == "How many schools are there?"
        assert case.expected_sql == "SELECT COUNT(*) FROM schools"
        assert case.db_id == "test_schools"
        assert case.evidence == "Count all rows in the schools table"
        assert case.difficulty == Difficulty.SIMPLE
        assert case.metadata["source"] == "BIRD"

    def test_load_expected_results_not_populated(self, bird_data_dir):
        """Adapter should populate expected_sql but NOT expected_results."""
        adapter = BirdAdapter(data_dir=bird_data_dir)
        cases = adapter.load(split="mini_dev")
        for case in cases:
            assert case.expected_sql is not None
            assert case.expected_results is None


class TestBirdAdapterFiltering:
    """Test filtering by difficulty and domain."""

    def test_filter_by_difficulty(self, bird_data_dir):
        adapter = BirdAdapter(data_dir=bird_data_dir)
        cases = adapter.load(split="mini_dev", difficulty=["simple"])
        assert len(cases) == 1
        assert cases[0].difficulty == Difficulty.SIMPLE

    def test_filter_by_multiple_difficulties(self, bird_data_dir):
        adapter = BirdAdapter(data_dir=bird_data_dir)
        cases = adapter.load(split="mini_dev", difficulty=["simple", "moderate"])
        assert len(cases) == 3

    def test_filter_by_domain(self, bird_data_dir):
        adapter = BirdAdapter(data_dir=bird_data_dir)
        cases = adapter.load(split="mini_dev", domain=["test_finance"])
        assert len(cases) == 1
        assert cases[0].db_id == "test_finance"

    def test_limit(self, bird_data_dir):
        adapter = BirdAdapter(data_dir=bird_data_dir)
        cases = adapter.load(split="mini_dev", limit=2)
        assert len(cases) == 2


class TestBirdAdapterExecutor:
    """Test SQL executor integration."""

    def test_get_executor_for_db(self, bird_data_dir):
        adapter = BirdAdapter(data_dir=bird_data_dir)
        executor = adapter.get_executor_for_db("test_schools")
        assert executor is not None
        assert executor.db_id == "test_schools"
        assert executor.dialect == "sqlite"

    def test_executor_can_execute_sql(self, bird_data_dir):
        adapter = BirdAdapter(data_dir=bird_data_dir)
        executor = adapter.get_executor_for_db("test_schools")
        result = executor.execute("SELECT COUNT(*) as cnt FROM schools")
        assert result.iloc[0]["cnt"] == 3

    def test_executor_is_cached(self, bird_data_dir):
        adapter = BirdAdapter(data_dir=bird_data_dir)
        executor1 = adapter.get_executor_for_db("test_schools")
        executor2 = adapter.get_executor_for_db("test_schools")
        assert executor1 is executor2

    def test_get_db_ids(self, bird_data_dir):
        adapter = BirdAdapter(data_dir=bird_data_dir)
        db_ids = adapter.get_db_ids(split="mini_dev")
        assert "test_schools" in db_ids
        assert "test_finance" in db_ids

    def test_available_splits(self, bird_data_dir):
        adapter = BirdAdapter(data_dir=bird_data_dir)
        assert "mini_dev" in adapter.available_splits


class TestBirdAdapterEndToEnd:
    """End-to-end: load BIRD cases, execute SQL, evaluate metrics."""

    def test_load_execute_and_evaluate(self, bird_data_dir):
        """Full pipeline: load cases → execute expected SQL → evaluate."""
        from analytics_eval.metrics import ExecutionAccuracy
        from analytics_eval.pytest_plugin import assert_test

        adapter = BirdAdapter(data_dir=bird_data_dir)
        cases = adapter.load(split="mini_dev", limit=2)

        # For each case, execute expected SQL and populate expected_results
        for case in cases:
            executor = adapter.get_executor_for_db(case.db_id)
            if executor and case.has_expected_sql():
                case.expected_results = executor.execute(case.expected_sql)
                # Simulate a perfect agent
                case.actual_sql = case.expected_sql
                case.actual_results = case.expected_results

        # Now evaluate with metrics
        for case in cases:
            if case.has_expected_results() and case.has_actual_results():
                assert_test(case, [ExecutionAccuracy()])


class TestBirdAdapterErrors:
    """Error handling tests."""

    def test_nonexistent_data_dir(self):
        with pytest.raises(FileNotFoundError):
            BirdAdapter(data_dir="/tmp/nonexistent_bird_dir_12345")

    def test_nonexistent_split(self, bird_data_dir):
        adapter = BirdAdapter(data_dir=bird_data_dir)
        with pytest.raises(FileNotFoundError):
            adapter.load(split="nonexistent_split")
