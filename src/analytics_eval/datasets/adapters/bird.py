"""BIRD Dataset Adapter — loads BIRD benchmark data into AnalyticsTestCases.

BIRD (BIg Bench for LaRge-Scale Database Grounded Text-to-SQL) is a benchmark
with 12,751 question-SQL pairs across 95 databases. It uses SQLite as the
database backend and includes natural language questions, gold SQL queries,
external knowledge evidence, and difficulty ratings.

Usage:
    adapter = BirdAdapter(data_dir="/data/bird")
    cases = adapter.load(split="mini_dev")
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from analytics_eval.datasets.adapters.base import DatasetAdapter
from analytics_eval.execution.sqlite_executor import SQLiteExecutor
from analytics_eval.test_case.case import AnalyticsTestCase, Difficulty

_BIRD_DIFFICULTY_MAP = {
    "simple": Difficulty.SIMPLE,
    "moderate": Difficulty.MODERATE,
    "challenging": Difficulty.CHALLENGING,
}


class BirdAdapter(DatasetAdapter):
    """Loads the BIRD benchmark into AnalyticsTestCase objects.

    This adapter:
    1. Parses BIRD's JSON format (question, SQL, db_id, evidence, difficulty)
    2. Maps fields to AnalyticsTestCase
    3. Provides SQLiteExecutor for each database via get_executor_for_db()
    4. Supports filtering by split, difficulty, domain, and limit

    Important: expected_results is NOT populated by the adapter. The pipeline
    should use the provided SQLiteExecutor to execute expected_sql.
    """

    def __init__(self, data_dir: str | Path) -> None:
        """Initialize the BIRD adapter.

        Args:
            data_dir: Path to the BIRD data directory containing
                      train/, dev/, mini_dev/ subdirectories.
        """
        self._data_dir = Path(data_dir)
        if not self._data_dir.exists():
            raise FileNotFoundError(f"BIRD data directory not found: {self._data_dir}")

        self._executor_cache: dict[str, SQLiteExecutor] = {}

    @property
    def name(self) -> str:
        return "BIRD"

    @property
    def available_splits(self) -> list[str]:
        """List available splits based on what directories exist."""
        splits = []
        for split_name in ("train", "dev", "mini_dev", "test"):
            split_dir = self._data_dir / split_name
            if split_dir.exists():
                splits.append(split_name)
        return splits

    def load(
        self,
        split: str = "dev",
        difficulty: Sequence[str] | None = None,
        domain: Sequence[str] | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> list[AnalyticsTestCase]:
        """Load BIRD test cases from the specified split."""
        json_path = self._find_json_file(split)
        if json_path is None:
            raise FileNotFoundError(
                f"BIRD JSON file not found for split '{split}' in {self._data_dir}"
            )

        with open(json_path) as f:
            raw_data = json.load(f)

        if isinstance(raw_data, dict):
            raw_data = [raw_data]

        difficulty_set = set(difficulty) if difficulty else None
        domain_set = set(domain) if domain else None

        cases: list[AnalyticsTestCase] = []
        for item in raw_data:
            item_difficulty = item.get("difficulty", "").lower()
            if difficulty_set and item_difficulty not in difficulty_set:
                continue

            item_db_id = item.get("db_id", "")
            if domain_set and item_db_id not in domain_set:
                continue

            case = self._item_to_case(item, split)
            cases.append(case)

            if limit and len(cases) >= limit:
                break

        return cases

    def get_db_ids(self, split: str = "dev") -> list[str]:
        """Get available database IDs for a given split."""
        json_path = self._find_json_file(split)
        if json_path is None:
            return []

        with open(json_path) as f:
            raw_data = json.load(f)

        if isinstance(raw_data, dict):
            raw_data = [raw_data]

        return sorted(set(item.get("db_id", "") for item in raw_data))

    def get_executor_for_db(self, db_id: str) -> SQLiteExecutor | None:
        """Get a SQLiteExecutor for the specified BIRD database."""
        if db_id in self._executor_cache:
            return self._executor_cache[db_id]

        db_path = self._find_sqlite_path(db_id)
        if db_path is None:
            return None

        executor = SQLiteExecutor(db_path=db_path, db_id_override=db_id)
        self._executor_cache[db_id] = executor
        return executor

    def _item_to_case(self, item: dict, split: str) -> AnalyticsTestCase:
        """Convert a BIRD JSON item to an AnalyticsTestCase."""
        difficulty_str = item.get("difficulty", "").lower()
        difficulty = _BIRD_DIFFICULTY_MAP.get(difficulty_str)

        return AnalyticsTestCase(
            input=item.get("question", ""),
            db_id=item.get("db_id"),
            expected_sql=item.get("SQL") or item.get("sql"),
            evidence=item.get("evidence"),
            difficulty=difficulty,
            metadata={
                "source": "BIRD",
                "split": split,
                "question_id": item.get("question_id"),
            },
        )

    def _find_json_file(self, split: str) -> Path | None:
        """Find the JSON file for a given split."""
        split_dir = self._data_dir / split
        if not split_dir.exists():
            return None

        candidates = [
            split_dir / f"{split}.json",
            split_dir / "data.json",
        ]

        if split_dir.exists():
            json_files = list(split_dir.glob("*.json"))
            candidates.extend(json_files)

        for candidate in candidates:
            if candidate.exists():
                return candidate

        return None

    def _find_sqlite_path(self, db_id: str) -> Path | None:
        """Find the SQLite file for a given db_id."""
        for split_dir in self._data_dir.iterdir():
            if not split_dir.is_dir():
                continue

            db_dir = split_dir / f"{split_dir.name}_databases" / db_id
            if db_dir.exists():
                db_file = db_dir / f"{db_id}.sqlite"
                if db_file.exists():
                    return db_file

        return None
