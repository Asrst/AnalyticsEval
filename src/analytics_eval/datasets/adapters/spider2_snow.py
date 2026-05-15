"""Spider 2.0-Snow Dataset Adapter — loads Spider 2.0-Snow benchmark data.

Spider 2.0-Snow is an enterprise-grade text-to-SQL benchmark with 547
tasks across real-world Snowflake databases. Key characteristics:
- Very low baselines (GPT-4 = 2.2%)
- Multi-step SQL generation workflows
- Snowflake SQL dialect
- Requires Snowflake credentials for evaluation

Usage:
    adapter = Spider2SnowAdapter(
        data_dir="/data/spider2_snow",
        snowflake_executor=executor,
    )
    cases = adapter.load(split="test")
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from analytics_eval.datasets.adapters.base import DatasetAdapter
from analytics_eval.test_case.case import AnalyticsTestCase


class Spider2SnowAdapter(DatasetAdapter):
    """Loads the Spider 2.0-Snow benchmark into AnalyticsTestCase objects.

    Unlike BIRD which manages its own SQLite executors, Spider 2.0-Snow
    requires an externally configured SnowflakeExecutor because Snowflake
    credentials and warehouse setup are environment-specific.
    """

    def __init__(
        self,
        data_dir: str | Path,
        snowflake_executor: Any | None = None,
    ) -> None:
        """Initialize the Spider 2.0-Snow adapter.

        Args:
            data_dir: Path to the Spider 2.0-Snow data directory.
            snowflake_executor: Optional pre-configured SnowflakeExecutor.
        """
        self._data_dir = Path(data_dir)
        self._default_executor = snowflake_executor
        self._executor_map: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "Spider 2.0-Snow"

    @property
    def available_splits(self) -> list[str]:
        return ["test"]

    def set_executor(self, db_id: str, executor: Any) -> None:
        """Set a SQLExecutor for a specific database."""
        self._executor_map[db_id] = executor

    def load(
        self,
        split: str = "test",
        difficulty: Sequence[str] | None = None,
        domain: Sequence[str] | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> list[AnalyticsTestCase]:
        """Load Spider 2.0-Snow test cases."""
        jsonl_path = self._find_jsonl_file()
        if jsonl_path is None:
            raise FileNotFoundError(f"Spider 2.0-Snow JSONL file not found in {self._data_dir}")

        domain_set = set(domain) if domain else None
        cases: list[AnalyticsTestCase] = []

        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                item = json.loads(line)

                item_db_id = item.get("db_id", "")
                if domain_set and item_db_id not in domain_set:
                    continue

                case = self._item_to_case(item, split)
                cases.append(case)

                if limit and len(cases) >= limit:
                    break

        return cases

    def get_db_ids(self, split: str = "test") -> list[str]:
        """Get available database IDs."""
        jsonl_path = self._find_jsonl_file()
        if jsonl_path is None:
            return []

        db_ids: set[str] = set()
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                db_ids.add(item.get("db_id", ""))

        return sorted(db_ids)

    def get_executor_for_db(self, db_id: str) -> Any | None:
        """Get the SQLExecutor for a specific database."""
        if db_id in self._executor_map:
            return self._executor_map[db_id]
        return self._default_executor

    def _item_to_case(self, item: dict, split: str) -> AnalyticsTestCase:
        """Convert a JSONL item to an AnalyticsTestCase."""
        return AnalyticsTestCase(
            input=item.get("instruction", ""),
            db_id=item.get("db_id"),
            expected_sql=item.get("query"),
            evidence=item.get("evidence"),
            semantic_context={
                "dialect": item.get("dialect", "snowflake"),
                "schema_info": item.get("schema"),
            },
            metadata={
                "source": "Spider 2.0-Snow",
                "split": split,
                "instance_id": item.get("instance_id"),
            },
        )

    def _find_jsonl_file(self) -> Path | None:
        """Find the JSONL file in the data directory."""
        candidates = [
            self._data_dir / "spider2-snow.jsonl",
            self._data_dir / "test.jsonl",
            self._data_dir / "data.jsonl",
        ]

        jsonl_files = list(self._data_dir.glob("*.jsonl"))
        candidates.extend(jsonl_files)

        for candidate in candidates:
            if candidate.exists():
                return candidate

        return None
