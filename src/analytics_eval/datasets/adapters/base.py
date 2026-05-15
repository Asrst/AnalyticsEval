"""Dataset Adapter base class — abstract interface for loading benchmark datasets.

Design principles:
- One method: load() -> list[AnalyticsTestCase]
- Backend-agnostic: BIRD, Spider 2.0-Snow, Custom YAML all implement this
- Lazy SQL execution: adapters populate expected_sql but NOT expected_results
- Configurable: filter by difficulty, domain, split, etc.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from analytics_eval.test_case.case import AnalyticsTestCase


class DatasetAdapter(ABC):
    """Abstract interface for loading benchmark datasets into AnalyticsTestCases.

    Every benchmark format (BIRD, Spider 2.0-Snow, Custom YAML) implements
    this interface. The adapter converts the benchmark's native format into
    AnalyticsTestCase objects that the pipeline can evaluate.

    Key design choice: Adapters populate `expected_sql` but NOT `expected_results`.
    The pipeline's SQLExecutor runs expected_sql against the database to produce
    the ground truth DataFrames.
    """

    @abstractmethod
    def load(
        self,
        split: str = "dev",
        difficulty: Sequence[str] | None = None,
        domain: Sequence[str] | None = None,
        limit: int | None = None,
        **kwargs: Any,
    ) -> list[AnalyticsTestCase]:
        """Load test cases from the benchmark dataset.

        Args:
            split: Dataset split to load (e.g., "dev", "test", "mini_dev").
            difficulty: Optional filter by difficulty level.
            domain: Optional filter by database domain.
            limit: Max number of test cases to load.
            **kwargs: Adapter-specific options.

        Returns:
            List of AnalyticsTestCase objects with expected_sql populated.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable dataset name (e.g., 'BIRD', 'Spider 2.0-Snow')."""
        ...

    @property
    @abstractmethod
    def available_splits(self) -> list[str]:
        """List of available dataset splits."""
        ...

    def get_db_ids(self, split: str = "dev") -> list[str]:
        """Get available database IDs for a given split."""
        return []

    def get_executor_for_db(self, db_id: str) -> Any | None:
        """Get a SQLExecutor for a specific database."""
        return None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
