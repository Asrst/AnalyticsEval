# Development Guide

How to set up, test, and contribute to analytics-eval.

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) for dependency management

### Installation

```bash
git clone https://github.com/your-org/analytics-eval.git
cd analytics-eval
uv sync --all-extras
```

### Verify

```bash
uv run pytest
```

All tests should pass (currently 164).

## Running Tests

```bash
# All tests
uv run pytest

# Verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_test_case.py

# Run specific test class
uv run pytest tests/test_metrics_base.py::TestMetricResult

# Run specific test method
uv run pytest tests/test_metrics_base.py::TestMetricResult::test_passing_result

# Run only analytics evaluation tests (uses pytest markers)
uv run pytest -m analytics

# Run with coverage
uv run pytest --cov=analytics_eval

# Run with coverage report
uv run pytest --cov=analytics_eval --cov-report=html
```

## Code Style

The project uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```bash
# Check for lint errors
uv run ruff check src/ tests/

# Auto-fix fixable issues
uv run ruff check --fix src/ tests/

# Format code
uv run ruff format src/ tests/

# Check formatting without changing files
uv run ruff format --check src/ tests/
```

**Rules enforced** (`pyproject.toml`):
- `E` — pycodestyle errors
- `F` — pyflakes
- `I` — isort (import ordering)
- `N` — pep8-naming
- `W` — pycodestyle warnings
- `UP` — pyupgrade (modern Python syntax)

Line length: 100 characters. Target Python: 3.12+.

## Adding a New Metric

1. **Create the metric file** under the appropriate subpackage:
   - SQL correctness → `src/analytics_eval/metrics/sql/`
   - Result quality → `src/analytics_eval/metrics/result/`
   - Insight quality → `src/analytics_eval/metrics/insight/`
   - etc.

2. **Subclass `AnalyticsMetric`** and implement:
   - `name` — Human-readable metric name
   - `category` — Which `MetricCategory` it belongs to
   - `mode` — `EvaluationMode.DETERMINISTIC`, `LLM_AS_JUDGE`, or `HYBRID`
   - `required_fields` — List of `AnalyticsTestCase` field names needed
   - `measure(test_case)` — Core evaluation logic returning `MetricResult`

3. **Export from `__init__.py`** in the subpackage and the parent `metrics/__init__.py`.

4. **Write tests** under `tests/metrics/<category>/`. Test:
   - Exact match / passing case
   - Mismatch / failing case
   - Edge cases (empty data, missing fields)
   - Configuration options (threshold, tolerance, etc.)
   - `required_fields` and skip behavior

5. **Run tests and lint:**
   ```bash
   uv run pytest tests/metrics/
   uv run ruff check src/analytics_eval/metrics/ tests/metrics/
   ```

### Example: Adding a New SQL Metric

```python
# src/analytics_eval/metrics/sql/query_safety.py
from analytics_eval.metrics.base import (
    AnalyticsMetric, MetricResult, MetricCategory, EvaluationMode,
)
from analytics_eval.test_case.case import AnalyticsTestCase

class QuerySafety(AnalyticsMetric):
    """Detects potentially dangerous SQL patterns."""

    def __init__(self, threshold: float = 0.5, **kwargs):
        super().__init__(threshold=threshold, **kwargs)
        self.dangerous_keywords = ["DROP", "DELETE", "TRUNCATE"]

    @property
    def name(self) -> str:
        return "QuerySafety"

    @property
    def category(self) -> MetricCategory:
        return MetricCategory.SQL_CORRECTNESS

    @property
    def mode(self) -> EvaluationMode:
        return EvaluationMode.DETERMINISTIC

    @property
    def required_fields(self) -> list[str]:
        return ["actual_sql"]

    def measure(self, test_case: AnalyticsTestCase) -> MetricResult:
        sql = test_case.actual_sql.upper()
        found = [kw for kw in self.dangerous_keywords if kw in sql]
        if found:
            return MetricResult(
                metric_name=self.name, score=0.0,
                reason=f"Dangerous SQL keywords found: {', '.join(found)}",
                category=self.category, mode=self.mode,
                threshold=self.threshold,
                details={"dangerous_keywords": found},
            )
        return MetricResult(
            metric_name=self.name, score=1.0,
            reason="No dangerous SQL patterns detected.",
            category=self.category, mode=self.mode,
            threshold=self.threshold,
        )
```

Then export from `src/analytics_eval/metrics/sql/__init__.py`:

```python
from analytics_eval.metrics.sql.query_safety import QuerySafety

__all__ = ["ExecutionAccuracy", "SchemaAdherence", "QuerySafety"]
```

And from `src/analytics_eval/metrics/__init__.py`:

```python
from analytics_eval.metrics.sql.query_safety import QuerySafety

__all__ = [
    # ... existing ...
    "QuerySafety",
]
```

## Adding a New Dataset Adapter

1. **Create the adapter** under `src/analytics_eval/datasets/adapters/`
2. **Subclass `DatasetAdapter`** and implement:
   - `name` — Dataset name
   - `available_splits` — List of available splits
   - `load(split, ...)` — Load test cases
3. **Write tests** under `tests/datasets/adapters/`
4. **Export** from `adapters/__init__.py`

## Project Structure

```
src/analytics_eval/
├── test_case/          # AnalyticsTestCase data model
├── metrics/            # Evaluation metrics
│   ├── base.py         # AnalyticsMetric ABC, MetricResult
│   ├── sql/            # SQL correctness metrics
│   ├── result/         # Result quality metrics
│   ├── insight/        # (planned)
│   ├── viz/            # (planned)
│   └── model/          # (planned)
├── agents/             # Agent interface and implementations
├── pipeline/           # EvalPipeline orchestration
├── execution/          # SQL execution backends
├── datasets/           # Benchmark dataset adapters
├── integrations/       # (planned)
└── pytest_plugin.py    # DeepEval-style pytest integration
```

## Pull Request Process

1. Create a branch for your changes
2. Write tests for new functionality
3. Run the full test suite: `uv run pytest`
4. Run the linter: `uv run ruff check src/ tests/`
5. Format code: `uv run ruff format src/ tests/`
6. Submit a pull request with a clear description

## Dependency Management

All dependencies are managed via `pyproject.toml` and `uv`.

```bash
# Add a runtime dependency
uv add <package>

# Add a development dependency
uv add --dev <package>

# Add an optional dependency group
uv add --optional <package>

# Sync all dependencies
uv sync

# Sync with optional groups
uv sync --all-extras
```

The project has these optional dependency groups:
- `bird` — BIRD benchmark support (SQLite is built-in)
- `spider2` — Spider 2.0-Snow support (requires `snowflake-connector-python`)
- `llm` — LLM-as-judge metrics (requires `openai`)
- `all` — All optional dependencies
