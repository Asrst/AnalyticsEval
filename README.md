# Analytics Eval

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)]

An open-source evaluation framework for analytics agents and text-to-SQL systems. Uses semantic models as ground truth for test case generation and metric evaluation.

## Quick Start

```python
import pandas as pd
from analytics_eval import AnalyticsTestCase, assert_test
from analytics_eval.metrics import ExecutionAccuracy, ResultAccuracy

def test_revenue_query():
    case = AnalyticsTestCase(
        input="What is total revenue by region?",
        expected_results=pd.DataFrame({
            "region": ["East", "West"],
            "total": [1000.0, 2000.0],
        }),
        actual_results=pd.DataFrame({
            "region": ["East", "West"],
            "total": [1000.0, 2000.0],
        }),
    )
    assert_test(case, [ExecutionAccuracy(), ResultAccuracy()])
```

## Installation

Requires [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Clone the repository
git clone https://github.com/your-org/analytics-eval.git
cd analytics-eval

# Install with uv
uv sync

# Run tests
uv run pytest
```

Install optional dependencies for specific benchmarks:

```bash
# BIRD benchmark support (SQLite is built-in)
uv sync --extra bird

# Spider 2.0-Snow support (requires Snowflake connector)
uv sync --extra spider2

# LLM-as-judge metrics
uv sync --extra llm

# All optional dependencies
uv sync --all-extras
```

## Core Concepts

### AnalyticsTestCase

The primary data model representing a single analytics interaction. Captures the natural language question, ground truth SQL and results, agent output, and semantic context.

```python
from analytics_eval import AnalyticsTestCase, Difficulty

case = AnalyticsTestCase(
    input="What is total revenue by region?",
    db_id="sales_db",
    expected_sql="SELECT region, SUM(amount) FROM orders GROUP BY region",
    expected_results=ground_truth_df,
    actual_sql=agent_generated_sql,
    actual_results=agent_result_df,
    evidence="Revenue excludes returns",
    difficulty=Difficulty.MODERATE,
    semantic_context={"metrics": ["revenue"], "dimensions": ["region"]},
)
```

All fields except `input` are optional. The framework degrades gracefully — metrics that require missing data are skipped with a score of 0.

### AnalyticsMetric

Abstract base class for evaluation metrics. Every metric produces a `MetricResult` with a score (0–1), a human-readable reason, and optional structured details.

```python
from analytics_eval.metrics import AnalyticsMetric, MetricResult, MetricCategory, EvaluationMode

class MyCustomMetric(AnalyticsMetric):
    @property
    def name(self) -> str:
        return "MyCustomMetric"

    @property
    def category(self) -> MetricCategory:
        return MetricCategory.SQL_CORRECTNESS

    @property
    def mode(self) -> EvaluationMode:
        return EvaluationMode.DETERMINISTIC

    @property
    def required_fields(self) -> list[str]:
        return ["expected_sql", "actual_sql"]

    def measure(self, test_case: AnalyticsTestCase) -> MetricResult:
        # Your evaluation logic here
        return MetricResult(
            metric_name=self.name,
            score=1.0,
            reason="Custom evaluation passed.",
            category=self.category,
            mode=self.mode,
            threshold=self.threshold,
        )
```

### assert_test()

The primary assertion function for pytest integration. Evaluates all metrics against a test case and raises `AssertionError` if any metric fails.

```python
from analytics_eval import assert_test

def test_my_query():
    case = AnalyticsTestCase(input="...", expected_results=..., actual_results=...)
    assert_test(case, [ExecutionAccuracy(), ResultAccuracy()])
```

## Available Metrics

### SQL Correctness

| Metric | Description | Required Fields |
|--------|-------------|-----------------|
| `ExecutionAccuracy` | Compares result sets of expected vs actual SQL | `expected_results`, `actual_results` |
| `SchemaAdherence` | Validates SQL references against database schema | `actual_sql`, `semantic_context` |

### Result Quality

| Metric | Description | Required Fields |
|--------|-------------|-----------------|
| `ResultAccuracy` | Numerical comparison with configurable tolerance | `expected_results`, `actual_results` |
| `GrainCorrectness` | Checks aggregation level and dimension matching | `expected_results`, `actual_results` |

Additional metric families (insight quality, visualization quality, model quality) are planned for future releases.

## Usage Examples

### Evaluating a Single Query

```python
import pandas as pd
from analytics_eval import AnalyticsTestCase, assert_test
from analytics_eval.metrics import ExecutionAccuracy

def test_simple_aggregation():
    case = AnalyticsTestCase(
        input="What is total revenue?",
        expected_results=pd.DataFrame({"total": [4500.0]}),
        actual_results=pd.DataFrame({"total": [4500.0]}),
    )
    assert_test(case, [ExecutionAccuracy()])
```

### Writing a Custom Metric

```python
from analytics_eval.metrics import AnalyticsMetric, MetricResult, MetricCategory, EvaluationMode

class QuerySafety(AnalyticsMetric):
    """Detects potentially dangerous SQL patterns."""

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
        dangerous = ["DROP", "DELETE", "TRUNCATE"]
        found = [kw for kw in dangerous if kw in sql]
        if found:
            return MetricResult(
                metric_name=self.name, score=0.0,
                reason=f"Dangerous SQL keywords found: {', '.join(found)}",
                category=self.category, mode=self.mode, threshold=self.threshold,
            )
        return MetricResult(
            metric_name=self.name, score=1.0,
            reason="No dangerous SQL patterns detected.",
            category=self.category, mode=self.mode, threshold=self.threshold,
        )
```

### Running with pytest

```bash
# Run all tests
uv run pytest

# Run only analytics evaluation tests
uv run pytest -m analytics

# Export results to JSON
uv run pytest --analytics-json=results.json

# Suppress terminal summary
uv run pytest --analytics-no-summary
```

## Architecture

```
analytics_eval/
├── test_case/          # AnalyticsTestCase, ConversationalAnalyticsCase
├── metrics/
│   ├── base.py         # AnalyticsMetric (ABC), MetricResult
│   ├── sql/            # ExecutionAccuracy, SchemaAdherence
│   ├── result/         # ResultAccuracy, GrainCorrectness
│   ├── insight/        # (planned)
│   ├── viz/            # (planned)
│   └── model/          # (planned)
├── agents/             # AgentInterface, LLMSQLAgent
├── pipeline/           # EvalPipeline, EvalConfig, EvalResult
├── execution/          # SQLExecutor, SQLiteExecutor, SnowflakeExecutor
├── datasets/           # DatasetAdapter, BirdAdapter, Spider2SnowAdapter
├── integrations/       # (planned)
└── pytest_plugin.py    # assert_test(), fixtures, hooks
```

Five core abstractions:

| Abstraction | Purpose |
|-------------|---------|
| `AnalyticsTestCase` | Captures full analytics interaction context |
| `AnalyticsMetric` | Evaluates one dimension, produces 0–1 score + reason |
| `DatasetAdapter` | Loads benchmark datasets into test cases |
| `AgentInterface` | BYOA — any agent implements one async method |
| `EvalPipeline` | Orchestrates plan → execute → evaluate cycle |

## Running Tests

```bash
# Install dev dependencies
uv sync --extra dev

# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=analytics_eval

# Run specific test file
uv run pytest tests/test_test_case.py

# Run with verbose output
uv run pytest -v
```

## Contributing

Contributions are welcome. Please follow these guidelines:

1. **Fork** the repository and create a branch for your changes.
2. **Write tests** for new functionality. The project uses test-driven development — tests define the contract.
3. **Run the test suite** before submitting: `uv run pytest`
4. **Run the linter**: `uv run ruff check .`
5. **Format code**: `uv run ruff format .`
6. **Submit a pull request** with a clear description of the changes.

### Adding a New Metric

1. Create a new file under `src/analytics_eval/metrics/<category>/`.
2. Subclass `AnalyticsMetric` and implement `name`, `category`, `mode`, `required_fields`, and `measure()`.
3. Write tests under `tests/metrics/<category>/`.
4. Export the metric from the appropriate `__init__.py`.

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
