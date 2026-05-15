# Analytics Eval — Architecture

> Living document. Updated as implementation evolves.
> Last updated: 2026-05-15

## Project Structure

```
analytics-eval/
├── pyproject.toml                      # Project config, deps, pytest integration
├── README.md
├── docs/
│   ├── architecture.md                 # This file
│   ├── metrics.md                      # Detailed metric documentation
│   └── development.md                  # Contributor guide
├── src/
│   └── analytics_eval/
│       ├── __init__.py                 # Top-level exports
│       ├── pytest_plugin.py            # assert_test(), fixtures, hooks, markers
│       ├── test_case/
│       │   ├── __init__.py
│       │   └── case.py                 # AnalyticsTestCase, ConversationalAnalyticsCase
│       ├── metrics/
│       │   ├── __init__.py             # Re-exports all metrics
│       │   ├── base.py                 # AnalyticsMetric (ABC), MetricResult
│       │   ├── sql/
│       │   │   ├── __init__.py
│       │   │   ├── execution_accuracy.py
│       │   │   └── schema_adherence.py
│       │   ├── result/
│       │   │   ├── __init__.py
│       │   │   ├── result_accuracy.py
│       │   │   └── grain_correctness.py
│       │   ├── insight/                # (stub)
│       │   ├── viz/                    # (stub)
│       │   └── model/                  # (stub)
│       ├── agents/
│       │   ├── __init__.py
│       │   ├── base.py                 # AgentInterface (ABC), AgentResponse
│       │   └── llm_sql_agent.py        # LLM-based SQL generation agent
│       ├── pipeline/
│       │   ├── __init__.py
│       │   └── base.py                 # EvalPipeline, EvalConfig, EvalResult
│       ├── execution/
│       │   ├── __init__.py
│       │   ├── base.py                 # SQLExecutor (ABC), SQLExecutionError
│       │   ├── sqlite_executor.py      # SQLite backend (BIRD)
│       │   └── snowflake_executor.py   # Snowflake backend (Spider 2.0)
│       ├── datasets/
│       │   ├── adapters/
│       │   │   ├── __init__.py
│       │   │   ├── base.py             # DatasetAdapter (ABC)
│       │   │   ├── bird.py             # BIRD benchmark adapter
│       │   │   └── spider2_snow.py     # Spider 2.0-Snow adapter
│       │   └── generation/             # (stub)
│       └── integrations/               # (stub)
└── tests/
    ├── conftest.py                     # Shared fixtures
    ├── test_test_case.py               # AnalyticsTestCase tests
    ├── test_metrics_base.py            # AnalyticsMetric + MetricResult tests
    ├── test_pytest_plugin.py           # Pytest integration tests
    ├── execution/
    │   └── test_sqlite_executor.py     # SQLite executor tests
    ├── metrics/
    │   ├── sql/
    │   │   ├── test_execution_accuracy.py
    │   │   └── test_schema_adherence.py
    │   └── result/
    │       ├── test_result_accuracy.py
    │       └── test_grain_correctness.py
    ├── agents/
    │   ├── test_agent_interface.py
    │   └── test_llm_sql_agent.py
    ├── pipeline/
    │   ├── test_pipeline.py
    │   └── test_pipeline_sql.py
    └── datasets/
        └── adapters/
            └── test_bird_adapter.py
```

## Five Core Abstractions

### 1. AnalyticsTestCase (Pydantic Model)

The central data model. All fields optional except `input` — the framework degrades gracefully.

**Key design choice:** Pydantic BaseModel for validation + serialization + IDE support. Not frozen because pipeline stages populate fields incrementally (agent fills `actual_sql`, execution fills `actual_results`, etc.).

**Field categories:**
- **Ground truth:** `expected_sql`, `expected_results`, `evidence`, `difficulty`
- **Agent output:** `actual_sql`, `actual_results`, `insight_text`, `chart_spec`
- **Semantic context:** `semantic_context`, `ir_manifest`
- **Execution metadata:** `latency_ms`, `token_count`, `cost_usd`
- **Extensibility:** `metadata` dict

**Helper methods:** `has_expected_sql()`, `has_actual_sql()`, `has_expected_results()`, etc. — used by metrics' `can_evaluate()` for graceful skip.

### 2. AnalyticsMetric (Abstract Base Class)

Every metric produces a `MetricResult` (score 0-1 + reason + details).

**Key design choices:**
- `required_fields` property: declares which `AnalyticsTestCase` fields must be populated
- `can_evaluate()`: checks required fields before running — enables graceful degradation
- `evaluate()` wraps `measure()` with timing, validation, and error handling
- `measure()` is the abstract method subclasses implement — the actual scoring logic
- `threshold` is per-metric and configurable at construction time

**Three evaluation modes:** Deterministic (rule-based), LLM-as-Judge, Hybrid

**Implemented metrics (4):**

| Metric | Category | Mode | Required Fields |
|---|---|---|---|
| ExecutionAccuracy | SQL Correctness | Deterministic | expected_results, actual_results |
| SchemaAdherence | SQL Correctness | Deterministic | actual_sql (+ semantic_context) |
| ResultAccuracy | Result Quality | Deterministic | expected_results, actual_results |
| GrainCorrectness | Result Quality | Deterministic | expected_results, actual_results |

See [metrics.md](metrics.md) for detailed documentation of each metric.

### 3. AgentInterface (ABC) + AgentResponse

BYOA model — any agent implements one async method: `query(question, db_id, context) -> AgentResponse`.

**Key design choice:** Minimal interface. Framework never assumes how the agent works internally. `LLMSQLAgent` is provided as a reference implementation using a pluggable LLM callable.

### 4. EvalPipeline + EvalConfig + EvalResult

Orchestrates the evaluation cycle.

**Flow:**
1. For each test case: if no `actual_sql`, call agent
2. If executor provided: execute `expected_sql` and `actual_sql` to populate DataFrames
3. For each metric: check `can_evaluate()`, then `evaluate()`
4. Aggregate: per-metric mean, per-category mean, overall mean, pass rate
5. Return `EvalResult` with everything

**Config:** Pydantic model for reproducibility. Thresholds override metric defaults. Supports timeout, parallelism, skip-on-missing.

### 5. SQLExecutor (ABC)

Backend-agnostic SQL execution interface. Returns `pd.DataFrame` for metric comparison.

**Implemented backends:**
- `SQLiteExecutor` — for BIRD benchmark (uses built-in `sqlite3`)
- `SnowflakeExecutor` — for Spider 2.0-Snow (requires `snowflake-connector-python`)

### 6. DatasetAdapter (ABC)

Loads benchmark datasets into `AnalyticsTestCase` objects.

**Implemented adapters:**
- `BirdAdapter` — BIRD benchmark (12,751 pairs, 95 SQLite databases)
- `Spider2SnowAdapter` — Spider 2.0-Snow (547 tasks, Snowflake)

## DeepEval-Style Pytest Integration

### Primary API: `assert_test()`

```python
from analytics_eval import AnalyticsTestCase, assert_test
from analytics_eval.metrics import ExecutionAccuracy, ResultAccuracy

def test_revenue_query():
    case = AnalyticsTestCase(
        input="What is total revenue?",
        expected_results=pd.DataFrame({"total": [1000]}),
        actual_results=pd.DataFrame({"total": [1000]}),
    )
    assert_test(case, [ExecutionAccuracy(), ResultAccuracy()])
```

**How it works (following DeepEval's pattern):**

1. `assert_test()` evaluates each metric against the test case
2. Results are captured in a session-scoped `_TestRunResult` collector
3. If any metric fails its threshold, `AssertionError` is raised with detailed failure info
4. Pytest reports the assertion failure normally — no custom runner needed
5. After all tests, `pytest_terminal_summary` hook prints a rich summary table

### Pytest Markers

- `@pytest.mark.analytics` — Mark a test as an analytics evaluation test
- Auto-detection: Tests that call `assert_test()` are auto-marked during collection
- Filter with: `pytest -m analytics` or `pytest -m "not analytics"`

### Fixtures

- `analytics_case` — Quick builder: `case = analytics_case(input="Q", ...)`
- `analytics_dataset` — Load from YAML/JSON: `cases = analytics_dataset.from_yaml("config.yaml")`

### CLI Options

- `--analytics-json=PATH` — Export results to JSON file
- `--analytics-no-summary` — Suppress the terminal summary table

### Terminal Output

```
============================ ANALYTICS EVAL RESULTS ============================

  Overall Score:  0.856  [PASS]
  Pass Rate:      85.0% (20 cases)

  ┌─────────────────────────┬──────────┐
  │ Metric                  │ Avg Score│
  ├─────────────────────────┼──────────┤
  │ ExecutionAccuracy       │ 0.920 PASS │
  │ GrainCorrectness        │ 0.850 PASS │
  │ ResultAccuracy          │ 0.780 PASS │
  │ SchemaAdherence         │ 0.875 PASS │
  └─────────────────────────┴──────────┘

  Per-Case Results:
    test_revenue: +ExecutionAccuracy=1.00 +ResultAccuracy=1.00
    test_orders: -ExecutionAccuracy=0.00 +ResultAccuracy=0.95
```

## Design Decisions

### DECISION-011: Pydantic over Dataclasses
- **Choice:** Use Pydantic BaseModel for all data models
- **Rationale:** Built-in validation, serialization (JSON/dict), IDE auto-complete, type coercion
- **Trade-off:** Pydantic is a dependency, but it's ubiquitous in the Python data ecosystem

### DECISION-012: DataFrame as First-Class Result Type
- **Choice:** Use pandas DataFrames for expected_results and actual_results
- **Rationale:** Analytics results ARE tabular data. DataFrames enable rich comparison (shape, column names, cell values, float tolerance)
- **Trade-off:** Pandas is a heavy dependency, but it's already universally installed in data teams

### DECISION-013: Skip-over-Fail for Missing Fields
- **Choice:** When a metric lacks required data, produce a skip result (score=0, reason explains skip) rather than raising an exception
- **Rationale:** In practice, not every test case will have every field. An insight metric shouldn't crash the pipeline when the agent didn't produce insights
- **Implication:** Tests that skip get score=0, which WILL fail threshold checks. This is intentional — missing data is a signal

### DECISION-014: Sync measure() with async query()
- **Choice:** `AnalyticsMetric.measure()` is synchronous; `AgentInterface.query()` is async
- **Rationale:** Metrics are CPU-bound (comparison, computation) and should be fast. Agent calls are I/O-bound (network, database) and should be async. The pipeline handles the async→sync boundary

### DECISION-015: src/ Layout with Hatchling
- **Choice:** Use `src/analytics_eval/` layout with hatchling build backend
- **Rationale:** src layout prevents accidental imports from the project root. Hatchling is fast and simple

### DECISION-016: assert_test as Primary API (DeepEval-style)
- **Choice:** `assert_test(test_case, metrics)` is the primary API, following DeepEval's naming
- **Rationale:** DeepEval proved this pattern — developers write `assert_test(case, [metric])` as a natural pytest assertion. It's discoverable, type-safe, and integrates with pytest's assertion rewriting
- **Implication:** `assert_analytics()` kept as backward-compatible alias. All docs/examples use `assert_test()`

### DECISION-017: Session-Scoped Result Capture
- **Choice:** Global `_TestRunResult` singleton collects all metric results across the pytest session
- **Rationale:** Enables rich terminal summary, JSON export, and future dashboard integration. DeepEval uses the same pattern
- **Trade-off:** Global state is generally avoided, but pytest's session model makes this safe (one session per run). `reset_test_run()` is available for testing the plugin itself

### DECISION-018: Auto-Mark Detection via Source Inspection
- **Choice:** Tests that call `assert_test()` are auto-marked with `@pytest.mark.analytics`
- **Rationale:** Reduces boilerplate — developers don't need to manually add markers. Enables filtering with `pytest -m analytics`
- **Trade-off:** Source inspection is heuristic-based (looking for "assert_test" in function source). Could miss dynamic calls. Explicit markers still work and take precedence

## Test Coverage

**164 tests across 15 test files:**

| Test File | Tests | What It Covers |
|---|---|---|
| test_test_case.py | 22 | AnalyticsTestCase creation, field checks, serialization, validation |
| test_metrics_base.py | 19 | MetricResult validation, AnalyticsMetric contract, skip/error handling |
| test_execution_accuracy.py | 12 | ExecutionAccuracy: exact match, mismatch, config, required fields |
| test_schema_adherence.py | 10 | SchemaAdherence: valid/invalid tables, columns, strict mode, partial credit |
| test_result_accuracy.py | 7 | ResultAccuracy: tolerance, partial credit, shape mismatch |
| test_grain_correctness.py | 10 | GrainCorrectness: matching/mismatched grain, cardinality, edge cases |
| test_sqlite_executor.py | 16 | SQLite execution, error handling, connectivity, schema introspection |
| test_agent_interface.py | 6 | AgentInterface contract, AgentResponse model |
| test_llm_sql_agent.py | 11 | LLM SQL generation, prompt construction, error handling, SQL extraction |
| test_pipeline.py | 9 | EvalPipeline orchestration, aggregation, sync wrapper, config |
| test_pipeline_sql.py | 8 | Pipeline SQL execution, error handling, executor factory, schema injection |
| test_pytest_plugin.py | 14 | assert_test, assert_analytics, result capture, markers |
| test_bird_adapter.py | 15 | BIRD loading, filtering, executor integration, end-to-end evaluation |
