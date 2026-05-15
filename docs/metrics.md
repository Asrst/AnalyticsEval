# Metrics

Detailed documentation of all evaluation metrics in analytics-eval.

## Metric System Overview

Every metric inherits from `AnalyticsMetric` and produces a `MetricResult`:

```python
class MetricResult:
    metric_name: str       # e.g., "ExecutionAccuracy"
    score: float           # 0.0 to 1.0
    reason: str            # Human-readable explanation
    category: MetricCategory  # sql_correctness, result_quality, etc.
    mode: EvaluationMode      # deterministic, llm_as_judge, hybrid
    threshold: float          # Pass threshold (default 0.5)
    details: dict             # Metric-specific structured output
    evaluation_time_ms: float # How long evaluation took
```

Metrics declare `required_fields` — the `AnalyticsTestCase` fields they need. If any are missing, the metric is skipped with `score=0.0`.

## SQL Correctness Metrics

### ExecutionAccuracy

**Purpose:** Measures whether the agent's SQL produces the same results as ground truth SQL.

This is the most fundamental SQL evaluation metric. It compares result sets deterministically, correctly handling semantically equivalent SQL that uses different syntax (e.g., `INNER JOIN` vs. `WHERE` clause joins).

**Required fields:** `expected_results`, `actual_results`

**Comparison logic:**
1. Compare shapes (row/column count)
2. Sort both DataFrames by all columns to handle ordering differences
3. Compare values with configurable float tolerance
4. Score = fraction of matching rows

**Configuration:**
- `threshold` — Pass threshold (default: `0.5`)
- `float_tolerance` — Relative tolerance for floating-point comparison (default: `1e-6`)
- `ignore_order` — Whether to ignore row ordering (default: `True`)
- `ignore_column_order` — Whether to ignore column ordering (default: `True`)

**Scoring:**
- `1.0` — Result sets match exactly
- `0.0` — Column count mismatch, column name mismatch, or no matching rows
- Partial credit — Row count mismatch gives `min_rows / max_rows`

**Example:**
```python
from analytics_eval import AnalyticsTestCase, assert_test
from analytics_eval.metrics import ExecutionAccuracy

case = AnalyticsTestCase(
    input="What is total revenue?",
    expected_results=pd.DataFrame({"total": [4500.0]}),
    actual_results=pd.DataFrame({"total": [4500.0]}),
)
assert_test(case, [ExecutionAccuracy()])
```

---

### SchemaAdherence

**Purpose:** Validates that generated SQL references tables and columns that actually exist in the database schema. Catches "hallucinated" SQL — a common failure mode for text-to-SQL agents.

Unlike `ExecutionAccuracy`, this metric does NOT execute the SQL. It's a purely static analysis using schema metadata.

**Required fields:** `actual_sql`

**Configuration:**
- `threshold` — Pass threshold (default: `0.7`)
- `strict_mode` — If `True`, any unknown reference scores `0.0` (default: `False`)

**Schema source:** Provided via `semantic_context` on the test case. Supports multiple formats:
```python
# Nested under "schema"
semantic_context={"schema": {"tables": ["orders"], "columns": {"orders": ["id", "amount"]}}}

# Nested under "database_schema"
semantic_context={"database_schema": {"tables": ["orders"]}}

# Flat at top level
semantic_context={"tables": ["orders"], "columns": {"orders": ["id"]}}
```

**Scoring:**
- `1.0` — All referenced tables and columns exist in schema
- `0.5+` — Some elements valid, some not (partial credit)
- `0.0` — No schema context available, or strict mode with violations

**Example:**
```python
case = AnalyticsTestCase(
    input="What is revenue?",
    actual_sql="SELECT SUM(amount) FROM orders",
    semantic_context={
        "schema": {
            "tables": ["orders", "customers"],
            "columns": {"orders": ["id", "amount", "customer_id"]},
        },
    },
)
result = SchemaAdherence().evaluate(case)
# score=1.0 — "orders" table and "amount" column both exist
```

---

## Result Quality Metrics

### ResultAccuracy

**Purpose:** Measures numerical accuracy of query results against ground truth with configurable tolerance. Unlike `ExecutionAccuracy` which requires exact (or near-exact) match, `ResultAccuracy` provides graded scoring based on how close values are.

Essential for analytics queries involving aggregations, percentages, and statistical calculations where exact bit-level equality is too strict.

**Required fields:** `expected_results`, `actual_results`

**Configuration:**
- `threshold` — Pass threshold (default: `0.8`, higher than ExecutionAccuracy)
- `relative_tolerance` — Acceptable relative error per cell (default: `0.01` = 1%)

**Scoring:**
- Each cell scored individually: `1.0` if within tolerance, else `max(0, 1 - relative_error)`
- Final score = mean of all cell scores
- Shape mismatch → `0.0`

**Example:**
```python
case = AnalyticsTestCase(
    input="What is average revenue?",
    expected_results=pd.DataFrame({"avg": [1000.0]}),
    actual_results=pd.DataFrame({"avg": [1005.0]}),  # 0.5% off
)
metric = ResultAccuracy(relative_tolerance=0.01)
result = metric.evaluate(case)
# score=1.0 — within 1% tolerance
```

---

### GrainCorrectness

**Purpose:** Checks whether results are at the correct aggregation level (grain). A common failure mode is generating SQL that aggregates at the wrong level — e.g., daily instead of monthly, or missing a `GROUP BY` dimension.

**Required fields:** `expected_results`, `actual_results`

**Configuration:**
- `threshold` — Pass threshold (default: `0.7`)
- `cardinality_tolerance` — How much row count can differ (default: `0.2` = 20%)

**How grain is determined:**
1. **Dimension columns** — Non-numeric columns, or low-cardinality numeric columns (heuristic: unique count <= min(10, rows * 0.3))
2. **Cardinality** — Number of rows
3. **Uniqueness** — Unique combinations of dimension columns

**Scoring (weighted):**
- Column structure: 60% of score
  - Dimensions match exactly → `1.0`
  - Extra dimensions only → `0.6`
  - Missing dimensions only → `0.4`
  - Both missing and extra → `0.2`
- Cardinality: 40% of score
  - Ratio = `min(expected_rows, actual_rows) / max(expected_rows, actual_rows)`

**Example:**
```python
case = AnalyticsTestCase(
    input="Revenue by region and year",
    expected_results=pd.DataFrame({
        "region": ["East", "West"],
        "year": [2023, 2023],
        "revenue": [1000.0, 2000.0],
    }),
    actual_results=pd.DataFrame({
        "region": ["East", "West"],
        "revenue": [1000.0, 2000.0],  # missing 'year' dimension
    }),
)
result = GrainCorrectness().evaluate(case)
# score < 1.0 — missing "year" dimension
```

---

## Writing a Custom Metric

Subclass `AnalyticsMetric` and implement five properties and one method:

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

Then use it like any built-in metric:

```python
assert_test(case, [ExecutionAccuracy(), QuerySafety()])
```

---

## Metric Categories

| Category | Description | Implemented |
|---|---|---|
| `sql_correctness` | Validates SQL generation quality | ExecutionAccuracy, SchemaAdherence |
| `result_quality` | Validates query output correctness | ResultAccuracy, GrainCorrectness |
| `insight_quality` | Validates natural language insights | (planned) |
| `visualization_quality` | Validates chart specifications | (planned) |
| `model_quality` | Validates semantic model completeness | (planned) |

## Evaluation Modes

| Mode | Description | Use Case |
|---|---|---|
| `deterministic` | Rule-based, reproducible scoring | All current metrics |
| `llm_as_judge` | Uses an LLM to evaluate quality | SemanticSQL, InsightFaithfulness |
| `hybrid` | Combines rules with LLM judgment | QuerySafety |
