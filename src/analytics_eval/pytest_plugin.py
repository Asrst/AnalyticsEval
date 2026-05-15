"""Pytest plugin for analytics evaluation — DeepEval-style.

Provides:
- ``assert_test(test_case, metrics)`` — primary assertion function
- ``assert_analytics(...)`` — backward-compatible alias
- ``@pytest.mark.analytics`` — marker for analytics eval tests
- ``analytics_case`` fixture — quick AnalyticsTestCase builder
- ``analytics_dataset`` fixture — load test cases from config
- Pytest hooks for result capture, rich terminal reporting, JSON export

Design principles (following DeepEval):
1. Tests are regular pytest functions — no custom runner needed
2. ``assert_test()`` hooks into pytest's assertion rewriting
3. Results are captured automatically and displayed in a summary table
4. ``--analytics-json`` flag exports machine-readable results
5. ``@pytest.mark.analytics`` enables filtering with ``-m analytics``
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from analytics_eval.metrics.base import AnalyticsMetric, MetricResult
from analytics_eval.test_case.case import AnalyticsTestCase

# ── Global result capture (session-scoped) ──────────────────────────────────

class _TestRunResult:
    """Collects all metric results across the pytest session."""

    def __init__(self) -> None:
        self.cases: list[dict[str, Any]] = []
        self._current_test_id: str | None = None

    def record(
        self,
        test_id: str,
        test_case: AnalyticsTestCase,
        metric_results: list[MetricResult],
    ) -> None:
        """Record results for a single test case."""
        self.cases.append({
            "test_id": test_id,
            "input": test_case.input,
            "db_id": test_case.db_id,
            "metrics": [
                {
                    "name": r.metric_name,
                    "score": r.score,
                    "passed": r.passed,
                    "reason": r.reason,
                    "category": r.category.value,
                    "mode": r.mode.value,
                    "threshold": r.threshold,
                    "evaluation_time_ms": r.evaluation_time_ms,
                    "details": r.details,
                }
                for r in metric_results
            ],
        })

    def summary(self) -> dict[str, Any]:
        """Compute aggregate summary for the entire session."""
        if not self.cases:
            return {
                "total_cases": 0,
                "overall_score": 0.0,
                "pass_rate": 0.0,
            }

        all_scores: list[float] = []
        pass_count = 0
        total_count = 0
        by_metric: dict[str, list[float]] = {}

        for case in self.cases:
            for m in case["metrics"]:
                all_scores.append(m["score"])
                total_count += 1
                if m["passed"]:
                    pass_count += 1
                by_metric.setdefault(m["name"], []).append(m["score"])

        return {
            "total_cases": len(self.cases),
            "overall_score": (
                sum(all_scores) / len(all_scores) if all_scores else 0.0
            ),
            "pass_rate": pass_count / total_count if total_count > 0 else 0.0,
            "by_metric": {
                k: sum(v) / len(v) for k, v in by_metric.items()
            },
            "cases": self.cases,
        }

    def to_json(self) -> str:
        """Export full results as JSON."""
        return json.dumps(self.summary(), indent=2, default=str)


# Module-level singleton — shared across the session
_test_run = _TestRunResult()


def get_test_run() -> _TestRunResult:
    """Get the global test run result collector."""
    return _test_run


def reset_test_run() -> None:
    """Reset the global test run (useful between test sessions)."""
    global _test_run
    _test_run = _TestRunResult()


# ── Primary assertion function ──────────────────────────────────────────────

def assert_test(
    test_case: AnalyticsTestCase,
    metrics: Sequence[AnalyticsMetric],
) -> list[MetricResult]:
    """Assert that all metrics pass their thresholds for the given test case.

    How it works:
    1. Evaluates each metric against the test case
    2. Records results in the global test run (for reporting)
    3. Raises ``AssertionError`` with detailed failure info if any metric fails
    4. Returns all MetricResult objects on success

    Usage::

        from analytics_eval import AnalyticsTestCase, assert_test
        from analytics_eval.metrics import ExecutionAccuracy

        def test_revenue_query():
            case = AnalyticsTestCase(
                input="What is total revenue?",
                expected_results=pd.DataFrame({"total": [1000]}),
                actual_results=pd.DataFrame({"total": [1000]}),
            )
            assert_test(case, [ExecutionAccuracy()])

    Args:
        test_case: The analytics interaction to evaluate.
        metrics: Sequence of metrics to apply.

    Returns:
        List of MetricResult objects (one per metric).

    Raises:
        AssertionError: If any metric score is below its threshold.
    """
    results: list[MetricResult] = []
    failures: list[str] = []

    for metric in metrics:
        result = metric.evaluate(test_case)
        results.append(result)

        if not result.passed:
            failures.append(str(result))

    # Record in global test run
    test_id = _get_test_id()
    _test_run.record(test_id, test_case, results)

    if failures:
        failure_text = "\n".join(f"  - {f}" for f in failures)
        raise AssertionError(
            f"Analytics evaluation failed:\n{failure_text}"
        )

    return results


def assert_analytics(
    test_case: AnalyticsTestCase,
    metrics: Sequence[AnalyticsMetric],
) -> list[MetricResult]:
    """Backward-compatible alias for ``assert_test``.

    .. deprecated::
        Use ``assert_test`` instead.
    """
    return assert_test(test_case, metrics)


def _get_test_id() -> str:
    """Get a test identifier from the calling frame for result tracking."""
    frame = inspect.currentframe()
    for _ in range(10):
        if frame is None:
            break
        name = frame.f_code.co_name
        if name.startswith("test_"):
            return name
        frame = frame.f_back
    return "unknown"


# ── Pytest fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def analytics_case():
    """Fixture that provides a fresh AnalyticsTestCase builder.

    Usage::

        def test_with_fixture(analytics_case):
            case = analytics_case(
                input="What is revenue?",
                expected_results=pd.DataFrame({"total": [4500]}),
                actual_results=pd.DataFrame({"total": [4500]}),
            )
            assert_test(case, [ExecutionAccuracy()])
    """
    def _make_case(**kwargs):
        return AnalyticsTestCase(**kwargs)
    return _make_case


@pytest.fixture
def analytics_dataset(tmp_path):
    """Fixture that provides a dataset loader for test cases.

    Returns a helper that creates a list of AnalyticsTestCases from
    a YAML or JSON config file.
    """
    class _DatasetLoader:
        def from_yaml(self, path: str | Path) -> list[AnalyticsTestCase]:
            """Load test cases from a YAML config file."""
            import yaml  # optional dependency
            with open(path) as f:
                config = yaml.safe_load(f)
            return self._parse_config(config)

        def from_json(self, path: str | Path) -> list[AnalyticsTestCase]:
            """Load test cases from a JSON config file."""
            with open(path) as f:
                config = json.load(f)
            return self._parse_config(config)

        def _parse_config(self, config: dict) -> list[AnalyticsTestCase]:
            """Parse a dataset config into AnalyticsTestCases."""
            cases = []
            for item in config.get("cases", []):
                cases.append(AnalyticsTestCase(**item))
            return cases

    return _DatasetLoader()


# ── Pytest hooks ─────────────────────────────────────────────────────────────

def pytest_addoption(parser: pytest.Parser) -> None:
    """Add analytics-eval command line options."""
    group = parser.getgroup("analytics-eval", "Analytics Eval Framework")
    group.addoption(
        "--analytics-json",
        action="store",
        default=None,
        metavar="PATH",
        help="Export analytics evaluation results to JSON file",
    )
    group.addoption(
        "--analytics-no-summary",
        action="store_true",
        default=False,
        help="Suppress the analytics evaluation summary table",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register the analytics marker and configure the plugin."""
    config.addinivalue_line(
        "markers",
        "analytics: mark test as an analytics evaluation test "
        "(deselect with '-m \"not analytics\"')",
    )
    # Reset the test run for a fresh session
    reset_test_run()


def pytest_terminal_summary(
    terminalreporter: pytest.TerminalReporter,
    exitstatus: int,
    config: pytest.Config,
) -> None:
    """Print analytics evaluation summary table and export JSON if requested."""
    if config.getoption("--analytics-no-summary", default=False):
        return

    test_run = get_test_run()
    if not test_run.cases:
        return

    summary = test_run.summary()
    tw = terminalreporter

    tw.write("\n")
    tw.ensure_newline()
    tw.write_sep("=", "ANALYTICS EVAL RESULTS", bold=True)
    tw.write("\n")

    overall = summary["overall_score"]
    pass_rate = summary["pass_rate"]
    total = summary["total_cases"]

    if pass_rate >= 0.8:
        status_icon = "PASS"
    elif pass_rate >= 0.5:
        status_icon = "WARN"
    else:
        status_icon = "FAIL"

    tw.write(f"  Overall Score:  {overall:.3f}  [{status_icon}]\n")
    tw.write(f"  Pass Rate:      {pass_rate:.1%} ({total} cases)\n")
    tw.write("\n")

    if summary.get("by_metric"):
        tw.write("  ┌─────────────────────────┬──────────┐\n")
        tw.write("  │ Metric                  │ Avg Score│\n")
        tw.write("  ├─────────────────────────┼──────────┤\n")
        for metric_name, score in sorted(summary["by_metric"].items()):
            status = "PASS" if score >= 0.5 else "FAIL"
            tw.write(
                f"  │ {metric_name:<23} │ {score:.3f} {status} │\n"
            )
        tw.write("  └─────────────────────────┴──────────┘\n")
    tw.write("\n")

    if total <= 20:
        tw.write("  Per-Case Results:\n")
        for case in summary["cases"]:
            metrics_strs = []
            for m in case["metrics"]:
                icon = "+" if m["passed"] else "-"
                metrics_strs.append(f"{icon}{m['name']}={m['score']:.2f}")
            tw.write(f"    {case['test_id']}: {' '.join(metrics_strs)}\n")
        tw.write("\n")

    json_path = config.getoption("--analytics-json", default=None)
    if json_path:
        output_path = Path(json_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(test_run.to_json())
        tw.write(f"  Results exported to: {output_path}\n\n")


# ── Pytest marker ────────────────────────────────────────────────────────────

def pytest_collection_modifyitems(
    session: pytest.Session,
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Auto-add the analytics marker to tests using assert_test."""
    for item in items:
        if isinstance(item, pytest.Function):
            source = _inspect_getsource(item.function)
            if source and (
                "assert_test" in source or "assert_analytics" in source
            ):
                item.add_marker(pytest.mark.analytics)


def _inspect_getsource(func) -> str | None:
    """Safely get source code of a function."""
    try:
        return inspect.getsource(func)
    except (OSError, TypeError):
        return None
