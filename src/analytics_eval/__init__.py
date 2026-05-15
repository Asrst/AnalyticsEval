"""Analytics Eval — An open-source evaluation framework for analytics agents.

Primary API:

    from analytics_eval import AnalyticsTestCase, assert_test
    from analytics_eval.metrics import ExecutionAccuracy

    def test_revenue_query():
        case = AnalyticsTestCase(
            input="What is total revenue?",
            expected_results=pd.DataFrame({"total": [1000]}),
            actual_results=pd.DataFrame({"total": [1000]}),
        )
        assert_test(case, [ExecutionAccuracy()])
"""

from analytics_eval.metrics.base import (
    AnalyticsMetric,
    EvaluationMode,
    MetricCategory,
    MetricResult,
)
from analytics_eval.test_case import AnalyticsTestCase, ConversationalAnalyticsCase
from analytics_eval.test_case.case import Difficulty

__version__ = "0.1.0"

__all__ = [
    # Core abstractions
    "AnalyticsTestCase",
    "ConversationalAnalyticsCase",
    "Difficulty",
    "AnalyticsMetric",
    "MetricResult",
    "MetricCategory",
    "EvaluationMode",
]
