"""Analytics Eval — An open-source evaluation framework for analytics agents.

Primary API (DeepEval-style):

    from analytics_eval import AnalyticsTestCase, assert_test
    from analytics_eval.metrics import ExecutionAccuracy

    def test_revenue_query():
        case = AnalyticsTestCase(
            input="What is total revenue?",
            expected_results=pd.DataFrame({"total": [1000]}),
            actual_results=pd.DataFrame({"total": [1000]}),
        )
        assert_test(case, [ExecutionAccuracy()])

Benchmark Evaluation API:

    from analytics_eval import EvalPipeline, EvalConfig
    from analytics_eval.datasets.adapters import BirdAdapter
    from analytics_eval.agents import LLMSQLAgent
    from analytics_eval.metrics import ExecutionAccuracy, SchemaAdherence

    adapter = BirdAdapter(data_dir="/data/bird")
    agent = LLMSQLAgent(llm_call=my_llm_function)

    pipeline = EvalPipeline(
        agent=agent,
        metrics=[ExecutionAccuracy(), SchemaAdherence()],
        executor_factory=adapter.get_executor_for_db,
    )
    result = pipeline.evaluate_sync(adapter.load(split="mini_dev"))
"""

from analytics_eval.agents.base import AgentInterface, AgentResponse
from analytics_eval.agents.llm_sql_agent import LLMSQLAgent
from analytics_eval.datasets.adapters.base import DatasetAdapter
from analytics_eval.datasets.adapters.bird import BirdAdapter
from analytics_eval.datasets.adapters.spider2_snow import Spider2SnowAdapter
from analytics_eval.execution.base import SQLExecutor, SQLExecutionError
from analytics_eval.execution.sqlite_executor import SQLiteExecutor
from analytics_eval.metrics.base import (
    AnalyticsMetric,
    EvaluationMode,
    MetricCategory,
    MetricResult,
)
from analytics_eval.pipeline.base import EvalConfig, EvalPipeline, EvalResult
from analytics_eval.pytest_plugin import (
    assert_analytics,
    assert_test,
    get_test_run,
)
from analytics_eval.test_case import (
    AnalyticsTestCase,
    ConversationalAnalyticsCase,
)
from analytics_eval.test_case.case import Difficulty

__version__ = "0.1.0"

__all__ = [
    # Primary API (DeepEval-style)
    "assert_test",
    "assert_analytics",
    # Core abstractions
    "AnalyticsTestCase",
    "ConversationalAnalyticsCase",
    "Difficulty",
    "AnalyticsMetric",
    "MetricResult",
    "MetricCategory",
    "EvaluationMode",
    # Agents
    "AgentInterface",
    "AgentResponse",
    "LLMSQLAgent",
    # Pipeline
    "EvalPipeline",
    "EvalConfig",
    "EvalResult",
    # SQL Execution
    "SQLExecutor",
    "SQLExecutionError",
    "SQLiteExecutor",
    # Dataset Adapters
    "DatasetAdapter",
    "BirdAdapter",
    "Spider2SnowAdapter",
    # Reporting
    "get_test_run",
]
