"""Test case definitions for analytics evaluation.

AnalyticsTestCase captures the full context of an analytics interaction,
extending the concept of LLM test cases with SQL, results, insights,
charts, and IR manifests.
"""

from analytics_eval.test_case.case import AnalyticsTestCase, ConversationalAnalyticsCase

__all__ = ["AnalyticsTestCase", "ConversationalAnalyticsCase"]
