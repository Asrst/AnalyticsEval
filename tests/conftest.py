"""Shared test fixtures for analytics_eval tests."""

import pandas as pd
import pytest

from analytics_eval.test_case.case import AnalyticsTestCase, Difficulty


@pytest.fixture
def simple_case():
    """A minimal AnalyticsTestCase with just input."""
    return AnalyticsTestCase(input="What is total revenue?")


@pytest.fixture
def full_case():
    """A fully populated AnalyticsTestCase with all fields."""
    return AnalyticsTestCase(
        input="What is total revenue by region?",
        db_id="sales_db",
        expected_sql="SELECT region, SUM(amount) FROM orders GROUP BY region",
        expected_results=pd.DataFrame({
            "region": ["East", "West", "North"],
            "total_amount": [1000.0, 2000.0, 1500.0],
        }),
        actual_sql="SELECT region, SUM(amount) as total_amount FROM orders GROUP BY region",
        actual_results=pd.DataFrame({
            "region": ["East", "West", "North"],
            "total_amount": [1000.0, 2000.0, 1500.0],
        }),
        evidence="Revenue excludes returns",
        difficulty=Difficulty.MODERATE,
        insight_text="The West region leads with $2,000 in revenue.",
        semantic_context={"metrics": ["revenue"], "dimensions": ["region"]},
        metadata={"source": "test"},
    )


@pytest.fixture
def mismatched_case():
    """A case where expected and actual results differ."""
    return AnalyticsTestCase(
        input="What is total revenue?",
        expected_results=pd.DataFrame({
            "total": [4500.0],
        }),
        actual_results=pd.DataFrame({
            "total": [3000.0],
        }),
    )


@pytest.fixture
def empty_results_case():
    """A case where both result sets are empty."""
    return AnalyticsTestCase(
        input="What is revenue for region X?",
        expected_results=pd.DataFrame(),
        actual_results=pd.DataFrame(),
    )


@pytest.fixture
def shape_mismatch_case():
    """A case where result shapes differ."""
    return AnalyticsTestCase(
        input="What is revenue by region?",
        expected_results=pd.DataFrame({
            "region": ["East", "West"],
            "total": [1000.0, 2000.0],
        }),
        actual_results=pd.DataFrame({
            "region": ["East"],
            "total": [1000.0],
        }),
    )
