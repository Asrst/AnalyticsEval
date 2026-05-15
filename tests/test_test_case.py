"""Tests for AnalyticsTestCase — the core data model.

TDD: These tests define the contract that AnalyticsTestCase must fulfill.
Every test here represents a requirement that the implementation must satisfy.
"""

import pandas as pd
import pytest

from analytics_eval.test_case.case import AnalyticsTestCase, ConversationalAnalyticsCase, Difficulty


class TestAnalyticsTestCaseCreation:
    """Test that AnalyticsTestCase can be created with various field combinations."""

    def test_minimal_case_with_input_only(self):
        """A test case with just input is valid — everything else is optional."""
        case = AnalyticsTestCase(input="What is revenue?")
        assert case.input == "What is revenue?"
        assert case.db_id is None
        assert case.expected_sql is None
        assert case.actual_sql is None
        assert case.expected_results is None
        assert case.actual_results is None
        assert case.difficulty is None

    def test_full_case_with_all_fields(self, full_case):
        """A test case with all fields populated works correctly."""
        assert full_case.input == "What is total revenue by region?"
        assert full_case.db_id == "sales_db"
        assert full_case.expected_sql is not None
        assert full_case.actual_sql is not None
        assert full_case.expected_results is not None
        assert full_case.actual_results is not None
        assert full_case.difficulty == Difficulty.MODERATE

    def test_blank_input_raises_validation_error(self):
        """Input must not be blank — this is the only required field."""
        with pytest.raises(ValueError):
            AnalyticsTestCase(input="")

    def test_whitespace_only_input_raises_validation_error(self):
        """Input must not be just whitespace."""
        with pytest.raises(ValueError, match="input must not be blank"):
            AnalyticsTestCase(input="   ")

    def test_case_with_difficulty_enum(self):
        """Difficulty can be set using the Difficulty enum."""
        case = AnalyticsTestCase(input="Q", difficulty=Difficulty.CHALLENGING)
        assert case.difficulty == Difficulty.CHALLENGING
        assert case.difficulty.value == "challenging"

    def test_case_with_metadata_dict(self):
        """Metadata dict enables extensibility without subclassing."""
        case = AnalyticsTestCase(
            input="Q",
            metadata={"source": "bird", "domain": "finance", "custom_tag": 42},
        )
        assert case.metadata["source"] == "bird"
        assert case.metadata["custom_tag"] == 42

    def test_case_with_pandas_dataframe(self):
        """Expected and actual results can be pandas DataFrames."""
        df = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        case = AnalyticsTestCase(
            input="Q",
            expected_results=df,
            actual_results=df,
        )
        assert case.expected_results is not None
        assert len(case.expected_results) == 2


class TestAnalyticsTestCaseFieldChecks:
    """Test the has_* methods that check field availability.

    These methods are critical for the pipeline's graceful degradation:
    metrics declare required_fields, and the pipeline checks has_* before
    running metrics that need specific data.
    """

    def test_has_expected_sql_true(self):
        case = AnalyticsTestCase(input="Q", expected_sql="SELECT 1")
        assert case.has_expected_sql() is True

    def test_has_expected_sql_false_when_none(self):
        case = AnalyticsTestCase(input="Q")
        assert case.has_expected_sql() is False

    def test_has_expected_sql_false_when_blank(self):
        case = AnalyticsTestCase(input="Q", expected_sql="   ")
        assert case.has_expected_sql() is False

    def test_has_actual_sql_true(self):
        case = AnalyticsTestCase(input="Q", actual_sql="SELECT 1")
        assert case.has_actual_sql() is True

    def test_has_expected_results_true(self):
        case = AnalyticsTestCase(
            input="Q",
            expected_results=pd.DataFrame({"a": [1]}),
        )
        assert case.has_expected_results() is True

    def test_has_expected_results_false_when_empty(self):
        case = AnalyticsTestCase(
            input="Q",
            expected_results=pd.DataFrame(),
        )
        assert case.has_expected_results() is False

    def test_has_ir_manifest_true(self):
        case = AnalyticsTestCase(input="Q", ir_manifest={"strategy": "STAR_SCHEMA"})
        assert case.has_ir_manifest() is True

    def test_has_ir_manifest_false_when_empty_dict(self):
        case = AnalyticsTestCase(input="Q", ir_manifest={})
        assert case.has_ir_manifest() is False

    def test_has_insight_true(self):
        case = AnalyticsTestCase(input="Q", insight_text="Revenue increased by 15%")
        assert case.has_insight() is True

    def test_has_chart_true(self):
        case = AnalyticsTestCase(input="Q", chart_spec={"type": "bar", "x": "region"})
        assert case.has_chart() is True


class TestConversationalAnalyticsCase:
    """Test multi-turn conversational test cases."""

    def test_create_conversation(self):
        turns = [
            AnalyticsTestCase(input="What is revenue?"),
            AnalyticsTestCase(input="Break it down by region"),
        ]
        conv = ConversationalAnalyticsCase(turns=turns)
        assert conv.num_turns == 2

    def test_conversation_with_id(self):
        conv = ConversationalAnalyticsCase(
            turns=[AnalyticsTestCase(input="Q")],
            conversation_id="conv-123",
        )
        assert conv.conversation_id == "conv-123"

    def test_empty_turns_raises_validation_error(self):
        with pytest.raises(ValueError):
            ConversationalAnalyticsCase(turns=[])


class TestAnalyticsTestCaseSerialization:
    """Test that test cases can be serialized for logging and debugging."""

    def test_model_dump(self, full_case):
        """Pydantic model_dump produces a dict with all fields."""
        data = full_case.model_dump()
        assert "input" in data
        assert "db_id" in data
        assert data["input"] == "What is total revenue by region?"

    def test_model_dump_json(self, simple_case):
        """Test cases can be serialized to JSON (DataFrame handling)."""
        case = AnalyticsTestCase(input="Q")
        json_str = case.model_dump_json()
        assert "Q" in json_str
