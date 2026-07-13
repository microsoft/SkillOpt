"""Tests for Superpowers skill evaluation (offline, no API)."""
import re
import pytest

from skillopt_sleep.adapters.superpowers import (
    VERIFICATION_SCENARIOS,
    _get_scenarios,
    _score_check,
)


def test_scenarios_exist():
    scenarios = _get_scenarios("verification-before-completion")
    assert len(scenarios) >= 3


def test_scenarios_have_required_fields():
    for s in VERIFICATION_SCENARIOS:
        assert "id" in s
        assert "description" in s
        assert "prompt" in s
        assert "judge" in s


def test_unknown_skill_raises():
    with pytest.raises(ValueError):
        _get_scenarios("nonexistent-skill")


class TestJudgeLogic:
    """Test rule-based judge scoring."""

    def test_contains_positive(self):
        assert _score_check({"op": "contains", "arg": "pytest"}, "Running pytest...") is True

    def test_contains_negative(self):
        assert _score_check({"op": "contains", "arg": "pytest"}, "Running tests...") is False

    def test_not_contains_positive(self):
        assert _score_check({"op": "not_contains", "arg": "error"}, "All good!") is True

    def test_not_contains_negative(self):
        assert _score_check({"op": "not_contains", "arg": "error"}, "Got an error") is False

    def test_regex_positive(self):
        assert _score_check({"op": "regex", "arg": r"\d+ passed"}, "5 passed in 0.1s") is True

    def test_regex_negative(self):
        assert _score_check({"op": "regex", "arg": r"\d+ passed"}, "tests ran") is False

    def test_order_positive(self):
        check = {"op": "order", "args": ["pytest", "done|complete"]}
        assert _score_check(check, "Running pytest... 1 passed. Done!") is True

    def test_order_negative(self):
        check = {"op": "order", "args": ["pytest", "done|complete"]}
        assert _score_check(check, "Done! Should run pytest.") is False

    def test_any_of_first_match(self):
        check = {"op": "any_of", "args": [
            {"op": "contains", "arg": "python"},
            {"op": "contains", "arg": "pytest"},
        ]}
        assert _score_check(check, "Running python") is True

    def test_any_of_second_match(self):
        check = {"op": "any_of", "args": [
            {"op": "contains", "arg": "python"},
            {"op": "contains", "arg": "pytest"},
        ]}
        assert _score_check(check, "Running pytest") is True

    def test_any_of_no_match(self):
        check = {"op": "any_of", "args": [
            {"op": "contains", "arg": "python"},
            {"op": "contains", "arg": "pytest"},
        ]}
        assert _score_check(check, "Just checking") is False
