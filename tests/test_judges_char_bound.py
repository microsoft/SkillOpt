"""A malformed max_chars/min_chars bound is reported as invalid, not raised.

Rule judges imported without validation (for example a gbrain-style
benchmark.jsonl) can carry a bound validate_checks() would reject. Scoring
called int() on it, so the night crashed on a bad bound, or silently used a
truncated or boolean one, while a bad regex is already reported as invalid.
"""

from __future__ import annotations

import pytest

from skillopt_sleep.judges import score_rule_judge, score_rule_judge_with_feedback


def _judge(*checks: dict) -> dict:
    return {"kind": "rule", "checks": list(checks)}


@pytest.mark.parametrize("op", ["max_chars", "min_chars"])
@pytest.mark.parametrize("arg", ["abc", "500 chars", None, True, 500.5, float("inf"), -1])
def test_malformed_char_bound_fails_closed_and_is_labelled_invalid(op, arg):
    hard, soft, rationale, feedback = score_rule_judge_with_feedback(_judge({"op": op, "arg": arg}), "x" * 100)

    assert (hard, soft) == (0.0, 0.0)
    assert "invalid char bound" in rationale
    assert "evaluator configuration is invalid" in feedback


def test_other_checks_are_still_scored_next_to_a_malformed_bound():
    hard, soft, _rationale = score_rule_judge(
        _judge({"op": "max_chars", "arg": "abc"}, {"op": "contains", "arg": "risk"}), "Key risk: none."
    )

    assert hard == 0.0
    assert soft == 0.5


@pytest.mark.parametrize("arg", [500, "500", " 500 ", 500.0])
def test_valid_char_bounds_are_unchanged(arg):
    assert score_rule_judge(_judge({"op": "max_chars", "arg": arg}), "x" * 500)[0] == 1.0
    assert score_rule_judge(_judge({"op": "max_chars", "arg": arg}), "x" * 501)[0] == 0.0
    assert score_rule_judge(_judge({"op": "min_chars", "arg": arg}), "x" * 500)[0] == 1.0
    assert score_rule_judge(_judge({"op": "min_chars", "arg": arg}), "x" * 499)[0] == 0.0
