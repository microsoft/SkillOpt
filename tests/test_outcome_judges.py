from __future__ import annotations

import pytest

from skillopt_sleep.judges import (
    KNOWN_OPS,
    SHAPE_OPS,
    char_bound,
    is_shape_only,
    score_rule_judge,
    validate_checks,
)
from skillopt_sleep.llm_miner import _mk_task
from skillopt_sleep.types import SessionDigest


def _digest() -> SessionDigest:
    return SessionDigest(
        session_id="s1",
        project=r"C:\proj",
        user_prompts=["gather context for a task"],
        assistant_finals=["done"],
        n_user_turns=1,
        n_assistant_turns=1,
    )


# --- outcome ops -------------------------------------------------------------


def test_not_contains_passes_when_absent_and_fails_when_present() -> None:
    judge = {"kind": "rule", "checks": [{"op": "not_contains", "arg": "TODO"}]}
    assert score_rule_judge(judge, "a complete answer")[0] == 1.0
    assert score_rule_judge(judge, "still TODO")[0] == 0.0


@pytest.mark.parametrize(
    "response",
    [
        "Cannot complete this task. There is no benchmark spec in scope.",
        "I cannot help with that.",
        "I'm unable to do this.",
        "> I cannot help with that.",
        "- I cannot help with that.",
        "1. I cannot help with that.",
        "**Sorry, I can't** do this.",
        "> " * 100 + "I cannot help with that.",
        "",
    ],
)
def test_no_refusal_fails_on_bare_refusals(response: str) -> None:
    judge = {"kind": "rule", "checks": [{"op": "no_refusal"}]}
    assert score_rule_judge(judge, response)[0] == 0.0


def test_no_refusal_passes_on_substantive_answer() -> None:
    judge = {"kind": "rule", "checks": [{"op": "no_refusal"}]}
    assert score_rule_judge(judge, "Here is the context you asked for: ...")[0] == 1.0


def test_no_refusal_ignores_markdown_prefix_length() -> None:
    judge = {"kind": "rule", "checks": [{"op": "no_refusal"}]}
    response = "> " * 350 + "I cannot help with that."
    assert len(response) >= 600
    assert score_rule_judge(judge, response)[0] == 0.0


def test_no_refusal_length_is_measured_before_unicode_lowercasing() -> None:
    judge = {"kind": "rule", "checks": [{"op": "no_refusal"}]}
    # U+0130 lowercases to two code points. Case normalization must not turn a
    # short refusal into an apparently substantive response over 600 chars.
    response = "I cannot help. " + "İ" * 300
    assert len(response) < 600
    assert len(response.lower()) >= 600
    assert score_rule_judge(judge, response)[0] == 0.0


def test_no_refusal_accepts_a_refusal_that_still_does_the_work() -> None:
    # An abstention that explains what was searched and what is missing is a
    # useful answer, not a dead end.
    response = "Cannot complete this task. " + (
        "I searched the working directory, the session artifacts folder, and "
        "the benchmarks docs path, and none of them contain a spec. To proceed "
        "I would need the spec file or a path to it. Here is what I checked and "
        "what each location contained, so the gap is reproducible. " * 3
    )
    assert len(response) >= 600
    judge = {"kind": "rule", "checks": [{"op": "no_refusal"}]}
    assert score_rule_judge(judge, response)[0] == 1.0


def test_new_ops_are_registered_and_validate() -> None:
    assert {"not_contains", "no_refusal"} <= KNOWN_OPS
    errors, _ = validate_checks(
        {"checks": [{"op": "not_contains", "arg": "x"}, {"op": "no_refusal"}]}
    )
    assert errors == []


def test_not_contains_requires_an_arg() -> None:
    errors, _ = validate_checks({"checks": [{"op": "not_contains", "arg": "  "}]})
    assert errors and "not_contains" in errors[0]


# --- shape-only detection ----------------------------------------------------


def test_shape_ops_are_the_formatting_ops() -> None:
    assert SHAPE_OPS == {"section_present", "section_contains", "max_chars", "min_chars"}


def test_is_shape_only_flags_a_formatting_judge() -> None:
    assert is_shape_only({"checks": [{"op": "section_present", "arg": "Results"}]})
    assert is_shape_only(
        {"checks": [{"op": "section_present", "arg": "Results"}, {"op": "max_chars", "arg": 500}]}
    )


def test_is_shape_only_false_when_any_outcome_op_present() -> None:
    assert not is_shape_only(
        {
            "checks": [
                {"op": "section_present", "arg": "Results"},
                {"op": "contains", "arg": "answer"},
            ]
        }
    )
    assert not is_shape_only({"checks": []})
    assert not is_shape_only(None)


def test_validate_warns_on_shape_only_judge() -> None:
    _, warnings = validate_checks({"checks": [{"op": "section_present", "arg": "Results"}]})
    assert any("shape-only" in w for w in warnings)


def test_validate_does_not_warn_when_an_outcome_op_is_present() -> None:
    _, warnings = validate_checks(
        {"checks": [{"op": "section_present", "arg": "Results"}, {"op": "no_refusal"}]}
    )
    assert not any("shape-only" in w for w in warnings)


# --- miner preference: the actual reward-hack regression ---------------------


def test_shape_only_checks_lose_to_the_rubric() -> None:
    # Regression for the observed hack: a `section_present=Results` judge let
    # the optimizer score 1.0 by adding a heading. With a rubric available the
    # task must be graded on outcome instead.
    task = _mk_task(
        _digest(),
        {
            "intent": "gather context for a task",
            "checks": [{"op": "section_present", "arg": "Results"}],
            "rubric": "A good answer reports what was found and what is missing.",
            "satisfied": False,
        },
        0,
    )
    assert task is not None
    assert task.reference_kind == "rubric"
    assert "what is missing" in task.reference


def test_shape_only_checks_still_used_when_no_rubric_offered() -> None:
    task = _mk_task(
        _digest(),
        {
            "intent": "gather context for a task",
            "checks": [{"op": "section_present", "arg": "Results"}],
            "rubric": "",
            "satisfied": True,
        },
        0,
    )
    assert task is not None
    assert task.reference_kind == "rule"


def test_miner_keeps_section_contains_when_no_rubric_offered() -> None:
    task = _mk_task(
        _digest(),
        {
            "intent": "write a report with an annotated risks heading",
            "checks": [{"op": "section_contains", "arg": "  Key Risks  "}],
            "rubric": "",
            "satisfied": True,
        },
        0,
    )
    assert task is not None
    assert task.reference_kind == "rule"
    assert task.judge["checks"] == [{"op": "section_contains", "arg": "Key Risks"}]


def test_outcome_checks_also_lose_to_the_rubric() -> None:
    # Second-order regression: after shape checks were demoted, the miner
    # produced `contains=DEFAULT_ORGANIZATION` and the optimizer won by
    # injecting that literal. Any literal-string check is injectable through
    # skill text, so the rubric wins whenever one exists.
    task = _mk_task(
        _digest(),
        {
            "intent": "gather context for a task",
            "checks": [{"op": "contains", "arg": "DEFAULT_ORGANIZATION"}],
            "rubric": "A good answer reports what was found.",
            "satisfied": True,
        },
        0,
    )
    assert task is not None
    assert task.reference_kind == "rubric"


def test_outcome_checks_used_when_no_rubric_offered() -> None:
    task = _mk_task(
        _digest(),
        {
            "intent": "gather context for a task",
            "checks": [{"op": "no_refusal"}],
            "rubric": "",
            "satisfied": True,
        },
        0,
    )
    assert task is not None
    assert task.reference_kind == "rule"
    assert task.judge["checks"] == [{"op": "no_refusal", "arg": None}]


def test_miner_keeps_the_new_outcome_ops() -> None:
    task = _mk_task(
        _digest(),
        {
            "intent": "gather context for a task",
            "checks": [{"op": "not_contains", "arg": "TODO"}],
            "rubric": "",
            "satisfied": True,
        },
        0,
    )
    assert task is not None
    assert task.judge["checks"] == [{"op": "not_contains", "arg": "TODO"}]


# --- review-follow-up regressions -------------------------------------------


def test_no_refusal_does_not_flag_a_helpful_sorry() -> None:
    # "Sorry, I can help" opens with an apology but is not an abstention; the
    # refusal prefixes must not swallow it.
    judge = {"kind": "rule", "checks": [{"op": "no_refusal"}]}
    assert score_rule_judge(judge, "Sorry, I can help with that. Here it is.")[0] == 1.0


@pytest.mark.parametrize("bad", [[], ["not", "a", "dict"], "string", 7, True])
def test_is_shape_only_returns_false_for_non_dict(bad) -> None:
    # A public helper imported by tests must not raise on a truthy non-dict.
    assert is_shape_only(bad) is False


def test_miner_keeps_a_tool_called_check() -> None:
    # tool_called is advertised by the miner prompt and supported by the local
    # judge, so a tool-only task must not be dropped as uncheckable.
    task = _mk_task(
        _digest(),
        {
            "intent": "invoke the search tool for a task",
            "checks": [{"op": "tool_called", "arg": "search"}],
            "rubric": "",
            "satisfied": True,
        },
        0,
    )
    assert task is not None
    assert task.judge["checks"] == [{"op": "tool_called", "arg": "search"}]


def test_miner_ignores_a_non_string_rubric() -> None:
    # A JSON null rubric must not coerce to the literal "None" and win over the
    # real checks.
    task = _mk_task(
        _digest(),
        {
            "intent": "gather context for a task",
            "checks": [{"op": "not_contains", "arg": "TODO"}],
            "rubric": None,
            "satisfied": True,
        },
        0,
    )
    assert task is not None
    assert task.reference_kind == "rule"
    assert task.judge["checks"] == [{"op": "not_contains", "arg": "TODO"}]


def test_miner_drops_malformed_char_checks() -> None:
    # A max_chars with a non-integer arg would crash the scorer during replay;
    # it is dropped, and an arg-less contains is dropped too.
    task = _mk_task(
        _digest(),
        {
            "intent": "gather context for a task",
            "checks": [
                {"op": "max_chars", "arg": "lots"},
                {"op": "contains", "arg": ""},
                {"op": "min_chars", "arg": "50"},
            ],
            "rubric": "",
            "satisfied": True,
        },
        0,
    )
    assert task is not None
    # only the coercible min_chars survives, normalized to an int
    assert task.judge["checks"] == [{"op": "min_chars", "arg": 50}]


def test_miner_strips_string_args() -> None:
    # Stray whitespace would otherwise become part of the required substring.
    task = _mk_task(
        _digest(),
        {
            "intent": "gather context for a task",
            "checks": [{"op": "contains", "arg": "  DEFAULT  "}],
            "rubric": "",
            "satisfied": True,
        },
        0,
    )
    assert task is not None
    assert task.judge["checks"] == [{"op": "contains", "arg": "DEFAULT"}]


def test_miner_preserves_regex_whitespace() -> None:
    # Trimming a regex changes what it matches, so the pattern is kept verbatim
    # even though other string args are stripped.
    pattern = r"\bfoo\s+$"
    task = _mk_task(
        _digest(),
        {
            "intent": "gather context for a task",
            "checks": [{"op": "regex", "arg": pattern}, {"op": "contains", "arg": " x "}],
            "rubric": "",
            "satisfied": True,
        },
        0,
    )
    assert task is not None
    assert task.judge["checks"] == [
        {"op": "regex", "arg": pattern},
        {"op": "contains", "arg": "x"},
    ]
    errors, _ = validate_checks(task.judge)
    assert errors == []



def test_mined_checks_always_pass_validate_checks() -> None:
    # The miner must never emit a judge that validate_checks() later rejects:
    # bools (int subclass) and negative bounds are errors there.
    task = _mk_task(
        _digest(),
        {
            "intent": "gather context for a task",
            "checks": [
                {"op": "max_chars", "arg": True},
                {"op": "min_chars", "arg": -5},
                {"op": "contains", "arg": " ok "},
            ],
            "rubric": "",
            "satisfied": True,
        },
        0,
    )
    assert task is not None
    assert task.judge["checks"] == [{"op": "contains", "arg": "ok"}]
    errors, _ = validate_checks(task.judge)
    assert errors == []


@pytest.mark.parametrize(
    "bad", [1.9, float("inf"), float("nan"), float("-inf"), "5_0", "ten", None, True],
)
def test_miner_and_validator_agree_on_bad_char_bounds(bad) -> None:
    # Root cause of several drifts: the miner used int(arg) while the validator
    # applied stricter rules, so a mined bound could fail its own validation.
    # Both now share char_bound(), so anything the miner keeps must validate.
    task = _mk_task(
        _digest(),
        {
            "intent": "gather context for a task",
            "checks": [{"op": "max_chars", "arg": bad}, {"op": "no_refusal"}],
            "rubric": "",
            "satisfied": True,
        },
        0,
    )
    assert task is not None
    # The malformed bound is dropped, never silently truncated (1.9 -> 1).
    assert task.judge["checks"] == [{"op": "no_refusal", "arg": None}]
    errors, _ = validate_checks(task.judge)
    assert errors == []


@pytest.mark.parametrize(
    ("arg", "expected"), [(50, 50), (50.0, 50), (" 50 ", 50), ("+50", 50)],
)
def test_char_bound_accepts_integral_forms(arg, expected) -> None:
    assert char_bound(arg) == expected


@pytest.mark.parametrize("bad", [1.9, True, "5_0", "ten", None, []])
def test_char_bound_rejects_non_integers(bad) -> None:
    with pytest.raises((ValueError, TypeError)):
        char_bound(bad)
