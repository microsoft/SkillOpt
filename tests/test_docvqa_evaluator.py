"""Unit tests for the DocVQA ANLS evaluator.

ANLS (Biten et al., 2019): for each gold answer, NL is the Levenshtein distance
divided by the longer string's length; the score is 1 - NL when NL < 0.5 and 0
otherwise, and the item score is the maximum over gold answers.
"""

import pytest

from skillopt.envs.docvqa.evaluator import (
    DEFAULT_ANLS_THRESHOLD,
    _extract_answer_strings,
    _levenshtein_distance,
    _score_single_answer,
    evaluate,
    extract_answer,
)


@pytest.mark.parametrize(
    ("a", "b", "distance"),
    [("kitten", "sitting", 3), ("flaw", "lawn", 2), ("abc", "abc", 0), ("", "abc", 3), ("abc", "", 3)],
)
def test_levenshtein_distance_known_values(a, b, distance):
    assert _levenshtein_distance(a, b) == distance
    assert _levenshtein_distance(b, a) == distance


def test_threshold_is_the_standard_anls_value():
    assert DEFAULT_ANLS_THRESHOLD == 0.5


@pytest.mark.parametrize(
    ("predicted", "target", "expected"),
    [
        ("kitten", "sitting", 1 - 3 / 7),
        ("abcd", "abce", 0.75),
        ("abcd", "ab", 0.0),  # NL is exactly 0.5, which is not below the threshold
        ("abcde", "abcd", 0.8),
        ("ab", "cd", 0.0),
    ],
)
def test_score_single_answer_follows_anls(predicted, target, expected):
    assert _score_single_answer(predicted, target, DEFAULT_ANLS_THRESHOLD) == pytest.approx(expected)


def test_score_single_answer_empty_strings():
    assert _score_single_answer("", "", DEFAULT_ANLS_THRESHOLD) == 1.0
    assert _score_single_answer(None, "", DEFAULT_ANLS_THRESHOLD) == 1.0
    assert _score_single_answer("x", "", DEFAULT_ANLS_THRESHOLD) == 0.0
    assert _score_single_answer("", "x", DEFAULT_ANLS_THRESHOLD) == 0.0


def test_case_and_whitespace_are_normalized():
    assert evaluate("<answer>  Total   REVENUE </answer>", ["total revenue"])["anls"] == 1.0


@pytest.mark.parametrize("golds", [["2000", "1999", "abcd"], ["abcd", "1999", "2000"]])
def test_item_score_is_the_max_over_gold_answers(golds):
    result = evaluate("<answer>1999</answer>", golds)

    assert result["anls"] == 1.0
    assert result["predicted_answer"] == "1999"
    assert result["gold_answers"] == golds


def test_near_miss_gets_partial_credit_but_not_exact():
    # The rollout sets hard = anls >= 0.999 and soft = anls.
    anls = evaluate("<answer>abce</answer>", ["abcd"])["anls"]

    assert anls == pytest.approx(0.75)
    assert anls < 0.999


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (["a", "b"], ["a", "b"]),
        ('["a", "b"]', ["a", "b"]),
        ("['a', 'b']", ["a", "b"]),
        ({"answers": ["x"]}, ["x"]),
        ({"ground_truth": "y"}, ["y"]),
        ([{"text": "t"}, {"answer": "u"}, {"value": "v"}], ["t", "u", "v"]),
        ("plain answer", ["plain answer"]),
        ("[not json", ["[not json"]),
        (5, ["5"]),
        (None, [""]),
        ("", [""]),
        ([], [""]),
    ],
)
def test_extract_answer_strings_accepts_dataset_shapes(raw, expected):
    assert _extract_answer_strings(raw) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("reasoning\n<ANSWER>Foo</ANSWER>", "Foo"),
        ("<answer>x</answer> then <answer>y</answer>", "y"),
        ("<answer>x<answer>y</answer>", "y"),
        ("first line\n\nlast line  ", "last line"),
        ("", ""),
    ],
)
def test_extract_answer(text, expected):
    assert extract_answer(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "<answer>42</answer> (formatted with the <answer> tag as asked)",
        "The total is <answer>42</answer>\nNote: only one <answer> block.",
        "<answer>42</answer>\n<answer>",
    ],
)
def test_stray_answer_tag_after_the_last_complete_answer_is_ignored(text):
    assert extract_answer(text) == "42"
    assert evaluate(text, ["42"])["anls"] == 1.0
