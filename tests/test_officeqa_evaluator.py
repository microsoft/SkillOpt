import pytest

from skillopt.envs.officeqa.evaluator import evaluate


@pytest.mark.parametrize(
    ("prediction", "gold"),
    [
        ("42.", "42"),
        ("-3.5.", "-3.5"),
        ("$1,234.5 million.", "1234.5"),
        ("12.5%.", "12.5%"),
        ("U.S.", "US"),
    ],
)
def test_sentence_period_does_not_break_the_match(prediction, gold):
    result = evaluate(prediction, gold)

    assert result["em"] == 1.0
    assert result["f1"] == 1.0


@pytest.mark.parametrize(
    ("prediction", "gold"),
    [("1.5", "15"), ("0.5", "5"), ("-3", "3"), ("42", "43")],
)
def test_decimal_point_and_sign_still_matter(prediction, gold):
    assert evaluate(prediction, gold)["em"] == 0.0


def test_partial_answer_with_trailing_period_gets_token_credit():
    result = evaluate("revenue was 42.", "42")

    assert result["em"] == 0.0
    assert result["f1"] == pytest.approx(0.5)
