import pytest

from skillopt.envs.searchqa.evaluator import evaluate


@pytest.mark.parametrize("response", ["<answer></answer>", "<answer>The.</answer>", "<answer> ! </answer>"])
def test_sub_em_does_not_credit_an_empty_prediction(response):
    result = evaluate(response, ["Paris"])

    assert result["sub_em"] == 0.0
    assert result["em"] == 0.0 and result["f1"] == 0.0


def test_sub_em_gold_that_normalizes_to_empty_does_not_match_everything():
    assert evaluate("<answer>London</answer>", ["The"])["sub_em"] == 0.0
    assert evaluate("<answer>London</answer>", ["The", "London"])["sub_em"] == 1.0


def test_sub_em_keeps_substring_matches_in_both_directions():
    assert evaluate("<answer>Paris, France</answer>", ["Paris"])["sub_em"] == 1.0
    assert evaluate("<answer>Paris</answer>", ["Paris, France"])["sub_em"] == 1.0
    assert evaluate("<answer>Lyon</answer>", ["Paris"])["sub_em"] == 0.0


def test_sub_em_empty_prediction_matches_empty_gold():
    result = evaluate("<answer></answer>", [""])

    assert result["sub_em"] == result["em"] == result["f1"] == 1.0


@pytest.mark.parametrize("response", ["<answer>P</answer>", "<answer>s</answer>"])
def test_single_string_gold_is_not_split_into_characters(response):
    result = evaluate(response, "Paris")

    assert result["em"] == result["f1"] == 0.0


def test_single_string_gold_matches_the_full_answer():
    result = evaluate("<answer>Paris</answer>", "Paris")

    assert result["em"] == result["f1"] == result["sub_em"] == 1.0
    assert result["gold_answers"] == ["Paris"]
