import pytest

from skillopt.envs.livemathematicianbench.evaluator import evaluate

CHOICES = [
    {"label": "A", "text": "2"},
    {"label": "B", "text": "3"},
    {"label": "C", "text": "5"},
]


@pytest.mark.parametrize(
    "response",
    [
        "<answer>B</answer>",
        "<answer>B.</answer>",
        "<answer>B)</answer>",
        "<answer>(B)</answer>",
        "<answer>[B]</answer>",
        "<answer>**B**</answer>",
        "<answer>(B) 3</answer>",
    ],
)
def test_wrapped_choice_label_is_scored_correct(response):
    result = evaluate(response, CHOICES[1], CHOICES)

    assert result["predicted_label"] == "B"
    assert result["em"] == 1.0


@pytest.mark.parametrize("response", ["<answer>(A)</answer>", "<answer>Option B</answer>", "<answer></answer>"])
def test_wrong_or_unparseable_label_is_not_scored_correct(response):
    assert evaluate(response, CHOICES[1], CHOICES)["em"] == 0.0


@pytest.mark.parametrize(
    "response",
    ["Use the <answer> tag format.\n<answer>B</answer>", "<answer>A, no wait<answer>B</answer>"],
)
def test_opening_tag_before_the_answer_is_not_part_of_it(response):
    result = evaluate(response, CHOICES[1], CHOICES)

    assert result["predicted_label"] == "B"
    assert result["em"] == 1.0


@pytest.mark.parametrize(
    "response",
    [
        "<answer>\\boxed{B}</answer>",
        "<answer>$\\boxed{B}$</answer>",
        "<answer>\\boxed{(B)}</answer>",
        "<answer>\\boxed{3}</answer>",
        "The sum is 3, so the answer is\n$\\boxed{B}$",
    ],
)
def test_boxed_choice_is_scored_as_the_choice_it_names(response):
    result = evaluate(response, CHOICES[1], CHOICES)

    assert result["predicted_label"] == "B"
    assert result["em"] == 1.0


def test_boxed_wrong_choice_is_not_scored_correct():
    result = evaluate("<answer>\\boxed{A}</answer>", CHOICES[1], CHOICES)

    assert result["predicted_label"] == "A"
    assert result["em"] == 0.0
