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
