"""SkillOpt-Sleep local exact and keyword scores match whole tokens, not substrings."""

from __future__ import annotations

import pytest

from skillopt_sleep.backend import exact_score, keyword_soft_score


@pytest.mark.parametrize(
    ("reference", "response"),
    [
        ("42", "The answer is 420."),
        ("arXiv:1706.0376", "See arXiv:1706.03762 for details."),
        ("Paris", "A comparison of European capitals."),
    ],
)
def test_exact_score_does_not_credit_a_reference_inside_a_longer_token(reference, response):
    assert exact_score(reference, response) == 0.0


@pytest.mark.parametrize(
    ("reference", "response"),
    [
        ("42", "The answer is 42."),
        ("42", "42"),
        ("arXiv:1706.03762", "the id is arXiv:1706.03762 ok"),
        ("New York", "They moved to new  york last year."),
    ],
)
def test_exact_score_credits_the_reference_as_whole_tokens(reference, response):
    assert exact_score(reference, response) == 1.0


def test_keyword_soft_score_does_not_count_a_keyword_inside_a_longer_word():
    assert keyword_soft_score("form", "Please share more information.") == 0.0
    assert keyword_soft_score("add login form", "please add the login form") == 1.0
    assert keyword_soft_score("add login form", "please add the login information") == pytest.approx(2 / 3)
