"""Optimizer rewrites must not turn a non-string ``new_skill`` into the skill text."""

from __future__ import annotations

import json

import pytest

from skillopt.optimizer import rewrite
from skillopt.optimizer.rewrite import usable_new_skill

PATCH = {"revise_suggestions": [{"type": "add", "title": "Check units", "instruction": "Check units first."}]}


def _rewrite(monkeypatch, payload: dict):
    monkeypatch.setattr(rewrite, "chat_optimizer", lambda **kwargs: (json.dumps(payload), {}))
    return rewrite.rewrite_skill_from_suggestions("# Skill\n- Rule one.\n", PATCH, system_prompt="rewrite")


@pytest.mark.parametrize("new_skill", [None, ["# Skill", "- Rule one."], {"text": "# Skill"}, 42, "   "])
def test_unusable_new_skill_gives_no_rewrite(monkeypatch, new_skill):
    assert _rewrite(monkeypatch, {"new_skill": new_skill, "change_summary": ["x"]}) is None


def test_string_new_skill_is_returned(monkeypatch):
    result = _rewrite(
        monkeypatch, {"new_skill": "# Skill\n- Rule one.\n- Check units first.  ", "change_summary": ["x"]}
    )

    assert result["new_skill"] == "# Skill\n- Rule one.\n- Check units first.\n"
    assert result["change_summary"] == ["x"]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("# Skill\n- Rule.  ", "# Skill\n- Rule.\n"),
        (None, ""),
        ("", ""),
        ("  \n ", ""),
        (["# Skill"], ""),
        ({"new_skill": "# Skill"}, ""),
        (0, ""),
    ],
)
def test_usable_new_skill(value, expected):
    # The trainer uses this to pick a full_rewrite_minibatch candidate.
    assert usable_new_skill(value) == expected
