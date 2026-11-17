"""Slow-update and meta-skill outputs must be text, never a Python repr of a list."""

from __future__ import annotations

import json

import pytest

from skillopt.optimizer import meta_skill, slow_update


def _slow_update(monkeypatch, payload: dict):
    monkeypatch.setattr(slow_update, "chat_optimizer", lambda **kwargs: (json.dumps(payload), {}))
    return slow_update.run_slow_update("# Skill", [], [], [], comparison_pairs=[], system_prompt="slow")


def _meta_skill(monkeypatch, payload: dict):
    monkeypatch.setattr(meta_skill, "chat_optimizer", lambda **kwargs: (json.dumps(payload), {}))
    return meta_skill.run_meta_skill("# Prev", "# Curr", [], system_prompt="meta")


def test_slow_update_list_content_is_joined_into_lines(monkeypatch):
    result = _slow_update(
        monkeypatch, {"reasoning": "r", "slow_update_content": ["Check units first.", " Prefer exact spans. "]}
    )

    assert result["slow_update_content"] == "Check units first.\nPrefer exact spans."
    skill = slow_update.replace_slow_update_field("# Skill", result["slow_update_content"])
    assert "['" not in skill
    assert slow_update.extract_slow_update_field(skill) == "Check units first.\nPrefer exact spans."


def test_meta_skill_list_content_is_joined_into_lines(monkeypatch):
    result = _meta_skill(
        monkeypatch, {"reasoning": "r", "meta_skill_content": ["Rank by support.", "Merge duplicates."]}
    )

    assert result["meta_skill_content"] == "Rank by support.\nMerge duplicates."


@pytest.mark.parametrize("content", [{"text": "x"}, 42, ["ok", 3], [], ["", "  "], "   "])
def test_unusable_content_gives_no_update(monkeypatch, content):
    assert _slow_update(monkeypatch, {"reasoning": "r", "slow_update_content": content}) is None
    assert _meta_skill(monkeypatch, {"reasoning": "r", "meta_skill_content": content}) is None


def test_string_content_and_null_reasoning(monkeypatch):
    slow = _slow_update(monkeypatch, {"reasoning": None, "slow_update_content": "  Check units first.  "})
    meta = _meta_skill(monkeypatch, {"reasoning": None, "meta_skill_content": "Rank by support."})

    assert slow == {"reasoning": "", "slow_update_content": "Check units first."}
    assert meta == {"reasoning": "", "meta_skill_content": "Rank by support."}
