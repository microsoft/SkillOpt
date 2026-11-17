"""Unit tests for the optimizer update-mode helpers in ``skillopt.optimizer.update_modes``."""

from __future__ import annotations

import pytest

from skillopt.optimizer.update_modes import (
    FULL_REWRITE_MINIBATCH_MODE,
    PATCH_MODE,
    REWRITE_MODE,
    describe_item,
    get_payload_items,
    is_full_rewrite_minibatch_mode,
    is_rewrite_mode,
    normalize_update_mode,
    payload_key,
    payload_label,
    set_payload_items,
    short_item_summary,
    truncate_payload,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, PATCH_MODE),
        ("", PATCH_MODE),
        ("patch", PATCH_MODE),
        ("EDITS", PATCH_MODE),
        (" Rewrite ", REWRITE_MODE),
        ("suggestions", REWRITE_MODE),
        ("rewrite_suggestions", REWRITE_MODE),
        ("full_rewrite", FULL_REWRITE_MINIBATCH_MODE),
        ("minibatch_full_rewrite", FULL_REWRITE_MINIBATCH_MODE),
        ("skill_rewrite_minibatch", FULL_REWRITE_MINIBATCH_MODE),
        ("unknown_mode", PATCH_MODE),  # unknown names fall back to patch mode
    ],
)
def test_normalize_update_mode(raw, expected):
    assert normalize_update_mode(raw) == expected


@pytest.mark.parametrize(
    ("mode", "key", "plural", "singular_title"),
    [
        ("patch", "edits", "edits", "Edit"),
        ("rewrite", "revise_suggestions", "suggestions", "Suggestion"),
        ("full_rewrite", "skill_candidates", "skill candidates", "Skill Candidate"),
    ],
)
def test_payload_key_and_labels(mode, key, plural, singular_title):
    assert payload_key(mode) == key
    assert payload_label(mode) == plural
    assert payload_label(mode, singular=True, title=True) == singular_title


def test_mode_predicates_are_exclusive():
    assert is_rewrite_mode("suggestions") and not is_full_rewrite_minibatch_mode("suggestions")
    assert is_full_rewrite_minibatch_mode("full_rewrite") and not is_rewrite_mode("full_rewrite")
    assert not is_rewrite_mode("patch") and not is_full_rewrite_minibatch_mode("patch")


@pytest.mark.parametrize("container", [None, "edits", {}, {"edits": "not a list"}, {"edits": None}])
def test_get_payload_items_returns_empty_list_for_missing_or_malformed_payload(container):
    assert get_payload_items(container, "patch") == []


def test_get_and_set_payload_items_use_the_mode_key():
    container = set_payload_items({}, [{"title": "t"}], "full_rewrite")

    assert container == {"skill_candidates": [{"title": "t"}]}
    assert get_payload_items(container, "full_rewrite") == [{"title": "t"}]
    assert get_payload_items(container, "patch") == []


@pytest.mark.parametrize(
    ("items", "max_items", "expected"),
    [
        ([1, 2, 3], 2, [1, 2]),
        ([1, 2], 2, [1, 2]),
        ([1, 2], 5, [1, 2]),
        ([1, 2], 0, []),
        ([1, 2], -1, [1, 2]),  # a negative budget means no limit
    ],
)
def test_truncate_payload(items, max_items, expected):
    container = {"edits": list(items)}

    assert truncate_payload(container, max_items, "patch") is container
    assert container["edits"] == expected


def test_truncate_payload_only_touches_the_mode_key():
    container = {"edits": [1, 2, 3], "revise_suggestions": [1, 2, 3]}

    truncate_payload(container, 1, "rewrite")

    assert container == {"edits": [1, 2, 3], "revise_suggestions": [1]}


def test_describe_patch_item():
    assert describe_item({"op": "replace", "target": "a", "content": "b", "support_count": 2}, "patch") == (
        "op=replace  target='a'  content='b'  support=2"
    )
    assert describe_item({"op": "append", "content": "b"}, "patch") == "op=append  content='b'"


def test_describe_rewrite_item_keeps_zero_support():
    item = {"type": "add", "title": "t", "instruction": "i", "priority_hint": "high", "support_count": 0}

    assert describe_item(item, "rewrite") == "type=add  title='t'  instruction='i'  priority=high  support=0"


def test_describe_full_rewrite_item_shows_the_stripped_skill_preview():
    item = {"title": "t", "change_summary": ["x"], "source_type": "failure", "support_count": 1, "new_skill": " # S "}

    assert describe_item(item, "full_rewrite") == (
        "title='t'  change_summary=['x']  source=failure  support=1  new_skill_preview='# S'"
    )


def test_describe_item_ignores_non_dict():
    assert describe_item("not a dict", "patch") == ""


def test_short_item_summary_per_mode():
    assert short_item_summary({"op": "delete", "target": "x"}, "patch") == {
        "op": "delete",
        "content": "",
        "target": "x",
    }
    assert short_item_summary({"type": "add", "title": "t", "instruction": "i"}, "rewrite") == {
        "type": "add",
        "title": "t",
        "instruction": "i",
    }
    assert short_item_summary({"title": "t", "change_summary": "not a list"}, "full_rewrite") == {
        "title": "t",
        "change_summary": [],
        "source_type": "",
    }
