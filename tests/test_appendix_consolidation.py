"""Unit tests for consolidate_appendix_notes, the LLM pass that compacts the
skill-aware Execution Notes Appendix.

Its contract is that a failure can never lose the appendix: any exception,
unparseable reply, empty result or longer result returns the original notes.
The optimizer call is replaced by a fake, so no model is contacted.
"""

from __future__ import annotations

import json

import pytest

from skillopt.optimizer.skill_aware import consolidate_appendix_notes

NOTES = [
    "Check the units column before answering.",
    "Always check units before you answer.",
    "Read the sheet range before writing a formula.",
]


class FakeChat:
    """Record each call and reply with ``reply`` (or raise it when it is an exception)."""

    def __init__(self, reply):
        self.reply = reply
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.reply, BaseException):
            raise self.reply
        return self.reply, {}


def _reply(notes) -> str:
    return json.dumps({"appendix_notes": notes})


@pytest.mark.parametrize("notes", [[], ["Only one note."], ["  ", "Only one note.", ""], None])
def test_fewer_than_two_notes_skip_the_model_call(notes):
    chat = FakeChat(_reply(["unused"]))

    result = consolidate_appendix_notes(notes, chat_fn=chat)

    assert chat.calls == []
    assert result == [n.strip() for n in (notes or []) if n.strip()]


def test_compacted_notes_are_returned():
    compacted = ["Check units before answering.", "Read the sheet range before writing a formula."]
    chat = FakeChat(_reply(compacted))

    assert consolidate_appendix_notes(NOTES, chat_fn=chat) == compacted


def test_reply_in_a_json_fence_is_parsed():
    chat = FakeChat("Here you go:\n```json\n" + _reply(["Check units before answering."]) + "\n```")

    assert consolidate_appendix_notes(NOTES, chat_fn=chat) == ["Check units before answering."]


def test_note_objects_in_the_reply_are_accepted():
    chat = FakeChat(json.dumps({"appendix_notes": [{"note": "Check units."}, {"content": "Read the range."}]}))

    assert consolidate_appendix_notes(NOTES, chat_fn=chat) == ["Check units.", "Read the range."]


def test_one_call_with_the_numbered_notes_and_the_given_budget():
    chat = FakeChat(_reply(["Check units."]))

    consolidate_appendix_notes(NOTES, chat_fn=chat, max_completion_tokens=123)

    (call,) = chat.calls
    assert call["stage"] == "appendix_consolidate"
    assert call["max_completion_tokens"] == 123
    assert call["retries"] == 2
    assert "Current Execution Notes (3 total)" in call["user"]
    for i, note in enumerate(NOTES, 1):
        assert f"{i}. {note}" in call["user"]


def test_notes_are_stripped_and_blanks_dropped_before_the_call():
    chat = FakeChat(_reply(["Check units."]))

    consolidate_appendix_notes(["  Check units.  ", "", "   ", "Read the range."], chat_fn=chat)

    (call,) = chat.calls
    assert "(2 total)" in call["user"]
    assert "1. Check units.\n2. Read the range." in call["user"]


@pytest.mark.parametrize(
    "reply",
    [
        RuntimeError("optimizer unavailable"),
        "I could not compact these notes.",
        "",
        _reply([]),
        _reply(["  ", ""]),
        json.dumps({"notes": ["Check units."]}),
        _reply(NOTES + ["An extra note the model invented."]),
    ],
    ids=["exception", "prose", "empty-reply", "empty-list", "blank-notes", "wrong-key", "longer-than-input"],
)
def test_any_failure_returns_the_original_notes(reply):
    chat = FakeChat(reply)

    assert consolidate_appendix_notes(NOTES, chat_fn=chat) == NOTES


def test_a_reply_as_long_as_the_input_is_accepted():
    rewritten = ["Check units first.", "Verify units again.", "Read the sheet range first."]

    assert consolidate_appendix_notes(NOTES, chat_fn=FakeChat(_reply(rewritten))) == rewritten
