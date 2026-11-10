"""OfficeQA final-answer extraction when a response contains more than one <answer> tag."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from skillopt.envs.officeqa import rollout

ITEM = {"id": "treasury", "question": "When was the declaration signed?", "ground_truth": "1776"}


@pytest.fixture(autouse=True)
def offline_backend(monkeypatch):
    for name in ("OFFICEQA_CUSTOM_SEARCH_AUTH", "OFFICEQA_SEARCH_API_URL", "PARALLEL_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(rollout, "get_target_backend", lambda: "openai_chat")


def _run(tmp_path, monkeypatch, response: str) -> dict:
    docs = tmp_path / "docs"
    docs.mkdir()
    monkeypatch.setattr(rollout, "chat_target_messages", lambda **kwargs: (SimpleNamespace(content=response), {}))
    return rollout.process_one(
        ITEM, str(tmp_path / "out"), "", search_mode="offline", use_local_tools=False, data_dirs=[str(docs)]
    )


def test_revised_answer_is_scored_not_the_draft(tmp_path, monkeypatch):
    row = _run(tmp_path, monkeypatch, "<answer>1774</answer>\nChecking the date again.\n<answer>1776</answer>")

    assert row["predicted_answer"] == "1776"
    assert row["hard"] == 1


def test_draft_answer_is_not_credited_when_the_revision_is_wrong(tmp_path, monkeypatch):
    row = _run(tmp_path, monkeypatch, "<answer>1776</answer> On reflection: <answer>1774</answer>")

    assert row["predicted_answer"] == "1774"
    assert row["hard"] == 0


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("<answer>42</answer>", "42"),
        ("<ANSWER> 42 </ANSWER>", "42"),
        ("<answer>\n12.5%\n</answer>", "12.5%"),
        ("draft <answer>1</answer> final <answer>2</answer>", "2"),
        ("no tags\nlast line", "last line"),
    ],
)
def test_extract_answer(text, expected):
    assert rollout._extract_answer(text) == expected
