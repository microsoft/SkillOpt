"""Refinement turns of the chat rollouts resend the task.

``chat_target`` keeps no conversation history, so a later turn that sends only
the previous answer leaves the model without the question, context or choices.
"""

from __future__ import annotations

from skillopt.envs.livemathematicianbench import rollout as livemath_rollout
from skillopt.envs.searchqa import rollout as searchqa_rollout


def _recording_chat_target(responses):
    calls = []

    def fake_chat_target(**kwargs):
        calls.append(kwargs)
        return responses[len(calls) - 1], {}

    return fake_chat_target, calls


SEARCHQA_ITEM = {
    "id": "q1",
    "question": "Which river flows through Vienna?",
    "context": "[DOC] The Danube flows through Vienna and Budapest.",
    "answers": ["Danube"],
}


def test_searchqa_refinement_turn_includes_question_and_context(tmp_path, monkeypatch):
    fake, calls = _recording_chat_target(["<answer>Rhine</answer>", "<answer>Danube</answer>"])
    monkeypatch.setattr(searchqa_rollout, "is_target_exec_backend", lambda: False)
    monkeypatch.setattr(searchqa_rollout, "chat_target", fake)

    row = searchqa_rollout.process_one(
        dict(SEARCHQA_ITEM),
        str(tmp_path),
        "",
        max_turns=2,
        diagnostic_mode=True,
        diagnostic_instruction="READOUT-ONLY-ON-TURN-ONE",
    )

    assert len(calls) == 2
    refinement = calls[1]["user"]
    assert SEARCHQA_ITEM["question"] in refinement
    assert "The Danube flows through Vienna" in refinement
    assert "<answer>Rhine</answer>" in refinement
    # The diagnostic readout belongs to the first turn only, as on the Codex path.
    assert "READOUT-ONLY-ON-TURN-ONE" in calls[0]["user"]
    assert "READOUT-ONLY-ON-TURN-ONE" not in refinement
    assert row["n_turns"] == 2
    assert row["hard"] == 1


def test_searchqa_single_turn_makes_one_call(tmp_path, monkeypatch):
    fake, calls = _recording_chat_target(["<answer>Danube</answer>"])
    monkeypatch.setattr(searchqa_rollout, "is_target_exec_backend", lambda: False)
    monkeypatch.setattr(searchqa_rollout, "chat_target", fake)

    row = searchqa_rollout.process_one(dict(SEARCHQA_ITEM), str(tmp_path), "")

    assert len(calls) == 1
    assert row["hard"] == 1


LIVEMATH_ITEM = {
    "id": "m1",
    "question": "Which prime is the sum of 2 and 3?",
    "choices": [
        {"label": "A", "text": "4"},
        {"label": "B", "text": "5"},
        {"label": "C", "text": "6"},
    ],
    "correct_choice": {"label": "B", "text": "5"},
}


def test_livemath_refinement_turn_includes_question_and_choices(tmp_path, monkeypatch):
    fake, calls = _recording_chat_target(["I think it is five.", "<answer>B</answer>"])
    monkeypatch.setattr(livemath_rollout, "is_target_exec_backend", lambda: False)
    monkeypatch.setattr(livemath_rollout, "chat_target", fake)

    row = livemath_rollout.process_one(dict(LIVEMATH_ITEM), str(tmp_path), "", max_turns=2)

    assert len(calls) == 2
    refinement = calls[1]["user"]
    assert LIVEMATH_ITEM["question"] in refinement
    assert "B. 5" in refinement
    assert "I think it is five." in refinement
    assert row["hard"] == 1
