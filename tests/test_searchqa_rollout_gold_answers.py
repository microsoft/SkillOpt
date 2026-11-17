"""SearchQA rollout rows record a single string gold answer as a one-item list."""

from __future__ import annotations

from skillopt.envs.searchqa import rollout

ITEM = {
    "id": "q1",
    "question": "What is the capital of France?",
    "context": "Paris is the capital.",
    "answers": "Paris",
}


def test_process_one_records_string_gold_answer_as_a_list(tmp_path, monkeypatch):
    monkeypatch.setattr(rollout, "is_target_exec_backend", lambda: False)
    monkeypatch.setattr(rollout, "chat_target", lambda **kwargs: ("<answer>Paris</answer>", {}))

    row = rollout.process_one(dict(ITEM), str(tmp_path), "")

    assert row["gold_answers"] == ["Paris"]
    assert row["hard"] == 1


def test_error_row_records_string_gold_answer_as_a_list(tmp_path, monkeypatch):
    good = {"id": "ok", "agent_ok": True, "hard": 1, "soft": 1.0}

    def fake_process_one(item, *args, **kwargs):
        if item["id"] == "q1":
            raise ValueError("boom")
        return dict(good)

    monkeypatch.setattr(rollout, "process_one", fake_process_one)

    rows = rollout.run_batch(
        [dict(ITEM), {"id": "ok", "question": "q", "answers": ["x"]}], str(tmp_path), "", workers=1
    )

    (error_row,) = [row for row in rows if row["id"] == "q1"]
    assert error_row["phase"] == "error"
    assert error_row["gold_answer"] == ["Paris"]
