from __future__ import annotations

import json
import os

import pytest

from skillopt.envs.task_output import (
    confined_task_output_dir,
    task_output_segment,
)
from skillopt.optimizer.slow_update import _read_trajectory


@pytest.mark.parametrize(
    "module_name",
    [
        "skillopt.envs.docvqa.rollout",
        "skillopt.envs.searchqa.rollout",
        "skillopt.envs.officeqa.rollout",
    ],
)
def test_rollouts_reject_unconfined_task_ids_before_running(monkeypatch, tmp_path, module_name):
    module = __import__(module_name, fromlist=["process_one"])
    reached = []

    def _bomb(*args, **kwargs):
        reached.append((args, kwargs))
        raise AssertionError("model reached for an unsafe task id")

    for name in (
        "chat_target",
        "chat_target_messages",
        "_run_offline_no_tools_process",
    ):
        if hasattr(module, name):
            monkeypatch.setattr(module, name, _bomb)

    with pytest.raises(ValueError, match="unsafe task id"):
        module.process_one({"id": "../../escape"}, str(tmp_path / "out"), "")

    assert reached == []
    assert not (tmp_path / "escape").exists()
    assert not (tmp_path / "out").exists()


def test_live_math_id_uses_portable_directory_and_slow_update_reads_it(tmp_path):
    task_id = "202602:12"
    segment = task_output_segment(task_id, map_unsafe=True)
    assert segment.startswith("_mapped-")
    assert ":" not in segment

    pred_dir = confined_task_output_dir(
        str(tmp_path),
        task_id,
        map_unsafe=True,
    )
    os.makedirs(pred_dir)
    with open(os.path.join(pred_dir, "conversation.json"), "w", encoding="utf-8") as f:
        json.dump([{"type": "message", "content": "mapped trajectory"}], f)

    assert "mapped trajectory" in _read_trajectory(str(tmp_path), task_id)


def test_live_math_rollout_writes_and_reads_mapped_task_directory(monkeypatch, tmp_path):
    from skillopt.envs.livemathematicianbench import rollout

    monkeypatch.setattr(rollout, "is_target_exec_backend", lambda: False)
    monkeypatch.setattr(
        rollout,
        "chat_target",
        lambda **_kwargs: ("<answer>A</answer>", None),
    )
    task_id = "202602:12"
    item = {
        "id": task_id,
        "question": "Pick A.",
        "choices": [
            {"label": "A", "text": "A"},
            {"label": "B", "text": "B"},
        ],
        "correct_choice": {"label": "A", "text": "A"},
    }

    result = rollout.process_one(item, str(tmp_path), "")

    segment = task_output_segment(task_id, map_unsafe=True)
    assert result["hard"] == 1
    assert (tmp_path / "predictions" / segment / "conversation.json").is_file()
    assert not (tmp_path / "predictions" / task_id).exists()
    assert "<answer>A</answer>" in _read_trajectory(str(tmp_path), task_id)


def test_slow_update_reads_legacy_confined_live_math_directory(tmp_path):
    task_id = "202602:12"
    legacy_dir = tmp_path / "predictions" / task_id
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "conversation.json").write_text(
        json.dumps([{"type": "message", "content": "legacy trajectory"}]),
        encoding="utf-8",
    )

    assert "legacy trajectory" in _read_trajectory(str(tmp_path), task_id)


def test_slow_update_never_reads_escaping_legacy_directory(tmp_path):
    outside = tmp_path.parent / "escape" / "conversation.json"
    outside.parent.mkdir(exist_ok=True)
    outside.write_text(
        json.dumps([{"type": "message", "content": "outside trajectory"}]),
        encoding="utf-8",
    )

    assert _read_trajectory(str(tmp_path), "../../escape") == "(trajectory not available)"


def test_confined_task_output_dir_rejects_symlink_escape(tmp_path):
    predictions = tmp_path / "out" / "predictions"
    predictions.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        os.symlink(outside, predictions / "task-1", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are not available on this host")

    with pytest.raises(ValueError, match="escapes predictions"):
        confined_task_output_dir(str(tmp_path / "out"), "task-1")
