import json
from pathlib import Path

from skillopt.engine.trainer import (
    _atomic_write_json,
    _json_digest,
    _load_committed_step_buffer,
    _recover_committed_prefix,
)
from skillopt.utils import skill_hash


def _write_step(root: Path, step: int, content: str, *, input_hash: str) -> dict:
    step_dir = root / "steps" / f"step_{step:04d}"
    step_dir.mkdir(parents=True)
    skill_dir = root / "skills"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / f"skill_v{step:04d}.md").write_text(content, encoding="utf-8")
    record = {
        "step": step,
        "epoch": 1,
        "input_skill_hash": input_hash,
        "skill_hash": skill_hash(content),
        "commit_id": f"commit-{step}",
        "scheduler_state": {"current_step": step},
    }
    (step_dir / "step_record.json").write_text(
        json.dumps(record), encoding="utf-8",
    )
    return record


def _make_run(root: Path, count: int) -> list[dict]:
    (root / "skills").mkdir(parents=True)
    initial = "initial\n"
    (root / "skills" / "skill_v0000.md").write_text(initial, encoding="utf-8")
    previous_hash = skill_hash(initial)
    history = []
    for step in range(1, count + 1):
        content = f"skill {step}\n"
        record = _write_step(root, step, content, input_hash=previous_hash)
        history.append(dict(record))
        previous_hash = skill_hash(content)
    (root / "history.json").write_text(json.dumps(history), encoding="utf-8")
    return history


def test_recovery_uses_runtime_as_marker_and_removes_ahead_artifacts(tmp_path: Path):
    history = _make_run(tmp_path, 2)
    (tmp_path / "runtime_state.json").write_text(
        json.dumps({
            "last_completed_step": 1,
            "current_skill_hash": history[0]["skill_hash"],
            "phase": "step",
        }),
        encoding="utf-8",
    )
    (tmp_path / "lr_history.jsonl").write_text(
        '{"step": 1, "learning_rate": 4}\n'
        '{"step": 2, "learning_rate": 6}\n'
        '{broken tail',
        encoding="utf-8",
    )

    recovered, runtime, last_step = _recover_committed_prefix(tmp_path.as_posix())

    assert last_step == 1
    assert runtime is not None
    assert [row["step"] for row in recovered] == [1]
    assert not (tmp_path / "steps" / "step_0002").exists()
    assert not (tmp_path / "skills" / "skill_v0002.md").exists()
    assert json.loads((tmp_path / "history.json").read_text()) == history[:1]
    lr_rows = (tmp_path / "lr_history.jsonl").read_text().splitlines()
    assert [json.loads(line)["step"] for line in lr_rows] == [1]


def test_recovery_rolls_back_marker_when_step_record_is_missing(tmp_path: Path):
    history = _make_run(tmp_path, 2)
    (tmp_path / "steps" / "step_0002" / "step_record.json").unlink()
    (tmp_path / "runtime_state.json").write_text(
        json.dumps({
            "last_completed_step": 2,
            "current_skill_hash": history[1]["skill_hash"],
        }),
        encoding="utf-8",
    )

    recovered, runtime, last_step = _recover_committed_prefix(tmp_path.as_posix())

    assert last_step == 1
    assert runtime is None
    assert [row["step"] for row in recovered] == [1]
    assert not (tmp_path / "runtime_state.json").exists()
    assert not (tmp_path / "steps" / "step_0002").exists()


def test_recovery_rejects_skill_hash_mismatch(tmp_path: Path):
    history = _make_run(tmp_path, 2)
    (tmp_path / "skills" / "skill_v0002.md").write_text("tampered\n", encoding="utf-8")
    (tmp_path / "runtime_state.json").write_text(
        json.dumps({
            "last_completed_step": 2,
            "current_skill_hash": history[1]["skill_hash"],
        }),
        encoding="utf-8",
    )

    _, runtime, last_step = _recover_committed_prefix(tmp_path.as_posix())

    assert last_step == 1
    assert runtime is None
    assert not (tmp_path / "skills" / "skill_v0002.md").exists()


def test_recovery_cross_checks_runtime_history_and_step_record_hashes(tmp_path: Path):
    history = _make_run(tmp_path, 2)
    (tmp_path / "runtime_state.json").write_text(
        json.dumps({
            "last_completed_step": 2,
            "current_skill_hash": history[1]["skill_hash"],
            "history_hash": _json_digest(history),
            "step_record_hash": "wrong",
        }),
        encoding="utf-8",
    )

    recovered, runtime, last_step = _recover_committed_prefix(tmp_path.as_posix())

    assert last_step == 1
    assert runtime is None
    assert [row["step"] for row in recovered] == [1]


def test_epoch_artifacts_follow_runtime_phase(tmp_path: Path):
    history = _make_run(tmp_path, 2)
    for root_name in ("slow_update", "meta_skill"):
        for epoch in (1, 2):
            path = tmp_path / root_name / f"epoch_{epoch:02d}"
            path.mkdir(parents=True)
            (path / "partial.json").write_text("{}", encoding="utf-8")
    (tmp_path / "runtime_state.json").write_text(
        json.dumps({
            "last_completed_step": 2,
            "current_skill_hash": history[1]["skill_hash"],
            "phase": "step",
        }),
        encoding="utf-8",
    )

    _recover_committed_prefix(tmp_path.as_posix(), steps_per_epoch=2)

    assert not (tmp_path / "slow_update" / "epoch_01").exists()
    assert not (tmp_path / "meta_skill" / "epoch_01").exists()
    assert not (tmp_path / "slow_update" / "epoch_02").exists()


def test_committed_epoch_phase_keeps_completed_epoch_artifacts(tmp_path: Path):
    history = _make_run(tmp_path, 2)
    for epoch in (1, 2):
        path = tmp_path / "meta_skill" / f"epoch_{epoch:02d}"
        path.mkdir(parents=True)
    (tmp_path / "runtime_state.json").write_text(
        json.dumps({
            "last_completed_step": 2,
            "current_skill_hash": history[1]["skill_hash"],
            "phase": "meta_skill",
        }),
        encoding="utf-8",
    )

    _recover_committed_prefix(tmp_path.as_posix(), steps_per_epoch=2)

    assert (tmp_path / "meta_skill" / "epoch_01").exists()
    assert not (tmp_path / "meta_skill" / "epoch_02").exists()


def test_slow_phase_does_not_commit_meta_skill_artifacts(tmp_path: Path):
    history = _make_run(tmp_path, 2)
    for root_name in ("slow_update", "meta_skill"):
        path = tmp_path / root_name / "epoch_01"
        path.mkdir(parents=True)
    (tmp_path / "runtime_state.json").write_text(
        json.dumps({
            "last_completed_step": 2,
            "current_skill_hash": history[1]["skill_hash"],
            "phase": "slow_update",
        }),
        encoding="utf-8",
    )

    _recover_committed_prefix(tmp_path.as_posix(), steps_per_epoch=2)

    assert (tmp_path / "slow_update" / "epoch_01").exists()
    assert not (tmp_path / "meta_skill" / "epoch_01").exists()


def test_step_buffer_is_rebuilt_only_for_committed_steps_in_epoch(tmp_path: Path):
    _make_run(tmp_path, 4)
    for step in (1, 2, 3, 4):
        path = tmp_path / "steps" / f"step_{step:04d}" / "trajectory_digest.json"
        path.write_text(json.dumps({"step": step, "action": "reject"}), encoding="utf-8")

    buffer = _load_committed_step_buffer(
        tmp_path.as_posix(), epoch=2, steps_per_epoch=3, last_step=4,
    )

    assert [row["step"] for row in buffer] == [4]


def test_atomic_json_replaces_complete_document(tmp_path: Path):
    path = tmp_path / "state.json"
    path.write_text('{"old": true}', encoding="utf-8")

    _atomic_write_json(path.as_posix(), {"new": [1, 2, 3]})

    assert json.loads(path.read_text()) == {"new": [1, 2, 3]}
    assert not list(tmp_path.glob(".state.json.*"))
