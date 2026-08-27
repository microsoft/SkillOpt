import json
import shutil
from pathlib import Path

import pytest

from skillopt.engine.trainer import (
    _atomic_write_json,
    _json_digest,
    _load_committed_step_buffer,
    _load_resume_skills,
    _recover_committed_prefix,
    _restore_best_skill_from_snapshot,
    _runtime_float,
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


def test_recovery_fail_closes_non_numeric_runtime_marker(tmp_path: Path):
    _make_run(tmp_path, 1)
    runtime_path = tmp_path / "runtime_state.json"
    runtime_path.write_text(
        json.dumps({"last_completed_step": "corrupt"}), encoding="utf-8",
    )

    recovered, runtime, last_step = _recover_committed_prefix(tmp_path.as_posix())

    assert recovered == []
    assert runtime is None
    assert last_step == 0
    assert not runtime_path.exists()
    assert not (tmp_path / "steps" / "step_0001").exists()
    assert not (tmp_path / "skills" / "skill_v0001.md").exists()


def test_history_fallback_rebuilds_best_skill_from_committed_snapshot(tmp_path: Path):
    _make_run(tmp_path, 2)
    best_path = tmp_path / "best_skill.md"
    best_path.write_text("uncommitted best\n", encoding="utf-8")

    best_skill, best_step = _restore_best_skill_from_snapshot(
        tmp_path.as_posix(), 1, last_step=2,
    )

    assert best_step == 1
    assert best_skill == "skill 1\n"
    assert best_path.read_text(encoding="utf-8") == "skill 1\n"


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


def test_runtime_scores_preserve_zero():
    state = {"current_score": 0.0, "best_score": 0}

    assert _runtime_float(state, "current_score", -1.0) == 0.0
    assert _runtime_float(state, "best_score", -1.0) == 0.0


@pytest.mark.parametrize("remove_original", [False, True])
def test_resume_uses_moved_checkpoint_files(tmp_path: Path, remove_original: bool):
    original = tmp_path / "original"
    history = _make_run(original, 1)
    (original / "best_skill.md").write_text("initial\n", encoding="utf-8")
    runtime = {
        "last_completed_step": 1,
        "current_skill_path": str(original / "skills" / "skill_v0001.md"),
        "best_skill_path": str(original / "best_skill.md"),
        "current_skill_hash": history[0]["skill_hash"],
        "best_skill_hash": skill_hash("initial\n"),
        "best_step": 0,
    }
    (original / "runtime_state.json").write_text(
        json.dumps(runtime), encoding="utf-8",
    )
    moved = tmp_path / "moved"
    shutil.copytree(original, moved)

    if remove_original:
        shutil.rmtree(original)
    else:
        (original / "skills" / "skill_v0001.md").write_text(
            "stale current\n", encoding="utf-8",
        )
        (original / "best_skill.md").write_text("stale best\n", encoding="utf-8")

    recovered, moved_runtime, last_step = _recover_committed_prefix(str(moved))
    current_skill, best_skill = _load_resume_skills(
        str(moved), moved_runtime, last_step,
    )

    assert recovered == history
    assert current_skill == "skill 1\n"
    assert best_skill == "initial\n"


class _Interrupted(Exception):
    pass


class _ResumeAdapter:
    def __init__(self, baseline_calls: list[str] | None = None):
        self.baseline_calls = baseline_calls

    def setup(self, cfg):
        pass

    def get_dataloader(self):
        return None

    def requires_ray(self):
        return False

    def build_eval_env(self, **kwargs):
        return [{"id": "item", "instruction": "test"}]

    def build_train_env(self, **kwargs):
        return [{"id": "item", "instruction": "test"}]

    def rollout(self, env, skill, out_dir, **kwargs):
        if self.baseline_calls is not None and str(out_dir).endswith(
            "selection_eval_baseline"
        ):
            self.baseline_calls.append(str(out_dir))
        return [{
            "id": item["id"],
            "hard": 0.0,
            "soft": 0.0,
            "fail_reason": "expected failure",
        } for item in env]

    def reflect(self, *args, **kwargs):
        return []


def _resume_cfg(root: Path, skill_init: Path) -> dict:
    return {
        "out_root": str(root),
        "skill_init": str(skill_init),
        "model_backend": "openai_chat",
        "optimizer_model": "optimizer",
        "target_model": "target",
        "batch_size": 1,
        "num_epochs": 2,
        "accumulation": 1,
        "seed": 7,
        "merge_batch_size": 2,
        "train_size": 1,
        "edit_budget": 1,
        "min_edit_budget": 1,
        "lr_scheduler": "constant",
        "sel_env_num": 1,
        "test_env_num": 1,
        "analyst_workers": 1,
        "eval_test": False,
        "use_slow_update": True,
        "slow_update_samples": 1,
        "slow_update_gate_with_selection": False,
        "use_meta_skill": True,
    }


@pytest.mark.parametrize("phase", ["step", "slow_update", "meta_skill", "complete"])
def test_interrupted_epoch_phases_match_uninterrupted(
    tmp_path: Path, monkeypatch, phase: str,
):
    import skillopt.engine.trainer as trainer_module

    skill_init = tmp_path / "initial.md"
    skill_init.write_text("initial\n", encoding="utf-8")
    meta_inputs: dict[str, list[tuple[str, list[dict]]]] = {
        "full": [], "resumed": [],
    }
    active_run = "full"

    for name in (
        "configure_azure_openai",
        "configure_claude_code_exec",
        "configure_codex_exec_from_config",
        "configure_copilot_chat",
        "configure_copilot_exec",
        "configure_cursor_exec",
        "configure_minimax_chat",
        "configure_qwen_chat",
        "set_optimizer_backend",
        "set_optimizer_deployment",
        "set_reasoning_effort",
        "set_target_backend",
        "set_target_deployment",
    ):
        monkeypatch.setattr(trainer_module, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(trainer_module, "get_token_summary", lambda: {})
    monkeypatch.setattr(
        trainer_module,
        "run_slow_update",
        lambda *args, **kwargs: {
            "reasoning": "fixed",
            "slow_update_content": "fixed guidance",
        },
    )

    def run_meta_skill(*, curr_skill, comparison_pairs, **kwargs):
        meta_inputs[active_run].append((curr_skill, comparison_pairs))
        return {"meta_skill_content": "fixed meta"}

    monkeypatch.setattr(trainer_module, "run_meta_skill", run_meta_skill)
    real_save = trainer_module._save_runtime_state

    full_root = tmp_path / "full"
    full_baselines: list[str] = []
    trainer_module.ReflACTTrainer(
        _resume_cfg(full_root, skill_init), _ResumeAdapter(full_baselines),
    ).train()

    interrupted_root = tmp_path / "interrupted"
    resumed_baselines: list[str] = []
    active_run = "resumed"
    stopped = False

    def interrupt_save(root, state):
        nonlocal stopped
        real_save(root, state)
        if (
            not stopped
            and state.get("last_completed_step") == 2
            and state.get("phase") == phase
        ):
            stopped = True
            raise _Interrupted

    monkeypatch.setattr(trainer_module, "_save_runtime_state", interrupt_save)
    with pytest.raises(_Interrupted):
        trainer_module.ReflACTTrainer(
            _resume_cfg(interrupted_root, skill_init),
            _ResumeAdapter(resumed_baselines),
        ).train()
    monkeypatch.setattr(trainer_module, "_save_runtime_state", real_save)
    trainer_module.ReflACTTrainer(
        _resume_cfg(interrupted_root, skill_init), _ResumeAdapter(resumed_baselines),
    ).train()

    full_state = json.loads((full_root / "runtime_state.json").read_text())
    resumed_state = json.loads((interrupted_root / "runtime_state.json").read_text())
    assert full_state["current_score"] == resumed_state["current_score"] == 0.0
    assert full_state["best_score"] == resumed_state["best_score"] == 0.0
    assert len(full_baselines) == len(resumed_baselines) == 1
    assert (full_root / "skills" / "skill_v0002.md").read_text() == (
        interrupted_root / "skills" / "skill_v0002.md"
    ).read_text()
    assert (full_root / "best_skill.md").read_text() == (
        interrupted_root / "best_skill.md"
    ).read_text()
    assert json.loads(
        (full_root / "meta_skill" / "epoch_02" / "meta_skill_result.json").read_text()
    ) == json.loads(
        (interrupted_root / "meta_skill" / "epoch_02" / "meta_skill_result.json").read_text()
    )
    assert meta_inputs["full"] == meta_inputs["resumed"]
    assert len(meta_inputs["full"]) == 1
    assert "fixed guidance" not in meta_inputs["full"][0][0]
