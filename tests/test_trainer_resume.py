"""Resuming a run restores the persisted selection scores exactly."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from unittest import mock

import pytest

from skillopt.engine.trainer import ReflACTTrainer
from skillopt.envs.base import EnvAdapter
from skillopt.model import backend_config


@pytest.fixture(autouse=True)
def isolated_backend_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    saved_environ = os.environ.copy()
    for name in ("OPTIMIZER_BACKEND", "TARGET_BACKEND"):
        monkeypatch.setattr(backend_config, name, getattr(backend_config, name))
    yield
    os.environ.clear()
    os.environ.update(saved_environ)


def _write_completed_run(out_root: Path, *, current_score: float, best_score: float) -> None:
    (out_root / "skills").mkdir(parents=True)
    (out_root / "skills" / "skill_v0001.md").write_text("# Skill v1\n", encoding="utf-8")
    (out_root / "best_skill.md").write_text("# Skill v1\n", encoding="utf-8")
    (out_root / "runtime_state.json").write_text(
        json.dumps({
            "last_completed_step": 1,
            "current_score": current_score,
            "best_score": best_score,
            "best_step": 1,
        }),
        encoding="utf-8",
    )


def test_resume_keeps_a_stored_zero_score(tmp_path: Path) -> None:
    out_root = tmp_path / "run"
    _write_completed_run(out_root, current_score=0.0, best_score=0.0)
    adapter = mock.create_autospec(EnvAdapter, instance=True)
    adapter.requires_ray.return_value = False
    adapter.get_dataloader.return_value = None
    adapter.rollout.return_value = [{"id": "a", "hard": 1, "soft": 1.0}]
    cfg = {
        "skill_init": str(tmp_path / "missing.md"),
        "num_epochs": 1,
        "train_size": 1,
        "batch_size": 1,
        "accumulation": 1,
        "seed": 0,
        "merge_batch_size": 2,
        "edit_budget": 1,
        "sel_env_num": 1,
        "eval_test": False,
        "out_root": str(out_root),
    }

    summary = ReflACTTrainer(cfg, adapter).train()

    adapter.rollout.assert_not_called()
    assert summary["best_selection_hard"] == 0.0
    assert json.loads((out_root / "runtime_state.json").read_text())["current_score"] == 0.0
