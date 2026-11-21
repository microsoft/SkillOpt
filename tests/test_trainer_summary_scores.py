"""The run summary's *_selection_hard fields hold hard accuracy, not the gate score."""

from __future__ import annotations

from unittest import mock

import pytest

import skillopt.engine.trainer as trainer_mod
from skillopt.envs.base import EnvAdapter

# Two selection items: hard accuracy 0.5, soft score 0.75.
RESULTS = [
    {"id": "1", "hard": 0, "soft": 0.5, "task_type": "qa"},
    {"id": "2", "hard": 1, "soft": 1.0, "task_type": "qa"},
]


def _summary(tmp_path, gate_metric: str) -> dict:
    skill = tmp_path / "skill.md"
    skill.write_text("# Skill\n", encoding="utf-8")
    cfg = {
        "out_root": str(tmp_path / "out"),
        "backend": "openai_chat",
        "optimizer_backend": "openai_chat",
        "target_backend": "openai_chat",
        "skill_init": str(skill),
        "num_epochs": 0,  # baseline only: the initial skill is both best and final
        "train_size": 1,
        "batch_size": 1,
        "accumulation": 1,
        "merge_batch_size": 2,
        "edit_budget": 2,
        "seed": 42,
        "sel_env_num": 2,
        "test_env_num": 2,
        "eval_test": True,
        "gate_metric": gate_metric,
    }
    adapter = mock.create_autospec(EnvAdapter, instance=True)
    adapter.requires_ray.return_value = False
    adapter.get_dataloader.return_value = None
    adapter.get_task_types.return_value = []
    adapter.build_eval_env.return_value = ["1", "2"]
    adapter.rollout.return_value = [dict(row) for row in RESULTS]
    return trainer_mod.ReflACTTrainer(cfg, adapter).train()


@pytest.mark.parametrize("gate_metric", ["soft", "mixed"])
def test_selection_hard_fields_report_hard_accuracy_for_non_hard_gates(tmp_path, gate_metric):
    summary = _summary(tmp_path, gate_metric)

    assert summary["baseline_selection_hard"] == 0.5
    assert summary["best_selection_hard"] == 0.5
    assert summary["final_selection_hard"] == 0.5
    assert summary["final_selection_soft"] == 0.75


def test_hard_gate_summary_is_unchanged(tmp_path):
    summary = _summary(tmp_path, "hard")

    assert summary["best_selection_hard"] == summary["final_selection_hard"] == 0.5
