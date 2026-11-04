import json
from unittest import mock

import skillopt.engine.trainer as trainer_mod
from skillopt.envs.base import EnvAdapter


def test_resume_preserves_zero_score_without_baseline_rerun(tmp_path):
    out_root = str(tmp_path)
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "skill_v0000.md").write_text("# Initial\n", encoding="utf-8")
    (skills_dir / "skill_v0001.md").write_text("# Step 1\n", encoding="utf-8")
    (tmp_path / "best_skill.md").write_text("# Step 1\n", encoding="utf-8")

    state = {
        "last_completed_step": 1,
        "current_skill_path": str(skills_dir / "skill_v0001.md"),
        "current_score": 0.0,
        "current_origin": "step_0001",
        "best_skill_path": str(tmp_path / "best_skill.md"),
        "best_score": 0.0,
        "best_step": 1,
        "best_origin": "step_0001",
    }
    (tmp_path / "runtime_state.json").write_text(json.dumps(state), encoding="utf-8")

    cfg = {
        "out_root": out_root,
        "backend": "openai_chat",
        "optimizer_backend": "openai_chat",
        "target_backend": "openai_chat",
        "skill_init": str(skills_dir / "skill_v0000.md"),
        "num_epochs": 1,
        "train_size": 1,
        "batch_size": 1,
        "accumulation": 1,
        "merge_batch_size": 2,
        "edit_budget": 2,
        "seed": 42,
        "sel_env_num": 1,
        "eval_test": False,
    }

    mock_adapter = mock.create_autospec(EnvAdapter, instance=True)
    mock_adapter.requires_ray.return_value = False
    mock_adapter.get_dataloader.return_value = None

    trainer = trainer_mod.ReflACTTrainer(cfg, mock_adapter)
    summary = trainer.train()

    # Selection baseline must NOT be re-evaluated when resume has score 0.0
    mock_adapter.build_eval_env.assert_not_called()
    mock_adapter.rollout.assert_not_called()

    assert summary["best_selection_hard"] == 0.0


def test_resume_preserves_positive_scores_without_baseline_rerun(tmp_path):
    out_root = str(tmp_path)
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "skill_v0000.md").write_text("# Initial\n", encoding="utf-8")
    (skills_dir / "skill_v0001.md").write_text("# Step 1\n", encoding="utf-8")
    (tmp_path / "best_skill.md").write_text("# Step 1\n", encoding="utf-8")

    state = {
        "last_completed_step": 1,
        "current_skill_path": str(skills_dir / "skill_v0001.md"),
        "current_score": 0.65,
        "current_origin": "step_0001",
        "best_skill_path": str(tmp_path / "best_skill.md"),
        "best_score": 0.80,
        "best_step": 1,
        "best_origin": "step_0001",
    }
    (tmp_path / "runtime_state.json").write_text(json.dumps(state), encoding="utf-8")

    cfg = {
        "out_root": out_root,
        "backend": "openai_chat",
        "optimizer_backend": "openai_chat",
        "target_backend": "openai_chat",
        "skill_init": str(skills_dir / "skill_v0000.md"),
        "num_epochs": 1,
        "train_size": 1,
        "batch_size": 1,
        "accumulation": 1,
        "merge_batch_size": 2,
        "edit_budget": 2,
        "seed": 42,
        "sel_env_num": 1,
        "eval_test": False,
    }

    mock_adapter = mock.create_autospec(EnvAdapter, instance=True)
    mock_adapter.requires_ray.return_value = False
    mock_adapter.get_dataloader.return_value = None

    trainer = trainer_mod.ReflACTTrainer(cfg, mock_adapter)
    summary = trainer.train()

    mock_adapter.build_eval_env.assert_not_called()
    mock_adapter.rollout.assert_not_called()

    assert summary["best_selection_hard"] == 0.80


def test_resume_missing_score_runs_baseline(tmp_path):
    out_root = str(tmp_path)
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "skill_v0000.md").write_text("# Initial\n", encoding="utf-8")
    (skills_dir / "skill_v0001.md").write_text("# Step 1\n", encoding="utf-8")
    (tmp_path / "best_skill.md").write_text("# Step 1\n", encoding="utf-8")

    state = {
        "last_completed_step": 1,
        "current_skill_path": str(skills_dir / "skill_v0001.md"),
        "best_skill_path": str(tmp_path / "best_skill.md"),
        "best_step": 1,
    }
    (tmp_path / "runtime_state.json").write_text(json.dumps(state), encoding="utf-8")

    cfg = {
        "out_root": out_root,
        "backend": "openai_chat",
        "optimizer_backend": "openai_chat",
        "target_backend": "openai_chat",
        "skill_init": str(skills_dir / "skill_v0000.md"),
        "num_epochs": 1,
        "train_size": 1,
        "batch_size": 1,
        "accumulation": 1,
        "merge_batch_size": 2,
        "edit_budget": 2,
        "seed": 42,
        "sel_env_num": 1,
        "eval_test": False,
    }

    mock_adapter = mock.create_autospec(EnvAdapter, instance=True)
    mock_adapter.requires_ray.return_value = False
    mock_adapter.get_dataloader.return_value = None
    mock_adapter.build_eval_env.return_value = ["mock_task"]
    mock_adapter.rollout.return_value = [{"hard": 1.0, "soft": 1.0}]

    trainer = trainer_mod.ReflACTTrainer(cfg, mock_adapter)
    summary = trainer.train()

    # Selection baseline SHOULD be evaluated when resume score was missing
    mock_adapter.build_eval_env.assert_called_once()
    mock_adapter.rollout.assert_called_once()
    assert summary["best_selection_hard"] == 1.0
