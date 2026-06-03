from __future__ import annotations

import json

from gitmoot_skillopt.contracts import TrainingPackage
from gitmoot_skillopt.optimize import build_trainer_config
from skillopt.engine.trainer import ReflACTTrainer, _best_selection_scores
from skillopt.evaluation.gate import evaluate_gate, find_gate_block
from tests.test_gitmoot_dataloader import write_training_package


def test_gate_blocks_unscored_selection_result_with_trace_paths():
    block = find_gate_block(
        [
            {
                "id": "item-1",
                "hard": None,
                "soft": None,
                "score_status": "unscored",
                "blocker": "evaluator_failed",
                "target_trace_path": "predictions/item-1/conversation.json",
                "evaluator_trace_path": "predictions/item-1/result.json",
                "fail_reason": "judge backend unavailable",
            }
        ]
    )

    assert block is not None
    data = block.to_dict()
    assert data["blocker"] == "evaluator_failed"
    assert data["items"] == [
        {
            "id": "item-1",
            "blocker": "evaluator_failed",
            "score_status": "unscored",
            "target_trace_path": "predictions/item-1/conversation.json",
            "evaluator_trace_path": "predictions/item-1/result.json",
            "fail_reason": "judge backend unavailable",
        }
    ]


def test_gate_accept_reject_behavior_still_uses_scored_results():
    assert find_gate_block([{"id": "item-1", "hard": 1, "soft": 0.75, "score_status": "scored"}]) is None

    accepted = evaluate_gate(
        candidate_skill="new",
        cand_hard=1,
        cand_soft=0.75,
        current_skill="old",
        current_score=0.5,
        best_skill="old",
        best_score=0.5,
        best_step=0,
        global_step=3,
        metric="hard",
    )
    rejected = evaluate_gate(
        candidate_skill="worse",
        cand_hard=0,
        cand_soft=0.2,
        current_skill="old",
        current_score=0.5,
        best_skill="old",
        best_score=0.5,
        best_step=0,
        global_step=4,
        metric="hard",
    )

    assert accepted.action == "accept_new_best"
    assert accepted.best_step == 3
    assert rejected.action == "reject"
    assert rejected.current_skill == "old"


def test_gate_maps_evaluator_not_run_with_nullable_scores():
    block = find_gate_block(
        [
            {
                "id": "item-2",
                "hard": None,
                "soft": None,
                "score_status": "unscored",
                "target_status": "passed",
                "evaluator_status": "not_run",
            }
        ]
    )

    assert block is not None
    assert block.blocker == "evaluator_not_run"
    assert block.items[0]["blocker"] == "evaluator_not_run"


def test_trainer_baseline_gate_returns_blocked_summary(tmp_path):
    class FakeAdapter:
        def setup(self, cfg):
            pass

        def get_dataloader(self):
            return None

        def requires_ray(self):
            return False

        def build_eval_env(self, **kwargs):
            return ["val-1"]

        def rollout(self, env_manager, skill_content, out_dir, **kwargs):
            return [
                {
                    "id": "val-1",
                    "hard": None,
                    "soft": None,
                    "score_status": "unscored",
                    "target_status": "passed",
                    "evaluator_status": "not_run",
                    "target_trace_path": f"{out_dir}/predictions/val-1/conversation.json",
                    "evaluator_trace_path": f"{out_dir}/predictions/val-1/result.json",
                }
            ]

        def get_task_types(self):
            return ["gitmoot-skillopt"]

    package_path, artifact_root = write_training_package(tmp_path)
    package = TrainingPackage.load(package_path)
    out_root = tmp_path / "out"
    initial_skill_path = out_root / "initial_skill.md"
    out_root.mkdir(parents=True)
    initial_skill_path.write_text(package.template.content, encoding="utf-8")
    cfg = build_trainer_config(
        package_path=package_path,
        artifact_root=artifact_root,
        out_root=out_root,
        initial_skill_path=initial_skill_path,
        dry_run=False,
        num_epochs=1,
        batch_size=1,
        seed=1,
        optimizer_model="gpt-test",
        target_model="gpt-test",
        optimizer_backend="openai_chat",
        target_backend="openai_chat",
        evaluator_config={"mode": "fixture"},
        gate_metric="hard",
        reasoning_effort="",
        skill_update_mode="patch",
    )
    cfg["train_size"] = 1

    summary = ReflACTTrainer(cfg, FakeAdapter()).train()

    assert summary["gate_status"] == "blocked"
    assert summary["gate_blocker"] == "evaluator_not_run"
    assert summary["promotable"] is False
    gate_block = json.loads((out_root / "selection_eval_baseline" / "gate_block.json").read_text(encoding="utf-8"))
    assert gate_block["items"][0]["id"] == "val-1"
    assert gate_block["items"][0]["evaluator_trace_path"].endswith("predictions/val-1/result.json")


def test_trainer_scored_soft_gate_reports_best_selection_soft(tmp_path):
    class FakeAdapter:
        def setup(self, cfg):
            pass

        def get_dataloader(self):
            return None

        def requires_ray(self):
            return False

        def build_eval_env(self, **kwargs):
            return ["val-1"]

        def rollout(self, env_manager, skill_content, out_dir, **kwargs):
            return [
                {
                    "id": "val-1",
                    "hard": 0,
                    "soft": 0.83,
                    "score_status": "scored",
                    "target_status": "passed",
                    "evaluator_status": "passed",
                }
            ]

        def get_task_types(self):
            return ["gitmoot-skillopt"]

    package_path, artifact_root = write_training_package(tmp_path)
    package = TrainingPackage.load(package_path)
    out_root = tmp_path / "out"
    initial_skill_path = out_root / "initial_skill.md"
    out_root.mkdir(parents=True)
    initial_skill_path.write_text(package.template.content, encoding="utf-8")
    cfg = build_trainer_config(
        package_path=package_path,
        artifact_root=artifact_root,
        out_root=out_root,
        initial_skill_path=initial_skill_path,
        dry_run=False,
        num_epochs=1,
        batch_size=1,
        seed=1,
        optimizer_model="gpt-test",
        target_model="gpt-test",
        optimizer_backend="openai_chat",
        target_backend="openai_chat",
        evaluator_config={"mode": "fixture"},
        gate_metric="soft",
        reasoning_effort="",
        skill_update_mode="patch",
    )
    cfg["num_epochs"] = 0
    cfg["train_size"] = 1
    cfg["eval_test"] = False

    summary = ReflACTTrainer(cfg, FakeAdapter()).train()

    assert summary["gate_status"] == "passed"
    assert summary["promotable"] is True
    assert summary["baseline_selection_soft"] == 0.83
    assert summary["best_selection_soft"] == 0.83
    assert summary["best_selection_hard"] == 0.0


def test_best_selection_scores_prefers_slow_update_origin():
    hard, soft = _best_selection_scores(
        history=[
            {
                "step": 2,
                "selection_hard": 0.1,
                "selection_soft": 0.2,
            }
        ],
        best_step=2,
        best_origin="slow_update_epoch_01",
        baseline_scores=(0.0, 0.3),
        selection_scores_by_origin={"slow_update_epoch_01": (1.0, 0.9)},
        gate_metric="soft",
        best_score=0.9,
    )

    assert hard == 1.0
    assert soft == 0.9
