from __future__ import annotations

import json

from gitmoot_skillopt.contracts import TrainingPackage
from gitmoot_skillopt.optimize import build_trainer_config
from skillopt.datasets.base import BatchSpec
from skillopt.engine.trainer import ReflACTTrainer, _best_selection_scores, _detect_no_meaningful_change
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


class _RetryDataLoader:
    def __init__(self, item_id: str = "train-1", train_size: int = 1) -> None:
        self.item_id = item_id
        self.train_size = train_size
        self.train_items = [
            {
                "id": item_id,
                "ranked_feedback_events": [
                    {
                        "required_improvements": ["better mobile layout"],
                        "useful_traits": {"D": ["clear structure"]},
                        "rejected_traits": {"C": ["overlapping text"]},
                    }
                ],
            }
        ]

    def get_train_size(self):
        return self.train_size

    def make_base_seeds(self, *, steps_per_epoch, accumulation, seed):
        return [seed + 1 for _ in range(steps_per_epoch * accumulation)]

    def plan_train_epoch(self, *, epoch, steps_per_epoch, accumulation, batch_size, seed, **kwargs):
        del epoch, steps_per_epoch, accumulation, batch_size, seed, kwargs
        return [
            BatchSpec(
                phase="train",
                split="train",
                seed=11 + idx,
                batch_size=1,
                payload=[{"id": self.item_id}],
            )
            for idx in range(self.train_size)
        ]

    def build_eval_batch(self, env_num, split, seed, **kwargs):
        del env_num, split, seed, kwargs
        return BatchSpec(
            phase="eval",
            split="val",
            seed=12,
            batch_size=1,
            payload=[{"id": "val-1"}],
        )


class _RetryAdapter:
    def __init__(self, dataloader: _RetryDataLoader) -> None:
        self.dataloader = dataloader

    def setup(self, cfg):
        pass

    def get_dataloader(self):
        return self.dataloader

    def requires_ray(self):
        return False

    def build_env_from_batch(self, batch, **kwargs):
        del kwargs
        return list(batch.payload or [])

    def rollout(self, env_manager, skill_content, out_dir, **kwargs):
        del env_manager, out_dir, kwargs
        changed = "Mobile layout guidance" in skill_content
        return [
            {
                "id": "val-1" if changed else "train-1",
                "hard": 1 if changed else 0,
                "soft": 0.9 if changed else 0.1,
                "score_status": "scored",
                "target_status": "passed",
                "evaluator_status": "passed",
            }
        ]

    def reflect(self, results, skill_content, out_dir, **kwargs):
        del results, skill_content, out_dir, kwargs
        return [
            {
                "source_type": "failure",
                "patch": {
                    "skill_candidates": [
                        {
                            "title": "candidate",
                            "change_summary": ["candidate update"],
                            "new_skill": "",
                        }
                    ]
                },
            }
        ]

    def get_task_types(self):
        return ["gitmoot-skillopt"]


def _retry_trainer_config(tmp_path, *, package_content: str, artifact_root, package_path):
    out_root = tmp_path / "out"
    initial_skill_path = out_root / "initial_skill.md"
    out_root.mkdir(parents=True)
    initial_skill_path.write_text(package_content, encoding="utf-8")
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
        skill_update_mode="full_rewrite_minibatch",
    )
    cfg["eval_test"] = False
    cfg["noop_retry_budget"] = 1
    return cfg


def test_trainer_retries_noop_candidate_with_feedback_hints(tmp_path, monkeypatch):
    package_path, artifact_root = write_training_package(tmp_path)
    package = TrainingPackage.load(package_path)
    merge_contexts: list[str] = []

    def fake_merge(skill_content, failure_patches, success_patches, **kwargs):
        del failure_patches, success_patches
        context = kwargs.get("meta_skill_context", "")
        merge_contexts.append(context)
        new_skill = skill_content
        if "Optimizer No-Op Retry" in context:
            new_skill = skill_content.rstrip() + "\n\nMobile layout guidance: preserve clear structure.\n"
        return {
            "reasoning": "fake merge",
            "skill_candidates": [
                {
                    "title": "retry candidate",
                    "change_summary": ["mobile layout"],
                    "new_skill": new_skill,
                }
            ],
        }

    monkeypatch.setattr("skillopt.engine.trainer.merge_patches", fake_merge)
    cfg = _retry_trainer_config(
        tmp_path,
        package_content=package.template.content,
        artifact_root=artifact_root,
        package_path=package_path,
    )

    summary = ReflACTTrainer(cfg, _RetryAdapter(_RetryDataLoader())).train()

    assert summary["total_accepts"] == 1
    assert summary["best_origin"] == "step_0001"
    assert len(summary["noop_retry_attempts"]) == 1
    assert summary["noop_retry_attempts"][0]["reasons"] == [
        "no_meaningful_skill_change",
        "candidate_content_unchanged",
        "human_feedback_not_incorporated",
    ]
    assert "Improve: better mobile layout" in merge_contexts[1]
    assert "Preserve: D: clear structure" in merge_contexts[1]
    assert "Avoid: C: overlapping text" in merge_contexts[1]
    canonical_candidate = (tmp_path / "out" / "steps" / "step_0001" / "candidate_skill.md").read_text(
        encoding="utf-8"
    )
    assert "Mobile layout guidance" in canonical_candidate


def test_trainer_stops_repeated_noop_candidate_without_fake_candidate(tmp_path, monkeypatch):
    package_path, artifact_root = write_training_package(tmp_path)
    package = TrainingPackage.load(package_path)

    def fake_merge(skill_content, failure_patches, success_patches, **kwargs):
        del failure_patches, success_patches, kwargs
        return {
            "reasoning": "fake merge",
            "skill_candidates": [
                {
                    "title": "unchanged candidate",
                    "change_summary": ["same candidate"],
                    "new_skill": skill_content,
                }
            ],
        }

    monkeypatch.setattr("skillopt.engine.trainer.merge_patches", fake_merge)
    cfg = _retry_trainer_config(
        tmp_path,
        package_content=package.template.content,
        artifact_root=artifact_root,
        package_path=package_path,
    )

    summary = ReflACTTrainer(cfg, _RetryAdapter(_RetryDataLoader())).train()

    assert summary["total_accepts"] == 0
    assert summary["best_origin"] == "initial_skill"
    assert summary["no_candidate_reason"] == "no_meaningful_skill_change"
    assert "candidate_content_unchanged" in summary["no_candidate_triggers"]
    assert "human_feedback_not_incorporated" in summary["no_candidate_triggers"]
    assert summary["total_skips"] == 1
    assert summary["epoch_stats"][0]["skips"] == 1
    assert len(summary["noop_retry_attempts"]) == 2
    assert summary["noop_retry_attempts"][1]["attempt"] == 1


def test_trainer_noop_skip_after_accept_does_not_poison_final_candidate(tmp_path, monkeypatch):
    package_path, artifact_root = write_training_package(tmp_path)
    package = TrainingPackage.load(package_path)
    merge_calls = 0

    def fake_merge(skill_content, failure_patches, success_patches, **kwargs):
        nonlocal merge_calls
        del failure_patches, success_patches, kwargs
        merge_calls += 1
        new_skill = (
            skill_content.rstrip() + "\n\nMobile layout guidance: preserve clear structure.\n"
            if merge_calls == 1
            else skill_content
        )
        return {
            "reasoning": "fake merge",
            "skill_candidates": [
                {
                    "title": "candidate",
                    "change_summary": ["mobile layout"],
                    "new_skill": new_skill,
                }
            ],
        }

    monkeypatch.setattr("skillopt.engine.trainer.merge_patches", fake_merge)
    cfg = _retry_trainer_config(
        tmp_path,
        package_content=package.template.content,
        artifact_root=artifact_root,
        package_path=package_path,
    )
    cfg["train_size"] = 2

    summary = ReflACTTrainer(cfg, _RetryAdapter(_RetryDataLoader(train_size=2))).train()

    assert summary["total_accepts"] == 1
    assert summary["best_origin"] == "step_0001"
    assert summary["no_candidate_triggers"] == []
    assert summary["no_candidate_reason"] == ""
    assert summary["total_skips"] == 1
    assert summary["epoch_stats"][0]["skips"] == 1
    assert len(summary["noop_retry_attempts"]) == 2


def test_noop_detection_allows_applied_delete_patch():
    current_skill = "Keep useful guidance.\nRemove obsolete section.\n"
    candidate_skill = "Keep useful guidance.\n"

    check = _detect_no_meaningful_change(
        current_skill=current_skill,
        candidate_skill=candidate_skill,
        ranked_items=[{"op": "delete", "target": "Remove obsolete section.", "content": ""}],
        apply_report=[{"status": "applied_delete"}],
        update_mode="patch",
        dataloader=None,
        rollout_results=[],
    )

    assert check.reasons == []
