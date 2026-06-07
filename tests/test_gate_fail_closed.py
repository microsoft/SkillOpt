from __future__ import annotations

import json

from gitmoot_skillopt.contracts import TrainingPackage
from gitmoot_skillopt.optimize import build_trainer_config
from skillopt.datasets.base import BatchSpec
from skillopt.engine.trainer import (
    ReflACTTrainer,
    _best_selection_scores,
    _detect_no_meaningful_change,
    _format_duplicate_gate_retry_context_from_packet,
    _format_gate_reject_retry_context,
    _gate_reject_exhausted_reason,
    _gate_rejection_retry_decision,
    _non_feedback_direct_items,
    _ranked_feedback_context_packet,
    _ranked_feedback_item_ids,
    _selection_eval_context,
    _selection_reject_gate_rejection,
    _selection_rejection_signal,
    _should_skip_final_test_after_selection_reject,
)
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


class _TwoItemFeedbackDataLoader(_RetryDataLoader):
    def __init__(self) -> None:
        super().__init__(item_id="item-001", train_size=1)
        self.train_items = [
            {
                "id": "item-001",
                "ranked_feedback_events": [
                    {
                        "ranking": ["D > A > B > C"],
                        "quality": "poor",
                        "continue_mode": "refine",
                        "promote": "no",
                        "reasoning": "replies need a sharper real point, not obvious restatement",
                        "required_improvements": ["sharper mechanism question"],
                    }
                ],
            },
            {
                "id": "item-002",
                "ranked_feedback_events": [
                    {
                        "ranking": ["C > B > A > D"],
                        "quality": "acceptable",
                        "continue_mode": "refine",
                        "promote": "no",
                        "reasoning": "remove finally, avoid fake edginess and AI-written phrasing",
                        "required_improvements": ["remove filler words"],
                    }
                ],
            },
        ]


class _SplitFeedbackDataLoader(_RetryDataLoader):
    def __init__(self) -> None:
        super().__init__(item_id="item-002", train_size=1)
        self.train_items = [
            {
                "id": "item-002",
                "ranked_feedback_events": [
                    {
                        "ranking": ["C > B > A > D"],
                        "quality": "acceptable",
                        "continue_mode": "refine",
                        "promote": "no",
                        "reasoning": "remove finally, avoid fake edginess and AI-written phrasing",
                        "required_improvements": ["remove filler words"],
                    }
                ],
            }
        ]
        self.val_items = [
            {
                "id": "item-001",
                "ranked_feedback_events": [
                    {
                        "ranking": ["D > A > B > C"],
                        "quality": "poor",
                        "continue_mode": "refine",
                        "promote": "no",
                        "reasoning": "replies need a sharper real point, not obvious restatement",
                        "required_improvements": ["sharper mechanism question"],
                    }
                ],
            }
        ]
        self.test_items = []


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


def test_ranked_feedback_context_empty_filter_does_not_fallback_when_disabled():
    dataloader = _RetryDataLoader()

    assert _ranked_feedback_context_packet(dataloader, set(), fallback_to_all=False) == {}
    assert _ranked_feedback_context_packet(dataloader, {"missing"}, fallback_to_all=False) == {}
    assert _ranked_feedback_context_packet(dataloader, {"missing"}, fallback_to_all=True)["improve"] == [
        "better mobile layout"
    ]


def test_ranked_feedback_context_preserves_full_text_themes_without_ranking_labels():
    dataloader = _RetryDataLoader()
    dataloader.train_items[0]["ranked_feedback_events"][0].update(
        {
            "feedback_source": "imported_human_review",
            "feedback_target": "baseline_review_outputs",
            "review_issue": "owner/previews#21",
            "review_run_id": "landing-page-preview-trial-005-review-004",
            "reviewed_skill_version": "landing-page-builder@v12",
            "ranking": ["D > B > C > A"],
            "themes": ["premium hero visual system"],
            "reasoning": (
                "The page needs MoonAI-like premium branding, stronger product-relevant graphics, "
                "trust logos, scroll animations, and better mobile responsiveness."
            ),
            "required_improvements": [
                "dark high-contrast visual identity",
                "less generic SaaS layout",
            ],
        }
    )

    packet = _ranked_feedback_context_packet(dataloader)

    assert packet["feedback_target"] == ["baseline_review_outputs"]
    assert packet["review_issue"] == ["owner/previews#21"]
    assert "D > B > C > A" in packet["rankings"]
    assert "premium hero visual system" in packet["themes"]
    assert any("MoonAI-like premium branding" in theme for theme in packet["themes"])
    assert "dark high-contrast visual identity" in packet["themes"]
    assert "less generic SaaS layout" in packet["themes"]
    assert "D > B > C > A" not in packet["themes"]
    assert "D" not in packet["themes"]


class _NoPatchAdapter(_RetryAdapter):
    def reflect(self, results, skill_content, out_dir, **kwargs):
        del results, skill_content, out_dir, kwargs
        return []


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


def _scored_result(*, soft: float) -> dict:
    return {
        "id": "val-1",
        "hard": 1,
        "soft": soft,
        "score_status": "scored",
        "target_status": "passed",
        "evaluator_status": "passed",
    }


def _wrong_artifact_result() -> dict:
    return {
        "id": "val-1",
        "hard": 0,
        "soft": 0.0,
        "score_status": "scored",
        "target_status": "passed",
        "evaluator_status": "passed",
        "fail_reason": "Generated response must be a JSON object containing a Vue/Vite preview bundle.",
        "failure": {
            "primary_reason": "wrong_artifact_type",
            "human_reason": "The candidate returned a skill/template instead of the landing-page bundle.",
            "optimizer_hint": "Return a Vue/Vite preview bundle JSON.",
            "failed_dimensions": ["wrong_artifact_type", "artifact_contract"],
            "failed_checks": [
                {
                    "check": "vue_vite_bundle.required_files",
                    "reason": "The required Vue/Vite files are missing.",
                    "evidence": ["missing src/App.vue"],
                }
            ],
            "evidence": ["response appears to be a skill/template document"],
            "expected_artifact": "vue-vite bundle",
            "actual_artifact": "skill markdown/template",
        },
    }


def _wrong_artifact_dimension_only_result() -> dict:
    result = _wrong_artifact_result()
    result["failure"] = dict(result["failure"])
    result["failure"]["primary_reason"] = ""
    result["failure"]["optimizer_hint"] = ""
    return result


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


def test_feedback_direct_mode_uses_ranked_feedback_before_training_rollout(tmp_path, monkeypatch):
    package_path, artifact_root = write_training_package(tmp_path)
    package = TrainingPackage.load(package_path)
    merge_contexts: list[str] = []
    rollout_dirs: list[str] = []
    reflect_results: list[list[dict]] = []

    def fake_merge(skill_content, failure_patches, success_patches, **kwargs):
        del failure_patches, success_patches
        merge_contexts.append(kwargs.get("meta_skill_context", ""))
        return {
            "reasoning": "fake merge",
            "skill_candidates": [
                {
                    "title": "feedback direct candidate",
                    "change_summary": ["mobile layout"],
                    "new_skill": skill_content.rstrip() + "\n\nMobile layout guidance: preserve clear structure.\n",
                }
            ],
        }

    class FeedbackDirectAdapter(_RetryAdapter):
        def rollout(self, env_manager, skill_content, out_dir, **kwargs):
            del env_manager, kwargs
            rollout_dirs.append(str(out_dir))
            changed = "Mobile layout guidance" in skill_content
            return [
                {
                    "id": "val-1",
                    "hard": 1 if changed else 0,
                    "soft": 0.9 if changed else 0.1,
                    "score_status": "scored",
                    "target_status": "passed",
                    "evaluator_status": "passed",
                }
            ]

        def reflect(self, results, skill_content, out_dir, **kwargs):
            del skill_content, out_dir
            reflect_results.append(results)
            assert "Feedback-Direct Optimization" in kwargs.get("step_buffer_context", "")
            return [
                {
                    "source_type": "failure",
                    "patch": {
                        "skill_candidates": [
                            {
                                "title": "candidate",
                                "change_summary": ["mobile layout"],
                                "new_skill": "",
                            }
                        ]
                    },
                }
            ]

    monkeypatch.setattr("skillopt.engine.trainer.merge_patches", fake_merge)
    cfg = _retry_trainer_config(
        tmp_path,
        package_content=package.template.content,
        artifact_root=artifact_root,
        package_path=package_path,
    )
    cfg["feedback_direct_mode"] = "auto"
    cfg["gate_metric"] = "soft"

    summary = ReflACTTrainer(cfg, FeedbackDirectAdapter(_RetryDataLoader())).train()

    assert summary["total_accepts"] == 1
    assert summary["best_origin"] == "step_0001"
    assert "Feedback-Direct Optimization" in merge_contexts[0]
    assert reflect_results[0][0]["metadata"]["feedback_direct"] is True
    assert reflect_results[0][0]["target_status"] == "not_run"
    assert not any("/steps/step_0001/rollout" in path for path in rollout_dirs)
    assert any("/selection_eval" in path for path in rollout_dirs)
    history = json.loads((tmp_path / "out" / "history.json").read_text(encoding="utf-8"))
    assert history[0]["feedback_direct_mode"] == "auto"
    assert history[0]["feedback_direct_items"] == ["train-1"]


def test_feedback_direct_optimizer_context_includes_all_reviewed_items(tmp_path, monkeypatch):
    package_path, artifact_root = write_training_package(tmp_path)
    package = TrainingPackage.load(package_path)
    merge_contexts: list[str] = []
    reflect_contexts: list[str] = []
    reflect_results: list[list[dict]] = []

    def fake_merge(skill_content, failure_patches, success_patches, **kwargs):
        del failure_patches, success_patches
        merge_contexts.append(kwargs.get("meta_skill_context", ""))
        return {
            "reasoning": "fake merge",
            "skill_candidates": [
                {
                    "title": "feedback direct candidate",
                    "change_summary": ["all review feedback"],
                    "new_skill": skill_content.rstrip() + "\n\nAll reviewed feedback guidance.\n",
                }
            ],
        }

    class AllFeedbackAdapter(_RetryAdapter):
        def reflect(self, results, skill_content, out_dir, **kwargs):
            del skill_content, out_dir
            reflect_results.append(results)
            reflect_contexts.append(kwargs.get("step_buffer_context", ""))
            return [
                {
                    "source_type": "failure",
                    "patch": {
                        "skill_candidates": [
                            {
                                "title": "candidate",
                                "change_summary": ["all review feedback"],
                                "new_skill": "",
                            }
                        ]
                    },
                }
            ]

        def rollout(self, env_manager, skill_content, out_dir, **kwargs):
            del env_manager, out_dir, kwargs
            changed = "All reviewed feedback guidance" in skill_content
            return [_scored_result(soft=0.95 if changed else 0.1)]

    monkeypatch.setattr("skillopt.engine.trainer.merge_patches", fake_merge)
    cfg = _retry_trainer_config(
        tmp_path,
        package_content=package.template.content,
        artifact_root=artifact_root,
        package_path=package_path,
    )
    cfg["feedback_direct_mode"] = "auto"
    cfg["gate_metric"] = "soft"

    summary = ReflACTTrainer(cfg, AllFeedbackAdapter(_TwoItemFeedbackDataLoader())).train()

    assert summary["total_accepts"] == 1
    assert summary["review_feedback_items"] == ["item-001", "item-002"]
    assert summary["optimizer_context_items"] == ["item-001", "item-002"]
    assert [result["id"] for result in reflect_results[0]] == ["item-001", "item-002"]
    assert "Optimizer context items: item-001, item-002" in reflect_contexts[0]
    assert "item-001" in merge_contexts[0]
    assert "item-002" in merge_contexts[0]
    assert "sharper mechanism question" in merge_contexts[0]
    assert "remove filler words" in merge_contexts[0]
    history = json.loads((tmp_path / "out" / "history.json").read_text(encoding="utf-8"))
    assert history[0]["feedback_direct_items"] == ["item-001", "item-002"]
    assert history[0]["optimizer_context_items"] == ["item-001", "item-002"]


def test_optimizer_views_replicate_full_feedback_context(tmp_path, monkeypatch):
    package_path, artifact_root = write_training_package(tmp_path)
    package = TrainingPackage.load(package_path)
    reflect_contexts: list[str] = []
    reflect_results: list[list[dict]] = []
    reflect_minibatch_sizes: list[int | None] = []

    def fake_merge(skill_content, failure_patches, success_patches, **kwargs):
        del failure_patches, success_patches, kwargs
        return {
            "reasoning": "fake merge",
            "skill_candidates": [
                {
                    "title": "feedback direct candidate",
                    "change_summary": ["all review feedback"],
                    "new_skill": skill_content.rstrip() + "\n\nAll reviewed feedback guidance.\n",
                }
            ],
        }

    class ViewAdapter(_RetryAdapter):
        def reflect(self, results, skill_content, out_dir, **kwargs):
            del skill_content, out_dir
            reflect_results.append(results)
            reflect_contexts.append(kwargs.get("step_buffer_context", ""))
            reflect_minibatch_sizes.append(kwargs.get("minibatch_size"))
            return [
                {
                    "source_type": "failure",
                    "patch": {
                        "skill_candidates": [
                            {
                                "title": f"candidate {index}",
                                "change_summary": ["view feedback"],
                                "new_skill": "",
                            }
                        ]
                    },
                }
                for index, _result in enumerate(results, start=1)
            ]

        def rollout(self, env_manager, skill_content, out_dir, **kwargs):
            del env_manager, out_dir, kwargs
            changed = "All reviewed feedback guidance" in skill_content
            return [_scored_result(soft=0.95 if changed else 0.1)]

    monkeypatch.setattr("skillopt.engine.trainer.merge_patches", fake_merge)
    cfg = _retry_trainer_config(
        tmp_path,
        package_content=package.template.content,
        artifact_root=artifact_root,
        package_path=package_path,
    )
    cfg["feedback_direct_mode"] = "auto"
    cfg["gate_metric"] = "soft"
    cfg["optimizer_views"] = 4

    summary = ReflACTTrainer(cfg, ViewAdapter(_TwoItemFeedbackDataLoader())).train()

    assert summary["total_accepts"] == 1
    assert [result["id"] for result in reflect_results[0]] == [
        "optimizer_view_01",
        "optimizer_view_02",
        "optimizer_view_03",
        "optimizer_view_04",
    ]
    for result in reflect_results[0]:
        assert result["metadata"]["source_item_ids"] == ["item-001", "item-002"]
        assert len(result["metadata"]["ranked_feedback_events"]) == 2
    assert "Optimizer context items: item-001, item-002" in reflect_contexts[0]
    assert "View 1/4" in reflect_contexts[0]
    assert "View 4/4" in reflect_contexts[0]
    assert reflect_minibatch_sizes == [1]
    history = json.loads((tmp_path / "out" / "history.json").read_text(encoding="utf-8"))
    assert history[0]["optimizer_views"] == 4
    assert history[0]["optimizer_view_items_per_view"] == [
        ["item-001", "item-002"],
        ["item-001", "item-002"],
        ["item-001", "item-002"],
        ["item-001", "item-002"],
    ]


def test_feedback_direct_optimizer_context_includes_reviewed_items_across_splits(tmp_path, monkeypatch):
    package_path, artifact_root = write_training_package(tmp_path)
    package = TrainingPackage.load(package_path)
    merge_contexts: list[str] = []
    reflect_contexts: list[str] = []
    reflect_results: list[list[dict]] = []

    def fake_merge(skill_content, failure_patches, success_patches, **kwargs):
        del failure_patches, success_patches
        merge_contexts.append(kwargs.get("meta_skill_context", ""))
        return {
            "reasoning": "fake merge",
            "skill_candidates": [
                {
                    "title": "split feedback direct candidate",
                    "change_summary": ["all split review feedback"],
                    "new_skill": skill_content.rstrip() + "\n\nAll reviewed split feedback guidance.\n",
                }
            ],
        }

    class SplitFeedbackAdapter(_RetryAdapter):
        def reflect(self, results, skill_content, out_dir, **kwargs):
            del skill_content, out_dir
            reflect_results.append(results)
            reflect_contexts.append(kwargs.get("step_buffer_context", ""))
            return [
                {
                    "source_type": "failure",
                    "patch": {
                        "skill_candidates": [
                            {
                                "title": "candidate",
                                "change_summary": ["all split review feedback"],
                                "new_skill": "",
                            }
                        ]
                    },
                }
            ]

        def rollout(self, env_manager, skill_content, out_dir, **kwargs):
            del env_manager, out_dir, kwargs
            changed = "All reviewed split feedback guidance" in skill_content
            return [_scored_result(soft=0.95 if changed else 0.1)]

    monkeypatch.setattr("skillopt.engine.trainer.merge_patches", fake_merge)
    cfg = _retry_trainer_config(
        tmp_path,
        package_content=package.template.content,
        artifact_root=artifact_root,
        package_path=package_path,
    )
    cfg["feedback_direct_mode"] = "auto"
    cfg["gate_metric"] = "soft"

    summary = ReflACTTrainer(cfg, SplitFeedbackAdapter(_SplitFeedbackDataLoader())).train()

    assert summary["total_accepts"] == 1
    assert summary["review_feedback_items"] == ["item-002", "item-001"]
    assert summary["optimizer_context_items"] == ["item-002", "item-001"]
    assert [result["id"] for result in reflect_results[0]] == ["item-002", "item-001"]
    assert "Optimizer context items: item-001, item-002" in reflect_contexts[0]
    assert "item-001" in merge_contexts[0]
    assert "item-002" in merge_contexts[0]
    assert "sharper mechanism question" in merge_contexts[0]
    assert "remove filler words" in merge_contexts[0]
    history = json.loads((tmp_path / "out" / "history.json").read_text(encoding="utf-8"))
    assert history[0]["feedback_direct_items"] == ["item-002", "item-001"]
    assert history[0]["optimizer_context_items"] == ["item-002", "item-001"]


def test_review_feedback_item_ids_are_not_prompt_context_limited():
    class ManyItemDataLoader(_RetryDataLoader):
        def __init__(self) -> None:
            super().__init__(item_id="item-001", train_size=1)
            self.train_items = [
                {
                    "id": f"item-{idx:03d}",
                    "ranked_feedback_events": [{"choice": f"feedback {idx}"}],
                }
                for idx in range(1, 18)
            ]

    dataloader = ManyItemDataLoader()

    assert len(_ranked_feedback_context_packet(dataloader)["source_item_ids"]) == 12
    assert _ranked_feedback_item_ids(dataloader) == [f"item-{idx:03d}" for idx in range(1, 18)]


def test_ranked_feedback_context_does_not_leak_test_split_feedback():
    class SplitFeedbackDataLoader(_RetryDataLoader):
        def __init__(self) -> None:
            super().__init__(item_id="train-001", train_size=1)
            self.train_items = [
                {
                    "id": "train-001",
                    "ranked_feedback_events": [{"choice": "train feedback"}],
                }
            ]
            self.val_items = [
                {
                    "id": "val-001",
                    "ranked_feedback_events": [{"choice": "selection feedback"}],
                }
            ]
            self.test_items = [
                {
                    "id": "test-001",
                    "ranked_feedback_events": [{"choice": "held-out test feedback"}],
                }
            ]

    dataloader = SplitFeedbackDataLoader()
    packet = _ranked_feedback_context_packet(dataloader)

    assert packet["source_item_ids"] == ["train-001", "val-001"]
    assert packet["reviewer_reasoning"] == ["train feedback", "selection feedback"]
    assert _ranked_feedback_item_ids(dataloader) == ["train-001", "val-001"]


def test_optimizer_views_no_patch_preserves_source_feedback_hints(tmp_path):
    package_path, artifact_root = write_training_package(tmp_path)
    package = TrainingPackage.load(package_path)
    cfg = _retry_trainer_config(
        tmp_path,
        package_content=package.template.content,
        artifact_root=artifact_root,
        package_path=package_path,
    )
    cfg["feedback_direct_mode"] = "auto"
    cfg["optimizer_views"] = 4
    cfg["hard_failure_retry_budget"] = 0

    summary = ReflACTTrainer(cfg, _NoPatchAdapter(_TwoItemFeedbackDataLoader())).train()

    assert summary["no_candidate_reason"] == "human_feedback_not_distilled"
    assert summary["feedback_retry_hints"]["improve"] == [
        "sharper mechanism question",
        "remove filler words",
    ]
    assert summary["optimizer_context_items"] == ["item-001", "item-002"]


def test_trainer_classifies_no_patches_with_ranked_feedback_as_not_distilled(tmp_path):
    package_path, artifact_root = write_training_package(tmp_path)
    package = TrainingPackage.load(package_path)
    cfg = _retry_trainer_config(
        tmp_path,
        package_content=package.template.content,
        artifact_root=artifact_root,
        package_path=package_path,
    )

    summary = ReflACTTrainer(cfg, _NoPatchAdapter(_RetryDataLoader())).train()

    assert summary["total_accepts"] == 0
    assert summary["best_origin"] == "initial_skill"
    assert summary["no_candidate_reason"] == "human_feedback_not_distilled"
    assert "human_feedback_not_distilled" in summary["no_candidate_triggers"]
    assert summary["failure"]["primary_reason"] == "human_feedback_not_distilled"
    assert "ranked human feedback" in summary["optimizer_hint"].lower()
    assert summary["feedback_retry_hints"]["improve"] == ["better mobile layout"]


def test_feedback_direct_preserves_original_item_id_for_feedback_hints(tmp_path):
    package_path, artifact_root = write_training_package(tmp_path)
    package = TrainingPackage.load(package_path)
    item_id = "train item #1"
    cfg = _retry_trainer_config(
        tmp_path,
        package_content=package.template.content,
        artifact_root=artifact_root,
        package_path=package_path,
    )

    summary = ReflACTTrainer(cfg, _NoPatchAdapter(_RetryDataLoader(item_id=item_id))).train()

    assert summary["no_candidate_reason"] == "human_feedback_not_distilled"
    assert summary["feedback_retry_hints"]["improve"] == ["better mobile layout"]
    history = json.loads((tmp_path / "out" / "history.json").read_text(encoding="utf-8"))
    assert history[0]["feedback_direct_items"] == [item_id]


def test_feedback_direct_records_accumulated_item_ids(tmp_path, monkeypatch):
    package_path, artifact_root = write_training_package(tmp_path)
    package = TrainingPackage.load(package_path)

    class TwoItemDataLoader(_RetryDataLoader):
        def __init__(self) -> None:
            super().__init__(item_id="item-a", train_size=2)
            self.train_items = [
                {
                    "id": "item-a",
                    "ranked_feedback_events": [
                        {
                            "required_improvements": ["better mobile layout"],
                            "continue_mode": "refine",
                            "promote": "no",
                        }
                    ],
                },
                {
                    "id": "item-b",
                    "ranked_feedback_events": [
                        {
                            "required_improvements": ["stronger product visuals"],
                            "continue_mode": "refine",
                            "promote": "no",
                        }
                    ],
                },
            ]

        def plan_train_epoch(self, *, epoch, steps_per_epoch, accumulation, batch_size, seed, **kwargs):
            del epoch, steps_per_epoch, accumulation, batch_size, seed, kwargs
            return [
                BatchSpec(phase="train", split="train", seed=11, batch_size=1, payload=[{"id": "item-a"}]),
                BatchSpec(phase="train", split="train", seed=12, batch_size=1, payload=[{"id": "item-b"}]),
            ]

    def fake_merge(skill_content, failure_patches, success_patches, **kwargs):
        del failure_patches, success_patches, kwargs
        return {
            "reasoning": "fake merge",
            "skill_candidates": [
                {
                    "title": "candidate",
                    "change_summary": ["mobile visual guidance"],
                    "new_skill": skill_content.rstrip() + "\n\nMobile layout guidance: preserve clear structure.\n",
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
    cfg["accumulation"] = 2
    cfg["train_size"] = 2

    summary = ReflACTTrainer(cfg, _RetryAdapter(TwoItemDataLoader())).train()

    assert summary["total_accepts"] == 1
    history = json.loads((tmp_path / "out" / "history.json").read_text(encoding="utf-8"))
    assert history[0]["feedback_direct_items"] == ["item-a", "item-b"]


def test_feedback_direct_preserves_normal_items_in_mixed_batch():
    normal_items = _non_feedback_direct_items(
        batch_items=[
            {"id": "feedback-item", "prompt": "ranked feedback"},
            {"id": "normal-item", "prompt": "normal rollout"},
        ],
        feedback_direct_items=[{"id": "feedback-item"}],
    )

    assert normal_items == [{"id": "normal-item", "prompt": "normal rollout"}]


def test_feedback_direct_off_allows_opaque_batch_payload(tmp_path, monkeypatch):
    package_path, artifact_root = write_training_package(tmp_path)
    package = TrainingPackage.load(package_path)

    class OpaquePayloadDataLoader(_RetryDataLoader):
        def plan_train_epoch(self, *, epoch, steps_per_epoch, accumulation, batch_size, seed, **kwargs):
            del epoch, steps_per_epoch, accumulation, batch_size, seed, kwargs
            return [
                BatchSpec(
                    phase="train",
                    split="train",
                    seed=11,
                    batch_size=1,
                    payload=object(),
                )
            ]

    class OpaquePayloadAdapter(_RetryAdapter):
        def build_env_from_batch(self, batch, **kwargs):
            del batch, kwargs
            return {"opaque_env": True}

        def rollout(self, env_manager, skill_content, out_dir, **kwargs):
            assert env_manager == {"opaque_env": True}
            return super().rollout(env_manager, skill_content, out_dir, **kwargs)

    def fake_merge(skill_content, failure_patches, success_patches, **kwargs):
        del failure_patches, success_patches, kwargs
        return {
            "reasoning": "fake merge",
            "skill_candidates": [
                {
                    "title": "candidate",
                    "change_summary": ["mobile layout"],
                    "new_skill": skill_content.rstrip() + "\n\nMobile layout guidance: preserve clear structure.\n",
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
    cfg["feedback_direct_mode"] = "off"

    summary = ReflACTTrainer(cfg, OpaquePayloadAdapter(OpaquePayloadDataLoader())).train()

    assert summary["total_accepts"] == 1


def test_feedback_direct_mixed_batch_preserves_adapter_built_remainder_env(tmp_path, monkeypatch):
    package_path, artifact_root = write_training_package(tmp_path)
    package = TrainingPackage.load(package_path)
    rollout_envs: list[object] = []

    class MixedDataLoader(_RetryDataLoader):
        def __init__(self) -> None:
            super().__init__(item_id="feedback-item", train_size=2)
            self.train_items = [
                {
                    "id": "feedback-item",
                    "ranked_feedback_events": [
                        {
                            "required_improvements": ["better mobile layout"],
                            "continue_mode": "refine",
                            "promote": "no",
                        }
                    ],
                },
                {"id": "normal-item", "ranked_feedback_events": []},
            ]

        def plan_train_epoch(self, *, epoch, steps_per_epoch, accumulation, batch_size, seed, **kwargs):
            del epoch, steps_per_epoch, accumulation, batch_size, seed, kwargs
            return [
                BatchSpec(
                    phase="train",
                    split="train",
                    seed=11,
                    batch_size=2,
                    payload=[{"id": "feedback-item"}, {"id": "normal-item"}],
                )
            ]

    class WrappedEnvAdapter(_RetryAdapter):
        def build_env_from_batch(self, batch, **kwargs):
            del kwargs
            return {"ids": [item["id"] for item in list(batch.payload or [])]}

        def rollout(self, env_manager, skill_content, out_dir, **kwargs):
            rollout_envs.append(env_manager)
            if env_manager == {"ids": ["normal-item"]}:
                return [
                    {
                        "id": "normal-item",
                        "hard": 0,
                        "soft": 0.2,
                        "score_status": "scored",
                        "target_status": "passed",
                        "evaluator_status": "passed",
                    }
                ]
            return super().rollout(env_manager, skill_content, out_dir, **kwargs)

    def fake_merge(skill_content, failure_patches, success_patches, **kwargs):
        del failure_patches, success_patches, kwargs
        return {
            "reasoning": "fake merge",
            "skill_candidates": [
                {
                    "title": "candidate",
                    "change_summary": ["mobile layout"],
                    "new_skill": skill_content.rstrip() + "\n\nMobile layout guidance: preserve clear structure.\n",
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
    cfg["batch_size"] = 2
    cfg["train_size"] = 2

    summary = ReflACTTrainer(cfg, WrappedEnvAdapter(MixedDataLoader())).train()

    assert summary["total_accepts"] == 1
    assert {"ids": ["normal-item"]} in rollout_envs


def test_feedback_direct_off_does_not_materialize_train_env(tmp_path, monkeypatch):
    package_path, artifact_root = write_training_package(tmp_path)
    package = TrainingPackage.load(package_path)

    class OpaqueEnv:
        def __iter__(self):
            raise AssertionError("trainer must not iterate opaque env before rollout")

    class OpaqueEnvAdapter(_RetryAdapter):
        def build_env_from_batch(self, batch, **kwargs):
            del batch, kwargs
            return OpaqueEnv()

        def rollout(self, env_manager, skill_content, out_dir, **kwargs):
            assert isinstance(env_manager, OpaqueEnv)
            return super().rollout(env_manager, skill_content, out_dir, **kwargs)

    def fake_merge(skill_content, failure_patches, success_patches, **kwargs):
        del failure_patches, success_patches, kwargs
        return {
            "reasoning": "fake merge",
            "skill_candidates": [
                {
                    "title": "candidate",
                    "change_summary": ["mobile layout"],
                    "new_skill": skill_content.rstrip() + "\n\nMobile layout guidance: preserve clear structure.\n",
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
    cfg["feedback_direct_mode"] = "off"

    summary = ReflACTTrainer(cfg, OpaqueEnvAdapter(_RetryDataLoader())).train()

    assert summary["total_accepts"] == 1
    assert "feedback_direct_items" not in json.loads(
        (tmp_path / "out" / "history.json").read_text(encoding="utf-8")
    )[0]


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
    assert "retry_budget_exhausted" in summary["no_candidate_diagnostics"]["categories"]
    assert summary["no_candidate_diagnostics"]["retry_budget_exhausted"] is True
    assert summary["no_candidate_diagnostics"]["retry_stop_reasons"] == ["noop_retry_budget_exhausted"]


def test_trainer_skips_final_test_eval_after_selection_reject(tmp_path, monkeypatch):
    package_path, artifact_root = write_training_package(tmp_path)
    package = TrainingPackage.load(package_path)
    final_test_rollouts: list[str] = []

    def fake_merge(skill_content, failure_patches, success_patches, **kwargs):
        del failure_patches, success_patches, kwargs
        return {
            "reasoning": "fake merge",
            "skill_candidates": [
                {
                    "title": "weak candidate",
                    "change_summary": ["artifact delivery only"],
                    "new_skill": skill_content.rstrip() + "\n\nArtifact delivery only.\n",
                }
            ],
        }

    class RejectingAdapter(_RetryAdapter):
        def rollout(self, env_manager, skill_content, out_dir, **kwargs):
            del env_manager, kwargs
            out_dir_text = str(out_dir)
            if "test_eval" in out_dir_text:
                final_test_rollouts.append(out_dir_text)
            changed = "Artifact delivery only" in skill_content
            return [
                {
                    "id": "val-1",
                    "hard": 1,
                    "soft": 0.84 if changed else 0.89,
                    "score_status": "scored",
                    "target_status": "passed",
                    "evaluator_status": "passed",
                }
            ]

    monkeypatch.setattr("skillopt.engine.trainer.merge_patches", fake_merge)
    cfg = _retry_trainer_config(
        tmp_path,
        package_content=package.template.content,
        artifact_root=artifact_root,
        package_path=package_path,
    )
    cfg["eval_test"] = True
    cfg["gate_metric"] = "soft"
    cfg["gate_reject_retry_budget"] = 1
    cfg["gate_reject_retry_close_gap"] = 0.1

    summary = ReflACTTrainer(cfg, RejectingAdapter(_RetryDataLoader())).train()

    assert summary["total_accepts"] == 0
    assert summary["total_rejects"] == 1
    assert summary["best_origin"] == "initial_skill"
    assert summary["final_test_skipped_reason"] == "selection_gate_rejected_candidate"
    assert len(summary["gate_reject_retry_attempts"]) == 1
    assert summary["gate_reject_retry_attempts"][0]["action"] == "stop"
    assert summary["gate_reject_retry_attempts"][0]["result"] == "duplicate_candidate"
    assert summary["gate_reject_retry_attempts"][0]["fresh_reflect_retry"] is True
    assert summary["baseline_test_hard"] is None
    assert summary["test_hard"] is None
    assert final_test_rollouts == []
    assert not (tmp_path / "out" / "test_eval").exists()
    rejection = summary["gate_rejection"]
    assert rejection["rejection_type"] == "candidate_score_regression"
    assert rejection["baseline"]["gate_score"] == 0.89
    assert rejection["candidate"]["gate_score"] == 0.84
    assert rejection["attempted_patch"] == "artifact delivery only"
    assert rejection["retry_attempts"] == "1/1"


def test_final_selection_reject_packet_uses_configured_gate_retry_budget():
    history = [
        {
            "action": "reject",
            "selection_hard": 1.0,
            "selection_soft": 0.68,
            "candidate_gate_score": 0.84,
            "rewrite_change_summary": ["artifact delivery only"],
        }
    ]

    assert _should_skip_final_test_after_selection_reject(
        history=history,
        best_origin="initial_skill",
        best_skill="base skill",
        skill_init="base skill",
    )
    rejection = _selection_reject_gate_rejection(
        history=history,
        baseline_scores=(1.0, 0.89),
        gate_metric="mixed",
        gate_mixed_weight=0.5,
        retry_used=0,
        retry_budget=1,
    )

    assert rejection is not None
    assert rejection["retryable"] is True
    assert rejection["candidate"]["gate_score"] == 0.84
    assert round(rejection["baseline"]["gate_score"], 3) == 0.945
    assert rejection["retry_attempts"] == "0/1"
    assert "retry" in rejection["next_action"].lower()


def test_selection_reject_packet_marks_evaluator_contract_failure_non_retryable():
    history = [
        {
            "action": "reject",
            "selection_hard": 0.0,
            "selection_soft": 0.0,
            "candidate_gate_score": 0.0,
            "rewrite_change_summary": ["human feedback update"],
        }
    ]
    rejection_signal = {
        "primary_reason": "evaluator_missing_human_feedback_dimensions",
        "source_item_id": "item-001",
        "human_reason": "Human feedback exists, but the judge did not return structured dimensions.",
        "optimizer_hint": "Retry or fix the evaluator schema.",
        "failed_dimensions": ["human_feedback_alignment"],
        "failed_checks": [
            {
                "check": "llm_judge.human_feedback_dimensions",
                "severity": "evaluator_contract_failure",
                "reason": "judge output omitted required human-feedback readiness fields",
            }
        ],
    }

    rejection = _selection_reject_gate_rejection(
        history=history,
        baseline_scores=(0.0, 0.0),
        gate_metric="soft",
        gate_mixed_weight=0.5,
        retry_used=0,
        retry_budget=3,
        rejection_signal=rejection_signal,
    )

    assert rejection is not None
    assert rejection["rejection_type"] == "evaluator_contract_failure"
    assert rejection["retryable"] is False
    assert rejection["primary_reason"] == "evaluator_missing_human_feedback_dimensions"
    assert rejection["selection_failed_item"] == "item-001"
    assert "do not retry the optimizer" in rejection["next_action"].lower()
    can_retry, reason = _gate_rejection_retry_decision(
        rejection,
        attempt=0,
        budget=3,
        seen_reasons=set(),
    )
    assert can_retry is False
    assert reason == "non_retryable_gate_rejection"
    assert (
        _gate_reject_exhausted_reason(
            rejection,
            gate_reject_stop_reason="budget_exhausted",
            wrong_artifact_retry=False,
        )
        == "evaluator_contract_failure"
    )


def test_selection_reject_packet_includes_evaluator_reasoning_and_delta_summary():
    history = [
        {
            "action": "reject",
            "selection_hard": 1.0,
            "selection_soft": 0.88,
            "candidate_gate_score": 0.94,
            "rewrite_change_summary": ["artifact-first rewrite"],
        }
    ]

    rejection = _selection_reject_gate_rejection(
        history=history,
        baseline_scores=(1.0, 0.9),
        gate_metric="mixed",
        gate_mixed_weight=0.5,
        retry_used=0,
        retry_budget=3,
        baseline_context={
            "evaluator_reasoning": "Baseline had a strong full-screen hero and complete footer."
        },
        candidate_context={
            "evaluator_reasoning": "Candidate was valid but still CSS-dashboard-only and structurally close."
        },
    )

    assert rejection is not None
    assert rejection["baseline"]["evaluator_reasoning"] == (
        "Baseline had a strong full-screen hero and complete footer."
    )
    assert rejection["candidate"]["evaluator_reasoning"] == (
        "Candidate was valid but still CSS-dashboard-only and structurally close."
    )
    assert rejection["delta_summary"]["strengths"] == [
        "Candidate evaluator rationale: Candidate was valid but still CSS-dashboard-only and structurally close."
    ]
    assert rejection["delta_summary"]["weaknesses"] == [
        "Baseline evaluator rationale to beat: Baseline had a strong full-screen hero and complete footer."
    ]


def test_selection_reject_packet_includes_dimension_deltas():
    baseline_context = _selection_eval_context(
        [
            {
                "id": "baseline",
                "reasoning": "baseline rationale",
                "dimension_scores": {
                    "hero_quality": 0.68,
                    "visual_images_relevance": 0.58,
                },
            }
        ]
    )
    candidate_context = _selection_eval_context(
        [
            {
                "id": "candidate",
                "reasoning": "candidate rationale",
                "metadata": {
                    "dimension_scores": {
                        "hero_quality": 0.72,
                        "visual_images_relevance": 0.58,
                    }
                },
            }
        ]
    )
    rejection = _selection_reject_gate_rejection(
        history=[
            {
                "action": "reject",
                "selection_hard": 1.0,
                "selection_soft": 0.62,
                "candidate_gate_score": 0.81,
                "rewrite_change_summary": ["brand and imagery guidance"],
            }
        ],
        baseline_scores=(1.0, 0.62),
        gate_metric="mixed",
        gate_mixed_weight=0.5,
        baseline_context=baseline_context,
        candidate_context=candidate_context,
    )

    assert rejection is not None
    assert rejection["baseline"]["dimension_scores"]["hero_quality"] == 0.68
    assert rejection["candidate"]["dimension_scores"]["hero_quality"] == 0.72
    assert round(rejection["dimension_scores"]["delta"]["hero_quality"], 2) == 0.04
    assert rejection["dimension_scores"]["delta"]["visual_images_relevance"] == 0.0

    retry_context = _format_gate_reject_retry_context(rejection, attempt=1, budget=3)

    assert "Dimension deltas candidate-baseline:" in retry_context
    assert "hero_quality=+0.0400" in retry_context
    assert "visual_images_relevance=+0.0000" in retry_context


def test_trainer_retries_actionable_gate_rejection_with_optimizer_hint(tmp_path, monkeypatch):
    package_path, artifact_root = write_training_package(tmp_path)
    package = TrainingPackage.load(package_path)
    merge_contexts: list[str] = []

    def fake_merge(skill_content, failure_patches, success_patches, **kwargs):
        del failure_patches, success_patches
        context = kwargs.get("meta_skill_context", "")
        merge_contexts.append(context)
        if "Gate Rejection Retry" in context:
            new_skill = skill_content.rstrip() + "\n\nMobile layout guidance: preserve clear structure.\n"
            change_summary = ["mobile layout"]
        else:
            new_skill = skill_content.rstrip() + "\n\nArtifact delivery only.\n"
            change_summary = ["artifact delivery only"]
        return {
            "reasoning": "fake merge",
            "skill_candidates": [
                {
                    "title": "candidate",
                    "change_summary": change_summary,
                    "new_skill": new_skill,
                }
            ],
        }

    class GateRetryAdapter(_RetryAdapter):
        def rollout(self, env_manager, skill_content, out_dir, **kwargs):
            del env_manager, out_dir, kwargs
            if "Mobile layout guidance" in skill_content:
                soft = 0.95
            elif "Artifact delivery only" in skill_content:
                soft = 0.84
            else:
                soft = 0.89
            return [
                {
                    "id": "val-1",
                    "hard": 1,
                    "soft": soft,
                    "score_status": "scored",
                    "target_status": "passed",
                    "evaluator_status": "passed",
                }
            ]

    monkeypatch.setattr("skillopt.engine.trainer.merge_patches", fake_merge)
    cfg = _retry_trainer_config(
        tmp_path,
        package_content=package.template.content,
        artifact_root=artifact_root,
        package_path=package_path,
    )
    cfg["gate_metric"] = "soft"
    cfg["gate_reject_retry_budget"] = 1
    cfg["gate_reject_retry_close_gap"] = 0.1

    summary = ReflACTTrainer(cfg, GateRetryAdapter(_RetryDataLoader())).train()

    assert summary["total_accepts"] == 1
    assert summary["total_rejects"] == 0
    assert summary["best_origin"] == "step_0001"
    assert len(summary["gate_reject_retry_attempts"]) == 1
    retry_attempt = summary["gate_reject_retry_attempts"][0]
    assert retry_attempt["action"] == "retry"
    assert retry_attempt["gate_rejection"]["candidate"]["gate_score"] == 0.84
    assert retry_attempt["gate_rejection"]["baseline"]["gate_score"] == 0.89
    assert "Gate Rejection Retry" in merge_contexts[1]
    assert "Previous patch summary: artifact delivery only" in merge_contexts[1]
    assert "Do not repeat this failed patch direction" in merge_contexts[1]


def test_gate_rejection_retry_preserves_noop_retry_context(tmp_path, monkeypatch):
    package_path, artifact_root = write_training_package(tmp_path)
    package = TrainingPackage.load(package_path)
    merge_contexts: list[str] = []

    def fake_merge(skill_content, failure_patches, success_patches, **kwargs):
        del failure_patches, success_patches
        context = kwargs.get("meta_skill_context", "")
        merge_contexts.append(context)
        if "Gate Rejection Retry" in context:
            new_skill = skill_content.rstrip() + "\n\nMobile layout guidance: preserve clear structure.\n"
            change_summary = ["mobile layout"]
        elif "Optimizer No-Op Retry" in context:
            new_skill = skill_content.rstrip() + "\n\nArtifact delivery only.\n"
            change_summary = ["artifact delivery only"]
        else:
            new_skill = skill_content
            change_summary = ["same candidate"]
        return {
            "reasoning": "fake merge",
            "skill_candidates": [
                {
                    "title": "candidate",
                    "change_summary": change_summary,
                    "new_skill": new_skill,
                }
            ],
        }

    class CombinedRetryAdapter(_RetryAdapter):
        def rollout(self, env_manager, skill_content, out_dir, **kwargs):
            del env_manager, out_dir, kwargs
            if "Mobile layout guidance" in skill_content:
                soft = 0.95
            elif "Artifact delivery only" in skill_content:
                soft = 0.84
            else:
                soft = 0.89
            return [
                {
                    "id": "val-1",
                    "hard": 1,
                    "soft": soft,
                    "score_status": "scored",
                    "target_status": "passed",
                    "evaluator_status": "passed",
                }
            ]

    monkeypatch.setattr("skillopt.engine.trainer.merge_patches", fake_merge)
    cfg = _retry_trainer_config(
        tmp_path,
        package_content=package.template.content,
        artifact_root=artifact_root,
        package_path=package_path,
    )
    cfg["gate_metric"] = "soft"
    cfg["gate_reject_retry_budget"] = 1
    cfg["gate_reject_retry_close_gap"] = 0.1

    summary = ReflACTTrainer(cfg, CombinedRetryAdapter(_RetryDataLoader())).train()

    assert summary["total_accepts"] == 1
    assert len(summary["noop_retry_attempts"]) == 1
    assert len(summary["gate_reject_retry_attempts"]) == 1
    gate_retry_context = merge_contexts[2]
    assert "Optimizer No-Op Retry" in gate_retry_context
    assert "Gate Rejection Retry" in gate_retry_context


def test_trainer_retries_actionable_wrong_artifact_rejection(tmp_path, monkeypatch):
    package_path, artifact_root = write_training_package(tmp_path)
    package = TrainingPackage.load(package_path)
    merge_contexts: list[str] = []
    dataloader = _RetryDataLoader()
    dataloader.train_items[0]["ranked_feedback_events"][0].update(
        {
            "required_improvements": [
                "MoonAI-style premium branding",
                "stronger product-relevant graphics",
                "scroll animations",
            ],
            "ranking": ["D > B > C > A"],
            "choice": "The page needs stronger branding, motion, and mobile polish.",
        }
    )

    def fake_merge(skill_content, failure_patches, success_patches, **kwargs):
        del failure_patches, success_patches
        context = kwargs.get("meta_skill_context", "")
        merge_contexts.append(context)
        if "wrong_artifact_type" in context:
            new_skill = skill_content.rstrip() + "\n\nVue bundle guidance.\n"
            change_summary = ["vue bundle guidance"]
        else:
            new_skill = skill_content.rstrip() + "\n\nWrong artifact patch.\n"
            change_summary = ["wrong artifact patch"]
        return {
            "reasoning": "fake merge",
            "skill_candidates": [
                {
                    "title": "candidate",
                    "change_summary": change_summary,
                    "new_skill": new_skill,
                }
            ],
        }

    class WrongArtifactAdapter(_RetryAdapter):
        def rollout(self, env_manager, skill_content, out_dir, **kwargs):
            del env_manager, out_dir, kwargs
            if "Vue bundle guidance" in skill_content:
                return [_scored_result(soft=0.95)]
            if "Wrong artifact patch" in skill_content:
                return [_wrong_artifact_result()]
            return [_scored_result(soft=0.89)]

    monkeypatch.setattr("skillopt.engine.trainer.merge_patches", fake_merge)
    cfg = _retry_trainer_config(
        tmp_path,
        package_content=package.template.content,
        artifact_root=artifact_root,
        package_path=package_path,
    )
    cfg["gate_metric"] = "soft"
    cfg["wrong_artifact_retry_budget"] = 1
    cfg["gate_reject_retry_budget"] = 0

    summary = ReflACTTrainer(cfg, WrongArtifactAdapter(dataloader)).train()

    assert summary["total_accepts"] == 1
    assert summary["best_origin"] == "step_0001"
    assert len(summary["wrong_artifact_retry_attempts"]) == 1
    retry_attempt = summary["wrong_artifact_retry_attempts"][0]
    assert retry_attempt["retry_class"] == "wrong_artifact_type"
    assert retry_attempt["expected_artifact"] == "vue-vite bundle"
    assert retry_attempt["actual_artifact"] == "skill markdown/template"
    assert retry_attempt["action"] == "retry"
    assert retry_attempt["gate_rejection"]["primary_reason"] == "wrong_artifact_type"
    assert retry_attempt["gate_rejection"]["retry_attempts"] == "0/1"
    assert retry_attempt["gate_rejection"]["failed_checks"][0]["check"] == "vue_vite_bundle.required_files"
    assert retry_attempt["gate_rejection"]["human_feedback_context"]["improve"] == [
        "MoonAI-style premium branding",
        "stronger product-relevant graphics",
        "scroll animations",
    ]
    assert "MoonAI-style premium branding" in retry_attempt["gate_rejection"]["optimizer_hint"]
    assert "Primary reason: wrong_artifact_type" in merge_contexts[1]
    assert "Expected artifact: vue-vite bundle" in merge_contexts[1]
    assert "Actual artifact: skill markdown/template" in merge_contexts[1]
    assert "vue_vite_bundle.required_files" in merge_contexts[1]
    assert "Return a Vue/Vite preview bundle JSON." in merge_contexts[1]
    assert "Rankings / pairwise preferences: D > B > C > A" in merge_contexts[1]
    assert "Required improvements: MoonAI-style premium branding" in merge_contexts[1]


def test_failure_hint_without_patch_is_recorded(tmp_path):
    package_path, artifact_root = write_training_package(tmp_path)
    package = TrainingPackage.load(package_path)

    class HintNoPatchAdapter(_RetryAdapter):
        def rollout(self, env_manager, skill_content, out_dir, **kwargs):
            del env_manager, skill_content, kwargs
            return [
                {
                    "id": str(out_dir).rsplit("/", 1)[-1],
                    "hard": 0,
                    "soft": 0.0,
                    "score_status": "scored",
                    "target_status": "passed",
                    "evaluator_status": "passed",
                    "failed_checks": [
                        {
                            "check": "top_level.required_files",
                            "reason": "Top-level failed check should be preserved.",
                            "evidence": ["top-level evidence"],
                        }
                    ],
                    "failure": {
                        "primary_reason": "artifact_contract_failure",
                        "optimizer_hint": "Return the required Vue/Vite preview bundle.",
                        "failed_checks": [
                            {
                                "check": "vue_vite_bundle.required_files",
                                "reason": "src/App.vue is missing.",
                                "evidence": ["missing src/App.vue"],
                            }
                        ],
                        "evidence": ["missing src/App.vue"],
                    },
                    "metadata": {
                        "failed_checks": [
                            {
                                "check": "metadata.required_files",
                                "reason": "Metadata failed check should be preserved.",
                                "evidence": ["metadata evidence"],
                            }
                        ]
                    },
                }
            ]

        def reflect(self, results, skill_content, out_dir, **kwargs):
            del results, skill_content, out_dir, kwargs
            return []

    cfg = _retry_trainer_config(
        tmp_path,
        package_content=package.template.content,
        artifact_root=artifact_root,
        package_path=package_path,
    )
    cfg["accumulation"] = 2
    cfg["train_size"] = 2
    cfg["feedback_direct_mode"] = "off"

    summary = ReflACTTrainer(cfg, HintNoPatchAdapter(_RetryDataLoader(train_size=2))).train()

    assert summary["total_skips"] == 1
    assert summary["no_candidate_reason"] == "failure_hint_not_converted_to_patch"
    assert "failure_hint_not_converted_to_patch" in summary["no_candidate_triggers"]
    history = json.loads((tmp_path / "out" / "history.json").read_text(encoding="utf-8"))
    assert history[0]["failure_hint_not_converted_to_patch"] is True
    assert len(history[0]["unconverted_failure_hints"]) == 2
    first_hint = history[0]["unconverted_failure_hints"][0]
    assert first_hint["optimizer_hint"].startswith("Return the required")
    checks = [check["check"] for check in first_hint["failed_checks"]]
    assert checks == [
        "top_level.required_files",
        "metadata.required_files",
        "vue_vite_bundle.required_files",
    ]


def test_actionable_hard_failure_retry_converts_hint_to_patch(tmp_path, monkeypatch):
    package_path, artifact_root = write_training_package(tmp_path)
    package = TrainingPackage.load(package_path)
    reflect_contexts: list[str] = []
    retry_result_ids: list[list[str]] = []
    dataloader = _RetryDataLoader()
    dataloader.train_items[0]["ranked_feedback_events"][0].update(
        {
            "required_improvements": [
                "MoonAI-style premium branding",
                "mobile polish",
                "Tailwind-style UI polish",
            ],
            "ranking": ["C > D > B > A"],
            "choice": "The candidate needs premium branding, mobile polish, and better UI details.",
        }
    )
    dataloader.train_items.append(
        {
            "id": "train-2",
            "ranked_feedback_events": [
                {
                    "required_improvements": ["unrelated successful item theme"],
                    "ranking": ["A > B > C > D"],
                }
            ],
        }
    )

    class HardFailureRetryAdapter(_RetryAdapter):
        def rollout(self, env_manager, skill_content, out_dir, **kwargs):
            del env_manager, out_dir, kwargs
            changed = "Vue artifact contract guidance" in skill_content
            return [
                {
                    "id": "val-1" if changed else "train-1",
                    "hard": 1 if changed else 0,
                    "soft": 0.9 if changed else 0.0,
                    "score_status": "scored",
                    "target_status": "passed",
                    "evaluator_status": "passed",
                    "failure": {
                        "primary_reason": "artifact_contract_failure",
                        "optimizer_hint": "Require the target to return the Vue/Vite preview bundle.",
                        "failed_checks": [
                            {
                                "check": "vue_vite_bundle.required_files",
                                "reason": "src/App.vue is missing.",
                                "evidence": ["missing src/App.vue"],
                            }
                        ],
                        "failed_dimensions": ["artifact_contract"],
                    },
                },
                {
                    "id": "train-2",
                    "hard": 1,
                    "soft": 0.8,
                    "score_status": "scored",
                    "target_status": "passed",
                    "evaluator_status": "passed",
                },
            ]

        def reflect(self, results, skill_content, out_dir, **kwargs):
            del skill_content, out_dir
            context = kwargs.get("step_buffer_context", "")
            reflect_contexts.append(context)
            if "Actionable Hard-Failure Retry" not in context:
                return []
            retry_result_ids.append([str(result.get("id")) for result in results])
            return [
                {
                    "source_type": "failure",
                    "patch": {
                        "skill_candidates": [
                            {
                                "title": "artifact contract guidance",
                                "change_summary": ["Vue artifact contract guidance"],
                                "new_skill": "",
                            }
                        ]
                    },
                }
            ]

    def fake_merge(skill_content, failure_patches, success_patches, **kwargs):
        del failure_patches, success_patches, kwargs
        return {
            "reasoning": "fake merge",
            "skill_candidates": [
                {
                    "title": "candidate",
                    "change_summary": ["Vue artifact contract guidance"],
                    "new_skill": skill_content.rstrip() + "\n\nVue artifact contract guidance.\n",
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
    cfg["feedback_direct_mode"] = "off"

    summary = ReflACTTrainer(cfg, HardFailureRetryAdapter(dataloader)).train()

    assert summary["total_accepts"] == 1
    assert any("Actionable Hard-Failure Retry" in context for context in reflect_contexts)
    retry_context = next(context for context in reflect_contexts if "Actionable Hard-Failure Retry" in context)
    assert "vue_vite_bundle.required_files" in retry_context
    assert "Required improvements: MoonAI-style premium branding" in retry_context
    assert "Rankings / pairwise preferences: C > D > B > A" in retry_context
    assert "unrelated successful item theme" not in retry_context
    history = json.loads((tmp_path / "out" / "history.json").read_text(encoding="utf-8"))
    assert history[0]["hard_failure_retry_attempts"][0]["status"] == "converted_to_patch"
    feedback_context = history[0]["hard_failure_retry_attempts"][0]["failure_hints"][0]["human_feedback_context"]
    assert feedback_context["improve"] == [
        "MoonAI-style premium branding",
        "mobile polish",
        "Tailwind-style UI polish",
    ]
    assert "failure_hint_not_converted_to_patch" not in history[0]
    assert retry_result_ids == [["train-1"]]


def test_actionable_hard_failure_retry_ignores_success_only_retry_patch(tmp_path, monkeypatch):
    package_path, artifact_root = write_training_package(tmp_path)
    package = TrainingPackage.load(package_path)
    retry_result_ids: list[list[str]] = []

    class SuccessOnlyRetryAdapter(_RetryAdapter):
        def rollout(self, env_manager, skill_content, out_dir, **kwargs):
            del env_manager, skill_content, out_dir, kwargs
            return [
                {
                    "id": "train-1",
                    "hard": 0,
                    "soft": 0.0,
                    "score_status": "scored",
                    "target_status": "passed",
                    "evaluator_status": "passed",
                    "failure": {
                        "primary_reason": "artifact_contract_failure",
                        "optimizer_hint": "Require the target to return the Vue/Vite preview bundle.",
                        "failed_checks": [
                            {
                                "check": "vue_vite_bundle.required_files",
                                "reason": "src/App.vue is missing.",
                                "evidence": ["missing src/App.vue"],
                            }
                        ],
                        "failed_dimensions": ["artifact_contract"],
                    },
                }
            ]

        def reflect(self, results, skill_content, out_dir, **kwargs):
            del skill_content, out_dir
            context = kwargs.get("step_buffer_context", "")
            if "Actionable Hard-Failure Retry" not in context:
                return []
            retry_result_ids.append([str(result.get("id")) for result in results])
            return [
                {
                    "source_type": "success",
                    "patch": {
                        "skill_candidates": [
                            {
                                "title": "success-only guidance",
                                "change_summary": ["success-only guidance"],
                                "new_skill": "",
                            }
                        ]
                    },
                }
            ]

    def fail_merge(*args, **kwargs):
        raise AssertionError("merge should not run without a converted failure patch")

    monkeypatch.setattr("skillopt.engine.trainer.merge_patches", fail_merge)
    cfg = _retry_trainer_config(
        tmp_path,
        package_content=package.template.content,
        artifact_root=artifact_root,
        package_path=package_path,
    )
    cfg["feedback_direct_mode"] = "off"

    summary = ReflACTTrainer(cfg, SuccessOnlyRetryAdapter(_RetryDataLoader())).train()

    assert summary["total_skips"] == 1
    assert "failure_hint_not_converted_to_patch" in summary["no_candidate_triggers"]
    history = json.loads((tmp_path / "out" / "history.json").read_text(encoding="utf-8"))
    assert history[0]["hard_failure_retry_attempts"][0]["status"] == "no_patch"
    assert history[0]["hard_failure_retry_attempts"][0]["n_success_patches"] == 1
    assert history[0]["failure_hint_not_converted_to_patch"] is True
    assert retry_result_ids == [["train-1"]]


def test_trainer_detects_wrong_artifact_from_failed_dimensions(tmp_path, monkeypatch):
    package_path, artifact_root = write_training_package(tmp_path)
    package = TrainingPackage.load(package_path)
    merge_contexts: list[str] = []

    def fake_merge(skill_content, failure_patches, success_patches, **kwargs):
        del failure_patches, success_patches
        context = kwargs.get("meta_skill_context", "")
        merge_contexts.append(context)
        if "wrong_artifact_type" in context:
            new_skill = skill_content.rstrip() + "\n\nVue bundle guidance.\n"
            change_summary = ["vue bundle guidance"]
        else:
            new_skill = skill_content.rstrip() + "\n\nWrong artifact patch.\n"
            change_summary = ["wrong artifact patch"]
        return {
            "reasoning": "fake merge",
            "skill_candidates": [
                {
                    "title": "candidate",
                    "change_summary": change_summary,
                    "new_skill": new_skill,
                }
            ],
        }

    class WrongArtifactDimensionAdapter(_RetryAdapter):
        def rollout(self, env_manager, skill_content, out_dir, **kwargs):
            del env_manager, out_dir, kwargs
            if "Vue bundle guidance" in skill_content:
                return [_scored_result(soft=0.95)]
            if "Wrong artifact patch" in skill_content:
                return [_wrong_artifact_dimension_only_result()]
            return [_scored_result(soft=0.89)]

    monkeypatch.setattr("skillopt.engine.trainer.merge_patches", fake_merge)
    cfg = _retry_trainer_config(
        tmp_path,
        package_content=package.template.content,
        artifact_root=artifact_root,
        package_path=package_path,
    )
    cfg["gate_metric"] = "soft"
    cfg["wrong_artifact_retry_budget"] = 1
    cfg["gate_reject_retry_budget"] = 0

    summary = ReflACTTrainer(cfg, WrongArtifactDimensionAdapter(_RetryDataLoader())).train()

    assert summary["total_accepts"] == 1
    assert len(summary["wrong_artifact_retry_attempts"]) == 1
    retry_attempt = summary["wrong_artifact_retry_attempts"][0]
    assert retry_attempt["retry_class"] == "wrong_artifact_type"
    assert retry_attempt["gate_rejection"]["primary_reason"] == "wrong_artifact_type"
    assert "Primary reason: wrong_artifact_type" in merge_contexts[1]


def test_trainer_stops_repeated_wrong_artifact_rejection(tmp_path, monkeypatch):
    package_path, artifact_root = write_training_package(tmp_path)
    package = TrainingPackage.load(package_path)
    merge_calls = 0

    def fake_merge(skill_content, failure_patches, success_patches, **kwargs):
        nonlocal merge_calls
        del failure_patches, success_patches, kwargs
        merge_calls += 1
        return {
            "reasoning": "fake merge",
            "skill_candidates": [
                {
                    "title": "candidate",
                    "change_summary": ["wrong artifact patch"],
                    "new_skill": skill_content.rstrip() + "\n\nWrong artifact patch.\n",
                }
            ],
        }

    class RepeatedWrongArtifactAdapter(_RetryAdapter):
        def rollout(self, env_manager, skill_content, out_dir, **kwargs):
            del env_manager, out_dir, kwargs
            if "Wrong artifact patch" in skill_content:
                return [_wrong_artifact_result()]
            return [_scored_result(soft=0.89)]

    monkeypatch.setattr("skillopt.engine.trainer.merge_patches", fake_merge)
    cfg = _retry_trainer_config(
        tmp_path,
        package_content=package.template.content,
        artifact_root=artifact_root,
        package_path=package_path,
    )
    cfg["gate_metric"] = "soft"
    cfg["wrong_artifact_retry_budget"] = 1
    cfg["gate_reject_retry_budget"] = 0

    summary = ReflACTTrainer(cfg, RepeatedWrongArtifactAdapter(_RetryDataLoader())).train()

    assert summary["total_accepts"] == 0
    assert summary["total_rejects"] == 1
    assert len(summary["wrong_artifact_retry_attempts"]) == 2
    assert summary["wrong_artifact_retry_attempts"][0]["action"] == "retry"
    assert summary["wrong_artifact_retry_attempts"][1]["action"] == "stop"
    assert summary["wrong_artifact_retry_attempts"][1]["stop_reason"] == "budget_exhausted"
    assert "budget_exhausted" in summary["no_candidate_triggers"]
    assert merge_calls == 2


def test_wrong_artifact_retry_does_not_consume_generic_gate_budget(tmp_path, monkeypatch):
    package_path, artifact_root = write_training_package(tmp_path)
    package = TrainingPackage.load(package_path)
    merge_contexts: list[str] = []

    def fake_merge(skill_content, failure_patches, success_patches, **kwargs):
        del failure_patches, success_patches
        context = kwargs.get("meta_skill_context", "")
        merge_contexts.append(context)
        if "Primary reason: candidate_quality_regressed" in context:
            new_skill = skill_content.rstrip() + "\n\nStrong visual guidance.\n"
            change_summary = ["strong visual guidance"]
        elif "wrong_artifact_type" in context:
            new_skill = skill_content.rstrip() + "\n\nWeak Vue bundle guidance.\n"
            change_summary = ["weak vue bundle guidance"]
        else:
            new_skill = skill_content.rstrip() + "\n\nWrong artifact patch.\n"
            change_summary = ["wrong artifact patch"]
        return {
            "reasoning": "fake merge",
            "skill_candidates": [
                {
                    "title": "candidate",
                    "change_summary": change_summary,
                    "new_skill": new_skill,
                }
            ],
        }

    class MixedRetryAdapter(_RetryAdapter):
        def rollout(self, env_manager, skill_content, out_dir, **kwargs):
            del env_manager, out_dir, kwargs
            if "Strong visual guidance" in skill_content:
                return [_scored_result(soft=0.95)]
            if "Weak Vue bundle guidance" in skill_content:
                return [_scored_result(soft=0.84)]
            if "Wrong artifact patch" in skill_content:
                return [_wrong_artifact_result()]
            return [_scored_result(soft=0.89)]

    monkeypatch.setattr("skillopt.engine.trainer.merge_patches", fake_merge)
    cfg = _retry_trainer_config(
        tmp_path,
        package_content=package.template.content,
        artifact_root=artifact_root,
        package_path=package_path,
    )
    cfg["gate_metric"] = "soft"
    cfg["wrong_artifact_retry_budget"] = 1
    cfg["gate_reject_retry_budget"] = 1
    cfg["gate_reject_retry_close_gap"] = 0.1

    summary = ReflACTTrainer(cfg, MixedRetryAdapter(_RetryDataLoader())).train()

    assert summary["total_accepts"] == 1
    assert summary["best_origin"] == "step_0001"
    assert len(summary["wrong_artifact_retry_attempts"]) == 1
    assert len(summary["gate_reject_retry_attempts"]) == 1
    assert summary["wrong_artifact_retry_attempts"][0]["retry_class"] == "wrong_artifact_type"
    assert summary["gate_reject_retry_attempts"][0]["retry_class"] == "gate_reject"
    assert summary["gate_reject_retry_attempts"][0]["attempt"] == 0
    assert len(merge_contexts) == 3
    step_dir = tmp_path / "out" / "steps" / "step_0001"
    assert (step_dir / "wrong_artifact_retry_00.json").exists()
    assert (step_dir / "gate_reject_retry_00.json").exists()
    assert (step_dir / "merged_patch_wrong_artifact_retry_01.json").exists()
    assert (step_dir / "merged_patch_gate_reject_retry_01.json").exists()
    assert (step_dir / "candidate_skill_wrong_artifact_retry_01.md").exists()
    assert (step_dir / "candidate_skill_gate_reject_retry_01.md").exists()


def test_trainer_stops_gate_rejection_after_retry_budget(tmp_path, monkeypatch):
    package_path, artifact_root = write_training_package(tmp_path)
    package = TrainingPackage.load(package_path)

    def fake_merge(skill_content, failure_patches, success_patches, **kwargs):
        del failure_patches, success_patches, kwargs
        return {
            "reasoning": "fake merge",
            "skill_candidates": [
                {
                    "title": "weak candidate",
                    "change_summary": ["artifact delivery only"],
                    "new_skill": skill_content.rstrip() + "\n\nArtifact delivery only.\n",
                }
            ],
        }

    class RejectingAdapter(_RetryAdapter):
        def rollout(self, env_manager, skill_content, out_dir, **kwargs):
            del env_manager, out_dir, kwargs
            changed = "Artifact delivery only" in skill_content
            return [
                {
                    "id": "val-1",
                    "hard": 1,
                    "soft": 0.84 if changed else 0.89,
                    "score_status": "scored",
                    "target_status": "passed",
                    "evaluator_status": "passed",
                }
            ]

    monkeypatch.setattr("skillopt.engine.trainer.merge_patches", fake_merge)
    cfg = _retry_trainer_config(
        tmp_path,
        package_content=package.template.content,
        artifact_root=artifact_root,
        package_path=package_path,
    )
    cfg["gate_metric"] = "soft"
    cfg["gate_reject_retry_budget"] = 1
    cfg["gate_reject_retry_close_gap"] = 0.1

    summary = ReflACTTrainer(cfg, RejectingAdapter(_RetryDataLoader())).train()

    assert summary["total_accepts"] == 0
    assert summary["total_rejects"] == 1
    assert summary["best_origin"] == "initial_skill"
    assert summary["no_candidate_reason"] == "gate_retry_duplicate_budget_exhausted"
    assert "retry_budget_exhausted" in summary["no_candidate_triggers"]
    assert len(summary["gate_reject_retry_attempts"]) == 1
    assert summary["gate_reject_retry_attempts"][0]["action"] == "stop"
    assert summary["gate_reject_retry_attempts"][0]["result"] == "duplicate_candidate"
    assert summary["gate_rejection"]["retry_attempts"] == "1/1"
    assert "retry_budget_exhausted" in summary["no_candidate_diagnostics"]["categories"]
    assert summary["no_candidate_diagnostics"]["retry_budget_exhausted"] is True
    assert summary["no_candidate_diagnostics"]["retry_stop_reasons"] == [
        "gate_retry_duplicate_budget_exhausted"
    ]
    assert summary["no_candidate_diagnostics"]["retry_attempts"][0]["fresh_reflect_retry"] is True
    assert summary["no_candidate_diagnostics"]["selection_gate_relation"] == "candidate_below_baseline"


def test_gate_rejection_duplicate_retry_forces_stronger_context(tmp_path, monkeypatch):
    package_path, artifact_root = write_training_package(tmp_path)
    package = TrainingPackage.load(package_path)
    merge_contexts: list[str] = []
    reflect_contexts: list[str] = []

    def fake_merge(skill_content, failure_patches, success_patches, **kwargs):
        del failure_patches, success_patches
        context = kwargs.get("meta_skill_context", "")
        merge_contexts.append(context)
        if "Duplicate Gate Retry Candidate" in context:
            new_skill = skill_content.rstrip() + "\n\nMobile visual system guidance.\n"
            change_summary = ["mobile visual system"]
        else:
            new_skill = skill_content.rstrip() + "\n\nArtifact delivery only.\n"
            change_summary = ["artifact delivery only"]
        return {
            "reasoning": "fake merge",
            "skill_candidates": [
                {
                    "title": "candidate",
                    "change_summary": change_summary,
                    "new_skill": new_skill,
                }
            ],
        }

    class DuplicateThenAcceptAdapter(_RetryAdapter):
        def reflect(self, results, skill_content, out_dir, **kwargs):
            reflect_contexts.append(
                "\n\n".join(
                    str(kwargs.get(key) or "")
                    for key in ("step_buffer_context", "meta_skill_context")
                )
            )
            return super().reflect(results, skill_content, out_dir, **kwargs)

        def rollout(self, env_manager, skill_content, out_dir, **kwargs):
            del env_manager, out_dir, kwargs
            if "Mobile visual system guidance" in skill_content:
                return [_scored_result(soft=0.95)]
            if "Artifact delivery only" in skill_content:
                return [_scored_result(soft=0.84)]
            return [_scored_result(soft=0.89)]

    monkeypatch.setattr("skillopt.engine.trainer.merge_patches", fake_merge)
    cfg = _retry_trainer_config(
        tmp_path,
        package_content=package.template.content,
        artifact_root=artifact_root,
        package_path=package_path,
    )
    cfg["gate_metric"] = "soft"
    cfg["gate_reject_retry_budget"] = 3
    cfg["gate_reject_retry_close_gap"] = 0.1

    summary = ReflACTTrainer(cfg, DuplicateThenAcceptAdapter(_RetryDataLoader())).train()

    assert summary["total_accepts"] == 1
    assert len(summary["gate_reject_retry_attempts"]) == 2
    duplicate_attempt = summary["gate_reject_retry_attempts"][0]
    assert duplicate_attempt["action"] == "retry"
    assert duplicate_attempt["retry_produced_duplicate_candidate"] is True
    assert duplicate_attempt["duplicate_of"] == duplicate_attempt["duplicate_candidate_hash"]
    assert "Duplicate Gate Retry Candidate" in merge_contexts[2]
    assert "Do not repeat the same structural update" in merge_contexts[2]
    assert "Repeated patch direction: artifact delivery only" in merge_contexts[2]
    assert "Unresolved human feedback themes:" in merge_contexts[2]
    assert "better mobile layout" in merge_contexts[2]
    assert any("Gate Rejection Retry" in context for context in reflect_contexts[1:])
    assert "Duplicate Gate Retry Candidate" in reflect_contexts[2]


def test_gate_rejection_retry_reflect_preserves_accumulation_batch_dirs(tmp_path, monkeypatch):
    package_path, artifact_root = write_training_package(tmp_path)
    package = TrainingPackage.load(package_path)
    reflect_calls: list[dict] = []

    class TwoBatchDataLoader(_RetryDataLoader):
        def __init__(self) -> None:
            super().__init__(item_id="item-a", train_size=2)
            self.train_items = [
                {
                    "id": "item-a",
                    "ranked_feedback_events": [
                        {"required_improvements": ["better mobile layout"]}
                    ],
                },
                {
                    "id": "item-b",
                    "ranked_feedback_events": [
                        {"required_improvements": ["better product imagery"]}
                    ],
                },
            ]

        def plan_train_epoch(self, *, epoch, steps_per_epoch, accumulation, batch_size, seed, **kwargs):
            del epoch, steps_per_epoch, accumulation, batch_size, seed, kwargs
            return [
                BatchSpec(phase="train", split="train", seed=11, batch_size=1, payload=[{"id": "item-a"}]),
                BatchSpec(phase="train", split="train", seed=12, batch_size=1, payload=[{"id": "item-b"}]),
            ]

    class RecordingAdapter(_RetryAdapter):
        def reflect(self, results, skill_content, out_dir, **kwargs):
            reflect_calls.append(
                {
                    "ids": [str(result.get("id")) for result in results],
                    "out_dir": str(out_dir),
                    "prediction_dir": str(kwargs.get("prediction_dir")),
                    "has_gate_retry_context": "Gate Rejection Retry" in str(kwargs.get("step_buffer_context", "")),
                }
            )
            return super().reflect(results, skill_content, out_dir, **kwargs)

        def rollout(self, env_manager, skill_content, out_dir, **kwargs):
            del env_manager, out_dir, kwargs
            changed = "Artifact delivery only" in skill_content
            return [_scored_result(soft=0.84 if changed else 0.89)]

    def fake_merge(skill_content, failure_patches, success_patches, **kwargs):
        del failure_patches, success_patches, kwargs
        return {
            "reasoning": "fake merge",
            "skill_candidates": [
                {
                    "title": "weak candidate",
                    "change_summary": ["artifact delivery only"],
                    "new_skill": skill_content.rstrip() + "\n\nArtifact delivery only.\n",
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
    cfg["accumulation"] = 2
    cfg["train_size"] = 2
    cfg["gate_metric"] = "soft"
    cfg["gate_reject_retry_budget"] = 1
    cfg["gate_reject_retry_close_gap"] = 0.1

    summary = ReflACTTrainer(cfg, RecordingAdapter(TwoBatchDataLoader())).train()

    retry_calls = [call for call in reflect_calls if call["has_gate_retry_context"]]
    assert summary["total_accepts"] == 0
    assert len(retry_calls) == 2
    assert retry_calls[0]["ids"] == ["item-a", "item-b"]
    assert retry_calls[1]["ids"] == ["item-a", "item-b"]
    assert "/batch_0/rollout/predictions" in retry_calls[0]["prediction_dir"]
    assert "/batch_1/rollout/predictions" in retry_calls[1]["prediction_dir"]


def test_duplicate_gate_retry_context_includes_unresolved_feedback_themes():
    context = _format_duplicate_gate_retry_context_from_packet(
        duplicate_of="abc123",
        attempt=1,
        packet={
            "attempted_patch": "artifact delivery only",
            "optimizer_hint": "Address the visual quality gap.",
            "failed_dimensions": ["human_feedback_alignment", "visual_quality"],
            "human_feedback_context": {
                "themes": ["MoonAI-like premium branding"],
                "improve": ["product-relevant graphics"],
            },
        },
    )

    assert "Duplicate Gate Retry Candidate" in context
    assert "Repeated patch direction: artifact delivery only" in context
    assert "Optimizer hint still unresolved: Address the visual quality gap." in context
    assert "Still failed dimensions: human_feedback_alignment, visual_quality" in context
    assert "MoonAI-like premium branding" in context
    assert "product-relevant graphics" in context


def test_gate_rejection_repeated_duplicate_retry_stops(tmp_path, monkeypatch):
    package_path, artifact_root = write_training_package(tmp_path)
    package = TrainingPackage.load(package_path)

    def fake_merge(skill_content, failure_patches, success_patches, **kwargs):
        del failure_patches, success_patches, kwargs
        return {
            "reasoning": "fake merge",
            "skill_candidates": [
                {
                    "title": "duplicate candidate",
                    "change_summary": ["artifact delivery only"],
                    "new_skill": skill_content.rstrip() + "\n\nArtifact delivery only.\n",
                }
            ],
        }

    class DuplicateRejectAdapter(_RetryAdapter):
        def rollout(self, env_manager, skill_content, out_dir, **kwargs):
            del env_manager, out_dir, kwargs
            if "Artifact delivery only" in skill_content:
                return [_scored_result(soft=0.84)]
            return [_scored_result(soft=0.89)]

    monkeypatch.setattr("skillopt.engine.trainer.merge_patches", fake_merge)
    cfg = _retry_trainer_config(
        tmp_path,
        package_content=package.template.content,
        artifact_root=artifact_root,
        package_path=package_path,
    )
    cfg["gate_metric"] = "soft"
    cfg["gate_reject_retry_budget"] = 3
    cfg["gate_reject_retry_close_gap"] = 0.1

    summary = ReflACTTrainer(cfg, DuplicateRejectAdapter(_RetryDataLoader())).train()

    assert summary["total_accepts"] == 0
    assert summary["no_candidate_reason"] == "gate_retry_duplicate_budget_exhausted"
    assert "duplicate_candidate" in summary["no_candidate_triggers"]
    assert len(summary["gate_reject_retry_attempts"]) == 3
    assert summary["gate_reject_retry_attempts"][0]["retry_produced_duplicate_candidate"] is True
    assert summary["gate_reject_retry_attempts"][1]["retry_produced_duplicate_candidate"] is True
    assert summary["gate_reject_retry_attempts"][1]["action"] == "retry"
    stop_attempt = summary["gate_reject_retry_attempts"][2]
    assert stop_attempt["retry_produced_duplicate_candidate"] is True
    assert stop_attempt["action"] == "stop"
    assert stop_attempt["stop_reason"] == "gate_retry_duplicate_budget_exhausted"


def test_gate_rejection_retry_uses_current_skill_scores_after_accept(tmp_path, monkeypatch):
    package_path, artifact_root = write_training_package(tmp_path)
    package = TrainingPackage.load(package_path)
    merge_calls = 0

    def fake_merge(skill_content, failure_patches, success_patches, **kwargs):
        nonlocal merge_calls
        del failure_patches, success_patches, kwargs
        merge_calls += 1
        if merge_calls == 1:
            new_skill = skill_content.rstrip() + "\n\nMobile layout guidance: preserve clear structure.\n"
            change_summary = ["mobile layout"]
        else:
            new_skill = skill_content.rstrip() + "\n\nArtifact delivery only.\n"
            change_summary = ["artifact delivery only"]
        return {
            "reasoning": "fake merge",
            "skill_candidates": [
                {
                    "title": "candidate",
                    "change_summary": change_summary,
                    "new_skill": new_skill,
                }
            ],
        }

    class CurrentScoreAdapter(_RetryAdapter):
        def rollout(self, env_manager, skill_content, out_dir, **kwargs):
            del env_manager, out_dir, kwargs
            if "Artifact delivery only" in skill_content:
                soft = 0.84
            elif "Mobile layout guidance" in skill_content:
                soft = 0.95
            else:
                soft = 0.89
            return [
                {
                    "id": "val-1",
                    "hard": 1,
                    "soft": soft,
                    "score_status": "scored",
                    "target_status": "passed",
                    "evaluator_status": "passed",
                }
            ]

    monkeypatch.setattr("skillopt.engine.trainer.merge_patches", fake_merge)
    cfg = _retry_trainer_config(
        tmp_path,
        package_content=package.template.content,
        artifact_root=artifact_root,
        package_path=package_path,
    )
    cfg["train_size"] = 2
    cfg["gate_metric"] = "soft"
    cfg["gate_reject_retry_budget"] = 1
    cfg["gate_reject_retry_close_gap"] = 0.2

    summary = ReflACTTrainer(cfg, CurrentScoreAdapter(_RetryDataLoader(train_size=2))).train()

    assert summary["total_accepts"] == 1
    assert summary["total_rejects"] == 1
    assert summary["best_origin"] == "step_0001"
    rejection = summary["gate_reject_retry_attempts"][0]["gate_rejection"]
    assert rejection["baseline"]["gate_score"] == 0.95
    assert rejection["candidate"]["gate_score"] == 0.84


def test_gate_rejection_retry_noop_preserves_fail_closed_skip(tmp_path, monkeypatch):
    package_path, artifact_root = write_training_package(tmp_path)
    package = TrainingPackage.load(package_path)
    final_test_rollouts: list[str] = []

    def fake_merge(skill_content, failure_patches, success_patches, **kwargs):
        del failure_patches, success_patches
        context = kwargs.get("meta_skill_context", "")
        if "Gate Rejection Retry" in context:
            new_skill = skill_content
            change_summary = ["same candidate"]
        else:
            new_skill = skill_content.rstrip() + "\n\nArtifact delivery only.\n"
            change_summary = ["artifact delivery only"]
        return {
            "reasoning": "fake merge",
            "skill_candidates": [
                {
                    "title": "candidate",
                    "change_summary": change_summary,
                    "new_skill": new_skill,
                }
            ],
        }

    class RejectThenNoopAdapter(_RetryAdapter):
        def rollout(self, env_manager, skill_content, out_dir, **kwargs):
            del env_manager, kwargs
            out_dir_text = str(out_dir)
            if "test_eval" in out_dir_text:
                final_test_rollouts.append(out_dir_text)
            changed = "Artifact delivery only" in skill_content
            return [
                {
                    "id": "val-1",
                    "hard": 1,
                    "soft": 0.84 if changed else 0.89,
                    "score_status": "scored",
                    "target_status": "passed",
                    "evaluator_status": "passed",
                }
            ]

    monkeypatch.setattr("skillopt.engine.trainer.merge_patches", fake_merge)
    cfg = _retry_trainer_config(
        tmp_path,
        package_content=package.template.content,
        artifact_root=artifact_root,
        package_path=package_path,
    )
    cfg["eval_test"] = True
    cfg["gate_metric"] = "soft"
    cfg["gate_reject_retry_budget"] = 1
    cfg["gate_reject_retry_close_gap"] = 0.1

    summary = ReflACTTrainer(cfg, RejectThenNoopAdapter(_RetryDataLoader())).train()

    assert summary["total_accepts"] == 0
    assert summary["total_rejects"] == 1
    assert summary["final_test_skipped_reason"] == "selection_gate_rejected_candidate"
    assert summary["baseline_test_hard"] is None
    assert summary["test_hard"] is None
    assert final_test_rollouts == []
    assert summary["no_candidate_reason"] == "gate_rejected_best_origin_initial_skill"
    assert "gate_retry_no_meaningful_change" in summary["no_candidate_triggers"]


def test_gate_rejection_retry_block_preserves_fail_closed_skip(tmp_path, monkeypatch):
    package_path, artifact_root = write_training_package(tmp_path)
    package = TrainingPackage.load(package_path)
    final_test_rollouts: list[str] = []

    def fake_merge(skill_content, failure_patches, success_patches, **kwargs):
        del failure_patches, success_patches
        context = kwargs.get("meta_skill_context", "")
        if "Gate Rejection Retry" in context:
            new_skill = skill_content.rstrip() + "\n\nMobile layout guidance: preserve clear structure.\n"
            change_summary = ["mobile layout"]
        else:
            new_skill = skill_content.rstrip() + "\n\nArtifact delivery only.\n"
            change_summary = ["artifact delivery only"]
        return {
            "reasoning": "fake merge",
            "skill_candidates": [
                {
                    "title": "candidate",
                    "change_summary": change_summary,
                    "new_skill": new_skill,
                }
            ],
        }

    class RejectThenBlockAdapter(_RetryAdapter):
        def rollout(self, env_manager, skill_content, out_dir, **kwargs):
            del env_manager, kwargs
            out_dir_text = str(out_dir)
            if "test_eval" in out_dir_text:
                final_test_rollouts.append(out_dir_text)
            if "selection_eval_gate_reject_retry" in out_dir_text:
                return [
                    {
                        "id": "val-1",
                        "hard": None,
                        "soft": None,
                        "score_status": "unscored",
                        "target_status": "passed",
                        "evaluator_status": "failed",
                    }
                ]
            changed = "Artifact delivery only" in skill_content
            return [
                {
                    "id": "val-1",
                    "hard": 1,
                    "soft": 0.84 if changed else 0.89,
                    "score_status": "scored",
                    "target_status": "passed",
                    "evaluator_status": "passed",
                }
            ]

    monkeypatch.setattr("skillopt.engine.trainer.merge_patches", fake_merge)
    cfg = _retry_trainer_config(
        tmp_path,
        package_content=package.template.content,
        artifact_root=artifact_root,
        package_path=package_path,
    )
    cfg["eval_test"] = True
    cfg["gate_metric"] = "soft"
    cfg["gate_reject_retry_budget"] = 1
    cfg["gate_reject_retry_close_gap"] = 0.1

    summary = ReflACTTrainer(cfg, RejectThenBlockAdapter(_RetryDataLoader())).train()

    assert summary["total_accepts"] == 0
    assert summary["total_rejects"] == 1
    assert summary["total_blocks"] == 1
    assert summary["gate_status"] == "blocked"
    assert summary["final_test_skipped_reason"] == "selection_gate_rejected_candidate"
    assert final_test_rollouts == []
    assert summary["no_candidate_reason"] == "gate_rejected_best_origin_initial_skill"
    assert "gate_retry_blocked:evaluator_failed" in summary["no_candidate_triggers"]


def test_selection_reject_skip_requires_unchanged_best_skill():
    history = [{"action": "reject"}]

    assert _should_skip_final_test_after_selection_reject(
        history=history,
        best_origin="initial_skill",
        best_skill="Initial skill\n",
        skill_init="Initial skill\n",
    )
    assert not _should_skip_final_test_after_selection_reject(
        history=history,
        best_origin="initial_skill",
        best_skill="Initial skill\n\nSlow update guidance.\n",
        skill_init="Initial skill\n",
    )


def test_gate_rejection_packet_handles_malformed_change_summary():
    rejection = _selection_reject_gate_rejection(
        history=[
            {
                "action": "reject",
                "selection_hard": 1.0,
                "selection_soft": 0.84,
                "candidate_gate_score": 0.84,
                "rewrite_change_summary": None,
            }
        ],
        baseline_scores=(1.0, 0.89),
        gate_metric="soft",
        gate_mixed_weight=0.5,
    )

    assert rejection is not None
    assert rejection["attempted_patch"] == "selection-gated candidate update"


def test_gate_rejection_prefers_signal_specific_human_feedback_context():
    rejection = _selection_reject_gate_rejection(
        history=[
            {
                "action": "reject",
                "selection_hard": 0,
                "selection_soft": 0.0,
                "candidate_gate_score": 0.0,
                "rewrite_change_summary": ["wrong artifact patch"],
            }
        ],
        baseline_scores=(1.0, 0.9),
        gate_metric="hard",
        gate_mixed_weight=0.5,
        rejection_signal={
            "primary_reason": "wrong_artifact_type",
            "optimizer_hint": "Return the Vue/Vite bundle.",
            "failed_dimensions": ["wrong_artifact_type"],
            "human_feedback_context": {
                "improve": ["signal-specific product graphics"],
                "rankings": ["C > A > B > D"],
            },
        },
        human_feedback_context={
            "improve": ["broad unrelated training theme"],
            "rankings": ["A > B > C > D"],
        },
    )

    assert rejection is not None
    assert rejection["human_feedback_context"]["improve"] == ["signal-specific product graphics"]
    assert "signal-specific product graphics" in rejection["optimizer_hint"]
    assert "broad unrelated training theme" not in rejection["optimizer_hint"]
    assert "human_feedback_alignment" in rejection["failed_dimensions"]


def test_selection_rejection_signal_keeps_non_artifact_failure_after_wrong_artifact_priority():
    signal = _selection_rejection_signal(
        [
            {
                "id": "item-1",
                "hard": 0,
                "soft": 0.45,
                "primary_reason": "human_feedback_not_resolved",
                "optimizer_hint": "Improve mobile responsiveness and motion.",
                "failed_dimensions": ["human_feedback_resolution", "mobile_responsiveness"],
            }
        ]
    )

    assert signal is not None
    assert signal["primary_reason"] == "human_feedback_not_resolved"
    assert signal["failed_dimensions"] == ["human_feedback_resolution", "mobile_responsiveness"]

    wrong_artifact = _selection_rejection_signal(
        [
            {
                "id": "item-1",
                "hard": 0,
                "soft": 0.45,
                "primary_reason": "human_feedback_not_resolved",
                "optimizer_hint": "Improve motion.",
                "failed_dimensions": ["human_feedback_resolution"],
            },
            {
                "id": "item-2",
                "hard": 0,
                "soft": 0.0,
                "primary_reason": "wrong_artifact_type",
                "optimizer_hint": "Return a Vue/Vite bundle.",
                "failed_dimensions": ["wrong_artifact_type"],
            },
        ]
    )

    assert wrong_artifact is not None
    assert wrong_artifact["primary_reason"] == "wrong_artifact_type"

    evaluator_contract = _selection_rejection_signal(
        [
            {
                "id": "item-1",
                "hard": 0,
                "soft": 0.45,
                "primary_reason": "human_feedback_not_resolved",
                "optimizer_hint": "Improve motion.",
                "failed_dimensions": ["human_feedback_resolution"],
            },
            {
                "id": "item-2",
                "hard": 0,
                "soft": 0.0,
                "primary_reason": "evaluator_missing_human_feedback_dimensions",
                "optimizer_hint": "Retry evaluation with explicit dimensions.",
                "failed_checks": [
                    {
                        "check": "llm_judge.human_feedback_dimensions",
                        "severity": "evaluator_contract_failure",
                    }
                ],
            },
        ]
    )

    assert evaluator_contract is not None
    assert evaluator_contract["primary_reason"] == "evaluator_missing_human_feedback_dimensions"

    evaluator_contract_over_artifact = _selection_rejection_signal(
        [
            {
                "id": "item-1",
                "hard": 0,
                "soft": 0.0,
                "primary_reason": "wrong_artifact_type",
                "optimizer_hint": "Return a Vue/Vite bundle.",
                "failed_dimensions": ["wrong_artifact_type"],
            },
            {
                "id": "item-2",
                "hard": 0,
                "soft": 0.0,
                "primary_reason": "evaluator_missing_human_feedback_dimensions",
                "optimizer_hint": "Retry evaluation with explicit dimensions.",
                "failed_checks": [
                    {
                        "check": "llm_judge.human_feedback_dimensions",
                        "severity": "evaluator_contract_failure",
                    }
                ],
            },
        ]
    )

    assert evaluator_contract_over_artifact is not None
    assert evaluator_contract_over_artifact["primary_reason"] == "evaluator_missing_human_feedback_dimensions"


def test_gate_rejection_retry_decision_requires_actionable_new_information():
    packet = {
        "rejection_type": "candidate_score_regression",
        "retryable": True,
        "baseline": {"hard": 1.0, "soft": 0.89, "gate_score": 0.89},
        "candidate": {"hard": 1.0, "soft": 0.87, "gate_score": 0.87},
        "primary_reason": "candidate_quality_regressed",
        "attempted_patch": "visual polish",
        "optimizer_hint": "Change direction.",
    }

    assert _gate_rejection_retry_decision(
        packet,
        attempt=0,
        budget=1,
        seen_reasons=set(),
        close_gap=0.03,
    ) == (True, "retryable")
    assert _gate_rejection_retry_decision(
        {**packet, "optimizer_hint": ""},
        attempt=0,
        budget=1,
        seen_reasons=set(),
        close_gap=0.03,
    ) == (False, "missing_actionable_rationale")
    assert _gate_rejection_retry_decision(
        packet,
        attempt=0,
        budget=2,
        seen_reasons={"candidate_quality_regressed|visual polish|Change direction."},
        close_gap=0.03,
    ) == (False, "repeated_rejection_reason")
    assert _gate_rejection_retry_decision(
        {**packet, "attempted_patch": "different direction"},
        attempt=0,
        budget=2,
        seen_reasons={"candidate_quality_regressed|visual polish|Change direction."},
        close_gap=0.03,
    ) == (True, "retryable")
    assert _gate_rejection_retry_decision(
        {**packet, "candidate": {"hard": 1.0, "soft": 0.7, "gate_score": 0.7}},
        attempt=0,
        budget=2,
        seen_reasons=set(),
        close_gap=0.03,
    ) == (True, "retryable")
    assert _gate_rejection_retry_decision(
        {
            **packet,
            "candidate": {"hard": 0.0, "soft": 0.87, "gate_score": 0.87},
            "failed_dimensions": ["human_feedback_resolution"],
        },
        attempt=0,
        budget=2,
        seen_reasons=set(),
        close_gap=0.03,
    ) == (True, "retryable")
    assert _gate_rejection_retry_decision(
        {
            **packet,
            "primary_reason": "mobile_responsiveness_failed",
            "candidate": {"hard": 0.0, "soft": 0.87, "gate_score": 0.87},
            "failed_dimensions": [],
        },
        attempt=0,
        budget=2,
        seen_reasons=set(),
        close_gap=0.03,
    ) == (True, "retryable")
    assert _gate_rejection_retry_decision(
        {
            **packet,
            "primary_reason": "landing_page_judge_rejected",
            "candidate": {"hard": 0.0, "soft": 0.87, "gate_score": 0.87},
            "failed_dimensions": [],
            "failed_checks": [{"check": "landing_page_v1.mobile_responsiveness"}],
        },
        attempt=0,
        budget=2,
        seen_reasons=set(),
        close_gap=0.03,
    ) == (True, "retryable")
    assert _gate_rejection_retry_decision(
        {
            **packet,
            "baseline": {"hard": 0.0, "soft": 0.45, "gate_score": 0.225},
            "candidate": {"hard": 0.0, "soft": 0.45, "gate_score": 0.225},
            "failed_dimensions": ["selection_gate", "human_feedback_alignment"],
            "human_feedback_context": {"feedback_target": "baseline_review_outputs"},
        },
        attempt=0,
        budget=3,
        seen_reasons=set(),
        close_gap=0.03,
    ) == (True, "retryable")
    assert _gate_rejection_retry_decision(
        {
            **packet,
            "baseline": {"hard": 1.0, "soft": 0.45, "gate_score": 0.725},
            "candidate": {"hard": 0.0, "soft": 0.45, "gate_score": 0.225},
            "failed_dimensions": ["selection_gate", "human_feedback_alignment"],
            "human_feedback_context": {"feedback_target": "baseline_review_outputs"},
        },
        attempt=0,
        budget=3,
        seen_reasons=set(),
        close_gap=1.0,
    ) == (True, "retryable")
    assert _gate_rejection_retry_decision(
        {
            **packet,
            "baseline": {"hard": 0.0, "soft": 0.45, "gate_score": 0.225},
            "candidate": {"hard": 0.0, "soft": 0.45, "gate_score": 0.225},
            "failed_dimensions": ["selection_gate", "human_feedback_alignment"],
        },
        attempt=0,
        budget=3,
        seen_reasons=set(),
        close_gap=0.03,
    ) == (True, "retryable")
    assert _gate_rejection_retry_decision(
        {
            **packet,
            "primary_reason": "new_unclassified_hard_failure",
            "candidate": {"hard": 0.0, "soft": 0.87, "gate_score": 0.87},
            "failed_dimensions": ["new_unclassified_hard_failure", "human_feedback_alignment"],
            "human_feedback_context": {"feedback_target": "baseline_review_outputs"},
        },
        attempt=0,
        budget=2,
        seen_reasons=set(),
        close_gap=0.03,
    ) == (True, "retryable")
    assert _gate_rejection_retry_decision(
        {
            **packet,
            "rejection_type": "wrong_artifact_type",
            "primary_reason": "wrong_artifact_type",
            "candidate": {"hard": 0.0, "soft": 0.87, "gate_score": 0.87},
            "failed_dimensions": ["wrong_artifact_type", "artifact_contract"],
        },
        attempt=0,
        budget=2,
        seen_reasons=set(),
        close_gap=0.03,
    ) == (True, "retryable")
    assert _gate_rejection_retry_decision(
        packet,
        attempt=1,
        budget=1,
        seen_reasons=set(),
    ) == (False, "budget_exhausted")


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
