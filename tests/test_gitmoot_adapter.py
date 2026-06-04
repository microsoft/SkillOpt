from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from skillopt.envs.gitmoot.adapter import GitmootAdapter
from skillopt.envs.gitmoot.evaluator import (
    TRUSTED_VUE_RENDER_PACKAGE_JSON,
    _block_external_browser_requests,
    _prepare_trusted_vue_render_deps,
    _render_artifact_dir,
    _run_vue_render_smoke,
    _trusted_vue_render_deps_cache_dir,
    _write_vue_render_workspace,
    evaluate_response,
)
from skillopt.envs.gitmoot.rollout import process_one
from skillopt.gradient.reflect import fmt_minibatch_trajectories
from skillopt.model import get_optimizer_backend, set_optimizer_backend, set_optimizer_deployment
from skillopt.model.common import default_model_for_backend
from tests.test_gitmoot_dataloader import write_training_package


def test_train_registry_can_instantiate_gitmoot_adapter(tmp_path):
    package_path, artifact_root = write_training_package(tmp_path)

    from scripts.train import get_adapter

    adapter = get_adapter(
        {
            "env": "gitmoot",
            "training_package": str(package_path),
            "artifact_root": str(artifact_root),
        }
    )

    assert isinstance(adapter, GitmootAdapter)


def test_adapter_setup_uses_package_template_as_initial_skill(tmp_path):
    package_path, artifact_root = write_training_package(tmp_path)
    adapter = GitmootAdapter(str(package_path), str(artifact_root))
    cfg = {"out_root": str(tmp_path / "out"), "skill_init": ""}

    adapter.setup(cfg)

    assert cfg["skill_init"].endswith("gitmoot_initial_skill.md")
    content = (tmp_path / "out" / "gitmoot_initial_skill.md").read_text(encoding="utf-8")
    assert content == adapter.dataloader.initial_skill_content
    assert content.startswith("---\n")


def test_adapter_rollout_returns_skillopt_result_shape(tmp_path):
    package_path, artifact_root = write_training_package(tmp_path)
    adapter = GitmootAdapter(str(package_path), str(artifact_root))
    adapter.setup({})
    env = adapter.build_train_env(batch_size=1, seed=1)

    results = adapter.rollout(env, adapter.dataloader.initial_skill_content, str(tmp_path / "out"))

    assert len(results) == 1
    assert results[0]["id"] == "train-1"
    assert results[0]["hard"] == 1
    assert results[0]["soft"] == 1.0
    assert results[0]["target_status"] == "passed"
    assert results[0]["evaluator_status"] == "passed"
    assert results[0]["score_status"] == "scored"
    assert results[0]["evaluator_id"] == "fixture"
    assert results[0]["evaluator_version"] == "v0"
    assert results[0]["response"] == "better plan"
    assert results[0]["fail_reason"] == ""
    assert (tmp_path / "out" / "predictions" / "train-1" / "conversation.json").is_file()


def test_adapter_eval_batch_uses_fixture_evaluator(tmp_path):
    package_path, artifact_root = write_training_package(tmp_path)
    adapter = GitmootAdapter(str(package_path), str(artifact_root))
    adapter.setup({})
    env = adapter.build_eval_env(env_num=1, split="valid_seen", seed=1)

    results = adapter.rollout(env, adapter.dataloader.initial_skill_content, str(tmp_path / "out"))

    assert results[0]["id"] == "val-1"
    assert results[0]["hard"] == 0
    assert results[0]["soft"] == 0.25
    assert results[0]["score_status"] == "scored"
    assert results[0]["metadata"]["evaluator"] == "fixture"


def test_adapter_reflection_uses_template_body_not_frontmatter(tmp_path, monkeypatch):
    captured = {}

    def fake_reflect(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr("skillopt.envs.gitmoot.adapter.run_minibatch_reflect", fake_reflect)
    package_path, artifact_root = write_training_package(tmp_path)
    adapter = GitmootAdapter(str(package_path), str(artifact_root))
    adapter.setup({})

    adapter.reflect([], adapter.dataloader.initial_skill_content, str(tmp_path / "out"))

    assert captured["skill_content"].startswith("# Planner")
    assert "kind: agent-template" not in captured["skill_content"]


def test_full_rewrite_reflection_preserves_template_frontmatter(tmp_path, monkeypatch):
    def fake_reflect(**kwargs):
        return [
            {
                "patch": {
                    "skill_candidates": [
                        {"new_skill": "# Planner\n\nUse the new planning rules.\n"}
                    ]
                }
            }
        ]

    monkeypatch.setattr("skillopt.envs.gitmoot.adapter.run_minibatch_reflect", fake_reflect)
    package_path, artifact_root = write_training_package(tmp_path)
    adapter = GitmootAdapter(str(package_path), str(artifact_root))
    adapter.setup({"skill_update_mode": "full_rewrite_minibatch"})

    patches = adapter.reflect([], adapter.dataloader.initial_skill_content, str(tmp_path / "out"))
    new_skill = patches[0]["patch"]["skill_candidates"][0]["new_skill"]

    assert new_skill.startswith("---\n")
    assert "kind: agent-template" in new_skill.split("---", 2)[1]
    assert "\n---\n# Planner" in new_skill


def test_failed_agent_execution_overrides_fixture_success(tmp_path):
    item = {
        "id": "broken",
        "prompt": "Prompt",
        "metadata": {"expected_hard": True},
        "evaluator_config": {"mode": "fixture"},
    }

    result = process_one(item=item, skill_content="skill", out_root=str(tmp_path))

    assert result["agent_ok"] is False
    assert result["hard"] is None
    assert result["soft"] is None
    assert result["target_status"] == "failed"
    assert result["evaluator_status"] == "not_run"
    assert result["score_status"] == "unscored"
    assert result["blocker"] == "target_rollout_failed"
    assert result["fail_reason"]


def test_failed_agent_execution_does_not_call_judge(tmp_path, monkeypatch):
    def fail_evaluator(*args, **kwargs):
        raise AssertionError("evaluator should not run after agent failure")

    monkeypatch.setattr("skillopt.envs.gitmoot.rollout.evaluate_response", fail_evaluator)
    item = {"id": "broken", "prompt": "Prompt", "metadata": {"expected_hard": True}}

    result = process_one(item=item, skill_content="skill", out_root=str(tmp_path))

    assert result["hard"] is None
    assert result["soft"] is None
    assert result["score_status"] == "unscored"
    assert result["metadata"]["agent_error"] is True


def test_empty_message_agent_exception_is_still_failure(tmp_path, monkeypatch):
    def fail_agent(*args, **kwargs):
        raise Exception()

    monkeypatch.setattr("skillopt.envs.gitmoot.rollout._run_agent", fail_agent)
    item = {
        "id": "broken",
        "prompt": "Prompt",
        "metadata": {"expected_hard": True},
        "evaluator_config": {"mode": "fixture"},
    }

    result = process_one(item=item, skill_content="skill", out_root=str(tmp_path))

    assert result["agent_ok"] is False
    assert result["hard"] is None
    assert result["soft"] is None
    assert result["fail_reason"] == "agent execution failed"
    assert result["target_status"] == "failed"
    assert result["evaluator_status"] == "not_run"
    assert result["score_status"] == "unscored"
    assert result["blocker"] == "target_rollout_failed"
    assert result["metadata"]["agent_error"] is True


def test_evaluator_exception_is_unscored_failure(tmp_path, monkeypatch):
    def pass_agent(*args, **kwargs):
        return "Candidate response"

    def fail_evaluator(*args, **kwargs):
        raise RuntimeError("judge backend unavailable")

    monkeypatch.setattr("skillopt.envs.gitmoot.rollout._run_agent", pass_agent)
    monkeypatch.setattr("skillopt.envs.gitmoot.rollout.evaluate_response", fail_evaluator)
    item = {"id": "broken-eval", "prompt": "Prompt"}

    result = process_one(item=item, skill_content="skill", out_root=str(tmp_path))

    assert result["agent_ok"] is True
    assert result["hard"] is None
    assert result["soft"] is None
    assert result["target_status"] == "passed"
    assert result["evaluator_status"] == "failed"
    assert result["score_status"] == "unscored"
    assert result["blocker"] == "evaluator_failed"
    assert result["fail_reason"] == "judge backend unavailable"


def test_invalid_evaluator_score_is_unscored_failure(tmp_path, monkeypatch):
    def pass_agent(*args, **kwargs):
        return "Candidate response"

    def invalid_evaluator(*args, **kwargs):
        return {
            "hard": "not-a-score",
            "soft": "also-not-a-score",
            "metadata": {"evaluator": "fixture"},
        }

    monkeypatch.setattr("skillopt.envs.gitmoot.rollout._run_agent", pass_agent)
    monkeypatch.setattr("skillopt.envs.gitmoot.rollout.evaluate_response", invalid_evaluator)
    item = {"id": "invalid-score", "prompt": "Prompt"}

    result = process_one(item=item, skill_content="skill", out_root=str(tmp_path))

    assert result["hard"] is None
    assert result["soft"] is None
    assert result["target_status"] == "passed"
    assert result["evaluator_status"] == "failed"
    assert result["score_status"] == "unscored"
    assert result["blocker"] == "invalid_evaluator_score"
    assert result["evaluator_id"] == "fixture"
    assert result["fail_reason"] == "evaluator returned invalid hard/soft scores"


def test_structured_evaluator_feedback_reaches_rollout_result(tmp_path, monkeypatch):
    def pass_agent(*args, **kwargs):
        return "Candidate response"

    def structured_evaluator(*args, **kwargs):
        return {
            "hard": 0,
            "soft": 0.2,
            "fail_reason": "missing required artifact",
            "profile_id": "vue_landing_page_v1",
            "task_kind": "vue_landing_page",
            "failure": {
                "primary_reason": "missing_required_artifact",
                "optimizer_hint": "Return the required Vue/Vite preview bundle.",
            },
            "stage_status": [{"stage": "artifact_contract", "status": "failed"}],
        }

    monkeypatch.setattr("skillopt.envs.gitmoot.rollout._run_agent", pass_agent)
    monkeypatch.setattr("skillopt.envs.gitmoot.rollout.evaluate_response", structured_evaluator)

    result = process_one(item={"id": "structured", "prompt": "Prompt"}, skill_content="skill", out_root=str(tmp_path))

    assert result["failure"]["primary_reason"] == "missing_required_artifact"
    assert result["stage_status"][0]["stage"] == "artifact_contract"
    assert result["metadata"]["failure"]["optimizer_hint"].startswith("Return the required")


def test_rollout_sets_prediction_local_render_artifact_dir(tmp_path, monkeypatch):
    captured = {}

    def pass_agent(*args, **kwargs):
        return _valid_vue_bundle_response()

    def fake_evaluator(item, response, config):
        del item, response
        captured.update(config)
        return {
            "hard": 1,
            "soft": 0.9,
            "metadata": {"evaluator": "landing_page_v1"},
        }

    monkeypatch.setattr("skillopt.envs.gitmoot.rollout._run_agent", pass_agent)
    monkeypatch.setattr("skillopt.envs.gitmoot.rollout.evaluate_response", fake_evaluator)

    process_one(
        item={"id": "landing-page-render-dir", "prompt": "Build a Vue landing page."},
        skill_content="skill",
        out_root=str(tmp_path),
        evaluator_config={
            "mode": "landing_page_v1",
            "artifact_dir": str(tmp_path / "outside-artifacts"),
            "render_artifact_dir": str(tmp_path / "outside-render-artifacts"),
        },
    )

    render_dir = captured["render_artifact_dir"]
    expected_root = tmp_path / "predictions" / "landing-page-render-dir" / "render-smoke"
    assert render_dir.startswith(str(expected_root))
    assert "outside" not in render_dir
    assert len(os.path.basename(render_dir)) == 12


def test_structured_evaluator_feedback_reaches_reflection_prompt(tmp_path):
    pred_dir = tmp_path / "predictions" / "structured"
    pred_dir.mkdir(parents=True)
    (pred_dir / "conversation.json").write_text(
        json.dumps(
            [
                {"role": "user", "content": "Build the landing page."},
                {"role": "assistant", "content": "Candidate response"},
            ]
        ),
        encoding="utf-8",
    )

    text = fmt_minibatch_trajectories(
        [
            {
                "id": "structured",
                "task_description": "Landing page",
                "task_type": "gitmoot-skillopt",
                "fail_reason": "missing required artifact",
                "contract_status": "failed",
                "quality_status": "not_run",
                "human_feedback_alignment": {
                    "status": "feedback_available",
                    "required_improvements": ["stronger product visuals"],
                },
                "failure": {
                    "primary_reason": "missing_required_artifact",
                    "optimizer_hint": "Return the required Vue/Vite preview bundle.",
                },
                "stage_status": [{"stage": "artifact_contract", "status": "failed"}],
            }
        ],
        str(tmp_path / "predictions"),
    )

    assert "Structured Evaluator Feedback" in text
    assert "contract_status" in text
    assert "quality_status" in text
    assert "stronger product visuals" in text
    assert "missing_required_artifact" in text
    assert "Return the required Vue" in text


def test_unscored_result_serializes_null_scores(tmp_path):
    item = {
        "id": "broken",
        "prompt": "Prompt",
        "metadata": {"expected_hard": True},
        "evaluator_config": {"mode": "fixture"},
    }

    result = process_one(item=item, skill_content="skill", out_root=str(tmp_path))
    result_path = tmp_path / "predictions" / "broken" / "result.json"
    persisted = json.loads(result_path.read_text(encoding="utf-8"))

    assert result["hard"] is None
    assert persisted["hard"] is None
    assert persisted["soft"] is None
    assert persisted["score_status"] == "unscored"
    assert persisted["metadata"]["score_status"] == "unscored"


def test_exec_target_backend_runs_harness_and_persists_raw_trace(tmp_path, monkeypatch):
    captured = {}

    def fake_exec(**kwargs):
        captured.update(kwargs)
        return "exec response", "raw exec trace"

    monkeypatch.setenv("TARGET_DEPLOYMENT", "gpt-test")
    monkeypatch.setattr("skillopt.envs.gitmoot.rollout.is_target_exec_backend", lambda: True)
    monkeypatch.setattr("skillopt.envs.gitmoot.rollout.is_target_chat_backend", lambda: False)
    monkeypatch.setattr("skillopt.envs.gitmoot.rollout.get_target_backend", lambda: "codex_exec")
    monkeypatch.setattr("skillopt.envs.gitmoot.rollout.run_target_exec", fake_exec)
    item = {
        "id": "exec-item",
        "prompt": "Prompt",
        "metadata": {"expected_hard": True},
        "evaluator_config": {"mode": "fixture"},
    }

    result = process_one(item=item, skill_content="skill", out_root=str(tmp_path))

    assert result["response"] == "exec response"
    assert result["hard"] == 1
    assert result["score_status"] == "scored"
    assert captured["model"] == "gpt-test"
    assert captured["work_dir"].endswith("predictions/exec-item/target_exec")
    assert "## System Instructions" in captured["prompt"]
    assert (tmp_path / "predictions" / "exec-item" / "target_exec" / "task.md").is_file()
    assert (
        tmp_path
        / "predictions"
        / "exec-item"
        / "target_exec"
        / ".agents"
        / "skills"
        / "skillopt-target"
        / "SKILL.md"
    ).is_file()
    raw_path = tmp_path / "predictions" / "exec-item" / "target_exec_raw.txt"
    assert raw_path.read_text(encoding="utf-8") == "raw exec trace"
    assert result["target_trace_path"] == str(raw_path)


def test_mock_response_under_exec_backend_records_conversation_trace(tmp_path, monkeypatch):
    def fail_exec(**kwargs):
        raise AssertionError("mock response should not invoke exec target")

    monkeypatch.setattr("skillopt.envs.gitmoot.rollout.is_target_exec_backend", lambda: True)
    monkeypatch.setattr("skillopt.envs.gitmoot.rollout.is_target_chat_backend", lambda: False)
    monkeypatch.setattr("skillopt.envs.gitmoot.rollout.run_target_exec", fail_exec)
    item = {
        "id": "mock-exec",
        "prompt": "Prompt",
        "metadata": {"mock_response": "fixture response", "expected_hard": True},
        "evaluator_config": {"mode": "fixture"},
    }

    result = process_one(item=item, skill_content="skill", out_root=str(tmp_path))

    conversation_path = tmp_path / "predictions" / "mock-exec" / "conversation.json"
    assert result["response"] == "fixture response"
    assert result["hard"] == 1
    assert result["score_status"] == "scored"
    assert result["target_trace_path"] == str(conversation_path)
    assert conversation_path.is_file()
    assert not (tmp_path / "predictions" / "mock-exec" / "target_exec_raw.txt").exists()


def test_empty_exec_target_response_is_unscored_target_failure(tmp_path, monkeypatch):
    def fake_exec(**kwargs):
        return "", "raw exec trace"

    monkeypatch.setattr("skillopt.envs.gitmoot.rollout.is_target_exec_backend", lambda: True)
    monkeypatch.setattr("skillopt.envs.gitmoot.rollout.is_target_chat_backend", lambda: False)
    monkeypatch.setattr("skillopt.envs.gitmoot.rollout.get_target_backend", lambda: "codex_exec")
    monkeypatch.setattr("skillopt.envs.gitmoot.rollout.run_target_exec", fake_exec)
    item = {
        "id": "empty-exec",
        "prompt": "Prompt",
        "metadata": {"expected_hard": True},
        "evaluator_config": {"mode": "fixture"},
    }

    result = process_one(item=item, skill_content="skill", out_root=str(tmp_path))

    assert result["hard"] is None
    assert result["soft"] is None
    assert result["target_status"] == "failed"
    assert result["evaluator_status"] == "not_run"
    assert result["score_status"] == "unscored"
    assert result["blocker"] == "target_rollout_failed"
    assert result["fail_reason"] == "exec target returned empty response"
    assert (tmp_path / "predictions" / "empty-exec" / "target_exec_raw.txt").is_file()


def test_exec_target_exception_preserves_existing_raw_trace(tmp_path, monkeypatch):
    def fail_exec(**kwargs):
        pred_dir = tmp_path / "predictions" / "exec-error"
        (pred_dir / "codex_raw.txt").write_text("codex failure trace", encoding="utf-8")
        raise RuntimeError("codex failed")

    monkeypatch.setattr("skillopt.envs.gitmoot.rollout.is_target_exec_backend", lambda: True)
    monkeypatch.setattr("skillopt.envs.gitmoot.rollout.is_target_chat_backend", lambda: False)
    monkeypatch.setattr("skillopt.envs.gitmoot.rollout.get_target_backend", lambda: "codex_exec")
    monkeypatch.setattr("skillopt.envs.gitmoot.rollout.run_target_exec", fail_exec)
    item = {
        "id": "exec-error",
        "prompt": "Prompt",
        "metadata": {"expected_hard": True},
        "evaluator_config": {"mode": "fixture"},
    }

    result = process_one(item=item, skill_content="skill", out_root=str(tmp_path))

    raw_path = tmp_path / "predictions" / "exec-error" / "target_exec_raw.txt"
    assert result["hard"] is None
    assert result["target_status"] == "failed"
    assert result["score_status"] == "unscored"
    assert result["fail_reason"] == "codex failed"
    assert raw_path.read_text(encoding="utf-8") == "codex failure trace"
    assert result["target_trace_path"] == str(raw_path)


def test_unsupported_target_backend_is_unscored_target_failure(tmp_path, monkeypatch):
    monkeypatch.setattr("skillopt.envs.gitmoot.rollout.is_target_exec_backend", lambda: False)
    monkeypatch.setattr("skillopt.envs.gitmoot.rollout.is_target_chat_backend", lambda: False)
    item = {
        "id": "unsupported-target",
        "prompt": "Prompt",
        "metadata": {"expected_hard": True},
        "evaluator_config": {"mode": "fixture"},
    }

    result = process_one(item=item, skill_content="skill", out_root=str(tmp_path))

    assert result["hard"] is None
    assert result["soft"] is None
    assert result["target_status"] == "failed"
    assert result["evaluator_status"] == "not_run"
    assert result["score_status"] == "unscored"
    assert result["blocker"] == "target_rollout_failed"
    assert "chat target backend, exec target backend" in result["fail_reason"]


def test_contains_evaluator_is_deterministic():
    item = {"metadata": {"required_text": "ship it"}}

    score = evaluate_response(item, "We should ship it today.", {"mode": "contains"})

    assert score["hard"] == 1
    assert score["soft"] == 1.0
    assert score["metadata"]["evaluator"] == "contains"


def test_judge_evaluator_parses_string_hard_values(monkeypatch):
    def fake_chat_optimizer(**kwargs):
        return ('{"hard": "0", "soft": "0.2", "fail_reason": "bad", "reasoning": "No."}', {})

    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator.chat_optimizer", fake_chat_optimizer)

    score = evaluate_response({"prompt": "Prompt"}, "Response", {})

    assert score["hard"] == 0
    assert score["soft"] == 0.2
    assert score["fail_reason"] == "bad"


def test_landing_page_evaluator_returns_structured_score(tmp_path, monkeypatch):
    captured = {}

    def pass_agent(*args, **kwargs):
        return _valid_vue_bundle_response()

    def fake_chat_optimizer(**kwargs):
        captured.update(kwargs)
        return (
            json.dumps(
                {
                    "hard": 1,
                    "soft": 0.87,
                    "dimension_scores": _landing_dimension_scores(),
                    "rationale": "Clear hero, CTA, footer, and responsive layout.",
                    "fail_reason": "",
                }
            ),
            {},
        )

    monkeypatch.setattr("skillopt.envs.gitmoot.rollout._run_agent", pass_agent)
    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator.chat_optimizer", fake_chat_optimizer)
    item = {
        "id": "landing-page",
        "prompt": "Build a Vue landing page.\nHuman ranking: D > B > C > A.\nNeeds mobile responsiveness.",
        "metadata": {"kind": "landing_page"},
        "evaluator_config": {"mode": "landing_page_v1"},
    }

    result = process_one(item=item, skill_content="skill", out_root=str(tmp_path))

    assert result["hard"] == 1
    assert result["soft"] == 0.87
    assert result["score_status"] == "scored"
    assert result["evaluator_id"] == "landing_page_v1"
    assert result["evaluator_version"] == "v1"
    assert result["dimension_scores"]["mobile_responsiveness"] == 0.9
    assert result["stage_status"] == [{"stage": "llm_judge", "status": "passed"}]
    assert result["metadata"]["dimension_scores"]["mobile_responsiveness"] == 0.9
    assert result["metadata"]["rationale"] == "Clear hero, CTA, footer, and responsive layout."
    assert result["metadata"]["check_context"]["artifact_contract"]["status"] == "not_required"
    assert captured["stage"] == "gitmoot_landing_page_judge"
    assert "Human ranking: D > B > C > A" in captured["user"]
    assert "Deterministic And Render Check Context" in captured["user"]
    assert "Structured Task Context" in captured["user"]
    assert "Generated Landing Page Response" in captured["user"]
    assert "mobile_responsiveness" in captured["system"]


def test_landing_page_judge_prompt_includes_render_and_feedback_context(monkeypatch):
    captured = {}

    def pass_render(*args, **kwargs):
        return {
            "hard": 1,
            "soft": 1.0,
            "dimension_scores": {"render_smoke": 1.0},
            "stage_status": [{"stage": "render_smoke", "status": "passed"}],
            "metadata": {
                "dimension_scores": {"render_smoke": 1.0},
                "stage_status": [{"stage": "render_smoke", "status": "passed"}],
                "render_smoke": {"screenshots": [{"label": "mobile", "path": "/tmp/mobile.png"}]},
            },
        }

    def fake_chat_optimizer(**kwargs):
        captured.update(kwargs)
        return (
            json.dumps(
                {
                    "hard": 1,
                    "soft": 0.91,
                    "dimension_scores": _landing_dimension_scores(),
                    "rationale": "Render and human feedback context were considered.",
                    "fail_reason": "",
                    "contract_status": "failed",
                    "quality_status": "failed",
                }
            ),
            {},
        )

    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator._run_vue_render_smoke", pass_render)
    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator.chat_optimizer", fake_chat_optimizer)

    score = evaluate_response(
        {
            "id": "landing-page-context",
            "prompt": "Build a Vue landing page.",
            "artifacts": [{"id": "option-d", "role": "winner", "path": "D.vue"}],
            "ranked_feedback_events": [{"ranking": ["D > B > C > A"], "choice": "D has the cleanest hero."}],
            "metadata": {"source": "github_issue_109", "secret_token": "do-not-send-this"},
        },
        _valid_vue_bundle_response(),
        {"mode": "landing_page_v1", "artifact_contract": "vue_vite_bundle", "require_vue_render_smoke": True},
    )

    assert score["hard"] == 1
    assert score["contract_status"] == "passed"
    assert score["quality_status"] == "passed"
    assert score["human_feedback_alignment"]["status"] == "feedback_available"
    assert score["human_feedback_alignment"]["rankings"] == ["D > B > C > A"]
    assert score["human_feedback_alignment"]["reasoning"] == ["D has the cleanest hero."]
    assert score["stage_status"][0] == {"stage": "render_smoke", "status": "passed"}
    assert score["stage_status"][-1] == {"stage": "llm_judge", "status": "passed"}
    assert '"status": "passed"' in captured["user"]
    assert "/tmp/mobile.png" in captured["user"]
    assert "github_issue_109" in captured["user"]
    assert "D has the cleanest hero" in captured["user"]
    assert "do-not-send-this" not in captured["user"]
    assert "secret_token" not in captured["user"]


def test_landing_page_evaluator_accepts_numeric_string_hard(tmp_path, monkeypatch):
    def pass_agent(*args, **kwargs):
        return _valid_vue_bundle_response()

    def fake_chat_optimizer(**kwargs):
        return (
            json.dumps(
                {
                    "hard": "1.0",
                    "soft": "0.82",
                    "dimension_scores": {key: str(value) for key, value in _landing_dimension_scores().items()},
                    "rationale": "Strong enough to promote with responsive layout and footer.",
                    "fail_reason": "",
                }
            ),
            {},
        )

    monkeypatch.setattr("skillopt.envs.gitmoot.rollout._run_agent", pass_agent)
    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator.chat_optimizer", fake_chat_optimizer)
    item = {
        "id": "landing-page-numeric-hard",
        "prompt": "Build a Vue landing page.",
        "evaluator_config": {"mode": "landing_page_v1"},
    }

    result = process_one(item=item, skill_content="skill", out_root=str(tmp_path))

    assert result["hard"] == 1
    assert result["soft"] == 0.82
    assert result["score_status"] == "scored"
    assert result["evaluator_status"] == "passed"


def test_landing_page_judge_rejection_returns_optimizer_failure_packet(monkeypatch):
    def fake_chat_optimizer(**kwargs):
        return (
            json.dumps(
                {
                    "hard": 0,
                    "soft": 0.31,
                    "dimension_scores": {
                        **_landing_dimension_scores(),
                        "hero_quality": 0.2,
                        "mobile_responsiveness": 0.1,
                        "animation_motion_quality": 0.0,
                    },
                    "rationale": "The page is not mobile responsive and has no meaningful motion.",
                    "fail_reason": "mobile layout and hero motion are below promotion quality",
                    "contract_status": "failed",
                    "quality_status": "passed",
                    "failure": {
                        "primary_reason": "mobile_responsiveness_failed",
                        "human_reason": "Mobile layout overflows and hero motion is missing.",
                        "optimizer_hint": "Make the hero responsive, remove overflow, and add purposeful motion.",
                        "failed_checks": [
                            {
                                "check": "landing_page_v1.mobile_responsiveness",
                                "reason": "mobile overflow",
                                "evidence": ["mobile layout overflows"],
                            }
                        ],
                        "evidence": ["mobile layout overflows", "no meaningful motion"],
                    },
                }
            ),
            {},
        )

    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator.chat_optimizer", fake_chat_optimizer)

    score = evaluate_response(
        {
            "id": "landing-page-rejected",
            "prompt": "Build a Vue landing page.",
            "ranked_feedback_events": [
                {
                    "ranking": ["D", "B", "C", "A"],
                    "reasoning": "D had the best structure but still needed mobile polish.",
                    "required_improvements": ["better mobile layout", "purposeful animation"],
                }
            ],
        },
        _valid_vue_bundle_response(),
        {"mode": "landing_page_v1", "artifact_contract": "vue_vite_bundle"},
    )

    assert score["hard"] == 0
    assert score["soft"] == 0.31
    assert score["contract_status"] == "passed"
    assert score["quality_status"] == "failed"
    assert score["human_feedback_alignment"]["required_improvements"] == ["better mobile layout", "purposeful animation"]
    assert score["failure"]["primary_reason"] == "mobile_responsiveness_failed"
    assert score["failure"]["optimizer_hint"].startswith("Make the hero responsive")
    assert score["failure"]["failed_checks"][0]["check"] == "landing_page_v1.mobile_responsiveness"
    assert score["stage_status"] == [{"stage": "llm_judge", "status": "failed"}]
    assert score["metadata"]["failure"]["human_reason"].startswith("Mobile layout")


def test_landing_page_judge_rejection_normalizes_malformed_failure_packet(monkeypatch):
    def fake_chat_optimizer(**kwargs):
        return (
            json.dumps(
                {
                    "hard": 0,
                    "soft": 0.4,
                    "dimension_scores": _landing_dimension_scores(),
                    "rationale": "The judge rejected the page.",
                    "fail_reason": "judge rejection",
                    "failure": {
                        "primary_reason": "judge_rejected",
                        "failed_checks": ["not a failed check", {"check": 12, "evidence": "single evidence"}],
                        "evidence": "top-level evidence",
                    },
                }
            ),
            {},
        )

    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator.chat_optimizer", fake_chat_optimizer)

    score = evaluate_response(
        {"id": "landing-page-malformed-failure", "prompt": "Build a Vue landing page."},
        _valid_vue_bundle_response(),
        {"mode": "landing_page_v1", "artifact_contract": "vue_vite_bundle"},
    )

    assert score["hard"] == 0
    assert score["failure"]["failed_checks"][0]["check"] == "12"
    assert score["failure"]["failed_checks"][0]["evidence"] == ["single evidence"]
    assert score["failure"]["evidence"] == ["top-level evidence"]


def test_landing_page_evaluator_uses_configured_evaluator_model(monkeypatch):
    captured = {}
    previous_backend = get_optimizer_backend()

    def fake_chat_optimizer(**kwargs):
        captured.update(kwargs)
        captured["optimizer_deployment"] = os.environ.get("OPTIMIZER_DEPLOYMENT")
        captured["optimizer_backend"] = get_optimizer_backend()
        return (
            json.dumps(
                {
                    "hard": 1,
                    "soft": 0.9,
                    "dimension_scores": _landing_dimension_scores(),
                    "rationale": "Evaluator model judged the page as promotable.",
                    "fail_reason": "",
                }
            ),
            {},
        )

    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator.chat_optimizer", fake_chat_optimizer)

    score = evaluate_response(
        {"prompt": "Build a Vue landing page."},
        _valid_vue_bundle_response(),
        {
            "mode": "landing_page_v1",
            "evaluator_backend": "codex",
            "evaluator_model": "gpt-evaluator",
        },
    )

    assert score["hard"] == 1
    assert captured["optimizer_backend"] == "codex"
    assert captured["optimizer_deployment"] == "gpt-evaluator"
    assert get_optimizer_backend() == previous_backend


def test_landing_page_evaluator_restores_default_deployment_when_env_unset(monkeypatch):
    captured = {}
    previous_backend = get_optimizer_backend()
    previous_deployment = os.environ.get("OPTIMIZER_DEPLOYMENT")

    def fake_chat_optimizer(**kwargs):
        captured["optimizer_deployment"] = os.environ.get("OPTIMIZER_DEPLOYMENT")
        return (
            json.dumps(
                {
                    "hard": 1,
                    "soft": 0.9,
                    "dimension_scores": _landing_dimension_scores(),
                    "rationale": "Evaluator model judged the page as promotable.",
                    "fail_reason": "",
                }
            ),
            {},
        )

    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator.chat_optimizer", fake_chat_optimizer)
    monkeypatch.delenv("OPTIMIZER_DEPLOYMENT", raising=False)
    set_optimizer_backend("openai_chat")
    try:
        score = evaluate_response(
            {"prompt": "Build a Vue landing page."},
            _valid_vue_bundle_response(),
            {
                "mode": "landing_page_v1",
                "evaluator_model": "gpt-evaluator",
            },
        )

        assert score["hard"] == 1
        assert captured["optimizer_deployment"] == "gpt-evaluator"
        assert os.environ["OPTIMIZER_DEPLOYMENT"] == default_model_for_backend("openai_chat")
    finally:
        set_optimizer_backend(previous_backend)
        if previous_deployment:
            set_optimizer_deployment(previous_deployment)


def test_landing_page_evaluator_invalid_json_fails_closed(tmp_path, monkeypatch):
    def pass_agent(*args, **kwargs):
        return _valid_vue_bundle_response()

    def fake_chat_optimizer(**kwargs):
        return ("not json", {})

    monkeypatch.setattr("skillopt.envs.gitmoot.rollout._run_agent", pass_agent)
    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator.chat_optimizer", fake_chat_optimizer)
    item = {
        "id": "landing-page-invalid",
        "prompt": "Build a Vue landing page.",
        "evaluator_config": {"mode": "landing_page_v1"},
    }

    result = process_one(item=item, skill_content="skill", out_root=str(tmp_path))

    assert result["hard"] is None
    assert result["soft"] is None
    assert result["target_status"] == "passed"
    assert result["evaluator_status"] == "failed"
    assert result["score_status"] == "unscored"
    assert result["blocker"] == "evaluator_failed"
    assert result["fail_reason"] == "landing_page_v1 judge did not return JSON"


def test_landing_page_evaluator_rejects_missing_vue_bundle_file_before_judge(monkeypatch):
    def fail_chat_optimizer(**kwargs):
        raise AssertionError("landing page judge should not run for artifact contract failures")

    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator.chat_optimizer", fail_chat_optimizer)

    score = evaluate_response(
        {
            "prompt": "Build a Vue landing page.",
            "ranked_feedback_events": [],
            "metadata": {
                "ranked_feedback_events": [
                    {
                        "ranking": ["D", "B", "C", "A"],
                        "reasoning": "D had the clearest direction.",
                        "required_improvements": ["stronger product visuals"],
                    }
                ],
            },
        },
        _valid_vue_bundle_response(files=[("package.json", '{"scripts":{"build":"vite build"}}')]),
        {"mode": "landing_page_v1", "artifact_contract": "vue_vite_bundle"},
    )

    assert score["hard"] == 0
    assert score["soft"] == 0.0
    assert score["contract_status"] == "failed"
    assert score["quality_status"] == "not_run"
    assert score["human_feedback_alignment"]["required_improvements"] == ["stronger product visuals"]
    assert score["evaluator_id"] == "landing_page_v1"
    assert score["failure"]["primary_reason"] == "vue_vite_bundle_contract_failed"
    assert score["failure"]["failed_checks"][0]["check"] == "vue_vite_bundle.required_files"
    assert "src/App.vue" in score["failure"]["evidence"][-1]
    assert score["failure"]["optimizer_hint"].startswith("Return a JSON Vue/Vite preview bundle")
    assert score["stage_status"] == [{"stage": "artifact_contract", "status": "failed"}]


def test_landing_page_evaluator_rejects_external_href_before_judge(monkeypatch):
    def fail_chat_optimizer(**kwargs):
        raise AssertionError("landing page judge should not run for artifact contract failures")

    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator.chat_optimizer", fail_chat_optimizer)

    score = evaluate_response(
        {"prompt": "Build a Vue landing page."},
        _valid_vue_bundle_response(app_vue='<template><main><a href="https://example.com">Docs</a></main></template>'),
        {"mode": "landing_page_v1", "artifact_contract": "vue_vite_bundle"},
    )

    assert score["hard"] == 0
    assert score["failure"]["failed_checks"][0]["check"] == "vue_vite_bundle.local_hrefs"
    assert "href='https://example.com'" in repr(score["failure"]["evidence"])


def test_landing_page_evaluator_rejects_external_v_bind_href_before_judge(monkeypatch):
    def fail_chat_optimizer(**kwargs):
        raise AssertionError("landing page judge should not run for artifact contract failures")

    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator.chat_optimizer", fail_chat_optimizer)

    score = evaluate_response(
        {"prompt": "Build a Vue landing page."},
        _valid_vue_bundle_response(app_vue='<template><main><a v-bind:href="\'https://example.com\'">Docs</a></main></template>'),
        {"mode": "landing_page_v1", "artifact_contract": "vue_vite_bundle"},
    )

    assert score["hard"] == 0
    assert score["failure"]["failed_checks"][0]["check"] == "vue_vite_bundle.local_hrefs"
    assert "href='https://example.com'" in repr(score["failure"]["evidence"])


def test_landing_page_evaluator_rejects_dynamic_import_before_judge(monkeypatch):
    def fail_chat_optimizer(**kwargs):
        raise AssertionError("landing page judge should not run for artifact contract failures")

    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator.chat_optimizer", fail_chat_optimizer)

    score = evaluate_response(
        {"prompt": "Build a Vue landing page."},
        _valid_vue_bundle_response(
            app_vue='<template><main><button @click="import(\'https://example.com/x.js\')">Load</button></main></template>'
        ),
        {"mode": "landing_page_v1", "artifact_contract": "vue_vite_bundle"},
    )

    assert score["hard"] == 0
    assert score["failure"]["failed_checks"][0]["check"] == "vue_vite_bundle.app_vue.dynamic_import"


def test_landing_page_evaluator_rejects_build_script_that_only_mentions_vite(monkeypatch):
    def fail_chat_optimizer(**kwargs):
        raise AssertionError("landing page judge should not run for artifact contract failures")

    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator.chat_optimizer", fail_chat_optimizer)

    score = evaluate_response(
        {"prompt": "Build a Vue landing page."},
        _valid_vue_bundle_response(package_json='{"scripts":{"build":"echo vite build"}}'),
        {"mode": "landing_page_v1", "artifact_contract": "vue_vite_bundle"},
    )

    assert score["hard"] == 0
    assert score["failure"]["failed_checks"][0]["check"] == "vue_vite_bundle.package_json.build"


def test_landing_page_evaluator_rejects_non_string_file_content_before_judge(monkeypatch):
    def fail_chat_optimizer(**kwargs):
        raise AssertionError("landing page judge should not run for artifact contract failures")

    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator.chat_optimizer", fail_chat_optimizer)

    score = evaluate_response(
        {"prompt": "Build a Vue landing page."},
        _valid_vue_bundle_response(files=[
            ("package.json", '{"scripts":{"build":"vite build"}}'),
            ("index.html", '<div id="app"></div><script type="module" src="/src/main.js"></script>'),
            ("src/main.js", "import { createApp } from 'vue'; import App from './App.vue'; createApp(App).mount('#app');"),
            ("src/App.vue", None),
        ]),
        {"mode": "landing_page_v1", "artifact_contract": "vue_vite_bundle"},
    )

    assert score["hard"] == 0
    assert score["failure"]["failed_checks"][0]["check"] == "vue_vite_bundle.files"
    assert "content is NoneType" in score["failure"]["evidence"][0]


def test_landing_page_evaluator_rejects_unsafe_bundle_path_before_judge(monkeypatch):
    def fail_chat_optimizer(**kwargs):
        raise AssertionError("landing page judge should not run for artifact contract failures")

    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator.chat_optimizer", fail_chat_optimizer)

    score = evaluate_response(
        {"prompt": "Build a Vue landing page."},
        _valid_vue_bundle_response(files=[
            ("package.json", '{"scripts":{"build":"vite build"}}'),
            ("index.html", '<div id="app"></div><script type="module" src="/src/main.js"></script>'),
            ("src/main.js", "import { createApp } from 'vue'; import App from './App.vue'; createApp(App).mount('#app');"),
            ("src/App.vue", "<template><main><footer>Footer</footer></main></template>"),
            ("../escape.js", "window.escape = true;"),
        ]),
        {"mode": "landing_page_v1", "artifact_contract": "vue_vite_bundle"},
    )

    assert score["hard"] == 0
    assert score["failure"]["failed_checks"][0]["check"] == "vue_vite_bundle.file_path"


def test_landing_page_evaluator_rejects_normalized_unsafe_bundle_paths_before_judge(monkeypatch):
    def fail_chat_optimizer(**kwargs):
        raise AssertionError("landing page judge should not run for artifact contract failures")

    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator.chat_optimizer", fail_chat_optimizer)

    score = evaluate_response(
        {"prompt": "Build a Vue landing page."},
        _valid_vue_bundle_response(files=[
            ("package.json", '{"scripts":{"build":"vite build"}}'),
            ("index.html", '<div id="app"></div><script type="module" src="/src/main.js"></script>'),
            ("src/main.js", "import { createApp } from 'vue'; import App from './App.vue'; createApp(App).mount('#app');"),
            ("src/App.vue", "<template><main><footer>Footer</footer></main></template>"),
            ("./src/escape.js", "window.escape = true;"),
            ("src//double-slash.js", "window.escape = true;"),
        ]),
        {"mode": "landing_page_v1", "artifact_contract": "vue_vite_bundle"},
    )

    failed_paths = " ".join(score["failure"]["evidence"])
    assert score["hard"] == 0
    assert "./src/escape.js" in failed_paths
    assert "src//double-slash.js" in failed_paths


def test_landing_page_evaluator_allows_import_copy_and_resource_hrefs(monkeypatch):
    called = False

    def fake_chat_optimizer(**kwargs):
        nonlocal called
        called = True
        return (
            json.dumps(
                {
                    "hard": 1,
                    "soft": 0.9,
                    "dimension_scores": _landing_dimension_scores(),
                    "rationale": "Bundle passed artifact checks and reached the judge.",
                    "fail_reason": "",
                }
            ),
            {},
        )

    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator.chat_optimizer", fake_chat_optimizer)
    response = _valid_vue_bundle_response(
        app_vue=(
            "<template>\n"
            "  <main>\n"
            "    <h1>Import issues from GitHub</h1>\n"
            "    Import issues into a cleaner landing page.\n"
            "    <a data-href=\"https://example.com\" :href=\"'#start'\">Start</a>\n"
            "  </main>\n"
            "</template>"
        ),
        index_html='<div id="app"></div><link rel="icon" href="/vite.svg"><script type="module" src="/src/main.js"></script>',
    )

    score = evaluate_response(
        {"prompt": "Build a Vue landing page."},
        response,
        {"mode": "landing_page_v1", "artifact_contract": "vue_vite_bundle"},
    )

    assert called is True
    assert score["hard"] == 1


def test_landing_page_evaluator_keeps_text_scoring_when_bundle_not_required(monkeypatch):
    called = False

    def fake_chat_optimizer(**kwargs):
        nonlocal called
        called = True
        return (
            json.dumps(
                {
                    "hard": 1,
                    "soft": 0.88,
                    "dimension_scores": _landing_dimension_scores(),
                    "rationale": "Text feedback response is suitable.",
                    "fail_reason": "",
                }
            ),
            {},
        )

    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator.chat_optimizer", fake_chat_optimizer)

    score = evaluate_response(
        {"prompt": "Build a Vue landing page."},
        "Preserve option D and improve the hero animation.",
        {"mode": "landing_page_v1"},
    )

    assert called is True
    assert score["hard"] == 1


def test_landing_page_evaluator_render_smoke_failure_skips_judge(monkeypatch):
    def fail_chat_optimizer(**kwargs):
        raise AssertionError("landing page judge should not run for render smoke failures")

    def fail_render(*args, **kwargs):
        return {
            "hard": 0,
            "soft": 0.0,
            "fail_reason": "mobile render has horizontal overflow",
            "failure": {
                "primary_reason": "vue_render_smoke_failed",
                "optimizer_hint": "Fix the overflow before visual judging.",
                "failed_checks": [{"check": "vue_render_smoke.horizontal_overflow"}],
            },
            "stage_status": [{"stage": "render_smoke", "status": "failed"}],
            "metadata": {"evaluator": "landing_page_v1"},
        }

    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator.chat_optimizer", fail_chat_optimizer)
    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator._run_vue_render_smoke", fail_render)

    score = evaluate_response(
        {"id": "landing-page-render-fail", "prompt": "Build a Vue landing page."},
        _valid_vue_bundle_response(),
        {"mode": "landing_page_v1", "artifact_contract": "vue_vite_bundle", "require_vue_render_smoke": True},
    )

    assert score["hard"] == 0
    assert score["failure"]["primary_reason"] == "vue_render_smoke_failed"
    assert score["stage_status"][0]["stage"] == "render_smoke"


def test_landing_page_evaluator_required_render_smoke_implies_vue_bundle_contract(monkeypatch):
    def fail_chat_optimizer(**kwargs):
        raise AssertionError("landing page judge should not run before required render smoke validation")

    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator.chat_optimizer", fail_chat_optimizer)

    score = evaluate_response(
        {"id": "landing-page-render-contract", "prompt": "Build a Vue landing page."},
        "This is not a Vue/Vite bundle.",
        {"mode": "landing_page_v1", "require_vue_render_smoke": True},
    )

    assert score["hard"] == 0
    assert score["failure"]["primary_reason"] == "vue_vite_bundle_contract_failed"
    assert score["failure"]["failed_checks"][0]["check"] == "vue_vite_bundle.json"


def test_landing_page_evaluator_rejects_render_imports_outside_bundle_before_build(monkeypatch):
    def fail_chat_optimizer(**kwargs):
        raise AssertionError("landing page judge should not run for required render smoke failures")

    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator.chat_optimizer", fail_chat_optimizer)

    score = evaluate_response(
        {"id": "landing-page-unsafe-import", "prompt": "Build a Vue landing page."},
        _valid_vue_bundle_response(files=[
            ("package.json", '{"scripts":{"build":"vite build"}}'),
            ("index.html", '<div id="app"></div><script type="module" src="/src/main.js"></script>'),
            ("src/main.js", "import secret from '../../../../etc/passwd?raw'; window.secret = secret;"),
            ("src/App.vue", "<template><main><footer>Footer</footer></main></template>"),
        ]),
        {"mode": "landing_page_v1", "artifact_contract": "vue_vite_bundle", "require_vue_render_smoke": True},
    )

    assert score["hard"] == 0
    assert score["failure"]["failed_checks"][0]["check"] == "vue_render_smoke.import_safety"
    assert "../../../../etc/passwd?raw" in score["failure"]["evidence"][0]


def test_landing_page_evaluator_rejects_compact_render_import_syntax_before_build(monkeypatch):
    def fail_chat_optimizer(**kwargs):
        raise AssertionError("landing page judge should not run for required render smoke failures")

    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator.chat_optimizer", fail_chat_optimizer)

    score = evaluate_response(
        {"id": "landing-page-compact-import", "prompt": "Build a Vue landing page."},
        _valid_vue_bundle_response(files=[
            ("package.json", '{"scripts":{"build":"vite build"}}'),
            ("index.html", '<div id="app"></div><script type="module" src="/src/main.js"></script>'),
            ("src/main.js", "import{readFileSync}from'fs'; window.secret = readFileSync('/etc/passwd', 'utf8');"),
            ("src/App.vue", "<template><main><footer>Footer</footer></main></template>"),
        ]),
        {"mode": "landing_page_v1", "artifact_contract": "vue_vite_bundle", "require_vue_render_smoke": True},
    )

    assert score["hard"] == 0
    assert score["failure"]["failed_checks"][0]["check"] == "vue_render_smoke.import_safety"
    assert "fs" in score["failure"]["evidence"][0]


def test_landing_page_evaluator_rejects_template_literal_dynamic_import_before_build(monkeypatch):
    def fail_chat_optimizer(**kwargs):
        raise AssertionError("landing page judge should not run for required render smoke failures")

    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator.chat_optimizer", fail_chat_optimizer)

    score = evaluate_response(
        {"id": "landing-page-template-import", "prompt": "Build a Vue landing page."},
        _valid_vue_bundle_response(files=[
            ("package.json", '{"scripts":{"build":"vite build"}}'),
            ("index.html", '<div id="app"></div><script type="module" src="/src/main.js"></script>'),
            ("src/main.js", "const secret = await import(`../../../../etc/passwd?raw`); window.secret = secret;"),
            ("src/App.vue", "<template><main><footer>Footer</footer></main></template>"),
        ]),
        {"mode": "landing_page_v1", "artifact_contract": "vue_vite_bundle", "require_vue_render_smoke": True},
    )

    assert score["hard"] == 0
    assert score["failure"]["failed_checks"][0]["check"] == "vue_render_smoke.import_safety"
    assert "../../../../etc/passwd?raw" in score["failure"]["evidence"][0]


def test_landing_page_evaluator_rejects_import_meta_glob_before_build(monkeypatch):
    def fail_chat_optimizer(**kwargs):
        raise AssertionError("landing page judge should not run for required render smoke failures")

    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator.chat_optimizer", fail_chat_optimizer)

    score = evaluate_response(
        {"id": "landing-page-import-meta-glob", "prompt": "Build a Vue landing page."},
        _valid_vue_bundle_response(files=[
            ("package.json", '{"scripts":{"build":"vite build"}}'),
            ("index.html", '<div id="app"></div><script type="module" src="/src/main.js"></script>'),
            ("src/main.js", "const secrets = import.meta.glob(`../../../../etc/*`, { query: '?raw' }); window.secrets = secrets;"),
            ("src/App.vue", "<template><main><footer>Footer</footer></main></template>"),
        ]),
        {"mode": "landing_page_v1", "artifact_contract": "vue_vite_bundle", "require_vue_render_smoke": True},
    )

    assert score["hard"] == 0
    assert score["failure"]["failed_checks"][0]["check"] == "vue_render_smoke.import_safety"
    assert "../../../../etc/*" in score["failure"]["evidence"][0]


def test_landing_page_evaluator_rejects_inline_html_module_imports_before_build(monkeypatch):
    def fail_chat_optimizer(**kwargs):
        raise AssertionError("landing page judge should not run for required render smoke failures")

    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator.chat_optimizer", fail_chat_optimizer)

    score = evaluate_response(
        {"id": "landing-page-html-import", "prompt": "Build a Vue landing page."},
        _valid_vue_bundle_response(files=[
            ("package.json", '{"scripts":{"build":"vite build"}}'),
            (
                "index.html",
                "<div id=\"app\"></div><script type=\"module\">import secret from '../../../../etc/passwd?raw';</script>",
            ),
            ("src/main.js", "import { createApp } from 'vue'; import App from './App.vue'; createApp(App).mount('#app');"),
            ("src/App.vue", "<template><main><footer>Footer</footer></main></template>"),
        ]),
        {"mode": "landing_page_v1", "artifact_contract": "vue_vite_bundle", "require_vue_render_smoke": True},
    )

    assert score["hard"] == 0
    assert score["failure"]["failed_checks"][0]["check"] == "vue_render_smoke.import_safety"
    assert "../../../../etc/passwd?raw" in score["failure"]["evidence"][0]


def test_vue_render_smoke_allows_public_asset_root_urls(tmp_path, monkeypatch):
    def fake_prepare_deps(work_path, timeout):
        del timeout
        vite_bin = work_path / "node_modules" / ".bin" / "vite"
        vite_bin.parent.mkdir(parents=True)
        vite_bin.write_text("#!/bin/sh\n", encoding="utf-8")
        return vite_bin

    def fake_run_render_command(command, cwd, timeout):
        del command, timeout
        dist_index = cwd / "dist" / "index.html"
        dist_index.parent.mkdir(parents=True, exist_ok=True)
        dist_index.write_text("<div id=\"app\"></div>", encoding="utf-8")

    def fake_smoke_check_dist(sync_playwright, dist_index, artifact_dir):
        del sync_playwright, dist_index, artifact_dir
        return {"hard": 1, "soft": 1.0, "metadata": {}}

    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator._import_sync_playwright", lambda: object())
    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator._prepare_trusted_vue_render_deps", fake_prepare_deps)
    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator._run_render_command", fake_run_render_command)
    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator._smoke_check_dist", fake_smoke_check_dist)

    score = _run_vue_render_smoke(
        _valid_vue_bundle_response(files=[
            ("package.json", '{"scripts":{"build":"vite build"}}'),
            ("index.html", '<div id="app"></div><script type="module" src="/src/main.js"></script>'),
            ("src/main.js", "import './style.css'; import { createApp } from 'vue'; import App from './App.vue'; createApp(App).mount('#app');"),
            ("src/App.vue", "<template><main><footer>Footer</footer></main></template>"),
            ("src/style.css", "main { background-image: url('/logo.svg'); }"),
            ("public/logo.svg", "<svg></svg>"),
        ]),
        {"id": "public-asset"},
        {"_render_smoke_required": True, "render_artifact_dir": str(tmp_path / "artifacts")},
    )

    assert score["hard"] == 1


def test_vue_render_smoke_allows_relative_css_asset_urls(tmp_path, monkeypatch):
    def fake_prepare_deps(work_path, timeout):
        del timeout
        vite_bin = work_path / "node_modules" / ".bin" / "vite"
        vite_bin.parent.mkdir(parents=True)
        vite_bin.write_text("#!/bin/sh\n", encoding="utf-8")
        return vite_bin

    def fake_run_render_command(command, cwd, timeout):
        del command, timeout
        dist_index = cwd / "dist" / "index.html"
        dist_index.parent.mkdir(parents=True, exist_ok=True)
        dist_index.write_text("<div id=\"app\"></div>", encoding="utf-8")

    def fake_smoke_check_dist(sync_playwright, dist_index, artifact_dir):
        del sync_playwright, dist_index, artifact_dir
        return {"hard": 1, "soft": 1.0, "metadata": {}}

    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator._import_sync_playwright", lambda: object())
    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator._prepare_trusted_vue_render_deps", fake_prepare_deps)
    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator._run_render_command", fake_run_render_command)
    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator._smoke_check_dist", fake_smoke_check_dist)

    score = _run_vue_render_smoke(
        _valid_vue_bundle_response(files=[
            ("package.json", '{"scripts":{"build":"vite build"}}'),
            ("index.html", '<div id="app"></div><script type="module" src="/src/main.js"></script>'),
            ("src/main.js", "import './style.css'; import { createApp } from 'vue'; import App from './App.vue'; createApp(App).mount('#app');"),
            ("src/App.vue", "<template><main><footer>Footer</footer></main></template>"),
            ("src/style.css", 'main { background-image: url("hero.svg"); }'),
            ("src/hero.svg", "<svg></svg>"),
        ]),
        {"id": "relative-css-asset"},
        {"_render_smoke_required": True, "render_artifact_dir": str(tmp_path / "artifacts")},
    )

    assert score["hard"] == 1


def test_landing_page_evaluator_optional_render_smoke_failure_reaches_judge(monkeypatch):
    called = False

    def fail_render(*args, **kwargs):
        assert kwargs == {}
        config = args[2]
        assert config["_render_smoke_required"] is False
        return {
            "hard": 0,
            "soft": 0.0,
            "fail_reason": "mobile render has horizontal overflow",
            "dimension_scores": {"render_smoke": 0.0},
            "failure": {
                "primary_reason": "vue_render_smoke_failed",
                "optimizer_hint": "Fix the overflow before visual judging.",
                "failed_checks": [{"check": "vue_render_smoke.horizontal_overflow"}],
            },
            "stage_status": [{"stage": "render_smoke", "status": "failed"}],
            "metadata": {
                "dimension_scores": {"render_smoke": 0.0},
                "render_smoke": {
                    "failure": {
                        "failed_checks": [{"check": "vue_render_smoke.horizontal_overflow"}],
                    }
                },
                "stage_status": [{"stage": "render_smoke", "status": "failed"}],
            },
        }

    def fake_chat_optimizer(**kwargs):
        nonlocal called
        called = True
        return (
            json.dumps(
                {
                    "hard": 1,
                    "soft": 0.84,
                    "dimension_scores": _landing_dimension_scores(),
                    "rationale": "Optional render smoke failed, but the visual judge still ran.",
                    "fail_reason": "",
                    "contract_status": "passed",
                }
            ),
            {},
        )

    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator._run_vue_render_smoke", fail_render)
    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator.chat_optimizer", fake_chat_optimizer)

    score = evaluate_response(
        {"id": "landing-page-optional-render-fail", "prompt": "Build a Vue landing page."},
        _valid_vue_bundle_response(),
        {
            "mode": "landing_page_v1",
            "artifact_contract": "vue_vite_bundle",
            "checks": [{"id": "render_smoke", "type": "playwright", "required": False}],
        },
    )

    assert called is True
    assert score["hard"] == 1
    assert score["contract_status"] == "failed"
    assert score["quality_status"] == "passed"
    assert score["metadata"]["dimension_scores"]["render_smoke"] == 0.0
    assert score["metadata"]["render_smoke"]["failure"]["failed_checks"][0]["check"] == "vue_render_smoke.horizontal_overflow"
    assert score["stage_status"][0] == {"stage": "render_smoke", "status": "failed"}


def test_landing_page_evaluator_ignores_non_render_playwright_check(monkeypatch):
    called_render = False
    called_judge = False

    def fail_render(*args, **kwargs):
        nonlocal called_render
        called_render = True
        raise AssertionError("preview_capture should not enable render_smoke")

    def fake_chat_optimizer(**kwargs):
        nonlocal called_judge
        called_judge = True
        return (
            json.dumps(
                {
                    "hard": 1,
                    "soft": 0.84,
                    "dimension_scores": _landing_dimension_scores(),
                    "rationale": "The non-render playwright check did not run render smoke.",
                    "fail_reason": "",
                }
            ),
            {},
        )

    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator._run_vue_render_smoke", fail_render)
    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator.chat_optimizer", fake_chat_optimizer)

    score = evaluate_response(
        {"id": "landing-page-preview-only", "prompt": "Build a Vue landing page."},
        _valid_vue_bundle_response(),
        {
            "mode": "landing_page_v1",
            "artifact_contract": "vue_vite_bundle",
            "checks": [{"id": "preview_capture", "type": "playwright", "required": True}],
        },
    )

    assert called_render is False
    assert called_judge is True
    assert score["hard"] == 1


def test_landing_page_evaluator_required_render_smoke_wins_over_earlier_optional(monkeypatch):
    def fail_chat_optimizer(**kwargs):
        raise AssertionError("landing page judge should not run for required render smoke failures")

    def fail_render(*args, **kwargs):
        config = args[2]
        assert config["_render_smoke_required"] is True
        return {
            "hard": 0,
            "soft": 0.0,
            "fail_reason": "render smoke failed",
            "failure": {"primary_reason": "vue_render_smoke_failed"},
            "stage_status": [{"stage": "render_smoke", "status": "failed"}],
            "metadata": {"stage_status": [{"stage": "render_smoke", "status": "failed"}]},
        }

    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator.chat_optimizer", fail_chat_optimizer)
    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator._run_vue_render_smoke", fail_render)

    score = evaluate_response(
        {"id": "landing-page-required-render-after-optional", "prompt": "Build a Vue landing page."},
        _valid_vue_bundle_response(),
        {
            "mode": "landing_page_v1",
            "artifact_contract": "vue_vite_bundle",
            "checks": [
                {"id": "preview_capture", "type": "playwright", "required": False},
                {"id": "render_smoke", "type": "playwright", "required": True},
            ],
        },
    )

    assert score["hard"] == 0
    assert score["failure"]["primary_reason"] == "vue_render_smoke_failed"


def test_landing_page_evaluator_render_smoke_success_attaches_screenshots(monkeypatch):
    def pass_render(*args, **kwargs):
        return {
            "hard": 1,
            "soft": 1.0,
            "dimension_scores": {"render_smoke": 1.0},
            "stage_status": [{"stage": "render_smoke", "status": "passed"}],
            "metadata": {
                "dimension_scores": {"render_smoke": 1.0},
                "stage_status": [{"stage": "render_smoke", "status": "passed"}],
                "render_smoke": {"screenshots": [{"label": "desktop", "path": "/tmp/desktop.png"}]},
            },
        }

    def fake_chat_optimizer(**kwargs):
        return (
            json.dumps(
                {
                    "hard": 1,
                    "soft": 0.9,
                    "dimension_scores": _landing_dimension_scores(),
                    "rationale": "Render passed before visual judging.",
                    "fail_reason": "",
                }
            ),
            {},
        )

    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator._run_vue_render_smoke", pass_render)
    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator.chat_optimizer", fake_chat_optimizer)

    score = evaluate_response(
        {"id": "landing-page-render-pass", "prompt": "Build a Vue landing page."},
        _valid_vue_bundle_response(),
        {"mode": "landing_page_v1", "artifact_contract": "vue_vite_bundle", "require_vue_render_smoke": True},
    )

    assert score["hard"] == 1
    assert score["metadata"]["render_smoke"]["screenshots"][0]["label"] == "desktop"
    assert score["metadata"]["dimension_scores"]["render_smoke"] == 1.0
    assert score["stage_status"][0] == {"stage": "render_smoke", "status": "passed"}


def test_landing_page_evaluator_required_render_smoke_fails_when_playwright_missing(monkeypatch):
    def fail_chat_optimizer(**kwargs):
        raise AssertionError("landing page judge should not run when required Playwright is missing")

    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator.chat_optimizer", fail_chat_optimizer)
    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator._import_sync_playwright", lambda: None)

    score = evaluate_response(
        {"id": "landing-page-render-unavailable", "prompt": "Build a Vue landing page."},
        _valid_vue_bundle_response(),
        {"mode": "landing_page_v1", "artifact_contract": "vue_vite_bundle", "require_vue_render_smoke": True},
    )

    assert score["hard"] == 0
    assert score["failure"]["primary_reason"] == "vue_render_smoke_environment_unavailable"
    assert score["failure"]["failed_checks"][0]["check"] == "vue_render_smoke.environment_unavailable"
    assert "environment is unavailable" in score["fail_reason"]


def test_landing_page_evaluator_optional_render_smoke_skips_when_playwright_missing(monkeypatch):
    called = False

    def fake_chat_optimizer(**kwargs):
        nonlocal called
        called = True
        return (
            json.dumps(
                {
                    "hard": 1,
                    "soft": 0.9,
                    "dimension_scores": _landing_dimension_scores(),
                    "rationale": "Optional render smoke was skipped, judge still ran.",
                    "fail_reason": "",
                }
            ),
            {},
        )

    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator.chat_optimizer", fake_chat_optimizer)
    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator._import_sync_playwright", lambda: None)

    score = evaluate_response(
        {"id": "landing-page-render-optional", "prompt": "Build a Vue landing page."},
        _valid_vue_bundle_response(),
        {
            "mode": "landing_page_v1",
            "artifact_contract": "vue_vite_bundle",
            "checks": [{"id": "render_smoke", "type": "playwright", "required": False}],
        },
    )

    assert called is True
    assert score["hard"] == 1
    assert score["metadata"]["render_smoke"]["skipped"] is True
    assert "playwright is unavailable" in score["metadata"]["render_smoke"]["reason"]
    assert score["stage_status"][0]["status"] == "skipped"


def test_vue_render_smoke_required_reports_environment_failure_when_npm_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("GITMOOT_RENDER_DEPS_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator._import_sync_playwright", lambda: object())
    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator.shutil.which", lambda command: None if command == "npm" else f"/bin/{command}")

    score = _run_vue_render_smoke(
        _valid_vue_bundle_response(),
        {"id": "missing-npm"},
        {"_render_smoke_required": True, "render_artifact_dir": str(tmp_path / "artifacts")},
    )

    assert score["hard"] == 0
    assert score["failure"]["primary_reason"] == "vue_render_smoke_environment_unavailable"
    assert score["failure"]["failed_checks"][0]["check"] == "vue_render_smoke.environment_unavailable"
    assert "npm is not available" in score["failure"]["evidence"][0]


def test_vue_render_smoke_optional_skips_when_npm_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("GITMOOT_RENDER_DEPS_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator._import_sync_playwright", lambda: object())
    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator.shutil.which", lambda command: None if command == "npm" else f"/bin/{command}")

    score = _run_vue_render_smoke(
        _valid_vue_bundle_response(),
        {"id": "missing-npm-optional"},
        {"_render_smoke_required": False, "render_artifact_dir": str(tmp_path / "artifacts")},
    )

    assert score["hard"] == 1
    assert score["metadata"]["render_smoke"]["skipped"] is True
    assert "npm is not available" in score["metadata"]["render_smoke"]["reason"]


def test_vue_render_smoke_required_reports_environment_failure_when_chromium_missing(tmp_path, monkeypatch):
    class FakeChromium:
        def launch(self):
            raise RuntimeError("Executable doesn't exist")

    class FakePlaywright:
        chromium = FakeChromium()

    class FakeSyncPlaywright:
        def __enter__(self):
            return FakePlaywright()

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_run_render_command(command, cwd, timeout):
        del command, timeout
        dist_index = cwd / "dist" / "index.html"
        dist_index.parent.mkdir(parents=True, exist_ok=True)
        dist_index.write_text("<div id=\"app\"></div>", encoding="utf-8")

    trusted_vite = tmp_path / "node_modules" / ".bin" / "vite"
    trusted_vite.parent.mkdir(parents=True)
    trusted_vite.write_text("#!/bin/sh\n", encoding="utf-8")

    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator._import_sync_playwright", lambda: FakeSyncPlaywright)
    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator._prepare_trusted_vue_render_deps", lambda work_path, timeout: trusted_vite)
    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator._run_render_command", fake_run_render_command)

    score = _run_vue_render_smoke(
        _valid_vue_bundle_response(),
        {"id": "missing-chromium"},
        {"_render_smoke_required": True, "render_artifact_dir": str(tmp_path / "artifacts")},
    )

    assert score["hard"] == 0
    assert score["failure"]["primary_reason"] == "vue_render_smoke_environment_unavailable"
    assert "Playwright Chromium is not available" in score["failure"]["evidence"][0]


def test_vue_render_workspace_uses_submitted_runtime_bundle_files(tmp_path):
    file_map = {
        "package.json": '{"scripts":{"build":"vite build --emptyOutDir"}}',
        "index.html": '<main id="submitted"></main><script type="module" src="/src/main.js"></script>',
        "src/main.js": "import './style.css'; import Hero from './components/Hero.vue'; window.__submittedMain = Hero;",
        "src/App.vue": "<template><footer>Submitted app</footer></template>",
        "src/style.css": "body { margin: 0; }",
        "src/components/Hero.vue": "<template><h1>Hero</h1></template>",
        "public/logo.svg": "<svg></svg>",
    }

    _write_vue_render_workspace(tmp_path, file_map)

    assert '"@vitejs/plugin-vue"' in (tmp_path / "package.json").read_text(encoding="utf-8")
    assert (tmp_path / "index.html").read_text(encoding="utf-8") == file_map["index.html"]
    assert (tmp_path / "src" / "main.js").read_text(encoding="utf-8") == file_map["src/main.js"]
    assert (tmp_path / "src" / "App.vue").read_text(encoding="utf-8") == file_map["src/App.vue"]
    assert (tmp_path / "src" / "style.css").read_text(encoding="utf-8") == file_map["src/style.css"]
    assert (tmp_path / "src" / "components" / "Hero.vue").read_text(encoding="utf-8") == file_map["src/components/Hero.vue"]
    assert (tmp_path / "public" / "logo.svg").read_text(encoding="utf-8") == file_map["public/logo.svg"]
    assert "plugins: [vue()]" in (tmp_path / "vite.config.mjs").read_text(encoding="utf-8")


def test_vue_render_workspace_blocks_submitted_build_tool_configs(tmp_path):
    file_map = {
        "package.json": '{"scripts":{"build":"vite build && curl https://example.com/pwn"}}',
        "index.html": '<div id="app"></div><script type="module" src="/src/main.js"></script>',
        "src/main.js": "import { createApp } from 'vue'; import App from './App.vue'; createApp(App).mount('#app');",
        "src/App.vue": "<template><main>App</main></template>",
        "postcss.config.cjs": "throw new Error('must not run')",
        "tailwind.config.js": "throw new Error('must not run')",
        "vite.config.js": "throw new Error('must not run')",
        ".babelrc": '{"plugins":["must-not-load"]}',
    }

    _write_vue_render_workspace(tmp_path, file_map)

    assert '"@vitejs/plugin-vue"' in (tmp_path / "package.json").read_text(encoding="utf-8")
    assert not (tmp_path / "postcss.config.cjs").exists()
    assert not (tmp_path / "tailwind.config.js").exists()
    assert not (tmp_path / "vite.config.js").exists()
    assert not (tmp_path / ".babelrc").exists()


def test_trusted_vue_render_dependencies_are_pinned():
    package = json.loads(TRUSTED_VUE_RENDER_PACKAGE_JSON)

    assert package["dependencies"] == {
        "@vitejs/plugin-vue": "5.2.1",
        "vite": "5.4.11",
        "vue": "3.5.13",
    }


def test_trusted_vue_render_dependencies_are_reused_across_workspaces(tmp_path, monkeypatch):
    commands: list[Path] = []
    monkeypatch.setenv("GITMOOT_RENDER_DEPS_CACHE", str(tmp_path / "cache"))

    def fake_run_render_command(command, cwd, timeout):
        del command, timeout
        commands.append(cwd)
        vite_bin = cwd / "node_modules" / ".bin" / "vite"
        vite_bin.parent.mkdir(parents=True, exist_ok=True)
        vite_bin.write_text("#!/bin/sh\n", encoding="utf-8")

    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator._run_render_command", fake_run_render_command)

    first_work = tmp_path / "first"
    second_work = tmp_path / "second"
    first_work.mkdir()
    second_work.mkdir()

    first_vite = _prepare_trusted_vue_render_deps(first_work, timeout=30)
    second_vite = _prepare_trusted_vue_render_deps(second_work, timeout=30)

    assert first_vite == second_vite
    assert len(commands) == 1
    assert (first_work / "node_modules").is_symlink()
    assert (second_work / "node_modules").is_symlink()


def test_trusted_vue_render_dependency_cache_defaults_to_user_cache(tmp_path, monkeypatch):
    monkeypatch.delenv("GITMOOT_RENDER_DEPS_CACHE", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg-cache"))

    deps_dir = _trusted_vue_render_deps_cache_dir()

    assert str(deps_dir).startswith(str(tmp_path / "xdg-cache" / "gitmoot-skillopt" / "vue-render-deps"))


def test_vue_render_smoke_uses_trusted_vite_command_not_submitted_build_script(tmp_path, monkeypatch):
    commands: list[list[str]] = []
    monkeypatch.setenv("GITMOOT_RENDER_DEPS_CACHE", str(tmp_path / "cache"))

    def fake_run_render_command(command, cwd, timeout):
        del timeout
        commands.append([str(part) for part in command])
        if command[0] == "npm":
            vite_bin = cwd / "node_modules" / ".bin" / "vite"
            vite_bin.parent.mkdir(parents=True)
            vite_bin.write_text("#!/bin/sh\n", encoding="utf-8")
        else:
            dist_index = cwd / "dist" / "index.html"
            dist_index.parent.mkdir(parents=True)
            dist_index.write_text("<div id=\"app\"></div>", encoding="utf-8")

    def fake_smoke_check_dist(sync_playwright, dist_index, artifact_dir):
        del sync_playwright, dist_index, artifact_dir
        return {"hard": 1, "soft": 1.0, "metadata": {}}

    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator._run_render_command", fake_run_render_command)
    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator._smoke_check_dist", fake_smoke_check_dist)
    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator._import_sync_playwright", lambda: object())

    score = _run_vue_render_smoke(
        _valid_vue_bundle_response(package_json='{"scripts":{"build":"vite build && curl https://example.com/pwn"}}'),
        {"id": "trusted-command"},
        {"_render_smoke_required": True, "render_artifact_dir": str(tmp_path / "artifacts")},
    )

    assert score["hard"] == 1
    assert commands[0][:3] == ["npm", "install", "--ignore-scripts"]
    assert "npm run build" not in repr(commands)
    assert "curl" not in repr(commands)
    assert commands[1][0].endswith("node_modules/.bin/vite")
    assert commands[1][1] == "build"
    assert commands[1][2] == "--config"
    assert commands[1][3].endswith("vite.config.mjs")


def test_render_smoke_blocks_external_browser_requests():
    class FakeRequest:
        def __init__(self, url):
            self.url = url

    class FakeRoute:
        def __init__(self, url):
            self.request = FakeRequest(url)
            self.action = ""

        def continue_(self):
            self.action = "continue"

        def abort(self):
            self.action = "abort"

    class FakeContext:
        def __init__(self):
            self.pattern = ""
            self.handler = None
            self.init_script = ""

        def route(self, pattern, handler):
            self.pattern = pattern
            self.handler = handler

        def add_init_script(self, script):
            self.init_script = script

    context = FakeContext()
    _block_external_browser_requests(context, Path("/tmp/dist"))

    file_route = FakeRoute("file:///tmp/dist/index.html")
    outside_file_route = FakeRoute("file:///etc/passwd")
    https_route = FakeRoute("https://example.com/image.png")
    internal_route = FakeRoute("http://127.0.0.1:8080/secret")
    assert context.pattern == "**/*"
    assert context.handler is not None
    assert "WebSocket connections are blocked" in context.init_script
    context.handler(file_route)
    context.handler(outside_file_route)
    context.handler(https_route)
    context.handler(internal_route)

    assert file_route.action == "continue"
    assert outside_file_route.action == "abort"
    assert https_route.action == "abort"
    assert internal_route.action == "abort"


def test_render_smoke_fallback_artifact_dir_sanitizes_item_id():
    artifact_dir = _render_artifact_dir(
        {"id": "../../outside/item"},
        {"_render_smoke_response_hash": "abc123"},
    )

    expected_root = os.path.join(tempfile.gettempdir(), "gitmoot-render-smoke")
    assert str(artifact_dir).startswith(expected_root)
    assert ".." not in artifact_dir.parts
    assert artifact_dir.name == "abc123"


def _landing_dimension_scores():
    return {
        "mobile_responsiveness": 0.9,
        "footer_presence_clarity": 0.8,
        "hero_quality": 0.9,
        "cta_clarity": 0.85,
        "visual_images_relevance": 0.75,
        "animation_motion_quality": 0.7,
        "text_overlap_readability": 0.95,
        "ranked_strength_preservation": 0.85,
    }


def _valid_vue_bundle_response(
    *,
    app_vue: str = "<template><main><a href=\"#hero\">Hero</a><footer>Footer</footer></main></template>",
    index_html: str = '<div id="app"></div><script type="module" src="/src/main.js"></script>',
    package_json: str = '{"scripts":{"build":"vite build"}}',
    files: list[tuple[str, str]] | None = None,
) -> str:
    if files is None:
        files = [
            ("package.json", package_json),
            ("index.html", index_html),
            ("src/main.js", "import { createApp } from 'vue'; import App from './App.vue'; createApp(App).mount('#app');"),
            ("src/App.vue", app_vue),
        ]
    return json.dumps(
        {
            "renderer": "vue-vite",
            "build_command": "npm run build",
            "dist_dir": "dist",
            "files": [{"path": path, "content": content} for path, content in files],
        }
    )
