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
from skillopt.envs.gitmoot.rollout import extract_target_skill_content, process_one
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


def test_target_artifact_retry_repairs_vue_bundle_before_reflection(tmp_path, monkeypatch):
    calls: list[str] = []
    eval_responses: list[str] = []

    def fake_agent(item, skill_content, system_prompt, user_prompt, max_completion_tokens, pred_dir):
        del item, skill_content, system_prompt, max_completion_tokens, pred_dir
        calls.append(user_prompt)
        if len(calls) == 1:
            return "Here is a prose landing page instead of a bundle."
        return _valid_vue_bundle_response()

    def fake_evaluator(item, response, evaluator_config):
        del item, evaluator_config
        eval_responses.append(response)
        if len(eval_responses) == 1:
            return {
                "hard": 0,
                "soft": 0.0,
                "fail_reason": "Generated response must be a JSON object containing a Vue/Vite preview bundle.",
                "primary_reason": "wrong_artifact_type",
                "optimizer_hint": "Return a JSON Vue/Vite preview bundle.",
                "failed_dimensions": ["artifact_contract"],
                "failed_checks": [
                    {
                        "check": "vue_vite_bundle.json",
                        "reason": "Generated response must be a JSON object containing a Vue/Vite preview bundle.",
                        "evidence": ["response did not contain a parseable JSON object"],
                    }
                ],
                "stage_status": [{"stage": "artifact_contract", "status": "failed"}],
            }
        return {"hard": 1, "soft": 1.0, "fail_reason": "", "metadata": {"evaluator": "fixture"}}

    monkeypatch.setattr("skillopt.envs.gitmoot.rollout._run_agent", fake_agent)
    monkeypatch.setattr("skillopt.envs.gitmoot.rollout.evaluate_response", fake_evaluator)

    result = process_one(
        item={
            "id": "repair-vue",
            "prompt": "Build a Vue/Vite landing page preview.",
            "evaluator_config": {"artifact_contract": "vue_vite_bundle"},
        },
        skill_content="Return the requested preview.",
        out_root=str(tmp_path),
        target_artifact_retry_budget=1,
    )

    assert result["hard"] == 1
    assert result.get("primary_reason") != "wrong_artifact_type"
    assert len(calls) == 2
    assert "Artifact Contract Repair Attempt 1/1" in calls[1]
    assert "vue_vite_bundle.json" in calls[1]
    assert result["metadata"]["target_artifact_repair_attempts"][0]["status"] == "accepted"


def test_target_artifact_retry_budget_zero_keeps_first_contract_failure(tmp_path, monkeypatch):
    calls = 0

    def fake_agent(*args, **kwargs):
        nonlocal calls
        calls += 1
        return "Here is a prose landing page instead of a bundle."

    def fake_evaluator(*args, **kwargs):
        return {
            "hard": 0,
            "soft": 0.0,
            "fail_reason": "Generated response must be a JSON object containing a Vue/Vite preview bundle.",
            "primary_reason": "wrong_artifact_type",
            "failed_dimensions": ["artifact_contract"],
            "failed_checks": [
                {
                    "check": "vue_vite_bundle.json",
                    "reason": "Generated response must be a JSON object containing a Vue/Vite preview bundle.",
                    "evidence": ["response did not contain a parseable JSON object"],
                }
            ],
        }

    monkeypatch.setattr("skillopt.envs.gitmoot.rollout._run_agent", fake_agent)
    monkeypatch.setattr("skillopt.envs.gitmoot.rollout.evaluate_response", fake_evaluator)

    result = process_one(
        item={
            "id": "no-repair-vue",
            "prompt": "Build a Vue/Vite landing page preview.",
            "evaluator_config": {"artifact_contract": "vue_vite_bundle"},
        },
        skill_content="Return the requested preview.",
        out_root=str(tmp_path),
        target_artifact_retry_budget=0,
    )

    assert calls == 1
    assert result["primary_reason"] == "wrong_artifact_type"
    assert "target_artifact_repair_attempts" not in result["metadata"]


def test_extract_target_skill_content_uses_sectioned_target_only():
    skill = """
# Landing Page Builder

<!-- SKILLOPT_TARGET_START -->
Target rules.
<!-- SKILLOPT_TARGET_END -->

<!-- SKILLOPT_OPTIMIZER_START -->
Optimizer notes.
## Update Format
<!-- SKILLOPT_OPTIMIZER_END -->
"""

    context = extract_target_skill_content(skill)

    assert context.content == "Target rules."
    assert context.metadata["sectioned"] is True
    assert context.metadata["target_section_present"] is True
    assert context.metadata["optimizer_section_present"] is True
    assert context.metadata["isolation"] == "target_section"


def test_extract_target_skill_content_falls_back_for_legacy_skill():
    context = extract_target_skill_content("# Planner\n\nUse the full legacy skill.")

    assert context.content == "# Planner\n\nUse the full legacy skill."
    assert context.metadata["sectioned"] is False
    assert context.metadata["isolation"] == "legacy_full_skill"
    assert context.metadata["warning"] == "skillopt_sections_absent"


def test_extract_target_skill_content_falls_back_for_malformed_markers():
    skill = """
<!-- SKILLOPT_TARGET_START -->
Target rules without an end marker.
<!-- SKILLOPT_OPTIMIZER_START -->
Optimizer notes.
<!-- SKILLOPT_OPTIMIZER_END -->
"""

    context = extract_target_skill_content(skill)

    assert "Target rules without an end marker" in context.content
    assert context.metadata["sectioned"] is False
    assert context.metadata["warning"] == "malformed_target_section"


def test_extract_target_skill_content_keeps_valid_target_when_optimizer_markers_are_malformed():
    skill = """
<!-- SKILLOPT_TARGET_START -->
Target rules remain usable.
<!-- SKILLOPT_TARGET_END -->

<!-- SKILLOPT_OPTIMIZER_START -->
Optimizer notes missing the end marker.
## Update Format
"""

    context = extract_target_skill_content(skill)

    assert context.content == "Target rules remain usable."
    assert context.metadata["sectioned"] is True
    assert context.metadata["isolation"] == "target_section"
    assert context.metadata["warning"] == "malformed_optimizer_section"


def test_extract_target_skill_content_ignores_marker_examples_in_optimizer_section():
    skill = """
<!-- SKILLOPT_TARGET_START -->
Target rules survive marker examples.
<!-- SKILLOPT_TARGET_END -->

<!-- SKILLOPT_OPTIMIZER_START -->
When rewriting, preserve literal markers such as:
<!-- SKILLOPT_TARGET_START -->
<!-- SKILLOPT_TARGET_END -->
<!-- SKILLOPT_OPTIMIZER_START -->
<!-- SKILLOPT_OPTIMIZER_END -->
<!-- SKILLOPT_OPTIMIZER_END -->
"""

    context = extract_target_skill_content(skill)

    assert context.content == "Target rules survive marker examples."
    assert context.metadata["sectioned"] is True
    assert context.metadata["isolation"] == "target_section"
    assert context.metadata.get("warning") is None


def test_rollout_chat_prompt_uses_target_section_only(tmp_path, monkeypatch):
    captured = {}

    def fake_chat_target(**kwargs):
        captured.update(kwargs)
        return "target response", {}

    monkeypatch.setattr("skillopt.envs.gitmoot.rollout.is_target_exec_backend", lambda: False)
    monkeypatch.setattr("skillopt.envs.gitmoot.rollout.is_target_chat_backend", lambda: True)
    monkeypatch.setattr("skillopt.envs.gitmoot.rollout.chat_target", fake_chat_target)
    item = {
        "id": "sectioned-chat",
        "prompt": "Prompt",
        "metadata": {"expected_hard": True},
        "evaluator_config": {"mode": "fixture"},
    }
    skill = """
<!-- SKILLOPT_TARGET_START -->
Target-only landing page rules.
<!-- SKILLOPT_TARGET_END -->

<!-- SKILLOPT_OPTIMIZER_START -->
Optimizer-only notes.
## Update Format
<!-- SKILLOPT_OPTIMIZER_END -->
"""

    result = process_one(item=item, skill_content=skill, out_root=str(tmp_path))

    assert captured["system"].startswith("You are solving one Gitmoot task.")
    assert "## Skill" in captured["system"]
    assert "## Output Contract" in captured["system"]
    assert "Return exactly the required deliverable." in captured["system"]
    assert "Target-only landing page rules." in captured["system"]
    assert "Optimizer-only notes" not in captured["system"]
    assert "SKILLOPT_OPTIMIZER" not in captured["system"]
    assert "## Update Format" not in captured["system"]
    assert result["metadata"]["target_skill"]["isolation"] == "target_section"


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

    skill = """
<!-- SKILLOPT_TARGET_START -->
Target-only exec rules.
<!-- SKILLOPT_TARGET_END -->

<!-- SKILLOPT_OPTIMIZER_START -->
Optimizer-only exec notes.
## Update Format
<!-- SKILLOPT_OPTIMIZER_END -->
"""

    result = process_one(item=item, skill_content=skill, out_root=str(tmp_path))

    assert result["response"] == "exec response"
    assert result["hard"] == 1
    assert result["score_status"] == "scored"
    assert captured["model"] == "gpt-test"
    assert captured["work_dir"].endswith("predictions/exec-item/target_exec")
    assert "## System Instructions" in captured["prompt"]
    assert "You are solving one Gitmoot task." in captured["prompt"]
    assert "Return exactly the required deliverable." in captured["prompt"]
    assert (tmp_path / "predictions" / "exec-item" / "target_exec" / "task.md").is_file()
    skill_path = (
        tmp_path
        / "predictions"
        / "exec-item"
        / "target_exec"
        / ".agents"
        / "skills"
        / "skillopt-target"
        / "SKILL.md"
    )
    assert skill_path.is_file()
    workspace_skill = skill_path.read_text(encoding="utf-8")
    assert "Target-only exec rules." in workspace_skill
    assert "Optimizer-only exec notes" not in workspace_skill
    assert "SKILLOPT_OPTIMIZER" not in workspace_skill
    assert "## Update Format" not in workspace_skill
    assert "Target-only exec rules." in captured["prompt"]
    assert "Optimizer-only exec notes" not in captured["prompt"]
    raw_path = tmp_path / "predictions" / "exec-item" / "target_exec_raw.txt"
    assert raw_path.read_text(encoding="utf-8") == "raw exec trace"
    assert result["target_trace_path"] == str(raw_path)


def test_gitmoot_reflect_uses_env_specific_prompts(tmp_path, monkeypatch):
    captured = {}

    def fake_run_minibatch_reflect(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr("skillopt.envs.gitmoot.adapter.run_minibatch_reflect", fake_run_minibatch_reflect)

    adapter = GitmootAdapter()
    adapter._cfg = {"skill_update_mode": "patch"}
    patches = adapter.reflect([], "# Skill", str(tmp_path))

    assert patches == []
    assert "SKILLOPT_TARGET_START" in captured["error_system"]
    assert "wrong_artifact_type" in captured["error_system"]
    assert "SKILLOPT_OPTIMIZER_START" in captured["success_system"]
    assert "Never insert optimizer response-format sections" in captured["success_system"]


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

    assert score["hard"] == 0
    assert score["soft"] == 0.75
    assert score["contract_status"] == "passed"
    assert score["quality_status"] == "failed"
    assert score["human_feedback_alignment"]["status"] == "feedback_available"
    assert score["human_feedback_alignment"]["rankings"] == ["D > B > C > A"]
    assert score["human_feedback_alignment"]["reasoning"] == ["D has the cleanest hero."]
    assert score["stage_status"][0] == {"stage": "render_smoke", "status": "passed"}
    assert score["stage_status"][-1] == {"stage": "llm_judge", "status": "failed"}
    assert '"status": "passed"' in captured["user"]
    assert "/tmp/mobile.png" in captured["user"]
    assert "github_issue_109" in captured["user"]
    assert "D has the cleanest hero" in captured["user"]
    assert "do-not-send-this" not in captured["user"]
    assert "secret_token" not in captured["user"]


def test_landing_page_judge_is_inferred_for_legacy_vue_feedback_package(monkeypatch):
    captured = {}

    def fake_chat_optimizer(**kwargs):
        captured.update(kwargs)
        return (
            json.dumps(
                {
                    "hard": 1,
                    "soft": 0.92,
                    "dimension_scores": _landing_dimension_scores(),
                    "rationale": "The page is strong but the review asked to keep refining.",
                    "fail_reason": "",
                    "human_feedback_alignment": {"resolved": ["palette"], "unresolved": ["motion"]},
                }
            ),
            {},
        )

    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator.chat_optimizer", fake_chat_optimizer)

    score = evaluate_response(
        {
            "id": "legacy-landing-page",
            "prompt": "Build a Vue/Vite landing page preview.",
            "metadata": {"output_type": "vue_vite_bundle"},
            "ranked_feedback_events": [
                {
                    "ranking": ["C", "D", "B", "A"],
                    "reasoning": "C has the best palette, but it still needs animation.",
                    "quality": "high",
                    "continue_mode": "refine",
                    "promote": "no",
                    "required_improvements": ["more motion", "better product graphics"],
                }
            ],
        },
        _valid_vue_bundle_response(),
        {},
    )

    assert captured["stage"] == "gitmoot_landing_page_judge"
    assert score["hard"] == 0
    assert score["soft"] == 0.75
    assert score["fail_reason"] == "human feedback requested continued optimization; candidate is not ready to stop"
    assert score["human_feedback_alignment"]["required_improvements"] == ["more motion", "better product graphics"]
    assert score["failure"]["primary_reason"] == "human_feedback_not_resolved"
    assert "motion" in score["failure"]["optimizer_hint"]
    assert score["failure"]["evidence"] == ["motion"]


def test_landing_page_inference_requires_vue_bundle_for_profile_only_feedback(monkeypatch):
    def fail_chat_optimizer(**kwargs):
        raise AssertionError("landing page judge should not run before Vue bundle validation")

    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator.chat_optimizer", fail_chat_optimizer)

    score = evaluate_response(
        {
            "id": "legacy-profile-only-landing-page",
            "prompt": "Build a landing page preview.",
            "metadata": {"profile_id": "vue_landing_page_v1"},
            "ranked_feedback_events": [
                {
                    "ranking": ["C", "D", "B", "A"],
                    "reasoning": "C has the best palette, but it still needs animation.",
                    "continue_mode": "refine",
                    "promote": "no",
                    "required_improvements": ["more motion"],
                }
            ],
        },
        "Here is a prose landing page idea instead of a Vue bundle.",
        {},
    )

    assert score["hard"] == 0
    assert score["failure"]["primary_reason"] == "wrong_artifact_type"
    assert score["contract_status"] == "failed"
    assert score["quality_status"] == "not_run"


def test_landing_page_judge_allows_resolved_alignment_without_top_level_unresolved(monkeypatch):
    def fake_chat_optimizer(**kwargs):
        return (
            json.dumps(
                {
                    "hard": 1,
                    "soft": 0.92,
                    "dimension_scores": _landing_dimension_scores(),
                    "rationale": "The requested palette, motion, graphics, and mobile improvements are resolved.",
                    "fail_reason": "",
                    "human_feedback_alignment": {
                        "status": "resolved",
                        "resolved": ["palette", "motion", "product graphics", "mobile layout"],
                        "unresolved": [],
                    },
                }
            ),
            {},
        )

    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator.chat_optimizer", fake_chat_optimizer)

    score = evaluate_response(
        {
            "id": "legacy-landing-page-resolved",
            "prompt": "Build a Vue/Vite landing page preview.",
            "metadata": {"output_type": "vue_vite_bundle"},
            "ranked_feedback_events": [
                {
                    "ranking": ["C", "D", "B", "A"],
                    "reasoning": "C has the best palette, but it still needs animation.",
                    "quality": "high",
                    "continue_mode": "refine",
                    "promote": "no",
                    "required_improvements": ["more motion", "better product graphics"],
                }
            ],
        },
        _valid_vue_bundle_response(),
        {},
    )

    assert score["hard"] == 1
    assert score["soft"] == 0.92
    assert score["quality_status"] == "passed"
    assert "failure" not in score


def test_generic_judge_with_feedback_fails_closed_without_feedback_dimensions(monkeypatch):
    def fake_chat_optimizer(**kwargs):
        return (
            json.dumps(
                {
                    "hard": 1,
                    "soft": 0.95,
                    "reasoning": "Looks complete.",
                    "fail_reason": "",
                }
            ),
            {},
        )

    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator.chat_optimizer", fake_chat_optimizer)

    score = evaluate_response(
        {
            "id": "generic-feedback",
            "prompt": "Write an X post.",
            "ranked_feedback_events": [
                {
                    "ranking": ["B", "A"],
                    "reasoning": "B is better, but keep refining the hook.",
                    "quality": "strong",
                    "continue_mode": "refine",
                    "promote": "no",
                    "required_improvements": ["sharper hook"],
                }
            ],
        },
        "A solid post.",
        {},
    )

    assert score["hard"] == 0
    assert score["soft"] == 0.0
    assert score["fail_reason"] == "evaluator_missing_human_feedback_dimensions"
    assert score["primary_reason"] == "evaluator_missing_human_feedback_dimensions"
    assert score["dimension_scores"]["human_feedback_resolution"] == 0.0


def test_generic_judge_with_feedback_fails_closed_with_invalid_dimension_values(monkeypatch):
    def fake_chat_optimizer(**kwargs):
        return (
            json.dumps(
                {
                    "hard": 1,
                    "soft": 0.95,
                    "reasoning": "Claims complete.",
                    "fail_reason": "",
                    "human_feedback_alignment": {"status": "resolved"},
                    "dimension_scores": {
                        "human_feedback_resolution": "n/a",
                        "artifact_validity": 1.0,
                        "task_completeness": 0.9,
                    },
                    "unresolved_feedback": [],
                    "rejection_reason": "",
                    "optimizer_hint": "",
                }
            ),
            {},
        )

    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator.chat_optimizer", fake_chat_optimizer)

    score = evaluate_response(
        {
            "id": "generic-feedback-invalid-dimensions",
            "prompt": "Write an X post.",
            "ranked_feedback_events": [
                {
                    "ranking": ["B", "A"],
                    "reasoning": "B is ready.",
                    "quality": "strong",
                    "continue_mode": "validate",
                    "promote": "yes",
                }
            ],
        },
        "A solid post.",
        {},
    )

    assert score["hard"] == 0
    assert score["soft"] == 0.0
    assert score["fail_reason"] == "evaluator_missing_human_feedback_dimensions"


def test_generic_judge_with_feedback_requires_explicit_unresolved_feedback_list(monkeypatch):
    def fake_chat_optimizer(**kwargs):
        return (
            json.dumps(
                {
                    "hard": 1,
                    "soft": 0.95,
                    "reasoning": "Claims complete.",
                    "fail_reason": "",
                    "human_feedback_alignment": {"status": "resolved"},
                    "dimension_scores": {
                        "human_feedback_resolution": 0.95,
                        "artifact_validity": 1.0,
                        "task_completeness": 0.9,
                    },
                    "rejection_reason": "",
                    "optimizer_hint": "",
                }
            ),
            {},
        )

    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator.chat_optimizer", fake_chat_optimizer)

    score = evaluate_response(
        {
            "id": "generic-feedback-missing-unresolved",
            "prompt": "Write an X post.",
            "ranked_feedback_events": [
                {
                    "ranking": ["B", "A"],
                    "reasoning": "B is better but keep refining.",
                    "quality": "strong",
                    "continue_mode": "refine",
                    "promote": "no",
                }
            ],
        },
        "A solid post.",
        {},
    )

    assert score["hard"] == 0
    assert score["soft"] == 0.0
    assert score["fail_reason"] == "evaluator_missing_human_feedback_dimensions"


def test_generic_judge_with_resolved_feedback_does_not_emit_failure_hint(monkeypatch):
    captured = {}

    def fake_chat_optimizer(**kwargs):
        captured.update(kwargs)
        return (
            json.dumps(
                {
                    "hard": 1,
                    "soft": 0.93,
                    "reasoning": "The feedback is resolved and promotion was requested.",
                    "fail_reason": "",
                    "human_feedback_alignment": {"status": "resolved"},
                    "dimension_scores": {
                        "human_feedback_resolution": 0.95,
                        "artifact_validity": 1.0,
                        "task_completeness": 0.9,
                    },
                    "unresolved_feedback": [],
                    "rejection_reason": "",
                    "optimizer_hint": "",
                }
            ),
            {},
        )

    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator.chat_optimizer", fake_chat_optimizer)

    score = evaluate_response(
        {
            "id": "generic-promote",
            "prompt": "Write an X post.",
            "ranked_feedback_events": [
                {
                    "ranking": ["B", "A"],
                    "reasoning": "B is ready.",
                    "quality": "strong",
                    "continue_mode": "validate",
                    "promote": "yes",
                }
            ],
        },
        "A strong post.",
        {},
    )

    assert score["hard"] == 1
    assert score["soft"] == 0.93
    assert "failure" not in score
    assert "optimizer_hint" not in score
    assert score["stage_status"] == [{"stage": "llm_judge", "status": "passed"}]
    assert "Human Feedback Context" in captured["user"]
    assert "B is ready" in captured["user"]


def test_generic_judge_uses_top_level_feedback_context(monkeypatch):
    captured = {}

    def fake_chat_optimizer(**kwargs):
        captured.update(kwargs)
        return (
            json.dumps(
                {
                    "hard": 1,
                    "soft": 0.9,
                    "reasoning": "The candidate resolves the top-level feedback context.",
                    "fail_reason": "",
                    "human_feedback_alignment": {"status": "resolved", "unresolved": []},
                    "dimension_scores": {
                        "human_feedback_resolution": 0.91,
                        "artifact_validity": 1.0,
                        "task_completeness": 0.9,
                    },
                    "unresolved_feedback": [],
                    "rejection_reason": "",
                    "optimizer_hint": "",
                    "selection_decision": "candidate_resolves_baseline_feedback",
                }
            ),
            {},
        )

    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator.chat_optimizer", fake_chat_optimizer)

    score = evaluate_response(
        {
            "id": "generic-top-level-feedback-context",
            "prompt": "Write a concise launch post.",
            "feedback_context": {
                "feedback_source": ["imported_human_review"],
                "feedback_target": ["baseline_review_outputs"],
                "review_issue": ["owner/repo#21"],
                "quality": ["poor"],
                "continue_mode": ["refine"],
                "promote": ["no"],
                "themes": ["stronger launch hook"],
            },
        },
        "A concise launch post with a stronger hook.",
        {},
    )

    assert score["hard"] == 1
    assert score["metadata"]["feedback_target"] == "baseline_review_outputs"
    assert score["human_feedback_alignment"]["feedback_target"] == ["baseline_review_outputs"]
    assert score["human_feedback_alignment"]["themes"] == ["stronger launch hook"]
    assert score["soft"] == 0.9
    assert "Human Feedback Context" in captured["user"]
    assert "owner/repo#21" in captured["user"]


def test_generic_judge_allows_refine_feedback_when_resolution_is_proven(monkeypatch):
    def fake_chat_optimizer(**kwargs):
        return (
            json.dumps(
                {
                    "hard": 1,
                    "soft": 0.91,
                    "reasoning": "The requested hook and tone improvements are resolved.",
                    "fail_reason": "",
                    "human_feedback_alignment": {
                        "status": "resolved",
                        "resolved": ["sharper hook", "clearer tone"],
                        "unresolved": [],
                    },
                    "dimension_scores": {
                        "human_feedback_resolution": 0.92,
                        "artifact_validity": 1.0,
                        "task_completeness": 0.9,
                    },
                    "unresolved_feedback": [],
                    "rejection_reason": "",
                    "optimizer_hint": "",
                    "baseline_known_issues": ["old hook was weak"],
                    "candidate_resolution": ["hook is now sharper", "tone is clearer"],
                    "baseline_resolution": "old feedback is used as the target, not as a candidate veto",
                    "selection_decision": "candidate_resolves_baseline_feedback",
                    "score_delta_reason": "candidate resolves the prior feedback themes",
                }
            ),
            {},
        )

    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator.chat_optimizer", fake_chat_optimizer)

    score = evaluate_response(
        {
            "id": "generic-refine-resolved",
            "prompt": "Write an X post.",
            "ranked_feedback_events": [
                {
                    "ranking": ["B", "A"],
                    "reasoning": "B is better but refine the hook.",
                    "quality": "strong",
                    "continue_mode": "refine",
                    "promote": "no",
                    "feedback_target": "baseline_review_outputs",
                    "required_improvements": ["sharper hook", "clearer tone"],
                }
            ],
        },
        "A stronger post.",
        {},
    )

    assert score["hard"] == 1
    assert score["soft"] == 0.91
    assert score["quality_status"] == "passed"
    assert score["metadata"]["feedback_target"] == "baseline_review_outputs"
    assert score["human_feedback_alignment"]["feedback_target"] == ["baseline_review_outputs"]
    assert score["selection_decision"] == "candidate_resolves_baseline_feedback"
    assert score["candidate_resolution"] == ["hook is now sharper", "tone is clearer"]
    assert "failure" not in score


def test_generic_judge_rejects_inconsistent_resolved_feedback(monkeypatch):
    def fake_chat_optimizer(**kwargs):
        return (
            json.dumps(
                {
                    "hard": 1,
                    "soft": 0.91,
                    "reasoning": "Claims resolved but still lists missing graphics.",
                    "fail_reason": "",
                    "human_feedback_alignment": {
                        "status": "resolved",
                        "resolved": ["layout"],
                        "unresolved": ["better graphics"],
                    },
                    "dimension_scores": {
                        "human_feedback_resolution": 0.92,
                        "artifact_validity": 1.0,
                        "task_completeness": 0.9,
                    },
                    "unresolved_feedback": ["better graphics"],
                    "rejection_reason": "visuals_unresolved",
                    "optimizer_hint": "Add product-relevant graphics.",
                }
            ),
            {},
        )

    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator.chat_optimizer", fake_chat_optimizer)

    score = evaluate_response(
        {
            "id": "generic-refine-inconsistent",
            "prompt": "Write an X post.",
            "ranked_feedback_events": [
                {
                    "ranking": ["B", "A"],
                    "reasoning": "B is better but refine the visuals.",
                    "continue_mode": "refine",
                    "promote": "no",
                    "required_improvements": ["better graphics"],
                }
            ],
        },
        "A stronger post.",
        {},
    )

    assert score["hard"] == 0
    assert score["soft"] == 0.75
    assert score["quality_status"] == "failed"
    assert score["primary_reason"] == "visuals_unresolved"
    assert score["optimizer_hint"] == "Add product-relevant graphics."


def test_generic_judge_preserves_alignment_only_unresolved_feedback(monkeypatch):
    def fake_chat_optimizer(**kwargs):
        return (
            json.dumps(
                {
                    "hard": 1,
                    "soft": 0.91,
                    "reasoning": "Claims resolved but alignment lists missing motion.",
                    "fail_reason": "",
                    "human_feedback_alignment": {
                        "status": "partial",
                        "resolved": ["layout"],
                        "unresolved": ["scroll animation", "product graphics"],
                    },
                    "dimension_scores": {
                        "human_feedback_resolution": 0.62,
                        "artifact_validity": 1.0,
                        "task_completeness": 0.9,
                    },
                    "unresolved_feedback": [],
                    "rejection_reason": "",
                    "optimizer_hint": "",
                }
            ),
            {},
        )

    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator.chat_optimizer", fake_chat_optimizer)

    score = evaluate_response(
        {
            "id": "generic-refine-alignment-unresolved",
            "prompt": "Write an X post.",
            "ranked_feedback_events": [
                {
                    "ranking": ["B", "A"],
                    "reasoning": "B is better but refine the visuals.",
                    "continue_mode": "refine",
                    "promote": "no",
                    "required_improvements": ["better graphics"],
                }
            ],
        },
        "A stronger post.",
        {},
    )

    assert score["hard"] == 0
    assert score["quality_status"] == "failed"
    assert score["primary_reason"] == "human_feedback_not_resolved"
    assert score["evidence"] == ["scroll animation", "product graphics"]
    assert "scroll animation" in score["optimizer_hint"]


def test_generic_judge_rejects_non_list_unresolved_feedback(monkeypatch):
    def fake_chat_optimizer(**kwargs):
        return (
            json.dumps(
                {
                    "hard": 1,
                    "soft": 0.91,
                    "reasoning": "Claims resolved but unresolved feedback is not a list.",
                    "fail_reason": "",
                    "human_feedback_alignment": {
                        "status": "resolved",
                        "resolved": ["layout"],
                        "unresolved": {"visuals": "better graphics"},
                    },
                    "dimension_scores": {
                        "human_feedback_resolution": 0.92,
                        "artifact_validity": 1.0,
                        "task_completeness": 0.9,
                    },
                    "unresolved_feedback": "better graphics",
                    "rejection_reason": "visuals_unresolved",
                    "optimizer_hint": "Add product-relevant graphics.",
                }
            ),
            {},
        )

    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator.chat_optimizer", fake_chat_optimizer)

    score = evaluate_response(
        {
            "id": "generic-refine-string-unresolved",
            "prompt": "Write an X post.",
            "ranked_feedback_events": [
                {
                    "ranking": ["B", "A"],
                    "reasoning": "B is better but refine the visuals.",
                    "continue_mode": "refine",
                    "promote": "no",
                    "required_improvements": ["better graphics"],
                }
            ],
        },
        "A stronger post.",
        {},
    )

    assert score["hard"] == 0
    assert score["soft"] == 0.0
    assert score["fail_reason"] == "evaluator_missing_human_feedback_dimensions"
    assert score["primary_reason"] == "evaluator_missing_human_feedback_dimensions"


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


def test_landing_page_old_review_feedback_trains_candidate_without_veto(monkeypatch):
    def fake_chat_optimizer(**kwargs):
        return (
            json.dumps(
                {
                    "hard": 1,
                    "soft": 0.92,
                    "contract_status": "passed",
                    "quality_status": "strong",
                    "human_feedback_alignment": (
                        "The new candidate resolves the old review themes: it keeps the strongest ranked "
                        "direction, adds MoonAI-level dark premium art direction, stronger branding, "
                        "product-relevant graphics, meaningful motion, proof sections, and mobile-safe layout."
                    ),
                    "baseline_known_issues": [
                        "old options lacked memorable branding",
                        "old options lacked product-relevant graphics",
                    ],
                    "candidate_resolution": {
                        "branding": "resolved with a darker premium direction",
                        "visuals": "resolved with product-relevant graphics",
                    },
                    "baseline_resolution": "baseline review issues were treated as prior-output defects",
                    "selection_decision": "candidate_resolves_baseline_feedback",
                    "score_delta_reason": "candidate improves the old reviewed outputs rather than inheriting their poor label",
                    "dimension_scores": {
                        **_landing_dimension_scores(),
                        "brand_identity": 0.93,
                        "proof_trust_content": 0.9,
                    },
                    "rationale": (
                        "The Vue/Vite bundle passes contract checks and the candidate addresses the old "
                        "poor/refine/promote=no review feedback with a visibly stronger landing-page direction."
                    ),
                    "fail_reason": "",
                }
            ),
            {},
        )

    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator.chat_optimizer", fake_chat_optimizer)

    score = evaluate_response(
        {
            "id": "old-review-feedback-candidate",
            "prompt": "Build a Vue landing page using the old review feedback as the target.",
            "ranked_feedback_events": [
                {
                    "ranking": ["C > D > A > B"],
                    "choice": (
                        "All old options were poor. Preserve only the clearest direction and push much harder "
                        "toward MoonAI-level branding, product graphics, motion, proof, and mobile polish."
                    ),
                    "quality": "poor",
                    "continue_mode": "refine",
                    "promote": "no",
                    "feedback_target": "baseline_review_outputs",
                    "feedback_source": "imported_human_review",
                    "themes": ["MoonAI-level branding", "product graphics", "mobile polish"],
                }
            ],
        },
        _valid_vue_bundle_response(),
        {"mode": "landing_page_v1", "artifact_contract": "vue_vite_bundle"},
    )

    assert score["contract_status"] == "passed"
    assert score["hard"] == 1
    assert score["soft"] == 0.92
    assert score["quality_status"] == "passed"
    assert score["fail_reason"] == ""
    assert score["metadata"]["feedback_source"] == "old_review"
    assert score["metadata"]["feedback_target"] == "baseline_review_outputs"
    assert score["human_feedback_alignment"]["feedback_target"] == ["baseline_review_outputs"]
    assert score["human_feedback_alignment"]["themes"] == ["MoonAI-level branding", "product graphics", "mobile polish"]
    assert score["selection_decision"] == "candidate_resolves_baseline_feedback"
    assert score["candidate_resolution"]["branding"] == "resolved with a darker premium direction"
    assert score["metadata"]["score_delta_reason"].startswith("candidate improves")
    assert score["metadata"]["candidate_specific_failure"] is False


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
    assert score["failure"]["primary_reason"] == "artifact_contract_failure"
    assert score["failure"]["failed_checks"][0]["check"] == "vue_vite_bundle.required_files"
    assert "src/App.vue" in score["failure"]["evidence"][-1]
    assert score["failure"]["optimizer_hint"].startswith("Return a JSON Vue/Vite preview bundle")
    assert score["failure"]["failed_dimensions"] == ["artifact_contract"]
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
    assert score["failure"]["primary_reason"] == "wrong_artifact_type"
    assert score["failure"]["failed_checks"][0]["check"] == "vue_vite_bundle.json"
    assert score["failure"]["failed_dimensions"] == ["artifact_contract"]


def test_landing_page_evaluator_rejects_skill_template_output_as_wrong_artifact_type(monkeypatch):
    def fail_chat_optimizer(**kwargs):
        raise AssertionError("landing page judge should not run for wrong artifact type")

    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator.chat_optimizer", fail_chat_optimizer)

    response = """---
id: landing-page-builder
kind: agent-template
---

# Landing Page Builder

## Update Format
Return a complete replacement skill.
"""

    score = evaluate_response(
        {"prompt": "Build a Vue landing page."},
        response,
        {"mode": "landing_page_v1", "artifact_contract": "vue_vite_bundle"},
    )

    assert score["hard"] == 0
    assert score["failure"]["primary_reason"] == "wrong_artifact_type"
    assert "skill/template document" in " ".join(score["failure"]["evidence"])
    assert score["quality_status"] == "not_run"


def test_landing_page_evaluator_rejects_missing_top_level_bundle_metadata_before_judge(monkeypatch):
    def fail_chat_optimizer(**kwargs):
        raise AssertionError("landing page judge should not run for artifact contract failures")

    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator.chat_optimizer", fail_chat_optimizer)

    bundle = json.loads(_valid_vue_bundle_response())
    bundle.pop("renderer")
    bundle["build_command"] = "vite build"
    bundle["dist_dir"] = "build"

    score = evaluate_response(
        {"prompt": "Build a Vue landing page."},
        json.dumps(bundle),
        {"mode": "landing_page_v1", "artifact_contract": "vue_vite_bundle"},
    )

    assert score["hard"] == 0
    assert score["quality_status"] == "not_run"
    assert score["failure"]["primary_reason"] == "artifact_contract_failure"
    checks = [check["check"] for check in score["failure"]["failed_checks"]]
    assert "vue_vite_bundle.renderer" in checks
    assert "vue_vite_bundle.build_command" in checks
    assert "vue_vite_bundle.dist_dir" in checks


def test_landing_page_evaluator_reports_top_level_failures_when_files_missing(monkeypatch):
    def fail_chat_optimizer(**kwargs):
        raise AssertionError("landing page judge should not run for artifact contract failures")

    monkeypatch.setattr("skillopt.envs.gitmoot.evaluator.chat_optimizer", fail_chat_optimizer)

    score = evaluate_response(
        {"prompt": "Build a Vue landing page."},
        json.dumps({}),
        {"mode": "landing_page_v1", "artifact_contract": "vue_vite_bundle"},
    )

    checks = [check["check"] for check in score["failure"]["failed_checks"]]
    assert score["failure"]["primary_reason"] == "artifact_contract_failure"
    assert "vue_vite_bundle.renderer" in checks
    assert "vue_vite_bundle.build_command" in checks
    assert "vue_vite_bundle.dist_dir" in checks
    assert "vue_vite_bundle.files" in checks


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
