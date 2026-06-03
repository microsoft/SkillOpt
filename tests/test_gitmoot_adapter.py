from __future__ import annotations

import json

from skillopt.envs.gitmoot.adapter import GitmootAdapter
from skillopt.envs.gitmoot.evaluator import evaluate_response
from skillopt.envs.gitmoot.rollout import process_one
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
        return "Vue landing page with full hero, final CTA, responsive CSS, and footer."

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
    assert result["metadata"]["dimension_scores"]["mobile_responsiveness"] == 0.9
    assert result["metadata"]["rationale"] == "Clear hero, CTA, footer, and responsive layout."
    assert captured["stage"] == "gitmoot_landing_page_judge"
    assert "Human ranking: D > B > C > A" in captured["user"]
    assert "Generated Landing Page Response" in captured["user"]
    assert "mobile_responsiveness" in captured["system"]


def test_landing_page_evaluator_accepts_numeric_string_hard(tmp_path, monkeypatch):
    def pass_agent(*args, **kwargs):
        return "Vue landing page with responsive sections, footer, and clear CTA."

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


def test_landing_page_evaluator_invalid_json_fails_closed(tmp_path, monkeypatch):
    def pass_agent(*args, **kwargs):
        return "Vue landing page"

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
