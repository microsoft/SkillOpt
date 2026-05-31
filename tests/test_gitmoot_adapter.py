from __future__ import annotations

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
    assert result["hard"] == 0
    assert result["soft"] == 0.0
    assert result["fail_reason"]


def test_failed_agent_execution_does_not_call_judge(tmp_path, monkeypatch):
    def fail_evaluator(*args, **kwargs):
        raise AssertionError("evaluator should not run after agent failure")

    monkeypatch.setattr("skillopt.envs.gitmoot.rollout.evaluate_response", fail_evaluator)
    item = {"id": "broken", "prompt": "Prompt", "metadata": {"expected_hard": True}}

    result = process_one(item=item, skill_content="skill", out_root=str(tmp_path))

    assert result["hard"] == 0
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
    assert result["hard"] == 0
    assert result["fail_reason"] == "agent execution failed"
    assert result["metadata"]["agent_error"] is True


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
