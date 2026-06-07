from __future__ import annotations

import json
from pathlib import Path

from skillopt.gradient.reflect import run_error_analyst_minibatch
from skillopt.prompts import clear_cache, load_prompt


def test_error_analyst_prompts_instruct_structured_failure_generalization():
    clear_cache()

    for prompt_name in ("analyst_error", "analyst_error_rewrite", "analyst_error_full_rewrite"):
        prompt = load_prompt(prompt_name)

        assert "Structured Evaluator Feedback" in prompt
        assert "optimizer_hint" in prompt
        assert "failed_checks" in prompt
        assert "general" in prompt.lower()
        assert "artifact-contract blockers" in prompt


def test_gitmoot_prompts_preserve_markers_and_separate_optimizer_context():
    clear_cache()

    for prompt_name in (
        "analyst_error",
        "analyst_error_full_rewrite",
        "analyst_success",
        "analyst_success_full_rewrite",
    ):
        prompt = load_prompt(prompt_name, env="gitmoot")

        assert "SKILLOPT_TARGET_START" in prompt
        assert "SKILLOPT_OPTIMIZER_START" in prompt
        assert "target section" in prompt
        assert "optimizer section" in prompt
        assert "Never insert optimizer response-format sections" in prompt

    error_prompt = load_prompt("analyst_error", env="gitmoot")
    assert "wrong_artifact_type" in error_prompt
    assert "artifact_contract_failure" in error_prompt
    assert "human_feedback_misalignment" in error_prompt
    assert "failed_dimensions" in error_prompt
    assert "Tailwind-style UI polish" in error_prompt
    assert "Prefer `replace` or `delete`" in error_prompt
    normalized_error_prompt = " ".join(error_prompt.split())
    assert "Do not append duplicate guidance" in normalized_error_prompt
    success_prompt = load_prompt("analyst_success", env="gitmoot")
    normalized_success_prompt = " ".join(success_prompt.split())
    assert "Prefer `replace` or `delete`" in success_prompt
    assert "Delete stale, contradicted, or redundant guidance" in normalized_success_prompt


def test_gitmoot_prompt_files_are_packaged():
    project_root = Path(__file__).resolve().parents[1]
    pyproject = (project_root / "pyproject.toml").read_text(encoding="utf-8")

    assert '"skillopt.envs.gitmoot" = ["prompts/*.md"]' in pyproject


def test_structured_failure_feedback_reaches_error_analyst_prompt(tmp_path, monkeypatch):
    pred_dir = tmp_path / "predictions"
    item_dir = pred_dir / "missing-bundle"
    item_dir.mkdir(parents=True)
    (item_dir / "conversation.json").write_text(
        json.dumps(
            [
                {"role": "user", "content": "Build a Vue landing page."},
                {"role": "assistant", "content": "Returned plain text instead of bundle JSON."},
            ]
        ),
        encoding="utf-8",
    )
    captured = {}

    def fake_chat_optimizer(**kwargs):
        captured.update(kwargs)
        return (
            json.dumps(
                {
                    "batch_size": 1,
                    "failure_summary": [
                        {
                            "failure_type": "vue_vite_bundle_contract_failed",
                            "count": 1,
                            "description": "The response omitted the required Vue/Vite bundle.",
                        }
                    ],
                    "patch": {"reasoning": "Contract guidance is missing.", "edits": []},
                }
            ),
            {},
        )

    monkeypatch.setattr("skillopt.gradient.reflect.chat_optimizer", fake_chat_optimizer)

    result = run_error_analyst_minibatch(
        "Current skill",
        [
            {
                "id": "missing-bundle",
                "task_description": "Landing page",
                "task_type": "gitmoot-skillopt",
                "fail_reason": "Vue/Vite preview bundle is missing required files.",
                "contract_status": "failed",
                "quality_status": "not_run",
                "human_feedback_alignment": {
                    "status": "feedback_available",
                    "required_improvements": ["stronger product visuals"],
                },
                "failure": {
                    "primary_reason": "vue_vite_bundle_contract_failed",
                    "human_reason": "The response did not include required bundle files.",
                    "optimizer_hint": "Return JSON with package.json, index.html, src/main.js, and src/App.vue.",
                    "failed_checks": [
                        {
                            "check": "vue_vite_bundle.required_files",
                            "reason": "Required files missing.",
                            "evidence": ["missing src/App.vue"],
                        }
                    ],
                    "evidence": ["missing src/App.vue"],
                },
                "stage_status": [{"stage": "artifact_contract", "status": "failed"}],
            }
        ],
        str(pred_dir),
        system_prompt="system",
    )

    assert result is not None
    assert result["source_type"] == "failure"
    assert "Structured Evaluator Feedback" in captured["user"]
    assert "contract_status" in captured["user"]
    assert "quality_status" in captured["user"]
    assert "stronger product visuals" in captured["user"]
    assert "vue_vite_bundle_contract_failed" in captured["user"]
    assert "Return JSON with package.json" in captured["user"]
    assert "vue_vite_bundle.required_files" in captured["user"]
    assert "missing src/App.vue" in captured["user"]


def test_error_analyst_reads_prediction_id_when_result_id_differs(tmp_path, monkeypatch):
    pred_dir = tmp_path / "predictions"
    item_dir = pred_dir / "safe-id"
    item_dir.mkdir(parents=True)
    (item_dir / "conversation.json").write_text(
        json.dumps([{"role": "user", "content": "Ranked feedback says improve mobile layout."}]),
        encoding="utf-8",
    )
    captured = {}

    def fake_chat_optimizer(**kwargs):
        captured.update(kwargs)
        return (
            json.dumps(
                {
                    "batch_size": 1,
                    "failure_summary": [],
                    "patch": {"reasoning": "Use feedback.", "edits": []},
                }
            ),
            {},
        )

    monkeypatch.setattr("skillopt.gradient.reflect.chat_optimizer", fake_chat_optimizer)

    result = run_error_analyst_minibatch(
        "Current skill",
        [
            {
                "id": "raw item #1",
                "prediction_id": "safe-id",
                "task_description": "Landing page",
                "task_type": "gitmoot-skillopt",
                "hard": 0,
                "fail_reason": "ranked human feedback requests optimization",
            }
        ],
        str(pred_dir),
        system_prompt="system",
    )

    assert result is not None
    assert "raw item #1" in captured["user"]
    assert "Ranked feedback says improve mobile layout." in captured["user"]


def test_error_analyst_ignores_untrusted_metadata_prediction_id(tmp_path, monkeypatch):
    pred_dir = tmp_path / "predictions"
    item_dir = pred_dir / "normal-item"
    item_dir.mkdir(parents=True)
    (item_dir / "conversation.json").write_text(
        json.dumps([{"role": "user", "content": "Real rollout conversation."}]),
        encoding="utf-8",
    )
    captured = {}

    def fake_chat_optimizer(**kwargs):
        captured.update(kwargs)
        return (
            json.dumps(
                {
                    "batch_size": 1,
                    "failure_summary": [],
                    "patch": {"reasoning": "Use real rollout.", "edits": []},
                }
            ),
            {},
        )

    monkeypatch.setattr("skillopt.gradient.reflect.chat_optimizer", fake_chat_optimizer)

    result = run_error_analyst_minibatch(
        "Current skill",
        [
            {
                "id": "normal-item",
                "metadata": {"prediction_id": "wrong-dir"},
                "task_description": "Landing page",
                "task_type": "gitmoot-skillopt",
                "hard": 0,
                "fail_reason": "target failed",
            }
        ],
        str(pred_dir),
        system_prompt="system",
    )

    assert result is not None
    assert "Real rollout conversation." in captured["user"]
