from __future__ import annotations

import json

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
