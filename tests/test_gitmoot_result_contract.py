from __future__ import annotations

import pytest

from skillopt.engine.trainer import _compute_task_type_buckets, _extract_failure_patterns
from skillopt.envs.gitmoot.result_contract import (
    EVALUATOR_FAILED,
    TARGET_PASSED,
    make_unscored_evaluation,
    normalize_scored_evaluation,
)
from skillopt.optimizer.slow_update import format_comparison_text
from skillopt.utils.scoring import compute_score


class _ResultWithExtras:
    hard = 1
    soft = 0.5
    extras = {"quality_status": "failed"}


def test_compute_score_rejects_unscored_results():
    results = [
        {
            "id": "broken",
            "hard": None,
            "soft": None,
            "score_status": "unscored",
            "blocker": "evaluator_not_run",
            "fail_reason": "judge was not configured",
        }
    ]

    with pytest.raises(ValueError, match="cannot compute aggregate score for unscored result"):
        compute_score(results)


def test_compute_score_treats_quality_failed_as_hard_failure():
    hard, soft = compute_score(
        [
            {
                "id": "quality-failed",
                "hard": 1,
                "soft": 0.64,
                "score_status": "scored",
                "quality_status": "failed",
            }
        ]
    )

    assert hard == 0.0
    assert soft == 0.64


def test_compute_score_treats_quality_failed_extras_as_hard_failure():
    hard, soft = compute_score([_ResultWithExtras()])

    assert hard == 0.0
    assert soft == 0.5


def test_slow_update_context_labels_quality_failed_as_fail():
    text = format_comparison_text(
        [
            {
                "id": "quality-regression",
                "task": "Improve the landing page.",
                "category": "regressed",
                "prev": {
                    "hard": 1,
                    "quality_status": "passed",
                    "soft": 0.91,
                    "predicted_answer": "Strong page",
                    "fail_reason": "",
                },
                "curr": {
                    "hard": 1,
                    "quality_status": "failed",
                    "soft": 0.62,
                    "predicted_answer": "Valid but still weak page",
                    "fail_reason": "human feedback not resolved",
                },
            }
        ]
    )

    assert "- Prev epoch: PASS" in text
    assert "- Curr epoch: FAIL" in text
    assert "human feedback not resolved" in text


def test_task_type_buckets_count_unscored_results_without_numeric_score():
    buckets = _compute_task_type_buckets(
        [
            {
                "id": "broken",
                "task_type": "gitmoot-skillopt",
                "hard": None,
                "soft": None,
                "score_status": "unscored",
                "blocker": "target_rollout_failed",
            },
            {
                "id": "scored-fail",
                "task_type": "gitmoot-skillopt",
                "hard": 0,
                "soft": 0.25,
                "score_status": "scored",
            },
        ],
        ["gitmoot-skillopt"],
    )

    assert buckets["gitmoot-skillopt"]["total"] == 2
    assert buckets["gitmoot-skillopt"]["hard"] == 0
    assert buckets["gitmoot-skillopt"]["soft"] == 0.25
    assert buckets["gitmoot-skillopt"]["unscored"] == 1
    assert buckets["overall"]["unscored"] == 1


def test_failure_patterns_include_unscored_results(tmp_path):
    patterns = _extract_failure_patterns(
        [
            {
                "id": "broken",
                "hard": None,
                "soft": None,
                "score_status": "unscored",
                "fail_reason": "judge was not configured",
            }
        ],
        str(tmp_path),
    )

    assert patterns
    assert patterns[0]["pattern"] == "judge was not configured"


def test_scored_evaluation_preserves_structured_failure_feedback():
    result = normalize_scored_evaluation(
        {
            "hard": 0,
            "soft": 0.2,
            "fail_reason": "missing required artifact",
            "profile_id": "vue_landing_page_v1",
            "task_kind": "vue_landing_page",
            "contract_status": "failed",
            "quality_status": "not_run",
            "human_feedback_alignment": {
                "status": "feedback_available",
                "required_improvements": ["make the layout responsive"],
            },
            "dimension_scores": {"artifact_contract": 0, "visual": 0.2},
            "failed_dimensions": ["artifact_contract"],
            "failure": {
                "primary_reason": "missing_required_artifact",
                "optimizer_hint": "Return the required Vue/Vite preview bundle before optimizing visual polish.",
            },
            "stage_status": [{"stage": "artifact_contract", "status": "failed"}],
        }
    )

    assert result["hard"] == 0
    assert result["profile_id"] == "vue_landing_page_v1"
    assert result["contract_status"] == "failed"
    assert result["quality_status"] == "not_run"
    assert result["human_feedback_alignment"]["required_improvements"] == ["make the layout responsive"]
    assert result["dimension_scores"]["artifact_contract"] == 0
    assert result["failed_dimensions"] == ["artifact_contract"]
    assert result["failure"]["primary_reason"] == "missing_required_artifact"
    assert result["metadata"]["failed_dimensions"] == ["artifact_contract"]
    assert result["metadata"]["failure"]["optimizer_hint"].startswith("Return the required Vue")


def test_invalid_scored_evaluation_preserves_structured_failure_feedback():
    result = normalize_scored_evaluation(
        {
            "hard": "not-a-score",
            "soft": 0,
            "metadata": {"evaluator": "fixture"},
            "failure": {
                "primary_reason": "invalid_score_with_failure_context",
                "optimizer_hint": "Keep the evaluator failure packet even when score parsing fails.",
            },
        }
    )

    assert result["hard"] is None
    assert result["score_status"] == "unscored"
    assert result["failure"]["primary_reason"] == "invalid_score_with_failure_context"
    assert result["metadata"]["failure"]["optimizer_hint"].startswith("Keep the evaluator failure packet")


def test_unscored_evaluation_preserves_structured_failure_feedback_from_metadata():
    result = make_unscored_evaluation(
        fail_reason="render smoke failed",
        target_status=TARGET_PASSED,
        evaluator_status=EVALUATOR_FAILED,
        blocker="render_smoke_failed",
        metadata={
            "failure": {
                "primary_reason": "render_smoke_failed",
                "optimizer_hint": "Fix the runtime error before sending the page to the visual judge.",
            },
            "contract_status": "failed",
            "quality_status": "not_run",
            "human_feedback_alignment": {"status": "feedback_available"},
            "stage_status": [{"stage": "render_smoke", "status": "failed"}],
        },
    )

    assert result["hard"] is None
    assert result["failure"]["primary_reason"] == "render_smoke_failed"
    assert result["contract_status"] == "failed"
    assert result["quality_status"] == "not_run"
    assert result["human_feedback_alignment"]["status"] == "feedback_available"
    assert result["metadata"]["stage_status"][0]["stage"] == "render_smoke"
