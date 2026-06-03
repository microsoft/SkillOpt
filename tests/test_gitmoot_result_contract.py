from __future__ import annotations

import pytest

from skillopt.engine.trainer import _compute_task_type_buckets, _extract_failure_patterns
from skillopt.utils.scoring import compute_score


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
