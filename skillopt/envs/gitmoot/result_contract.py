"""Result contract helpers for Gitmoot rollouts."""

from __future__ import annotations

import math
from typing import Any

TARGET_PASSED = "passed"
TARGET_FAILED = "failed"
TARGET_NOT_RUN = "not_run"
EVALUATOR_PASSED = "passed"
EVALUATOR_FAILED = "failed"
EVALUATOR_NOT_RUN = "not_run"
SCORE_SCORED = "scored"
SCORE_UNSCORED = "unscored"
DEFAULT_EVALUATOR_VERSION = "v0"
STRUCTURED_EVALUATOR_FIELDS = (
    "profile_id",
    "task_kind",
    "dimension_scores",
    "failure",
    "primary_reason",
    "human_reason",
    "optimizer_hint",
    "failed_checks",
    "evidence",
    "stage_status",
)


def normalize_scored_evaluation(
    score: dict[str, Any],
    *,
    target_trace_path: str = "",
    evaluator_trace_path: str = "",
) -> dict[str, Any]:
    """Normalize an evaluator response into the Gitmoot result contract."""
    metadata = score.get("metadata") if isinstance(score.get("metadata"), dict) else {}
    hard = _parse_hard(score.get("hard"))
    soft = _parse_soft(score.get("soft"))
    evaluator_id = str(score.get("evaluator_id") or metadata.get("evaluator") or "").strip()
    evaluator_version = str(
        score.get("evaluator_version")
        or metadata.get("evaluator_version")
        or DEFAULT_EVALUATOR_VERSION
    )
    if hard is None or soft is None:
        structured = _structured_evaluator_fields(score, metadata)
        return make_unscored_evaluation(
            fail_reason="evaluator returned invalid hard/soft scores",
            target_status=TARGET_PASSED,
            evaluator_status=EVALUATOR_FAILED,
            blocker="invalid_evaluator_score",
            evaluator_id=evaluator_id,
            evaluator_version=evaluator_version,
            target_trace_path=target_trace_path,
            evaluator_trace_path=evaluator_trace_path,
            metadata={**metadata, **structured},
        )
    structured = _structured_evaluator_fields(score, metadata)
    return {
        "hard": hard,
        "soft": soft,
        "fail_reason": str(score.get("fail_reason") or ""),
        **structured,
        "target_status": TARGET_PASSED,
        "evaluator_status": EVALUATOR_PASSED,
        "score_status": SCORE_SCORED,
        "blocker": "",
        "evaluator_id": evaluator_id,
        "evaluator_version": evaluator_version,
        "target_trace_path": target_trace_path,
        "evaluator_trace_path": evaluator_trace_path,
        "metadata": {
            **metadata,
            **structured,
            "target_status": TARGET_PASSED,
            "evaluator_status": EVALUATOR_PASSED,
            "score_status": SCORE_SCORED,
            "evaluator_id": evaluator_id,
            "evaluator_version": evaluator_version,
        },
    }


def make_unscored_evaluation(
    *,
    fail_reason: str,
    target_status: str,
    evaluator_status: str,
    blocker: str,
    evaluator_id: str = "",
    evaluator_version: str = "",
    target_trace_path: str = "",
    evaluator_trace_path: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an unscored result that cannot be mistaken for a real score."""
    reason = fail_reason or blocker or "unscored evaluation"
    normalized_metadata = dict(metadata or {})
    normalized_metadata.setdefault("evaluator", evaluator_id or "not_run")
    structured = _structured_evaluator_fields(normalized_metadata)
    normalized_metadata.update(
        {
            **structured,
            "target_status": target_status,
            "evaluator_status": evaluator_status,
            "score_status": SCORE_UNSCORED,
            "blocker": blocker,
            "evaluator_id": evaluator_id,
            "evaluator_version": evaluator_version,
        }
    )
    return {
        "hard": None,
        "soft": None,
        "fail_reason": reason,
        **structured,
        "target_status": target_status,
        "evaluator_status": evaluator_status,
        "score_status": SCORE_UNSCORED,
        "blocker": blocker,
        "evaluator_id": evaluator_id,
        "evaluator_version": evaluator_version,
        "target_trace_path": target_trace_path,
        "evaluator_trace_path": evaluator_trace_path,
        "metadata": normalized_metadata,
    }


def _parse_hard(value: Any) -> int | None:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, int | float):
        number = float(value)
        if not math.isfinite(number):
            return None
        return 1 if number > 0 else 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "pass", "passed", "success"}:
            return 1
        if normalized in {"0", "false", "no", "fail", "failed", "failure"}:
            return 0
        try:
            number = float(normalized)
        except ValueError:
            return None
        if not math.isfinite(number):
            return None
        return 1 if number > 0 else 0
    return None


def _parse_soft(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, int | float):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    if not math.isfinite(number):
        return None
    return max(0.0, min(1.0, number))


def _structured_evaluator_fields(*sources: dict[str, Any]) -> dict[str, Any]:
    structured: dict[str, Any] = {}
    for key in STRUCTURED_EVALUATOR_FIELDS:
        for source in sources:
            value = source.get(key)
            if value is not None:
                structured[key] = value
                break
    return structured
