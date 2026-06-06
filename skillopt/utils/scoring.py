"""Scoring and hashing utilities."""
from __future__ import annotations

import hashlib
from typing import Any


def compute_score(results: list) -> tuple[float, float]:
    """Compute hard and soft accuracy from a list of episode results.

    Accepts both plain dicts and :class:    instances.  hard may be continuous
    (0.0-1.0) when using smoothed reward.
    """
    if not results:
        return 0.0, 0.0

    def _hard(r: object) -> float:
        if is_quality_failed_result(r):
            return 0.0
        return _score_value(r, "hard", 0)

    def _soft(r: object) -> float:
        return _score_value(r, "soft", 0.0)

    hard = sum(_hard(r) for r in results) / len(results)
    soft = sum(_soft(r) for r in results) / len(results)
    return hard, soft


def compute_structural_score(results: list) -> tuple[float, float]:
    """Compute raw structural hard/soft scores without applying quality-gate failure."""
    if not results:
        return 0.0, 0.0

    hard = sum(_score_value(r, "hard", 0) for r in results) / len(results)
    soft = sum(_score_value(r, "soft", 0.0) for r in results) / len(results)
    return hard, soft


def _score_value(result: object, key: str, default: float) -> float:
    if hasattr(result, key):
        value = getattr(result, key)
    elif isinstance(result, dict):
        value = result.get(key, default)
    else:
        value = default
    if _is_unscored_result(result) or value is None:
        raise ValueError(f"cannot compute aggregate score for unscored result {_result_label(result)}")
    return float(value)


def _is_unscored_result(result: object) -> bool:
    status = _result_field(result, "score_status", "")
    return str(status).strip().lower() == "unscored"


def is_quality_failed_result(result: object) -> bool:
    """Return true when a scored result failed the structured quality gate."""
    status = _result_field(result, "quality_status", "")
    return str(status).strip().lower().replace("-", "_") == "failed"


def _result_label(result: object) -> str:
    result_id = _result_field(result, "id", "unknown")
    blocker = _result_field(result, "blocker", "")
    fail_reason = _result_field(result, "fail_reason", "")
    details = [f"id={result_id!r}", "score_status='unscored'"]
    if blocker:
        details.append(f"blocker={blocker!r}")
    if fail_reason:
        details.append(f"fail_reason={str(fail_reason)[:120]!r}")
    return " ".join(details)


def _result_field(result: object, key: str, default: Any) -> Any:
    if hasattr(result, key):
        return getattr(result, key)
    extras = getattr(result, "extras", None)
    if isinstance(extras, dict) and key in extras:
        return extras.get(key, default)
    if isinstance(result, dict):
        return result.get(key, default)
    return default


def skill_hash(content: str) -> str:
    """Return a short deterministic hash of skill content (for caching)."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]
