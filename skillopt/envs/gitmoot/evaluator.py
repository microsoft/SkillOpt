"""Gitmoot rollout evaluators."""

from __future__ import annotations

import json
from typing import Any

from skillopt.model import chat_optimizer
from skillopt.utils import extract_json


def evaluate_response(item: dict[str, Any], response: str, evaluator_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Evaluate one Gitmoot response.

    Supported deterministic modes are intended for fixtures and CI. When no
    supported explicit mode is present, an LLM judge compares the response
    against source, baseline, candidate, and feedback context.
    """
    config = evaluator_config if isinstance(evaluator_config, dict) else {}
    mode = str(config.get("mode") or item.get("metadata", {}).get("evaluator_mode") or "").strip().lower()
    if mode in {"fixture", "deterministic", "mock"}:
        return _fixture_score(item)
    if mode in {"contains", "substring"}:
        return _contains_score(item, response, config)
    return _judge_score(item, response, config)


def _fixture_score(item: dict[str, Any]) -> dict[str, Any]:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    hard = bool(metadata.get("expected_hard", True))
    soft = float(metadata.get("expected_soft", 1.0 if hard else 0.0))
    return {
        "hard": 1 if hard else 0,
        "soft": max(0.0, min(1.0, soft)),
        "fail_reason": "" if hard else str(metadata.get("fail_reason") or "fixture evaluator marked this item failed"),
        "metadata": {"evaluator": "fixture"},
    }


def _contains_score(item: dict[str, Any], response: str, config: dict[str, Any]) -> dict[str, Any]:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    required = str(config.get("required_text") or metadata.get("required_text") or "").strip()
    if not required:
        return {
            "hard": 0,
            "soft": 0.0,
            "fail_reason": "contains evaluator requires required_text",
            "metadata": {"evaluator": "contains"},
        }
    ok = required.lower() in response.lower()
    return {
        "hard": 1 if ok else 0,
        "soft": 1.0 if ok else 0.0,
        "fail_reason": "" if ok else f"response did not contain required text {required!r}",
        "metadata": {"evaluator": "contains", "required_text": required},
    }


def _judge_score(item: dict[str, Any], response: str, config: dict[str, Any]) -> dict[str, Any]:
    system = (
        "You are evaluating a Gitmoot SkillOpt candidate response. "
        "Return only JSON with keys hard, soft, fail_reason, and reasoning. "
        "hard must be 0 or 1. soft must be a number from 0 to 1."
    )
    user = "\n\n".join(
        [
            "## Evaluation Config",
            json.dumps(config, indent=2, sort_keys=True),
            "## Task Prompt",
            str(item.get("prompt") or ""),
            "## Candidate Response",
            response,
        ]
    )
    raw, _usage = chat_optimizer(system=system, user=user, max_completion_tokens=2048, retries=2, stage="gitmoot_judge")
    parsed = extract_json(raw)
    if not isinstance(parsed, dict):
        return {
            "hard": 0,
            "soft": 0.0,
            "fail_reason": "judge did not return JSON",
            "metadata": {"evaluator": "llm_judge", "raw": raw[:1000]},
        }
    hard = _parse_hard(parsed.get("hard"))
    try:
        soft = float(parsed.get("soft", hard))
    except (TypeError, ValueError):
        soft = float(hard)
    soft = max(0.0, min(1.0, soft))
    fail_reason = str(parsed.get("fail_reason") or "")
    return {
        "hard": hard,
        "soft": soft,
        "fail_reason": "" if hard else fail_reason or "judge marked this item failed",
        "metadata": {
            "evaluator": "llm_judge",
            "judge_derived": True,
            "reasoning": str(parsed.get("reasoning") or ""),
        },
    }


def _parse_hard(value: Any) -> int:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, int | float):
        return 1 if float(value) > 0 else 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "pass", "passed", "success"}:
            return 1
        if normalized in {"0", "false", "no", "fail", "failed", "failure"}:
            return 0
        try:
            return 1 if float(normalized) > 0 else 0
        except ValueError:
            return 0
    return 0
