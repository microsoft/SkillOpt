"""Gitmoot rollout evaluators."""

from __future__ import annotations

import json
from typing import Any

from skillopt.model import chat_optimizer
from skillopt.utils import extract_json

LANDING_PAGE_EVALUATOR_ID = "landing_page_v1"
LANDING_PAGE_EVALUATOR_VERSION = "v1"
LANDING_PAGE_DIMENSIONS = (
    "mobile_responsiveness",
    "footer_presence_clarity",
    "hero_quality",
    "cta_clarity",
    "visual_images_relevance",
    "animation_motion_quality",
    "text_overlap_readability",
    "ranked_strength_preservation",
)


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
    if mode in {LANDING_PAGE_EVALUATOR_ID, "landing-page-v1", "landing_page"}:
        return _landing_page_score(item, response, config)
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


def _landing_page_score(item: dict[str, Any], response: str, config: dict[str, Any]) -> dict[str, Any]:
    raw, _usage = chat_optimizer(
        system=_landing_page_system_prompt(),
        user=_landing_page_user_prompt(item, response, config),
        max_completion_tokens=4096,
        retries=2,
        stage="gitmoot_landing_page_judge",
    )
    parsed = extract_json(raw)
    if not isinstance(parsed, dict):
        raise ValueError("landing_page_v1 judge did not return JSON")
    return _normalize_landing_page_score(parsed, raw=raw)


def _landing_page_system_prompt() -> str:
    dimensions = ", ".join(LANDING_PAGE_DIMENSIONS)
    return (
        "You are evaluating a generated landing page for Gitmoot SkillOpt. "
        "Return only JSON with keys evaluator_id, evaluator_version, hard, soft, "
        "dimension_scores, rationale, and fail_reason. "
        f"dimension_scores must contain exactly these 0-to-1 dimensions: {dimensions}. "
        "hard must be 1 only when the landing page is suitable to promote after review. "
        "soft must be a 0-to-1 overall quality score. Penalize missing mobile responsiveness, "
        "missing footer, weak hero/CTA, irrelevant or absent visuals, missing requested motion, "
        "text overlap/readability problems, and failure to preserve human-ranked strengths."
    )


def _landing_page_user_prompt(item: dict[str, Any], response: str, config: dict[str, Any]) -> str:
    return "\n\n".join(
        [
            "## Landing Page Evaluation Config",
            json.dumps(config, indent=2, sort_keys=True),
            "## Rubric",
            "\n".join(
                [
                    "- mobile_responsiveness: Works on mobile without overflow or unusable layout.",
                    "- footer_presence_clarity: Includes a clear footer with useful links or closing context.",
                    "- hero_quality: Hero is clear, polished, product-relevant, and visually strong.",
                    "- cta_clarity: Primary and final calls to action are obvious and well placed.",
                    "- visual_images_relevance: Graphics/images help explain the product and are not generic filler.",
                    "- animation_motion_quality: Motion exists when requested and supports comprehension.",
                    "- text_overlap_readability: Text does not overlap, occlude, or become unreadable.",
                    "- ranked_strength_preservation: Preserves strengths called out in human rankings/feedback.",
                ]
            ),
            "## Task Prompt, Artifacts, And Human Feedback",
            str(item.get("prompt") or ""),
            "## Generated Landing Page Response",
            response,
        ]
    )


def _normalize_landing_page_score(parsed: dict[str, Any], *, raw: str) -> dict[str, Any]:
    hard = _parse_landing_hard(parsed.get("hard"))
    soft = _parse_score(parsed.get("soft"), "soft")
    dimensions = _parse_dimension_scores(parsed.get("dimension_scores"))
    rationale = str(parsed.get("rationale") or parsed.get("reasoning") or "").strip()
    fail_reason = str(parsed.get("fail_reason") or "").strip()
    if hard not in {0, 1}:
        raise ValueError("landing_page_v1 hard must be 0 or 1")
    if not rationale:
        raise ValueError("landing_page_v1 rationale is required")
    if hard == 0 and not fail_reason:
        fail_reason = "landing_page_v1 judge marked this landing page below promotion quality"
    return {
        "hard": hard,
        "soft": soft,
        "fail_reason": "" if hard else fail_reason,
        "evaluator_id": LANDING_PAGE_EVALUATOR_ID,
        "evaluator_version": LANDING_PAGE_EVALUATOR_VERSION,
        "metadata": {
            "evaluator": LANDING_PAGE_EVALUATOR_ID,
            "evaluator_version": LANDING_PAGE_EVALUATOR_VERSION,
            "dimension_scores": dimensions,
            "rationale": rationale,
            "raw": raw[:1000],
        },
    }


def _parse_score(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"landing_page_v1 {label} must be numeric")
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"landing_page_v1 {label} must be numeric") from exc
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"landing_page_v1 {label} must be between 0 and 1")
    return score


def _parse_landing_hard(value: Any) -> int:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, int | float) and not isinstance(value, bool):
        if float(value) in {0.0, 1.0}:
            return int(float(value))
    if isinstance(value, str):
        normalized = value.strip().lower()
        try:
            numeric = float(normalized)
        except ValueError:
            numeric = None
        if numeric in {0.0, 1.0}:
            return int(numeric)
        if normalized in {"true", "yes", "pass", "passed", "success"}:
            return 1
        if normalized in {"false", "no", "fail", "failed", "failure"}:
            return 0
    raise ValueError("landing_page_v1 hard must be 0 or 1")


def _parse_dimension_scores(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ValueError("landing_page_v1 dimension_scores must be an object")
    missing = [dimension for dimension in LANDING_PAGE_DIMENSIONS if dimension not in value]
    if missing:
        raise ValueError(f"landing_page_v1 dimension_scores missing: {', '.join(missing)}")
    return {
        dimension: _parse_score(value[dimension], f"dimension_scores.{dimension}")
        for dimension in LANDING_PAGE_DIMENSIONS
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
