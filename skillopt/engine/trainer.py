"""ReflACT Trainer — the main training loop.

Orchestrates the 6-stage ReflACT pipeline:
  1. Rollout   — execute episodes with current skill
  2. Reflect   — analyze trajectories, generate patches
  3. Aggregate — hierarchical merge of patches
  4. Select    — rank and select top edits
  5. Update    — apply edits to skill document
  6. Evaluate  — validate candidate skill, accept/reject

The trainer is environment-agnostic; all environment-specific logic is
delegated to an :class:`~skillopt.envs.base.EnvAdapter` instance.
"""
from __future__ import annotations

import glob
import json
import math
import os
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass

from skillopt.datasets.base import BatchSpec
from skillopt.envs.base import EnvAdapter
from skillopt.envs.gitmoot.package import safe_item_path_segment
from skillopt.evaluation.gate import evaluate_gate, find_gate_block, select_gate_score
from skillopt.gradient.aggregate import merge_patches
from skillopt.model import (
    configure_azure_openai,
    configure_claude_code_exec,
    configure_codex_exec,
    configure_minimax_chat,
    configure_qwen_chat,
    get_token_summary,
    set_optimizer_backend,
    set_optimizer_deployment,
    set_reasoning_effort,
    set_target_backend,
    set_target_deployment,
)
from skillopt.optimizer.clip import rank_and_select
from skillopt.optimizer.lr_autonomous import decide_autonomous_learning_rate
from skillopt.optimizer.meta_skill import run_meta_skill
from skillopt.optimizer.rewrite import rewrite_skill_from_suggestions
from skillopt.optimizer.scheduler import build_scheduler
from skillopt.optimizer.skill import apply_patch_with_report
from skillopt.optimizer.slow_update import (
    build_comparison_pairs,
    extract_slow_update_field,
    inject_empty_slow_update_field,
    replace_slow_update_field,
    run_slow_update,
    save_comparison_pairs,
)
from skillopt.optimizer.update_modes import (
    get_payload_items,
    is_full_rewrite_minibatch_mode,
    normalize_update_mode,
    payload_label,
    short_item_summary,
)
from skillopt.utils import compute_score, skill_hash

# ── Patch normalization ───────────────────────────────────────────────────────

def _normalise_patches(
    raw_patches: list[dict | None],
    update_mode: str = "patch",
) -> tuple[list[dict], list[dict]]:
    """Extract inner 'patch' sub-dict, split into failure/success lists.

    Each element is expected to conform to :class:`~skillopt.types.RawPatch`.
    """
    mode = normalize_update_mode(update_mode)
    failure: list[dict] = []
    success: list[dict] = []
    for p in raw_patches:
        if not isinstance(p, dict):
            continue
        inner = p.get("patch", p)
        if not isinstance(inner, dict):
            continue
        items = get_payload_items(inner, mode)
        if not items:
            continue
        support = max(int(p.get("batch_size", 0) or 0), 1)
        for item in items:
            if isinstance(item, dict):
                item.setdefault("source_type", p.get("source_type", "failure"))
                item.setdefault("support_count", support)
        if p.get("source_type", "failure") == "success":
            success.append(inner)
        else:
            failure.append(inner)
    return failure, success


def _normalise_longitudinal_pair_policy(policy: str | None) -> str:
    raw = str(policy or "mixed").strip().lower()
    aliases = {
        "mixed": "mixed",
        "default": "mixed",
        "random": "mixed",
        "all": "mixed",
        "changed": "changed",
        "change": "changed",
        "delta": "changed",
        "10_01": "changed",
        "01_10": "changed",
        "unchanged": "unchanged",
        "stable": "unchanged",
        "same": "unchanged",
        "00_11": "unchanged",
    }
    if raw not in aliases:
        raise ValueError(
            "optimizer.longitudinal_pair_policy must be one of "
            "mixed, changed, unchanged"
        )
    return aliases[raw]


@dataclass
class NoMeaningfulChangeCheck:
    reasons: list[str]
    retry_hints: dict[str, list[str]]


def _dedupe_texts(values: list[str], *, limit: int = 8) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= limit:
            break
    return out


def _flatten_trait_values(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            out.extend(_flatten_trait_values(item))
        return out
    if isinstance(value, dict):
        out: list[str] = []
        for key, item in value.items():
            for text in _flatten_trait_values(item):
                out.append(f"{key}: {text}" if str(key).strip() else text)
        return out
    return []


def _feedback_retry_hints(dataloader, result_ids: set[str]) -> dict[str, list[str]]:
    hints = {
        "preserve": [],
        "improve": [],
        "avoid": [],
        "already_covered": [],
    }
    if dataloader is None or not hasattr(dataloader, "train_items"):
        return hints
    try:
        train_items = dataloader.train_items
    except Exception:  # noqa: BLE001
        return hints
    for item in train_items:
        if result_ids and str(item.get("id", "")) not in result_ids:
            continue
        events = item.get("ranked_feedback_events")
        if not isinstance(events, list):
            continue
        for event in events:
            if not isinstance(event, dict):
                continue
            hints["preserve"].extend(_flatten_trait_values(event.get("useful_traits")))
            hints["improve"].extend(_flatten_trait_values(event.get("required_improvements")))
            hints["avoid"].extend(_flatten_trait_values(event.get("rejected_traits")))
    return {key: _dedupe_texts(values) for key, values in hints.items()}


def _ranked_feedback_retry_hints(dataloader, result_ids: set[str] | None = None) -> dict[str, list[str]]:
    return _feedback_retry_hints(dataloader, result_ids or set())


def _ranked_feedback_context_from_events(events_with_item: list[tuple[str, dict]]) -> dict:
    if not events_with_item:
        return {}

    def _event_strings(*keys: str) -> list[str]:
        values: list[str] = []
        for _item_id, event in events_with_item:
            for key in keys:
                value = event.get(key)
                values.extend(_flatten_trait_values(value))
        return _dedupe_texts(values, limit=12)

    reasoning = _event_strings("choice", "reasoning", "reviewer_reasoning")
    packet = {
        "source_item_ids": _dedupe_texts([item_id for item_id, _event in events_with_item if item_id], limit=12),
        "preserve": _event_strings("useful_traits", "winning_traits", "preserve"),
        "improve": _event_strings("required_improvements", "improvements", "required_improvement_themes"),
        "avoid": _event_strings("rejected_traits", "losing_traits", "avoid"),
        "rankings": _event_strings("ranking", "rankings"),
        "reviewer_reasoning": reasoning[:8],
        "quality": _event_strings("quality"),
        "continue_mode": _event_strings("continue_mode"),
        "promote": _event_strings("promote"),
    }
    return {
        key: value
        for key, value in packet.items()
        if value
    }


def _ranked_feedback_context_packet(
    dataloader,
    result_ids: set[str] | None = None,
    *,
    fallback_to_all: bool = True,
) -> dict:
    if dataloader is None or not hasattr(dataloader, "train_items"):
        return {}
    try:
        train_items = [item for item in dataloader.train_items if isinstance(item, dict)]
    except Exception:  # noqa: BLE001
        return {}
    explicit_filter = result_ids is not None
    ids = {str(item_id or "").strip() for item_id in (result_ids or set()) if str(item_id or "").strip()}
    if explicit_filter and not ids and not fallback_to_all:
        return {}
    selected_items = [
        item
        for item in train_items
        if not ids or str(item.get("id") or "").strip() in ids
    ]
    if ids and not selected_items and fallback_to_all:
        selected_items = train_items
    if ids and not selected_items:
        return {}

    events_with_item: list[tuple[str, dict]] = []
    for item in selected_items:
        item_id = str(item.get("id") or "").strip()
        for event in _item_ranked_feedback_events(item):
            events_with_item.append((item_id, event))
    return _ranked_feedback_context_from_events(events_with_item)


def _format_ranked_feedback_packet(packet: dict | None) -> str:
    if not isinstance(packet, dict) or not packet:
        return ""
    labels = {
        "source_item_ids": "Feedback source items",
        "rankings": "Rankings / pairwise preferences",
        "preserve": "Preserve winning traits",
        "improve": "Required improvements",
        "avoid": "Avoid losing traits",
        "reviewer_reasoning": "Reviewer reasoning",
        "quality": "Quality labels",
        "continue_mode": "Continue modes",
        "promote": "Promote decisions",
    }
    lines = ["## Ranked Human Feedback Context"]
    for key, label in labels.items():
        values = packet.get(key)
        if isinstance(values, list) and values:
            lines.append(f"{label}: " + "; ".join(str(value) for value in values[:8]))
    return "\n".join(lines)


def _human_feedback_hint_suffix(packet: dict | None) -> str:
    if not isinstance(packet, dict) or not packet:
        return ""
    themes = _dedupe_texts(
        (packet.get("improve") if isinstance(packet.get("improve"), list) else [])
        + (packet.get("preserve") if isinstance(packet.get("preserve"), list) else [])
        + (packet.get("avoid") if isinstance(packet.get("avoid"), list) else []),
        limit=8,
    )
    if not themes:
        return ""
    return " Preserve and resolve ranked human feedback themes: " + "; ".join(str(theme) for theme in themes) + "."


def _with_human_feedback_context(packet: dict | None, feedback_context: dict | None) -> dict | None:
    if not isinstance(packet, dict):
        return packet
    if not isinstance(feedback_context, dict) or not feedback_context:
        return packet
    enriched = dict(packet)
    existing_context = enriched.get("human_feedback_context")
    if not isinstance(existing_context, dict) or not existing_context:
        enriched["human_feedback_context"] = feedback_context
    hint = str(enriched.get("optimizer_hint") or "").strip()
    suffix = _human_feedback_hint_suffix(feedback_context)
    if suffix and suffix.strip() not in hint:
        enriched["optimizer_hint"] = (hint + suffix).strip()
    failed_dimensions = enriched.get("failed_dimensions") if isinstance(enriched.get("failed_dimensions"), list) else []
    if "human_feedback_alignment" not in failed_dimensions:
        enriched["failed_dimensions"] = _dedupe_texts(
            [str(item) for item in failed_dimensions] + ["human_feedback_alignment"],
            limit=12,
        )
    return enriched


def _has_ranked_feedback(dataloader, result_ids: set[str] | None = None) -> bool:
    if dataloader is None or not hasattr(dataloader, "train_items"):
        return False
    try:
        train_items = dataloader.train_items
    except Exception:  # noqa: BLE001
        return False
    ids = result_ids or set()
    for item in train_items:
        if ids and str(item.get("id", "")) not in ids:
            continue
        events = item.get("ranked_feedback_events")
        if isinstance(events, list) and events:
            return True
    return False


def _normalize_feedback_direct_mode(value: str | None) -> str:
    raw = str(value or "auto").strip().lower()
    if raw not in {"auto", "on", "off"}:
        raise ValueError("optimizer.feedback_direct_mode must be one of auto, on, off")
    return raw


def _ranked_feedback_item_index(dataloader) -> dict[str, dict]:
    if dataloader is None or not hasattr(dataloader, "train_items"):
        return {}
    try:
        train_items = dataloader.train_items
    except Exception:  # noqa: BLE001
        return {}
    index: dict[str, dict] = {}
    for item in train_items:
        if isinstance(item, dict):
            item_id = str(item.get("id") or "").strip()
            if item_id:
                index[item_id] = item
    return index


def _item_ranked_feedback_events(item: dict | None) -> list[dict]:
    if not isinstance(item, dict):
        return []
    events = item.get("ranked_feedback_events")
    if isinstance(events, list):
        return [event for event in events if isinstance(event, dict)]
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    events = metadata.get("ranked_feedback_events")
    if isinstance(events, list):
        return [event for event in events if isinstance(event, dict)]
    return []


def _ranked_feedback_requests_optimization(events: list[dict]) -> bool:
    if not events:
        return False
    for event in events:
        promote = str(event.get("promote") or "").strip().lower()
        continue_mode = str(event.get("continue_mode") or "").strip().lower()
        if promote in {"no", "false", "0"}:
            return True
        if continue_mode in {"refine", "explore", "distill"}:
            return True
        required = _flatten_trait_values(event.get("required_improvements"))
        if required:
            return True
        if promote not in {"yes", "true", "1"} and continue_mode not in {"stop", "validate"}:
            return True
    return False


def _feedback_direct_items(
    *,
    dataloader,
    batch_items: list[dict],
    mode: str,
) -> list[dict]:
    normalized_mode = _normalize_feedback_direct_mode(mode)
    if normalized_mode == "off":
        return []
    index = _ranked_feedback_item_index(dataloader)
    direct_items: list[dict] = []
    for batch_item in batch_items:
        if not isinstance(batch_item, dict):
            continue
        item_id = str(batch_item.get("id") or "").strip()
        full_item = index.get(item_id, batch_item)
        events = _item_ranked_feedback_events(full_item)
        if normalized_mode == "auto" and not _ranked_feedback_requests_optimization(events):
            continue
        if normalized_mode == "on" and not events:
            continue
        merged = {**batch_item, **full_item}
        merged["ranked_feedback_events"] = events
        direct_items.append(merged)
    return direct_items


def _non_feedback_direct_items(
    *,
    batch_items: list[dict],
    feedback_direct_items: list[dict],
) -> list[dict]:
    direct_ids = {str(item.get("id") or "").strip() for item in feedback_direct_items}
    return [
        item
        for item in batch_items
        if isinstance(item, dict) and str(item.get("id") or "").strip() not in direct_ids
    ]


def _format_feedback_direct_context(items: list[dict], dataloader) -> str:
    hints = _ranked_feedback_retry_hints(dataloader, {str(item.get("id") or "") for item in items})
    lines = [
        "## Feedback-Direct Optimization",
        "Ranked human feedback is available before target rollout.",
        "Update the skill from the review feedback first; do not wait for a fresh old-skill artifact failure.",
        "The optimizer output must be a skill update. Do not output Vue files, JSON bundles, YAML review data, or target artifacts.",
    ]
    labels = {
        "preserve": "Preserve winning traits",
        "improve": "Required improvements",
        "avoid": "Avoid losing traits",
    }
    for key, label in labels.items():
        values = hints.get(key) or []
        if values:
            lines.append(f"{label}: " + "; ".join(values[:8]))
    return "\n".join(lines)


def _write_feedback_direct_predictions(
    *,
    items: list[dict],
    prediction_dir: str,
) -> list[dict]:
    os.makedirs(prediction_dir, exist_ok=True)
    results: list[dict] = []
    for index, item in enumerate(items, start=1):
        item_id = str(item.get("id") or f"feedback-{index}").strip()
        prediction_id = safe_item_path_segment(item_id)
        item_dir = os.path.join(prediction_dir, prediction_id)
        os.makedirs(item_dir, exist_ok=True)
        events = _item_ranked_feedback_events(item)
        feedback_json = json.dumps(events, indent=2, ensure_ascii=False)
        conversation = [
            {
                "role": "system",
                "content": (
                    "Feedback-direct optimizer item. Ranked human review feedback should drive "
                    "a skill update before target rollout."
                ),
            },
            {
                "role": "user",
                "content": feedback_json,
            },
        ]
        with open(os.path.join(item_dir, "conversation.json"), "w", encoding="utf-8") as f:
            json.dump(conversation, f, indent=2, ensure_ascii=False)
        prompt = str(item.get("prompt") or item.get("task_description") or item_id)
        with open(os.path.join(item_dir, "target_user_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(prompt)
        feedback_context = _ranked_feedback_context_from_events([(item_id, event) for event in events])
        failure = {
            "primary_reason": "ranked_human_feedback_requires_skill_update",
            "human_reason": (
                "Ranked human feedback was imported and asks the optimizer to refine the skill "
                "before another target rollout."
            ),
            "optimizer_hint": _human_feedback_not_distilled_hint(
                {
                    "preserve": _dedupe_texts(
                        [
                            text
                            for event in events
                            for text in _flatten_trait_values(event.get("useful_traits"))
                        ]
                    ),
                    "improve": _dedupe_texts(
                        [
                            text
                            for event in events
                            for text in _flatten_trait_values(event.get("required_improvements"))
                        ]
                    ),
                    "avoid": _dedupe_texts(
                        [
                            text
                            for event in events
                            for text in _flatten_trait_values(event.get("rejected_traits"))
                        ]
                    ),
                    "already_covered": [],
                }
            ),
            "failed_dimensions": ["human_feedback_alignment"],
            "evidence": [feedback_json[:2000]],
            "stage_status": [{"stage": "feedback_direct", "status": "failed"}],
            "human_feedback_context": feedback_context,
        }
        results.append(
            {
                "id": item_id,
                "prediction_id": prediction_id,
                "hard": 0,
                "soft": 0.0,
                "response": "",
                "fail_reason": "ranked human feedback requests feedback-direct skill optimization",
                "agent_ok": True,
                "n_turns": 1,
                "task_type": item.get("task_type", "gitmoot-skillopt"),
                "task_description": item.get("task_description", prompt),
                "metadata": {
                    **(item.get("metadata") if isinstance(item.get("metadata"), dict) else {}),
                    "ranked_feedback_events": events,
                    "feedback_direct": True,
                    "prediction_id": prediction_id,
                    "failure": failure,
                },
                "target_status": "not_run",
                "evaluator_status": "not_run",
                "score_status": "scored",
                "blocker": "",
                "failure": failure,
                "primary_reason": failure["primary_reason"],
                "human_reason": failure["human_reason"],
                "optimizer_hint": failure["optimizer_hint"],
                "failed_dimensions": failure["failed_dimensions"],
                "evidence": failure["evidence"],
                "stage_status": failure["stage_status"],
                "target_trace_path": os.path.join(item_dir, "conversation.json"),
                "evaluator_trace_path": "",
                "target_system_prompt": "Feedback-direct optimization input",
                "target_user_prompt": prompt,
            }
        )
    return results


def _human_feedback_not_distilled_hint(retry_hints: dict[str, list[str]]) -> str:
    themes = _dedupe_texts(
        (retry_hints.get("improve") or [])
        + (retry_hints.get("preserve") or [])
        + (retry_hints.get("avoid") or []),
        limit=8,
    )
    if themes:
        return "The optimizer ignored ranked human feedback. Update the skill using these requested themes: " + "; ".join(themes)
    return (
        "The optimizer ignored ranked human feedback. Update the skill using the requested themes: "
        "branding, product visuals, animation, mobile responsiveness, spacing, CTA/footer quality, "
        "and Tailwind-style polish."
    )


def _contains_topic(content: str, topic: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", " ", str(topic or "").lower()).strip()
    if not normalized:
        return False
    words = [word for word in normalized.split() if len(word) >= 4]
    if not words:
        return False
    haystack = re.sub(r"[^a-z0-9]+", " ", content.lower())
    return all(word in haystack for word in words[:4])


def _patch_texts(items: list[dict], update_mode: str) -> list[str]:
    texts: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if is_full_rewrite_minibatch_mode(update_mode):
            texts.extend(_flatten_trait_values(item.get("change_summary")))
            texts.append(str(item.get("title", "")))
        elif normalize_update_mode(update_mode) == "rewrite_from_suggestions":
            texts.append(str(item.get("title", "")))
            texts.append(str(item.get("instruction", "")))
        else:
            texts.append(str(item.get("content", "")))
            texts.append(str(item.get("target", "")))
    return _dedupe_texts(texts, limit=12)


def _detect_no_meaningful_change(
    *,
    current_skill: str,
    candidate_skill: str,
    ranked_items: list[dict],
    apply_report: list[dict],
    update_mode: str,
    dataloader,
    rollout_results: list[dict],
) -> NoMeaningfulChangeCheck:
    reasons: list[str] = []
    current_hash = skill_hash(current_skill)
    candidate_hash = skill_hash(candidate_skill)
    unchanged = current_hash == candidate_hash
    if not ranked_items:
        reasons.append("no_meaningful_skill_change")
    if unchanged:
        reasons.extend(["no_meaningful_skill_change", "candidate_content_unchanged"])
    if apply_report:
        applied = [
            row for row in apply_report
            if str(row.get("status", "")).startswith("applied")
        ]
        if not applied:
            reasons.extend(["no_meaningful_skill_change", "duplicate_or_already_covered_patch"])

    result_ids = {str(row.get("id", "")) for row in rollout_results if isinstance(row, dict)}
    retry_hints = _feedback_retry_hints(dataloader, result_ids)
    patch_texts = _patch_texts(ranked_items, update_mode)
    already_covered = [
        text for text in patch_texts
        if _contains_topic(current_skill, text)
    ]
    retry_hints["already_covered"] = _dedupe_texts(retry_hints["already_covered"] + already_covered)

    explicit_traits = bool(retry_hints["preserve"] or retry_hints["improve"] or retry_hints["avoid"])
    if explicit_traits and unchanged:
        reasons.append("human_feedback_not_incorporated")
    if unchanged and patch_texts and already_covered and len(already_covered) == len(patch_texts):
        reasons.extend(["no_meaningful_skill_change", "duplicate_or_already_covered_patch"])

    return NoMeaningfulChangeCheck(
        reasons=_dedupe_texts(reasons, limit=8),
        retry_hints=retry_hints,
    )


def _format_noop_retry_context(check: NoMeaningfulChangeCheck, attempt: int) -> str:
    lines = [
        "## Optimizer No-Op Retry",
        f"Attempt {attempt} was rejected before candidate import.",
        "Reasons: " + ", ".join(check.reasons),
        "Produce a meaningful skill change that incorporates the imported human feedback. "
        "Do not repeat unchanged or already-covered contract language.",
    ]
    labels = {
        "preserve": "Preserve",
        "improve": "Improve",
        "avoid": "Avoid",
        "already_covered": "Already covered",
    }
    for key, label in labels.items():
        values = check.retry_hints.get(key) or []
        if values:
            lines.append(f"{label}: " + "; ".join(values[:6]))
    return "\n".join(lines)


def _fmt_score(value: float | None) -> str:
    return "unknown" if value is None else f"{float(value):.4f}"


def _score_delta(before: float | None, after: float | None) -> str:
    if before is None or after is None:
        return "unknown"
    return f"{float(after) - float(before):+.4f}"


def _gate_rejection_retry_decision(
    packet: dict | None,
    *,
    attempt: int,
    budget: int,
    seen_reasons: set[str],
    close_gap: float | None = None,
) -> tuple[bool, str]:
    if attempt >= budget:
        return False, "budget_exhausted"
    if not isinstance(packet, dict):
        return False, "missing_structured_gate_rejection"
    if not packet.get("retryable", False):
        return False, "non_retryable_gate_rejection"
    baseline = packet.get("baseline") if isinstance(packet.get("baseline"), dict) else {}
    candidate = packet.get("candidate") if isinstance(packet.get("candidate"), dict) else {}
    if close_gap is not None:
        candidate_hard = candidate.get("hard")
        if isinstance(candidate_hard, bool) or not isinstance(candidate_hard, int | float):
            return False, "missing_candidate_hard_score"
        if float(candidate_hard) < 1.0 and not _candidate_hard_score_allows_gate_retry(packet):
            return False, "candidate_hard_score_failed"
        baseline_gate = baseline.get("gate_score")
        candidate_gate = candidate.get("gate_score")
        if (
            isinstance(baseline_gate, bool)
            or isinstance(candidate_gate, bool)
            or not isinstance(baseline_gate, int | float)
            or not isinstance(candidate_gate, int | float)
        ):
            return False, "missing_gate_score_delta"
        if float(baseline_gate) - float(candidate_gate) > max(0.0, float(close_gap)):
            return False, "gate_score_gap_too_large"
    has_actionable_context = any(
        str(value or "").strip()
        for value in (
            packet.get("optimizer_hint"),
            packet.get("human_reason"),
            baseline.get("evaluator_reasoning"),
            candidate.get("evaluator_reasoning"),
        )
    )
    if not has_actionable_context:
        return False, "missing_actionable_rationale"
    reason = str(packet.get("primary_reason") or packet.get("rejection_type") or "").strip()
    if not reason:
        return False, "missing_rejection_reason"
    signature = _gate_rejection_signature(packet)
    if signature in seen_reasons:
        return False, "repeated_rejection_reason"
    return True, "retryable"


def _candidate_hard_score_allows_gate_retry(packet: dict) -> bool:
    if _is_wrong_artifact_rejection(packet):
        return False
    failed_dimensions = packet.get("failed_dimensions") if isinstance(packet.get("failed_dimensions"), list) else []
    failed_checks = packet.get("failed_checks") if isinstance(packet.get("failed_checks"), list) else []
    fields = [
        packet.get("primary_reason"),
        packet.get("rejection_type"),
        *(failed_dimensions or []),
    ]
    for check in failed_checks:
        if isinstance(check, dict):
            fields.extend([check.get("check"), check.get("reason"), check.get("severity")])
    normalized = {
        str(field or "").strip().lower().replace("-", "_").replace(".", "_")
        for field in fields
        if str(field or "").strip()
    }
    retryable_hard_zero_tokens = {
        "animation_motion_quality",
        "brand_identity",
        "cta_clarity",
        "footer_presence_clarity",
        "hero_quality",
        "human_feedback_not_resolved",
        "human_feedback_resolution",
        "mobile_responsiveness",
        "proof_trust_content",
        "ranked_strength_preservation",
        "text_overlap_readability",
        "visual_images_relevance",
        "visual_quality",
    }
    return any(
        _matches_retryable_hard_zero_token(field, token)
        for field in normalized
        for token in retryable_hard_zero_tokens
    )


def _matches_retryable_hard_zero_token(field: str, token: str) -> bool:
    return (
        field == token
        or field.endswith(f"_{token}")
        or field.startswith(f"{token}_")
        or f"_{token}_" in field
    )


def _gate_rejection_signature(packet: dict | None) -> str:
    if not isinstance(packet, dict):
        return ""
    reason = str(packet.get("primary_reason") or packet.get("rejection_type") or "").strip()
    return "|".join(
        str(value or "").strip()
        for value in (
            reason,
            packet.get("attempted_patch"),
            packet.get("optimizer_hint"),
        )
    )


def _is_wrong_artifact_rejection(packet: dict | None) -> bool:
    if not isinstance(packet, dict):
        return False
    fields = [
        packet.get("primary_reason"),
        packet.get("rejection_type"),
        *(packet.get("failed_dimensions") if isinstance(packet.get("failed_dimensions"), list) else []),
    ]
    return any(str(field or "").strip().lower() == "wrong_artifact_type" for field in fields)


def _gate_rejection_with_retry_attempts(packet: dict | None, *, used: int, budget: int) -> dict | None:
    if not isinstance(packet, dict):
        return None
    enriched = dict(packet)
    enriched["retry_attempts"] = f"{used}/{budget}"
    if used >= budget:
        enriched["next_action"] = (
            "Stop this pass without final test eval; collect more feedback or rerun with a larger "
            "gate-reject retry budget."
        )
    else:
        enriched["next_action"] = (
            "Retry the optimizer with this gate-rejection packet before spending final test-eval budget."
        )
    return enriched


def _format_gate_reject_retry_context(packet: dict, attempt: int, budget: int) -> str:
    baseline = packet.get("baseline") if isinstance(packet.get("baseline"), dict) else {}
    candidate = packet.get("candidate") if isinstance(packet.get("candidate"), dict) else {}
    delta_summary = packet.get("delta_summary") if isinstance(packet.get("delta_summary"), dict) else {}
    failed_dimensions = packet.get("failed_dimensions") if isinstance(packet.get("failed_dimensions"), list) else []
    failed_checks = packet.get("failed_checks") if isinstance(packet.get("failed_checks"), list) else []
    evidence = packet.get("evidence") if isinstance(packet.get("evidence"), list) else []
    attempted_patch = str(packet.get("attempted_patch") or "selection-gated candidate update").strip()
    lines = [
        "## Gate Rejection Retry",
        f"Attempt {attempt}/{budget} was rejected by baseline-vs-candidate selection.",
        f"Rejection type: {packet.get('rejection_type', 'unknown')}",
        f"Primary reason: {packet.get('primary_reason', 'unknown')}",
        f"Human reason: {packet.get('human_reason', '')}",
        f"Previous patch summary: {attempted_patch}",
        "Baseline scores: "
        f"hard={_fmt_score(baseline.get('hard'))}, "
        f"soft={_fmt_score(baseline.get('soft'))}, "
        f"gate={_fmt_score(baseline.get('gate_score'))}",
        "Candidate scores: "
        f"hard={_fmt_score(candidate.get('hard'))}, "
        f"soft={_fmt_score(candidate.get('soft'))}, "
        f"gate={_fmt_score(candidate.get('gate_score'))}",
        "Score deltas candidate-baseline: "
        f"hard={_score_delta(baseline.get('hard'), candidate.get('hard'))}, "
        f"soft={_score_delta(baseline.get('soft'), candidate.get('soft'))}, "
        f"gate={_score_delta(baseline.get('gate_score'), candidate.get('gate_score'))}",
        f"Baseline evaluator reasoning: {baseline.get('evaluator_reasoning', '')}",
        f"Candidate evaluator reasoning: {candidate.get('evaluator_reasoning', '')}",
        f"Optimizer hint: {packet.get('optimizer_hint', '')}",
        "Do not repeat this failed patch direction. Make a different, meaningful skill update "
        "that addresses the failed dimensions and imported human feedback.",
    ]
    expected_artifact = str(packet.get("expected_artifact") or "").strip()
    actual_artifact = str(packet.get("actual_artifact") or "").strip()
    if expected_artifact:
        lines.append(f"Expected artifact: {expected_artifact}")
    if actual_artifact:
        lines.append(f"Actual artifact: {actual_artifact}")
    strengths = delta_summary.get("strengths") if isinstance(delta_summary.get("strengths"), list) else []
    weaknesses = delta_summary.get("weaknesses") if isinstance(delta_summary.get("weaknesses"), list) else []
    if strengths:
        lines.append("Candidate strengths to preserve: " + "; ".join(str(item) for item in strengths[:8]))
    if weaknesses:
        lines.append("Candidate weaknesses to fix: " + "; ".join(str(item) for item in weaknesses[:8]))
    if failed_dimensions:
        lines.append("Failed dimensions: " + ", ".join(str(item) for item in failed_dimensions[:8]))
    if failed_checks:
        lines.append("Failed checks: " + "; ".join(str(item) for item in failed_checks[:8]))
    if evidence:
        lines.append("Evidence: " + "; ".join(str(item) for item in evidence[:8]))
    feedback_text = _format_ranked_feedback_packet(
        packet.get("human_feedback_context") if isinstance(packet.get("human_feedback_context"), dict) else {}
    )
    if feedback_text:
        lines.append(feedback_text)
    return "\n".join(line for line in lines if str(line).strip())


def _format_duplicate_gate_retry_context(*, duplicate_of: str, attempt: int) -> str:
    return "\n".join(
        [
            "## Duplicate Gate Retry Candidate",
            f"The previous gate retry attempt {attempt} produced the same candidate hash: {duplicate_of}.",
            "Do not repeat the same structural update or patch direction.",
            "Preserve useful candidate strengths, but make a different skill change using evaluator rationale.",
            "Specifically address the candidate weaknesses from delta_summary before retrying.",
        ]
    )


def _selection_rejection_signal(results: list[dict] | None) -> dict | None:
    if not results:
        return None
    first_signal: dict | None = None
    for result in results:
        if not isinstance(result, dict):
            continue
        signal = _structured_result_failure(result)
        if not signal:
            continue
        if _is_wrong_artifact_rejection(signal):
            return signal
        if first_signal is None:
            first_signal = signal
    return first_signal


def _structured_result_failure(result: dict) -> dict | None:
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    failure = result.get("failure")
    if not isinstance(failure, dict):
        failure = metadata.get("failure") if isinstance(metadata.get("failure"), dict) else {}
    primary_reason = str(
        result.get("primary_reason")
        or metadata.get("primary_reason")
        or failure.get("primary_reason")
        or ""
    ).strip()
    optimizer_hint = str(
        result.get("optimizer_hint")
        or metadata.get("optimizer_hint")
        or failure.get("optimizer_hint")
        or ""
    ).strip()

    def _string_list(*values: object) -> list[str]:
        out: list[str] = []
        for value in values:
            if isinstance(value, list):
                out.extend(str(item).strip() for item in value if str(item).strip())
            elif isinstance(value, str) and value.strip():
                out.append(value.strip())
        return _dedupe_texts(out, limit=12)

    rejection_type = str(
        result.get("rejection_type")
        or metadata.get("rejection_type")
        or failure.get("rejection_type")
        or ""
    ).strip()
    failed_dimensions = _string_list(
        result.get("failed_dimensions"),
        metadata.get("failed_dimensions"),
        failure.get("failed_dimensions"),
    )
    if not primary_reason and not optimizer_hint and not rejection_type and not failed_dimensions:
        return None

    failed_checks = [
        check
        for source in (result.get("failed_checks"), metadata.get("failed_checks"), failure.get("failed_checks"))
        if isinstance(source, list)
        for check in source
    ]
    human_feedback_context = next(
        (
            source
            for source in (
                result.get("human_feedback_context"),
                metadata.get("human_feedback_context"),
                failure.get("human_feedback_context"),
                metadata.get("human_feedback_alignment"),
            )
            if isinstance(source, dict) and source
        ),
        {},
    )

    return {
        "primary_reason": primary_reason,
        "rejection_type": rejection_type,
        "human_reason": str(
            result.get("human_reason")
            or metadata.get("human_reason")
            or failure.get("human_reason")
            or result.get("fail_reason")
            or ""
        ).strip(),
        "optimizer_hint": optimizer_hint,
        "failed_dimensions": failed_dimensions,
        "failed_checks": failed_checks,
        "evidence": _string_list(result.get("evidence"), metadata.get("evidence"), failure.get("evidence")),
        "stage_status": failure.get("stage_status") if isinstance(failure.get("stage_status"), list) else [],
        "human_feedback_context": human_feedback_context,
        "expected_artifact": str(
            failure.get("expected_artifact")
            or metadata.get("expected_artifact")
            or ("vue-vite bundle" if _is_wrong_artifact_rejection(failure) else "")
        ).strip(),
        "actual_artifact": str(
            failure.get("actual_artifact")
            or metadata.get("actual_artifact")
            or ("skill markdown/template" if _is_wrong_artifact_rejection(failure) else "")
        ).strip(),
        "source_item_id": str(result.get("id") or "").strip(),
    }


def _structured_failure_hints(results: list[dict] | None) -> list[dict]:
    hints: list[dict] = []
    for result in results or []:
        if not isinstance(result, dict):
            continue
        signal = _structured_result_failure(result)
        if not signal or not str(signal.get("optimizer_hint") or "").strip():
            continue
        hints.append(signal)
    return hints


def _format_unconverted_failure_hints(hints: list[dict]) -> list[dict]:
    records: list[dict] = []
    for hint in hints[:8]:
        records.append(
            {
                "source_item_id": str(hint.get("source_item_id") or "").strip(),
                "primary_reason": str(hint.get("primary_reason") or hint.get("rejection_type") or "").strip(),
                "optimizer_hint": str(hint.get("optimizer_hint") or "").strip(),
                "failed_checks": hint.get("failed_checks") if isinstance(hint.get("failed_checks"), list) else [],
                "evidence": hint.get("evidence") if isinstance(hint.get("evidence"), list) else [],
                "human_feedback_context": (
                    hint.get("human_feedback_context")
                    if isinstance(hint.get("human_feedback_context"), dict)
                    else {}
                ),
            }
        )
    return records


def _format_hard_failure_retry_context(
    hints: list[dict],
    *,
    attempt: int,
    budget: int,
    feedback_context: dict | None = None,
) -> str:
    lines = [
        "## Actionable Hard-Failure Retry",
        f"Retry attempt: {attempt}/{budget}",
        (
            "The evaluator produced structured hard-failure guidance, but the previous reflection "
            "did not produce a usable skill patch. Convert these failure packets into a concrete "
            "skill update. Do not repeat a no-op response."
        ),
        json.dumps(_format_unconverted_failure_hints(hints), indent=2, ensure_ascii=False),
    ]
    feedback_text = _format_ranked_feedback_packet(feedback_context)
    if feedback_text:
        lines.append(feedback_text)
    return "\n".join(lines)


def _extract_evaluator_reasoning(result: dict) -> str:
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    failure = result.get("failure")
    if not isinstance(failure, dict):
        failure = metadata.get("failure") if isinstance(metadata.get("failure"), dict) else {}
    values = [
        result.get("reasoning"),
        metadata.get("reasoning"),
        result.get("rationale"),
        metadata.get("rationale"),
        failure.get("human_reason"),
        failure.get("primary_reason"),
        failure.get("optimizer_hint"),
        result.get("fail_reason"),
    ]
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _selection_eval_context(results: list[dict] | None) -> dict:
    if not results:
        return {}
    reasoning: list[str] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        item_id = str(result.get("id") or "").strip()
        text = _extract_evaluator_reasoning(result)
        if not text:
            continue
        if item_id:
            text = f"{item_id}: {text}"
        reasoning.append(text)
    return {"evaluator_reasoning": " | ".join(_dedupe_texts(reasoning, limit=4))}


def _gate_score_payload(
    *,
    hard: float | None,
    soft: float | None,
    gate_score: float | None,
    context: dict | None,
) -> dict:
    payload = {
        "hard": hard,
        "soft": soft,
        "gate_score": gate_score,
    }
    if isinstance(context, dict):
        evaluator_reasoning = str(context.get("evaluator_reasoning") or "").strip()
        if evaluator_reasoning:
            payload["evaluator_reasoning"] = evaluator_reasoning
    return payload


def _selection_delta_summary(
    *,
    baseline_context: dict | None,
    candidate_context: dict | None,
    rejection_signal: dict | None,
) -> dict:
    signal = rejection_signal if isinstance(rejection_signal, dict) else {}
    raw_summary = signal.get("delta_summary")
    if isinstance(raw_summary, dict):
        strengths = raw_summary.get("strengths") if isinstance(raw_summary.get("strengths"), list) else []
        weaknesses = raw_summary.get("weaknesses") if isinstance(raw_summary.get("weaknesses"), list) else []
        return {
            "strengths": _dedupe_texts([str(item) for item in strengths if str(item).strip()], limit=8),
            "weaknesses": _dedupe_texts([str(item) for item in weaknesses if str(item).strip()], limit=8),
        }
    strengths: list[str] = []
    weaknesses: list[str] = []
    candidate_reasoning = str((candidate_context or {}).get("evaluator_reasoning") or "").strip()
    baseline_reasoning = str((baseline_context or {}).get("evaluator_reasoning") or "").strip()
    if candidate_reasoning:
        strengths.append(f"Candidate evaluator rationale: {candidate_reasoning}")
    if baseline_reasoning:
        weaknesses.append(f"Baseline evaluator rationale to beat: {baseline_reasoning}")
    return {
        "strengths": _dedupe_texts(strengths, limit=4),
        "weaknesses": _dedupe_texts(weaknesses, limit=4),
    }


def _join_optimizer_context(*parts: str) -> str:
    return "\n\n".join(part.strip() for part in parts if str(part or "").strip())


def _is_skipped_step(action: str) -> bool:
    return str(action or "").startswith("skip_")


def _normalise_lr_control_mode(mode: str | None) -> str:
    raw = str(mode or "fixed").strip().lower()
    aliases = {
        "fixed": "fixed",
        "manual": "fixed",
        "scheduler": "fixed",
        "scheduled": "fixed",
        "autonomous": "autonomous",
        "auto": "autonomous",
        "optimizer": "autonomous",
        "none": "none",
        "off": "none",
        "no_lr": "none",
    }
    if raw not in aliases:
        raise ValueError("optimizer.lr_control_mode must be one of fixed, autonomous, none")
    return aliases[raw]


def _filter_longitudinal_pairs(pairs: list[dict], policy: str) -> list[dict]:
    if policy == "mixed":
        return pairs
    if policy == "changed":
        keep = {"improved", "regressed"}
    elif policy == "unchanged":
        keep = {"persistent_fail", "stable_success"}
    else:
        raise ValueError(f"Unknown longitudinal pair policy: {policy}")
    return [p for p in pairs if p.get("category") in keep]


def _pair_category_counts(pairs: list[dict]) -> dict[str, int]:
    counts = {
        "improved": 0,
        "regressed": 0,
        "persistent_fail": 0,
        "stable_success": 0,
    }
    for pair in pairs:
        cat = str(pair.get("category", ""))
        counts[cat] = counts.get(cat, 0) + 1
    return counts


def _safe_pair_id(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return safe[:80] or "item"


def _build_longitudinal_pairs(
    *,
    adapter: EnvAdapter,
    dataloader,
    prev_skill: str,
    curr_skill: str,
    initial_items: list[dict],
    initial_prev_results: list[dict],
    initial_curr_results: list[dict],
    prev_rollout_dir: str,
    curr_rollout_dir: str,
    policy: str,
    target_n: int,
    seed: int,
    out_root: str,
) -> tuple[list[dict], list[dict]]:
    """Build longitudinal pairs, optionally filtering by change category.

    ``mixed`` preserves the legacy behavior exactly. ``changed`` keeps only
    10/01 pairs and attempts to top up to ``target_n`` by scanning the train
    split once. ``unchanged`` keeps only 00/11 pairs and does not top up.
    """
    all_pairs = build_comparison_pairs(
        initial_prev_results,
        initial_curr_results,
        initial_items,
        prev_rollout_dir=prev_rollout_dir,
        curr_rollout_dir=curr_rollout_dir,
    )
    selected_pairs = _filter_longitudinal_pairs(all_pairs, policy)
    if policy != "changed" or len(selected_pairs) >= target_n or dataloader is None:
        return selected_pairs, all_pairs

    train_items = list(getattr(dataloader, "train_items", []) or [])
    if not train_items:
        return selected_pairs, all_pairs

    seen_ids = {str(p.get("id", "")) for p in all_pairs}
    rng = random.Random(seed)
    candidates = list(train_items)
    rng.shuffle(candidates)
    candidates = [item for item in candidates if str(item.get("id", "")) not in seen_ids]

    for idx, item in enumerate(candidates):
        if len(selected_pairs) >= target_n:
            break
        item_id = _safe_pair_id(str(item.get("id", f"item_{idx}")))
        batch = BatchSpec(
            phase="train",
            split="train",
            seed=seed + idx + 1,
            batch_size=1,
            payload=[item],
        )
        env = adapter.build_env_from_batch(batch, out_root=out_root)
        prev_dir = os.path.join(prev_rollout_dir, "topup", item_id)
        curr_dir = os.path.join(curr_rollout_dir, "topup", item_id)
        prev_results = adapter.rollout(env, prev_skill, prev_dir)
        curr_results = adapter.rollout(env, curr_skill, curr_dir)
        pair = build_comparison_pairs(
            prev_results,
            curr_results,
            [item],
            prev_rollout_dir=prev_dir,
            curr_rollout_dir=curr_dir,
        )
        all_pairs.extend(pair)
        selected_pairs.extend(_filter_longitudinal_pairs(pair, policy))

    return selected_pairs[:target_n], all_pairs


# ── History / persistence helpers ─────────────────────────────────────────────

_SECRET_KEYS = {
    "azure_api_key",
    "api_key",
    "openai_api_key",
}


def _redact_value(val: str) -> str:
    if len(val) <= 8:
        return "*" * len(val)
    return f"{val[:4]}...{val[-4:]}"


def _redact_cfg(cfg: dict) -> dict:
    redacted = dict(cfg)
    for key in list(redacted):
        if key.lower() in _SECRET_KEYS and redacted.get(key):
            redacted[key] = _redact_value(str(redacted[key]))
    return redacted

def _load_history(out_root: str) -> list[dict]:
    path = os.path.join(out_root, "history.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


def _save_history(out_root: str, history: list[dict]) -> None:
    path = os.path.join(out_root, "history.json")
    with open(path, "w") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def _save_skill(out_root: str, step: int, content: str) -> None:
    skills_dir = os.path.join(out_root, "skills")
    os.makedirs(skills_dir, exist_ok=True)
    with open(os.path.join(skills_dir, f"skill_v{step:04d}.md"), "w") as f:
        f.write(content)


def _load_skill(out_root: str, step: int) -> str:
    path = os.path.join(out_root, "skills", f"skill_v{step:04d}.md")
    with open(path) as f:
        return f.read()


def _load_meta_skill_content(out_root: str, epoch: int) -> str:
    if epoch <= 0:
        return ""
    path = os.path.join(
        out_root, "meta_skill", f"epoch_{epoch:02d}", "meta_skill_result.json",
    )
    if not os.path.exists(path):
        return ""
    try:
        with open(path) as f:
            result = json.load(f)
        return str(result.get("meta_skill_content", "")).strip()
    except Exception:
        return ""


def _load_runtime_state(out_root: str) -> dict | None:
    path = os.path.join(out_root, "runtime_state.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            state = json.load(f)
        return state if isinstance(state, dict) else None
    except Exception:
        return None


def _save_runtime_state(out_root: str, state: dict) -> None:
    path = os.path.join(out_root, "runtime_state.json")
    with open(path, "w") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def _resolve_train_size(cfg: dict, dataloader) -> int:
    configured = int(cfg.get("train_size", 0) or 0)
    inferred: int | None = None

    if dataloader is not None:
        getter = getattr(dataloader, "get_train_size", None)
        if callable(getter):
            try:
                value = getter()
            except Exception:
                value = None
            if value is not None:
                inferred = int(value)
        elif hasattr(dataloader, "train_items"):
            try:
                inferred = len(getattr(dataloader, "train_items"))
            except Exception:
                inferred = None

    if inferred is not None and inferred <= 0:
        inferred = None

    if configured > 0 and inferred is not None and configured != inferred:
        raise ValueError(
            f"Configured train_size={configured} does not match loaded train split "
            f"size={inferred}. Fix the config or the dataset split."
        )

    train_size = configured if configured > 0 else inferred
    if train_size is None or train_size <= 0:
        raise ValueError(
            "Unable to determine train_size automatically. "
            "Provide train.train_size in the config for this environment."
        )
    return int(train_size)


def _compute_task_type_buckets(results: list[dict], task_types: list[str]) -> dict[str, dict]:
    """Compute per-task-type success rates."""
    buckets: dict[str, dict] = {}
    for task in task_types + ["overall"]:
        buckets[task] = {"total": 0, "hard": 0, "soft": 0.0, "unscored": 0}
    for r in results:
        tt = r.get("task_type", "other")
        for key in [tt, "overall"]:
            if key not in buckets:
                buckets[key] = {"total": 0, "hard": 0, "soft": 0.0, "unscored": 0}
            buckets[key]["total"] += 1
            if _is_unscored_rollout_result(r):
                buckets[key]["unscored"] += 1
                continue
            buckets[key]["hard"] += float(r.get("hard", 0))
            buckets[key]["soft"] += float(r.get("soft", 0.0))
    return buckets


def _is_unscored_rollout_result(result: dict) -> bool:
    return str(result.get("score_status") or "").strip().lower() == "unscored" or result.get("hard") is None


def _is_failed_rollout_result(result: dict) -> bool:
    if _is_unscored_rollout_result(result):
        return True
    return not result.get("hard") or float(result.get("hard", 0)) < 1e-9


def _write_gate_block(directory: str, block: dict) -> None:
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "gate_block.json"), "w", encoding="utf-8") as f:
        json.dump(block, f, indent=2, ensure_ascii=False)


def _blocked_training_summary(
    *,
    cfg: dict,
    block: dict,
    total_steps: int = 0,
    best_step: int = 0,
    token_summary: dict | None = None,
) -> dict:
    return {
        "version": "skillopt-0.1.0",
        "config": _redact_cfg(cfg),
        "gate_status": "blocked",
        "gate_blocker": block.get("blocker", "unscored"),
        "gate_block": block,
        "gate_blockers": [block],
        "promotable": False,
        "baseline_selection_hard": None,
        "baseline_selection_soft": None,
        "best_selection_hard": None,
        "best_selection_soft": None,
        "best_step": best_step,
        "current_origin": "initial_skill",
        "best_origin": "initial_skill",
        "total_steps": total_steps,
        "total_accepts": 0,
        "total_rejects": 0,
        "total_blocks": 1,
        "total_skips": 0,
        "epoch_stats": [],
        "baseline_test_hard": None,
        "baseline_test_soft": None,
        "test_hard": None,
        "test_soft": None,
        "test_delta_hard": None,
        "total_wall_time_s": 0,
        "token_summary": token_summary or {},
    }


def _best_selection_scores(
    *,
    history: list[dict],
    best_step: int,
    best_origin: str,
    baseline_scores: tuple[float | None, float | None],
    selection_scores_by_origin: dict[str, tuple[float | None, float | None]],
    gate_metric: str,
    best_score: float,
) -> tuple[float | None, float | None]:
    if best_origin in selection_scores_by_origin:
        return selection_scores_by_origin[best_origin]
    if best_step == 0:
        return baseline_scores
    for rec in reversed(history):
        if rec.get("step") != best_step or rec.get("selection_hard") is None:
            continue
        return rec.get("selection_hard"), rec.get("selection_soft")
    hard, soft = baseline_scores
    if hard is None and gate_metric == "hard" and best_score >= 0:
        hard = best_score
    if soft is None and gate_metric == "soft" and best_score >= 0:
        soft = best_score
    return hard, soft


def _gate_score_for_summary(hard: float | None, soft: float | None, gate_metric: str, gate_mixed_weight: float) -> float | None:
    if hard is None or soft is None:
        return None
    return select_gate_score(float(hard), float(soft), gate_metric, gate_mixed_weight)


def _selection_reject_gate_rejection(
    *,
    history: list[dict],
    baseline_scores: tuple[float | None, float | None],
    gate_metric: str,
    gate_mixed_weight: float,
    retry_used: int = 0,
    retry_budget: int = 0,
    rejection_signal: dict | None = None,
    baseline_context: dict | None = None,
    candidate_context: dict | None = None,
    human_feedback_context: dict | None = None,
) -> dict | None:
    reject_steps = [record for record in history if record.get("action") == "reject"]
    if not reject_steps:
        return None
    rejected = reject_steps[-1]
    if candidate_context is None and isinstance(rejected.get("selection_eval_context"), dict):
        candidate_context = rejected.get("selection_eval_context")
    baseline_hard, baseline_soft = baseline_scores
    candidate_hard = rejected.get("selection_hard")
    candidate_soft = rejected.get("selection_soft")
    baseline_gate_score = _gate_score_for_summary(
        baseline_hard,
        baseline_soft,
        gate_metric,
        gate_mixed_weight,
    )
    candidate_gate_score = rejected.get("candidate_gate_score")
    if candidate_gate_score is None:
        candidate_gate_score = _gate_score_for_summary(
            candidate_hard,
            candidate_soft,
            gate_metric,
            gate_mixed_weight,
        )

    raw_change_summary = rejected.get("rewrite_change_summary", [])
    if raw_change_summary is None:
        change_summary: list = []
    elif isinstance(raw_change_summary, list):
        change_summary = raw_change_summary
    else:
        change_summary = [raw_change_summary]
    attempted_patch = ", ".join(
        str(item).strip()
        for item in change_summary
        if str(item).strip()
    )
    if not attempted_patch:
        attempted_patch = "selection-gated candidate update"

    evidence: list[str] = []
    if candidate_gate_score is not None and baseline_gate_score is not None:
        evidence.append(
            f"Candidate gate score {candidate_gate_score:.4f} <= baseline gate score {baseline_gate_score:.4f}."
        )
    if candidate_soft is not None and baseline_soft is not None:
        evidence.append(
            f"Candidate soft score {float(candidate_soft):.4f}; baseline soft score {float(baseline_soft):.4f}."
        )
    if not evidence:
        evidence.append("Candidate selection gate action was reject.")

    existing_gate_rejection = (
        rejected.get("gate_rejection")
        if isinstance(rejected.get("gate_rejection"), dict)
        else {}
    )
    retry_attempts = str(
        existing_gate_rejection.get("retry_attempts")
        or f"{max(0, int(retry_used))}/{max(0, int(retry_budget))}"
    )
    next_action = str(
        existing_gate_rejection.get("next_action")
        or "Stop this pass without final test eval; retry only when gate-rejection retry is enabled or collect more feedback."
    )

    signal = rejection_signal if isinstance(rejection_signal, dict) else {}
    signal_primary_reason = str(signal.get("primary_reason") or "").strip()
    signal_is_wrong_artifact = _is_wrong_artifact_rejection(signal)
    signal_feedback_context = signal.get("human_feedback_context")
    packet_feedback_context = (
        signal_feedback_context
        if isinstance(signal_feedback_context, dict) and signal_feedback_context
        else human_feedback_context
    )
    if not isinstance(packet_feedback_context, dict):
        packet_feedback_context = {}
    signal_optimizer_hint = str(signal.get("optimizer_hint") or "").strip()
    signal_human_reason = str(signal.get("human_reason") or "").strip()
    signal_failed_dimensions = (
        signal.get("failed_dimensions") if isinstance(signal.get("failed_dimensions"), list) else []
    )
    signal_evidence = signal.get("evidence") if isinstance(signal.get("evidence"), list) else []
    signal_failed_checks = signal.get("failed_checks") if isinstance(signal.get("failed_checks"), list) else []
    expected_artifact = str(signal.get("expected_artifact") or "").strip()
    actual_artifact = str(signal.get("actual_artifact") or "").strip()
    if signal_evidence:
        evidence.extend(str(item) for item in signal_evidence if str(item).strip())
    source_item_id = str(signal.get("source_item_id") or "").strip()
    if source_item_id:
        evidence.append(f"Structured rejection came from selection item {source_item_id}.")

    primary_reason = signal_primary_reason or (
        "wrong_artifact_type" if signal_is_wrong_artifact else "candidate_quality_regressed"
    )
    rejection_type = "wrong_artifact_type" if signal_is_wrong_artifact else "candidate_score_regression"
    optimizer_hint = signal_optimizer_hint or (
        "Use the gate rejection evidence and human feedback to change the skill direction before spending final test-eval budget."
    )
    hint_suffix = _human_feedback_hint_suffix(packet_feedback_context)
    if hint_suffix and hint_suffix.strip() not in optimizer_hint:
        optimizer_hint = (optimizer_hint + hint_suffix).strip()
    human_reason = signal_human_reason or (
        "The candidate lost selection evaluation against the baseline skill, so final test evaluation was skipped."
    )
    failed_dimensions = signal_failed_dimensions or ["selection_gate", "human_feedback_alignment"]
    if packet_feedback_context and "human_feedback_alignment" not in failed_dimensions:
        failed_dimensions = _dedupe_texts(
            [str(item) for item in failed_dimensions] + ["human_feedback_alignment"],
            limit=12,
        )

    packet = {
        "rejection_type": rejection_type,
        "retryable": True,
        "baseline": _gate_score_payload(
            hard=baseline_hard,
            soft=baseline_soft,
            gate_score=baseline_gate_score,
            context=baseline_context,
        ),
        "candidate": _gate_score_payload(
            hard=candidate_hard,
            soft=candidate_soft,
            gate_score=candidate_gate_score,
            context=candidate_context,
        ),
        "delta_summary": _selection_delta_summary(
            baseline_context=baseline_context,
            candidate_context=candidate_context,
            rejection_signal=rejection_signal,
        ),
        "primary_reason": primary_reason,
        "human_reason": human_reason,
        "optimizer_hint": optimizer_hint,
        "failed_dimensions": failed_dimensions,
        "failed_checks": signal_failed_checks,
        "evidence": _dedupe_texts(evidence, limit=12),
        "expected_artifact": expected_artifact,
        "actual_artifact": actual_artifact,
        "attempted_patch": attempted_patch,
        "retry_attempts": retry_attempts,
        "next_action": next_action,
    }
    if packet_feedback_context:
        packet["human_feedback_context"] = packet_feedback_context
    return packet


def _should_skip_final_test_after_selection_reject(
    *,
    history: list[dict],
    best_origin: str,
    best_skill: str,
    skill_init: str,
) -> bool:
    return (
        best_origin == "initial_skill"
        and skill_hash(best_skill) == skill_hash(skill_init)
        and any(record.get("action") == "reject" for record in history)
    )


def _format_rejection_buffer(buffer: list[dict]) -> str:
    """**DEPRECATED** — kept for backward compat; use _format_step_buffer."""
    return _format_step_buffer(buffer)


def _extract_failure_patterns(
    rollout_results: list[dict],
    step_dir: str,
) -> list[dict]:
    """Extract compact failure patterns from rollout results.

    Uses analyst ``failure_summary`` from minibatch patches when available,
    otherwise falls back to ``fail_reason`` prefix grouping.
    """
    failures = [r for r in rollout_results if _is_failed_rollout_result(r)]
    if not failures:
        return []

    # Group by fail_reason prefix
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in failures:
        reason = r.get("fail_reason", "unknown")
        prefix = reason.split(":")[0].strip() if ":" in reason else reason
        groups[prefix].append(r)

    # Try richer descriptions from analyst patches
    analyst_descs: list[str] = []
    patch_globs = [
        os.path.join(step_dir, "patches", "minibatch_fail_*.json"),
        os.path.join(step_dir, "batch_*", "patches", "minibatch_fail_*.json"),
    ]
    seen_patch_files: set[str] = set()
    for pattern in patch_globs:
        for fname in sorted(glob.glob(pattern)):
            if fname in seen_patch_files:
                continue
            seen_patch_files.add(fname)
            try:
                with open(fname) as f:
                    patch = json.load(f)
                for fs in patch.get("failure_summary", []):
                    ft = fs.get("failure_type", "")
                    sd = fs.get("description", "")
                    analyst_descs.append(f"{ft}: {sd}" if sd else ft)
            except Exception:
                pass

    patterns = []
    desc_iter = iter(analyst_descs)
    for prefix, items in groups.items():
        desc = next(desc_iter, None) or prefix
        patterns.append({
            "pattern": desc,
            "count": len(items),
            "task_ids": [str(r.get("id", "?")) for r in items],
        })
    return patterns


def _format_step_buffer(buffer: list[dict]) -> str:
    """Format the unified step buffer into a single context block.

    Each entry captures what happened at a previous step: failure patterns
    observed during rollout, and — when the step was rejected — the specific
    edits that were tried and the resulting score drop.

    Returns empty string when *buffer* is empty.
    """
    if not buffer:
        return ""

    parts = [
        "Below is a summary of previous steps in this epoch. "
        "Use it to avoid repeating ineffective edits and to prioritise "
        "failure patterns that remain unsolved.\n"
    ]

    for entry in buffer:
        step = entry["step"]
        action = entry["action"]
        n_fail = entry.get("n_fail", 0)
        n_total = entry.get("n_total", "?")

        parts.append(f"### Step {step} — {action.upper()} ({n_fail}/{n_total} failed)")

        # Failure patterns
        for p in entry.get("failure_patterns", []):
            ids = ", ".join(p["task_ids"][:3])
            parts.append(f'  - "{p["pattern"]}" (×{p["count"]}, tasks: {ids})')

        # Rejected edits (only present on reject)
        rejected = entry.get("rejected_edits", [])
        if rejected:
            score_before = entry.get("score_before", "?")
            score_after = entry.get("score_after", "?")
            parts.append(
                f"  Rejected edits (score {score_before} → {score_after}):"
            )
            for i, e in enumerate(rejected, 1):
                if e.get("op") is not None:
                    op = e.get("op", "?")
                    content = e.get("content", "")
                    target = e.get("target", "")
                    if target:
                        parts.append(f'    {i}. [{op}] target="{target[:80]}" → "{content}"')
                    else:
                        parts.append(f'    {i}. [{op}] "{content}"')
                else:
                    kind = e.get("type", "?")
                    title = e.get("title", "")
                    instruction = e.get("instruction", "")
                    parts.append(f'    {i}. [{kind}] "{title}" → "{instruction}"')

    return "\n".join(parts)


# ── Trainer ──────────────────────────────────────────────────────────────────

class ReflACTTrainer:
    """Main ReflACT training loop.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary. See ``configs/alfworld_default.yaml``
        for the full list of keys.
    adapter : EnvAdapter
        Environment adapter instance.
    """

    def __init__(self, cfg: dict, adapter: EnvAdapter) -> None:
        self.cfg = cfg
        self.adapter = adapter

    def train(self) -> dict:
        """Execute the full ReflACT training loop. Returns summary dict."""
        cfg = self.cfg
        adapter = self.adapter
        out_root = cfg["out_root"]
        os.makedirs(out_root, exist_ok=True)

        # ── Adapter setup (one-time init) ────────────────────────────
        adapter.setup(cfg)
        dataloader = adapter.get_dataloader()

        def _build_train_env(batch: BatchSpec):
            env_manager = adapter.build_env_from_batch(batch, out_root=out_root)
            return env_manager, batch.batch_size, batch.seed

        def _build_eval_env(split: str, env_num: int, seed: int):
            if dataloader is None:
                env_manager = adapter.build_eval_env(
                    env_num=env_num,
                    split=split,
                    seed=seed,
                    out_root=out_root,
                )
                actual_n = len(env_manager) if hasattr(env_manager, "__len__") else env_num
                return env_manager, actual_n

            batch = dataloader.build_eval_batch(
                env_num=env_num,
                split=split,
                seed=seed,
                out_root=out_root,
            )
            env_manager = adapter.build_env_from_batch(batch, out_root=out_root)
            return env_manager, batch.batch_size

        # ── Configure models ─────────────────────────────────────────────
        backend = cfg.get("model_backend", "azure_openai")
        configure_azure_openai(
            endpoint=(
                cfg.get("azure_openai_endpoint")
                or cfg.get("azure_endpoint")
                or None
            ),
            api_version=(
                cfg.get("azure_openai_api_version")
                or cfg.get("azure_api_version")
                or None
            ),
            api_key=(
                cfg.get("azure_openai_api_key")
                or cfg.get("azure_api_key")
                or None
            ),
            auth_mode=cfg.get("azure_openai_auth_mode") or None,
            ad_scope=cfg.get("azure_openai_ad_scope") or None,
            managed_identity_client_id=cfg.get("azure_openai_managed_identity_client_id") or None,
            optimizer_endpoint=cfg.get("optimizer_azure_openai_endpoint") or None,
            optimizer_api_version=cfg.get("optimizer_azure_openai_api_version") or None,
            optimizer_api_key=cfg.get("optimizer_azure_openai_api_key") or None,
            optimizer_auth_mode=cfg.get("optimizer_azure_openai_auth_mode") or None,
            optimizer_ad_scope=cfg.get("optimizer_azure_openai_ad_scope") or None,
            optimizer_managed_identity_client_id=(
                cfg.get("optimizer_azure_openai_managed_identity_client_id") or None
            ),
            target_endpoint=cfg.get("target_azure_openai_endpoint") or None,
            target_api_version=cfg.get("target_azure_openai_api_version") or None,
            target_api_key=cfg.get("target_azure_openai_api_key") or None,
            target_auth_mode=cfg.get("target_azure_openai_auth_mode") or None,
            target_ad_scope=cfg.get("target_azure_openai_ad_scope") or None,
            target_managed_identity_client_id=(
                cfg.get("target_azure_openai_managed_identity_client_id") or None
            ),
        )
        optimizer_backend = cfg.get("optimizer_backend")
        target_backend = cfg.get("target_backend")
        if not optimizer_backend or not target_backend:
            if backend in {"claude", "claude_chat"}:
                optimizer_backend = optimizer_backend or "claude_chat"
                target_backend = target_backend or "claude_chat"
            elif backend in {"codex", "codex_exec"}:
                optimizer_backend = optimizer_backend or "openai_chat"
                target_backend = target_backend or "codex_exec"
            elif backend == "claude_code_exec":
                optimizer_backend = optimizer_backend or "openai_chat"
                target_backend = target_backend or "claude_code_exec"
            elif backend in {"qwen", "qwen_chat"}:
                optimizer_backend = optimizer_backend or "openai_chat"
                target_backend = target_backend or "qwen_chat"
            else:
                optimizer_backend = optimizer_backend or "openai_chat"
                target_backend = target_backend or "openai_chat"
            cfg["optimizer_backend"] = optimizer_backend
            cfg["target_backend"] = target_backend
        set_optimizer_backend(optimizer_backend)
        set_target_backend(target_backend)
        set_optimizer_deployment(cfg["optimizer_model"])
        set_target_deployment(cfg["target_model"])
        configure_codex_exec(
            path=cfg.get("codex_exec_path", "codex"),
            sandbox=cfg.get("codex_exec_sandbox", "workspace-write"),
            profile=cfg.get("codex_exec_profile", ""),
            full_auto=cfg.get("codex_exec_full_auto", False),
            reasoning_effort=cfg.get("codex_exec_reasoning_effort", "none"),
            use_sdk=cfg.get("codex_exec_use_sdk", None),
            network_access=cfg.get("codex_exec_network_access", False),
            web_search=cfg.get("codex_exec_web_search", False),
            approval_policy=cfg.get("codex_exec_approval_policy", "never"),
        )
        configure_claude_code_exec(
            path=cfg.get("claude_code_exec_path", "claude"),
            profile=cfg.get("claude_code_exec_profile", ""),
            use_sdk=cfg.get("claude_code_exec_use_sdk", None),
            effort=cfg.get("claude_code_exec_effort", cfg.get("reasoning_effort", "medium")),
            max_thinking_tokens=cfg.get("claude_code_exec_max_thinking_tokens", 16384),
        )
        configure_qwen_chat(
            base_url=cfg.get("qwen_chat_base_url") or None,
            api_key=cfg.get("qwen_chat_api_key") or None,
            temperature=cfg.get("qwen_chat_temperature"),
            timeout_seconds=cfg.get("qwen_chat_timeout_seconds"),
            max_tokens=cfg.get("qwen_chat_max_tokens"),
            enable_thinking=cfg.get("qwen_chat_enable_thinking"),
        )
        configure_minimax_chat(
            base_url=cfg.get("minimax_base_url") or None,
            api_key=cfg.get("minimax_api_key") or None,
            temperature=cfg.get("minimax_temperature"),
            max_tokens=cfg.get("minimax_max_tokens"),
            enable_thinking=cfg.get("minimax_enable_thinking"),
        )
        minimax_model_cfg = cfg.get("minimax_model")
        if minimax_model_cfg and cfg.get("target_backend") == "minimax_chat":
            set_target_deployment(str(minimax_model_cfg))
        os.environ["REFLACT_CODEX_TRACE_TO_OPTIMIZER"] = (
            "1"
            if target_backend == "codex_exec" and cfg.get("codex_trace_to_optimizer", False)
            else "0"
        )
        reasoning = cfg.get("reasoning_effort", "") or None
        set_reasoning_effort(reasoning)
        print(
            f"  [model config] backend={backend}  "
            f"optimizer={cfg['optimizer_model']} ({optimizer_backend})  "
            f"target={cfg['target_model']} ({target_backend})  "
            f"reasoning={reasoning or 'off'}"
        )

        # ── Initialize Ray ───────────────────────────────────────────────
        if adapter.requires_ray():
            try:
                import ray
            except ImportError as e:
                raise ImportError(
                    "This environment requires ray, but ray is not installed."
                ) from e

            if not ray.is_initialized():
                ray.init(num_gpus=0)

        # ── Load initial skill ───────────────────────────────────────────
        skill_init_path = os.path.abspath(cfg["skill_init"])
        if os.path.exists(skill_init_path):
            with open(skill_init_path) as f:
                skill_init = f.read()
            print(f"  [initial skill] {skill_init_path} ({len(skill_init)} chars)")
        else:
            skill_init = ""
            print("  [initial skill] no initial skill file — starting from blank")

        # ── Training parameters ──────────────────────────────────────────
        batch_size = cfg["batch_size"]
        num_epochs = cfg["num_epochs"]
        accumulation = cfg["accumulation"]
        seed = cfg["seed"]
        merge_bs = cfg["merge_batch_size"]
        max_analyst_rounds = int(cfg.get("max_analyst_rounds", 3) or 3)
        update_mode = normalize_update_mode(cfg.get("skill_update_mode", "patch"))
        lr_control_mode = _normalise_lr_control_mode(cfg.get("lr_control_mode", "fixed"))
        if is_full_rewrite_minibatch_mode(update_mode):
            lr_control_mode = "none"
        longitudinal_pair_policy = _normalise_longitudinal_pair_policy(
            cfg.get("longitudinal_pair_policy", "mixed")
        )
        rewrite_reasoning_effort = cfg.get("rewrite_reasoning_effort", "high")
        if rewrite_reasoning_effort == "":
            rewrite_reasoning_effort = None
        rewrite_max_completion_tokens = int(cfg.get("rewrite_max_completion_tokens", 64000))
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if accumulation <= 0:
            raise ValueError(f"accumulation must be positive, got {accumulation}")

        train_size = _resolve_train_size(cfg, dataloader)
        steps_per_epoch = math.ceil(train_size / (batch_size * accumulation))
        batches_per_epoch = steps_per_epoch * accumulation
        total_steps = num_epochs * steps_per_epoch

        # Persist resolved derived fields so config.json / summary.json match
        # the actual runtime recipe.
        cfg["train_size"] = train_size
        cfg["steps_per_epoch"] = steps_per_epoch
        cfg["batches_per_epoch"] = batches_per_epoch
        cfg["samples_per_epoch"] = train_size
        cfg["skill_update_mode"] = update_mode
        cfg["lr_control_mode"] = lr_control_mode
        cfg["feedback_direct_mode"] = _normalize_feedback_direct_mode(cfg.get("feedback_direct_mode", "auto"))

        # Save config after deriving runtime values.
        with open(os.path.join(out_root, "config.json"), "w") as f:
            json.dump(_redact_cfg(cfg), f, indent=2, ensure_ascii=False)

        train_pool_size = train_size

        scheduler = build_scheduler(
            mode=cfg.get("lr_scheduler", "constant"),
            max_lr=cfg["edit_budget"],
            min_lr=cfg.get("min_edit_budget", 2),
            total_steps=total_steps,
        )

        # Fixed training pool: base seeds (each seed = one deterministic batch)
        if dataloader is not None:
            base_seeds = dataloader.make_base_seeds(
                steps_per_epoch=steps_per_epoch,
                accumulation=accumulation,
                seed=seed,
            )
        else:
            base_seeds = [seed + i + 1 for i in range(batches_per_epoch)]

        print(f"\n  [config] epochs={num_epochs} steps/epoch={steps_per_epoch} "
              f"(auto) accum={accumulation} batch_size={batch_size}")
        print(f"  [config] train_size={train_size}")
        print(f"  [config] batches/epoch={batches_per_epoch} "
              f"total_steps={total_steps} "
              f"games/epoch={train_pool_size}")
        print(f"  [config] lr_scheduler={cfg.get('lr_scheduler', 'constant')} "
              f"edit_budget={cfg['edit_budget']} "
              f"min_edit_budget={cfg.get('min_edit_budget', 2)}")
        print(f"  [config] skill_update_mode={update_mode} "
              f"lr_control_mode={lr_control_mode} "
              f"rewrite_reasoning_effort={rewrite_reasoning_effort or 'off'} "
              f"rewrite_max_completion_tokens={rewrite_max_completion_tokens} "
              f"max_analyst_rounds={max_analyst_rounds}")
        print(f"  [config] longitudinal_pair_policy={longitudinal_pair_policy}")
        print(f"  [config] base_seeds={base_seeds}")

        # ── Resume check ─────────────────────────────────────────────────
        history = _load_history(out_root)
        runtime_state = _load_runtime_state(out_root)
        if runtime_state:
            last_step = int(runtime_state.get("last_completed_step", 0) or 0)
            current_skill_path = runtime_state.get("current_skill_path") or os.path.join(
                out_root, "skills", f"skill_v{last_step:04d}.md",
            )
            with open(current_skill_path) as f:
                current_skill = f.read()
            best_skill_path = runtime_state.get("best_skill_path") or os.path.join(
                out_root, "best_skill.md",
            )
            if os.path.exists(best_skill_path):
                with open(best_skill_path) as f:
                    best_skill = f.read()
            else:
                best_skill = current_skill
            current_score = float(runtime_state.get("current_score", -1.0) or -1.0)
            best_score = float(runtime_state.get("best_score", current_score) or current_score)
            best_step = runtime_state.get("best_step", last_step)
            current_origin = str(
                runtime_state.get("current_origin")
                or (f"step_{last_step:04d}" if last_step > 0 else "initial_skill")
            )
            best_origin = str(runtime_state.get("best_origin") or current_origin)
            resume_from = last_step + 1
            scheduler.load_state_dict({"current_step": last_step})
            print(
                f"  [resume] from step {resume_from}  "
                f"current={current_score:.4f} best={best_score:.4f} "
                f"(origin={current_origin})"
            )
        elif history:
            last_step = history[-1]["step"]
            current_skill = _load_skill(out_root, last_step)
            best_rec = max(history, key=lambda h: h.get("best_score", 0.0))
            best_score = best_rec["best_score"]
            best_step = best_rec["best_step"]
            best_skill_path = os.path.join(out_root, "best_skill.md")
            if os.path.exists(best_skill_path):
                with open(best_skill_path) as f:
                    best_skill = f.read()
            else:
                best_skill = _load_skill(out_root, best_step)
            current_score = history[-1].get("current_score", best_score)
            current_origin = f"step_{last_step:04d}"
            best_origin = f"step_{int(best_step):04d}" if isinstance(best_step, int) else str(best_step)
            resume_from = last_step + 1
            scheduler.load_state_dict({"current_step": last_step})
            print(
                f"  [resume] from step {resume_from}  "
                f"current={current_score:.4f} best={best_score:.4f}"
            )
        else:
            current_skill = skill_init
            best_skill = skill_init
            best_score = -1.0
            current_score = -1.0
            best_step = 0
            current_origin = "initial_skill"
            best_origin = "initial_skill"
            resume_from = 1

        _save_skill(out_root, 0, skill_init)

        def _persist_runtime_state(last_completed_step: int) -> None:
            _save_runtime_state(
                out_root,
                {
                    "last_completed_step": last_completed_step,
                    "current_skill_path": os.path.join(
                        out_root, "skills", f"skill_v{last_completed_step:04d}.md",
                    ),
                    "current_score": current_score,
                    "current_origin": current_origin,
                    "best_skill_path": os.path.join(out_root, "best_skill.md"),
                    "best_score": best_score,
                    "best_step": best_step,
                    "best_origin": best_origin,
                },
            )

        # ── Selection cache ──────────────────────────────────────────────
        sel_cache: dict[str, tuple[float, float]] = {}
        sel_rejection_signal_cache: dict[str, dict] = {}
        sel_eval_context_cache: dict[str, dict] = {}
        for rec in history:
            sh = rec.get("candidate_hash", "")
            if sh and rec.get("selection_hard") is not None:
                sel_cache[sh] = (rec["selection_hard"], rec["selection_soft"])
                selection_eval_context = rec.get("selection_eval_context")
                if isinstance(selection_eval_context, dict):
                    sel_eval_context_cache[sh] = selection_eval_context
                gate_rejection = rec.get("gate_rejection") if isinstance(rec.get("gate_rejection"), dict) else {}
                if _is_wrong_artifact_rejection(gate_rejection):
                    sel_rejection_signal_cache[sh] = dict(gate_rejection)
        selection_scores_by_origin: dict[str, tuple[float | None, float | None]] = {}

        # ── Baseline evaluation on selection set ─────────────────────────
        if cfg.get("use_gate") is False:
            raise ValueError(
                "Gate validation is mandatory in this branch. Remove "
                "`evaluation.use_gate=false` from the config."
            )
        gate_metric = str(cfg.get("gate_metric", "hard")).strip().lower()
        if gate_metric not in {"hard", "soft", "mixed"}:
            raise ValueError(
                f"evaluation.gate_metric must be 'hard' | 'soft' | 'mixed', "
                f"got {gate_metric!r}"
            )
        gate_mixed_weight = float(cfg.get("gate_mixed_weight", 0.5))
        if not 0.0 <= gate_mixed_weight <= 1.0:
            raise ValueError(
                f"evaluation.gate_mixed_weight must be in [0, 1], "
                f"got {gate_mixed_weight}"
            )
        print(
            f"  [gate] metric={gate_metric}"
            + (
                f" mixed_weight={gate_mixed_weight}"
                if gate_metric == "mixed"
                else ""
            )
        )
        slow_gate_with_selection = bool(
            cfg.get("slow_update_gate_with_selection", False)
        )
        print(
            "  [slow update] acceptance="
            + ("gated (selection-set validation)"
               if slow_gate_with_selection
               else "force-accept (unconditional)")
        )
        if current_score < 0:
            print(f"\n{'='*60}")
            print("  BASELINE — evaluate initial skill on Selection set (valid_seen)")
            print(f"{'='*60}")
            sel_env, sel_n = _build_eval_env(
                split="valid_seen",
                env_num=cfg["sel_env_num"],
                seed=seed,
            )
            print(f"  Selection items: {sel_n}")
            baseline_dir = os.path.join(out_root, "selection_eval_baseline")
            baseline_results = adapter.rollout(sel_env, skill_init, baseline_dir)
            baseline_block = find_gate_block(baseline_results)
            if baseline_block is not None:
                block = baseline_block.to_dict()
                _write_gate_block(baseline_dir, block)
                with open(os.path.join(out_root, "best_skill.md"), "w", encoding="utf-8") as f:
                    f.write(best_skill)
                summary = _blocked_training_summary(
                    cfg=cfg,
                    block=block,
                    total_steps=0,
                    best_step=best_step,
                    token_summary=get_token_summary(),
                )
                with open(os.path.join(out_root, "summary.json"), "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)
                print(f"  [gate blocked] blocked:{block['blocker']} items={len(block.get('items', []))}")
                return summary
            baseline_hard, baseline_soft = compute_score(baseline_results)
            current_score = select_gate_score(
                baseline_hard, baseline_soft, gate_metric, gate_mixed_weight,
            )
            best_score = current_score
            sh = skill_hash(skill_init)
            sel_cache[sh] = (baseline_hard, baseline_soft)
            sel_eval_context_cache[sh] = _selection_eval_context(baseline_results)
            current_origin = "initial_skill"
            best_origin = "initial_skill"
            selection_scores_by_origin[best_origin] = (baseline_hard, baseline_soft)
            _persist_runtime_state(0)
            print(
                f"  [baseline result] selection hard={baseline_hard:.4f} "
                f"soft={baseline_soft:.4f} "
                f"gate[{gate_metric}]={current_score:.4f}"
            )

        # ── Training loop ────────────────────────────────────────────────
        t_loop_start = time.time()

        if resume_from > total_steps:
            print(f"\n  [skip] all {total_steps} steps complete — jumping to evaluation")

        global_step = 0
        run_gate_blocks: list[dict] = []
        for epoch in range(1, num_epochs + 1):
            if dataloader is not None:
                epoch_batches = dataloader.plan_train_epoch(
                    epoch=epoch,
                    steps_per_epoch=steps_per_epoch,
                    accumulation=accumulation,
                    batch_size=batch_size,
                    seed=seed,
                    out_root=out_root,
                )
                shuffled_seeds = [batch.seed for batch in epoch_batches]
            else:
                epoch_batches = []
                epoch_rng = random.Random(seed + epoch * 1000)
                shuffled_seeds = base_seeds.copy()
                epoch_rng.shuffle(shuffled_seeds)

            # Step buffer: accumulates per-step context (failure patterns +
            # rejected edits) within this epoch so optimizers see full history.
            step_buffer: list[dict] = []
            active_meta_skill = (
                _load_meta_skill_content(out_root, epoch - 1)
                if cfg.get("use_meta_skill", False)
                else ""
            )

            print(
                f"\n  [EPOCH {epoch}/{num_epochs}] "
                f"shuffled_seeds={shuffled_seeds}"
            )
            if active_meta_skill:
                print(
                    f"  [meta skill] loaded from epoch {epoch - 1} "
                    f"({len(active_meta_skill)} chars)"
                )

            for step_in_epoch in range(steps_per_epoch):
                global_step += 1
                if global_step < resume_from:
                    continue

                step_t0 = time.time()
                step_dir = os.path.join(out_root, "steps", f"step_{global_step:04d}")
                os.makedirs(step_dir, exist_ok=True)

                tokens_before = get_token_summary()

                print(
                    f"\n  [STEP {global_step}/{total_steps}] "
                    f"epoch={epoch} step_in_epoch={step_in_epoch} "
                    f"{'='*30}"
                )

                step_rec: dict = {
                    "step": global_step,
                    "epoch": epoch,
                    "step_in_epoch": step_in_epoch,
                    "timing": {},
                    "tokens": {},
                }

                # ── Accumulation: Rollout + Reflect ──────────────────────
                all_failure_patches: list[dict] = []
                all_success_patches: list[dict] = []
                all_raw_patches: list[dict | None] = []
                all_rollout_results: list[dict] = []
                accum_rollout_stats: list[dict] = []
                feedback_direct_contexts: list[str] = []
                total_rollout_time = 0.0
                total_reflect_time = 0.0

                for a in range(accumulation):
                    batch_idx = step_in_epoch * accumulation + a
                    if dataloader is not None:
                        batch_spec = epoch_batches[batch_idx]
                        train_env, train_n, batch_seed = _build_train_env(batch_spec)
                        batch_payload = batch_spec.payload
                        batch_items = (
                            list(batch_payload)
                            if isinstance(batch_payload, list)
                            and all(isinstance(item, dict) for item in batch_payload)
                            else []
                        )
                    else:
                        batch_seed = shuffled_seeds[batch_idx]
                        train_env = adapter.build_train_env(
                            batch_size=batch_size,
                            seed=batch_seed,
                            out_root=out_root,
                        )
                        train_n = len(train_env) if hasattr(train_env, "__len__") else batch_size
                        batch_items = []

                    # Directory routing
                    if accumulation > 1:
                        batch_dir = os.path.join(step_dir, f"batch_{a}")
                    else:
                        batch_dir = step_dir

                    rollout_dir = os.path.join(batch_dir, "rollout")
                    patches_dir = os.path.join(batch_dir, "patches")
                    feedback_direct_items = _feedback_direct_items(
                        dataloader=dataloader,
                        batch_items=batch_items,
                        mode=cfg.get("feedback_direct_mode", "auto"),
                    )

                    # ① ROLLOUT ────────────────────────────────────────────
                    t_phase = time.time()
                    if feedback_direct_items:
                        pred_dir = os.path.join(rollout_dir, "predictions")
                        normal_rollout_items = _non_feedback_direct_items(
                            batch_items=batch_items,
                            feedback_direct_items=feedback_direct_items,
                        )
                        print(
                            f"    [1/6 FEEDBACK-DIRECT] ranked feedback items={len(feedback_direct_items)} "
                            f"(batch_seed={batch_seed})"
                        )
                        rollout_results = _write_feedback_direct_predictions(
                            items=feedback_direct_items,
                            prediction_dir=pred_dir,
                        )
                        if normal_rollout_items:
                            normal_batch = BatchSpec(
                                phase=batch_spec.phase,
                                split=batch_spec.split,
                                seed=batch_spec.seed,
                                batch_size=len(normal_rollout_items),
                                payload=normal_rollout_items,
                                metadata=batch_spec.metadata,
                            )
                            normal_env, normal_n, _normal_seed = _build_train_env(normal_batch)
                            print(
                                f"    [1/6 ROLLOUT] remaining train items={normal_n} "
                                f"(from mixed feedback-direct batch)"
                            )
                            rollout_results.extend(
                                adapter.rollout(
                                    normal_env,
                                    current_skill,
                                    rollout_dir,
                                    use_eval_feedback=True,
                                )
                            )
                        step_rec["feedback_direct_mode"] = cfg.get("feedback_direct_mode", "auto")
                        step_rec["feedback_direct_items"] = _dedupe_texts(
                            list(step_rec.get("feedback_direct_items") or [])
                            + [str(item.get("id") or "") for item in feedback_direct_items],
                            limit=64,
                        )
                    else:
                        print(f"    [1/6 ROLLOUT] train items={train_n} (from pool, batch_seed={batch_seed})")
                        rollout_results = adapter.rollout(
                            train_env, current_skill, rollout_dir,
                            use_eval_feedback=True,
                        )
                    r_hard, r_soft = compute_score(rollout_results)
                    total_rollout_time += time.time() - t_phase
                    all_rollout_results.extend(rollout_results)
                    print(f"    [1/6 done] hard={r_hard:.4f} soft={r_soft:.4f}")

                    # ② REFLECT ────────────────────────────────────────────
                    t_phase = time.time()
                    pred_dir = os.path.join(rollout_dir, "predictions")

                    # Build step context from buffer
                    step_buffer_context = _format_step_buffer(step_buffer)
                    if feedback_direct_items:
                        feedback_direct_context = _format_feedback_direct_context(feedback_direct_items, dataloader)
                        feedback_direct_contexts.append(feedback_direct_context)
                        step_buffer_context = _join_optimizer_context(step_buffer_context, feedback_direct_context)

                    raw_patches = adapter.reflect(
                        rollout_results, current_skill, batch_dir,
                        prediction_dir=pred_dir, patches_dir=patches_dir,
                        random_seed=batch_seed,
                        step_buffer_context=step_buffer_context,
                        meta_skill_context=active_meta_skill,
                    )
                    failure_patches, success_patches = _normalise_patches(
                        raw_patches,
                        update_mode=update_mode,
                    )
                    structured_failure_hints = _structured_failure_hints(rollout_results)
                    structured_failure_ids = {
                        str(hint.get("source_item_id") or "").strip()
                        for hint in structured_failure_hints
                        if str(hint.get("source_item_id") or "").strip()
                    }
                    feedback_failure_context = _ranked_feedback_context_packet(
                        dataloader,
                        structured_failure_ids,
                        fallback_to_all=False,
                    )
                    if feedback_failure_context:
                        structured_failure_hints = [
                            _with_human_feedback_context(hint, feedback_failure_context) or hint
                            for hint in structured_failure_hints
                        ]
                    if structured_failure_hints and not failure_patches:
                        hard_failure_retry_budget = max(0, int(cfg.get("hard_failure_retry_budget", 1) or 0))
                        hard_failure_retry_attempts = list(step_rec.get("hard_failure_retry_attempts") or [])
                        hard_failure_results = [
                            result for result in rollout_results
                            if isinstance(result, dict) and not result.get("hard")
                        ]
                        for retry_attempt in range(1, hard_failure_retry_budget + 1):
                            retry_context = _format_hard_failure_retry_context(
                                structured_failure_hints,
                                attempt=retry_attempt,
                                budget=hard_failure_retry_budget,
                                feedback_context=feedback_failure_context,
                            )
                            retry_patches_dir = os.path.join(
                                patches_dir,
                                f"hard_failure_retry_{retry_attempt:02d}",
                            )
                            raw_retry_patches = adapter.reflect(
                                hard_failure_results, current_skill, batch_dir,
                                prediction_dir=pred_dir, patches_dir=retry_patches_dir,
                                random_seed=batch_seed + retry_attempt,
                                step_buffer_context=_join_optimizer_context(step_buffer_context, retry_context),
                                meta_skill_context=active_meta_skill,
                            )
                            retry_failure_patches, retry_success_patches = _normalise_patches(
                                raw_retry_patches,
                                update_mode=update_mode,
                            )
                            if retry_failure_patches:
                                raw_patches.extend(raw_retry_patches)
                                failure_patches.extend(retry_failure_patches)
                                success_patches.extend(retry_success_patches)
                            hard_failure_retry_attempts.append(
                                {
                                    "attempt": retry_attempt,
                                    "status": "converted_to_patch" if retry_failure_patches else "no_patch",
                                    "n_failure_patches": len(retry_failure_patches),
                                    "n_success_patches": len(retry_success_patches),
                                    "failure_hints": _format_unconverted_failure_hints(structured_failure_hints),
                                }
                            )
                            if retry_failure_patches:
                                break
                        if hard_failure_retry_attempts:
                            step_rec["hard_failure_retry_attempts"] = hard_failure_retry_attempts
                        if not failure_patches:
                            step_rec["failure_hint_not_converted_to_patch"] = True
                            step_rec["unconverted_failure_hints"] = (
                                step_rec.get("unconverted_failure_hints", [])
                                + _format_unconverted_failure_hints(structured_failure_hints)
                            )
                    all_failure_patches.extend(failure_patches)
                    all_success_patches.extend(success_patches)
                    all_raw_patches.extend(raw_patches)
                    total_reflect_time += time.time() - t_phase

                    print(
                        f"    [2/6 done] failure_patches={len(failure_patches)} "
                        f"success_patches={len(success_patches)}"
                    )

                    # Track per-batch stats
                    accum_rollout_stats.append({
                        "batch_idx": a,
                        "batch_seed": batch_seed,
                        "n_envs": len(rollout_results),
                        "hard": r_hard,
                        "soft": r_soft,
                        "n_failure_patches": len(failure_patches),
                        "n_success_patches": len(success_patches),
                    })

                # ── End of accumulation loop ─────────────────────────────

                # Aggregate rollout stats across batches
                total_n = sum(b["n_envs"] for b in accum_rollout_stats)
                agg_hard = sum(b["hard"] * b["n_envs"] for b in accum_rollout_stats) / max(total_n, 1)
                agg_soft = sum(b["soft"] * b["n_envs"] for b in accum_rollout_stats) / max(total_n, 1)

                step_rec["rollout_hard"] = round(agg_hard, 6)
                step_rec["rollout_soft"] = round(agg_soft, 6)
                step_rec["rollout_n"] = total_n
                step_rec["accumulation_batches"] = accum_rollout_stats
                step_rec["timing"]["rollout_s"] = round(total_rollout_time, 1)
                step_rec["timing"]["reflect_s"] = round(total_reflect_time, 1)

                n_total_patches = len(all_failure_patches) + len(all_success_patches)
                step_rec["n_patches"] = n_total_patches
                step_rec["n_failure_patches"] = len(all_failure_patches)
                step_rec["n_success_patches"] = len(all_success_patches)

                if accumulation > 1:
                    print(
                        f"    [accum done] total: failure={len(all_failure_patches)} "
                        f"success={len(all_success_patches)} "
                        f"from {accumulation} batches"
                    )

                # ── No patches? Skip ─────────────────────────────────────
                if not all_failure_patches and not all_success_patches:
                    step_rec["action"] = "skip_no_patches"
                    result_ids = {str(row.get("id", "")) for row in all_rollout_results if isinstance(row, dict)}
                    if _has_ranked_feedback(dataloader, result_ids):
                        retry_hints = _ranked_feedback_retry_hints(dataloader, result_ids)
                        optimizer_hint = _human_feedback_not_distilled_hint(retry_hints)
                        step_rec["no_candidate_reason"] = "human_feedback_not_distilled"
                        step_rec["no_candidate_triggers"] = _dedupe_texts(
                            ["human_feedback_not_distilled", "no_usable_patches_from_ranked_feedback"]
                        )
                        step_rec["optimizer_hint"] = optimizer_hint
                        step_rec["feedback_retry_hints"] = retry_hints
                        step_rec["failure"] = {
                            "primary_reason": "human_feedback_not_distilled",
                            "human_reason": (
                                "Ranked human feedback was imported, but reflection produced no usable skill patches."
                            ),
                            "optimizer_hint": optimizer_hint,
                            "failed_dimensions": ["human_feedback_alignment"],
                            "evidence": retry_hints.get("improve") or retry_hints.get("preserve") or [],
                            "stage_status": [{"stage": "reflect", "status": "failed"}],
                        }
                    if step_rec.get("failure_hint_not_converted_to_patch"):
                        step_rec.setdefault("no_candidate_reason", "failure_hint_not_converted_to_patch")
                        step_rec["no_candidate_triggers"] = _dedupe_texts(
                            list(step_rec.get("no_candidate_triggers") or [])
                            + ["failure_hint_not_converted_to_patch"]
                        )
                    step_rec["current_score"] = current_score
                    step_rec["best_score"] = best_score
                    step_rec["best_step"] = best_step
                    step_rec["skill_len"] = len(current_skill)
                    step_rec["wall_time_s"] = round(time.time() - step_t0, 1)
                    history.append(step_rec)
                    _save_history(out_root, history)
                    _save_skill(out_root, global_step, current_skill)
                    _persist_runtime_state(global_step)
                    with open(os.path.join(step_dir, "step_record.json"), "w") as f:
                        json.dump(step_rec, f, indent=2, ensure_ascii=False)
                    print("    [skip] no usable patches — skill unchanged")
                    continue

                feedback_direct_optimizer_context = _join_optimizer_context(*feedback_direct_contexts)

                gate_reject_retry_budget = max(0, int(cfg.get("gate_reject_retry_budget", 3) or 0))
                gate_reject_retry_close_gap = max(0.0, float(cfg.get("gate_reject_retry_close_gap", 0.03) or 0.0))
                gate_reject_retry_attempts: list[dict] = []
                seen_gate_rejection_reasons: set[str] = set()
                wrong_artifact_retry_budget = max(0, int(cfg.get("wrong_artifact_retry_budget", 1) or 0))
                wrong_artifact_retry_attempts: list[dict] = []
                seen_wrong_artifact_rejection_reasons: set[str] = set()
                duplicate_gate_retry_context = ""
                duplicate_gate_retry_hashes: set[str] = set()
                noop_retry_budget = max(0, int(cfg.get("noop_retry_budget", 1) or 0))
                noop_retry_attempts: list[dict] = []
                noop_retry_context = ""
                skip_step_after_noop = False

                for noop_attempt in range(noop_retry_budget + 1):
                    retry_suffix = "" if noop_attempt == 0 else f"_retry_{noop_attempt:02d}"
                    optimizer_context = _join_optimizer_context(
                        active_meta_skill,
                        feedback_direct_optimizer_context,
                        noop_retry_context,
                    )
                    rewrite_step_context = _join_optimizer_context(step_buffer_context, noop_retry_context)
                    if noop_attempt:
                        print(
                            f"    [noop retry {noop_attempt}/{noop_retry_budget}] "
                            "rerunning aggregate/select/update with no-op feedback"
                        )

                    # ③ AGGREGATE ──────────────────────────────────────────────
                    t_phase = time.time()
                    merged_patch = merge_patches(
                        current_skill, all_failure_patches, all_success_patches,
                        batch_size=merge_bs, verbose=True,
                        workers=cfg["analyst_workers"],
                        update_mode=update_mode,
                        meta_skill_context=optimizer_context,
                    )
                    with open(os.path.join(step_dir, f"merged_patch{retry_suffix}.json"), "w") as f:
                        json.dump(merged_patch, f, ensure_ascii=False, indent=2)

                    merged_items = get_payload_items(merged_patch, update_mode)
                    n_edits_merged = len(merged_items)
                    step_rec["n_edits_merged"] = n_edits_merged
                    step_rec["timing"]["aggregate_s"] = round(time.time() - t_phase, 1)
                    print(f"    [3/6 done] merged {n_edits_merged} {payload_label(update_mode)}")

                    # ④ SELECT ─────────────────────────────────────────────────
                    t_phase = time.time()
                    lr_decision = None
                    if is_full_rewrite_minibatch_mode(update_mode):
                        edit_budget = None
                        ranked_patch = merged_patch
                        ranked_items = merged_items
                        n_edits_ranked = len(ranked_items)
                        step_rec["n_edits_ranked"] = n_edits_ranked
                        step_rec["edit_budget"] = None
                        step_rec["lr_control_mode"] = "none"
                        with open(os.path.join(step_dir, f"ranked_edits{retry_suffix}.json"), "w") as f:
                            json.dump(ranked_patch, f, ensure_ascii=False, indent=2)
                    else:
                        if lr_control_mode == "autonomous":
                            lr_decision = decide_autonomous_learning_rate(
                                skill_content=current_skill,
                                merged_patch=merged_patch,
                                update_mode=update_mode,
                                rollout_hard=agg_hard,
                                rollout_soft=agg_soft,
                                rollout_n=total_n,
                                step_buffer_context=rewrite_step_context,
                                meta_skill_context=optimizer_context,
                            )
                            edit_budget = int(lr_decision["learning_rate"])
                            with open(os.path.join(step_dir, f"lr_decision{retry_suffix}.json"), "w") as f:
                                json.dump(lr_decision, f, ensure_ascii=False, indent=2)
                            with open(os.path.join(out_root, "lr_history.jsonl"), "a") as f:
                                f.write(json.dumps({
                                    "step": global_step,
                                    "epoch": epoch,
                                    "noop_retry_attempt": noop_attempt,
                                    **lr_decision,
                                }, ensure_ascii=False) + "\n")
                        else:
                            edit_budget = scheduler.step() if noop_attempt == 0 else int(step_rec.get("edit_budget") or cfg["edit_budget"])
                        ranked_patch = rank_and_select(
                            current_skill, merged_patch,
                            max_edits=edit_budget,
                            update_mode=update_mode,
                            meta_skill_context=optimizer_context,
                        )
                        with open(os.path.join(step_dir, f"ranked_edits{retry_suffix}.json"), "w") as f:
                            json.dump(ranked_patch, f, ensure_ascii=False, indent=2)

                        ranked_items = get_payload_items(ranked_patch, update_mode)
                        n_edits_ranked = len(ranked_items)
                        step_rec["n_edits_ranked"] = n_edits_ranked
                        step_rec["edit_budget"] = edit_budget
                        step_rec["lr_control_mode"] = lr_control_mode
                        if lr_decision is not None:
                            step_rec["lr_decision"] = lr_decision
                    step_rec["timing"]["select_s"] = round(time.time() - t_phase, 1)

                    support_counts = [
                        item.get("support_count", 0) for item in ranked_items if isinstance(item, dict)
                    ]
                    step_rec["support_counts"] = support_counts
                    if is_full_rewrite_minibatch_mode(update_mode):
                        print(
                            f"    [4/6 SELECT] skipped LR/select; "
                            f"using {n_edits_ranked} merged {payload_label(update_mode)}"
                        )
                    else:
                        print(
                            f"    [4/6 SELECT] "
                            f"{n_edits_merged} -> {n_edits_ranked} {payload_label(update_mode)} "
                            f"(budget={edit_budget}, lr_control={lr_control_mode})"
                        )

                    # ⑤ UPDATE ─────────────────────────────────────────────────
                    t_phase = time.time()
                    rewrite_result = None
                    if update_mode == "rewrite_from_suggestions":
                        rewrite_result = rewrite_skill_from_suggestions(
                            current_skill,
                            ranked_patch,
                            step_buffer_context=rewrite_step_context,
                            env=cfg.get("env"),
                            reasoning_effort=rewrite_reasoning_effort,
                            max_completion_tokens=rewrite_max_completion_tokens,
                        )
                        if rewrite_result and rewrite_result.get("new_skill"):
                            candidate_skill = rewrite_result["new_skill"]
                            apply_report = []
                            with open(os.path.join(step_dir, f"rewrite_result{retry_suffix}.json"), "w") as f:
                                json.dump(rewrite_result, f, ensure_ascii=False, indent=2)
                        else:
                            candidate_skill = current_skill
                            apply_report = []
                    elif is_full_rewrite_minibatch_mode(update_mode):
                        skill_candidates = get_payload_items(ranked_patch, update_mode)
                        selected_candidate = next(
                            (
                                item for item in skill_candidates
                                if isinstance(item, dict) and str(item.get("new_skill", "")).strip()
                            ),
                            None,
                        )
                        if selected_candidate:
                            candidate_skill = str(selected_candidate["new_skill"]).rstrip() + "\n"
                            apply_report = []
                            rewrite_result = {
                                "reasoning": ranked_patch.get("reasoning", ""),
                                "change_summary": selected_candidate.get("change_summary", []),
                                "title": selected_candidate.get("title", ""),
                                "source_type": selected_candidate.get("source_type", ""),
                            }
                            with open(os.path.join(step_dir, f"full_rewrite_result{retry_suffix}.json"), "w") as f:
                                json.dump(
                                    {
                                        "selected_candidate": selected_candidate,
                                        "merged_patch": ranked_patch,
                                    },
                                    f,
                                    ensure_ascii=False,
                                    indent=2,
                                )
                        else:
                            candidate_skill = current_skill
                            apply_report = []
                    else:
                        candidate_skill, apply_report = apply_patch_with_report(current_skill, ranked_patch)
                    candidate_skill_path = os.path.join(step_dir, f"candidate_skill{retry_suffix}.md")
                    with open(candidate_skill_path, "w") as f:
                        f.write(candidate_skill)
                    if retry_suffix:
                        with open(os.path.join(step_dir, "candidate_skill.md"), "w") as f:
                            f.write(candidate_skill)
                    if apply_report:
                        with open(os.path.join(step_dir, f"edit_apply_report{retry_suffix}.json"), "w") as f:
                            json.dump(apply_report, f, indent=2, ensure_ascii=False)

                    cand_hash = skill_hash(candidate_skill)
                    step_rec["candidate_hash"] = cand_hash
                    step_rec["candidate_skill_len"] = len(candidate_skill)
                    if rewrite_result:
                        step_rec["rewrite_change_summary"] = rewrite_result.get("change_summary", [])
                    if apply_report:
                        step_rec["edit_apply_summary"] = {
                            "total": len(apply_report),
                            "applied": sum(
                                1 for row in apply_report if str(row.get("status", "")).startswith("applied")
                            ),
                            "skipped": sum(
                                1 for row in apply_report if str(row.get("status", "")).startswith("skipped")
                            ),
                            "errors": sum(
                                1 for row in apply_report if row.get("status") == "error"
                            ),
                        }
                    step_rec["timing"]["update_s"] = round(time.time() - t_phase, 1)

                    no_rewrite = (
                        update_mode == "rewrite_from_suggestions"
                        and rewrite_result is None
                    ) or (
                        is_full_rewrite_minibatch_mode(update_mode)
                        and rewrite_result is None
                    )
                    change_check = _detect_no_meaningful_change(
                        current_skill=current_skill,
                        candidate_skill=candidate_skill,
                        ranked_items=ranked_items,
                        apply_report=apply_report,
                        update_mode=update_mode,
                        dataloader=dataloader,
                        rollout_results=all_rollout_results,
                    )
                    if no_rewrite and "no_meaningful_skill_change" not in change_check.reasons:
                        change_check.reasons.insert(0, "no_meaningful_skill_change")
                    if change_check.reasons:
                        attempt_record = {
                            "attempt": noop_attempt,
                            "reasons": change_check.reasons,
                            "retry_hints": change_check.retry_hints,
                            "candidate_hash": cand_hash,
                        }
                        noop_retry_attempts.append(attempt_record)
                        step_rec["noop_retry_attempts"] = noop_retry_attempts
                        with open(os.path.join(step_dir, f"noop_change_check{retry_suffix}.json"), "w") as f:
                            json.dump(attempt_record, f, indent=2, ensure_ascii=False)
                        if noop_attempt < noop_retry_budget:
                            noop_retry_context = _format_noop_retry_context(change_check, noop_attempt + 1)
                            print(
                                "    [noop] rejected optimizer update before evaluation: "
                                + ", ".join(change_check.reasons)
                            )
                            continue
                        step_rec["action"] = "skip_no_meaningful_change"
                        step_rec["no_candidate_reason"] = change_check.reasons[0]
                        step_rec["no_candidate_triggers"] = change_check.reasons
                        step_rec["current_score"] = current_score
                        step_rec["best_score"] = best_score
                        step_rec["best_step"] = best_step
                        step_rec["skill_len"] = len(current_skill)
                        step_rec["wall_time_s"] = round(time.time() - step_t0, 1)
                        history.append(step_rec)
                        _save_history(out_root, history)
                        _save_skill(out_root, global_step, current_skill)
                        _persist_runtime_state(global_step)
                        with open(os.path.join(step_dir, "step_record.json"), "w") as f:
                            json.dump(step_rec, f, indent=2, ensure_ascii=False)
                        print(
                            "    [skip] no meaningful skill change after retry budget: "
                            + ", ".join(change_check.reasons)
                        )
                        skip_step_after_noop = True
                    break

                if skip_step_after_noop:
                    continue

                print(
                    f"    [5/6 UPDATE] "
                    f"skill_len {len(current_skill)} -> {len(candidate_skill)}"
                )

                # ⑥ EVALUATE ───────────────────────────────────────────────
                t_phase = time.time()
                selection_rejection_signal = None
                selection_candidate_context = None
                if cand_hash in sel_cache:
                    cand_hard, cand_soft = sel_cache[cand_hash]
                    selection_rejection_signal = sel_rejection_signal_cache.get(cand_hash)
                    selection_candidate_context = sel_eval_context_cache.get(cand_hash)
                    print(
                        f"    [6/6 EVALUATE] "
                        f"cache hit {cand_hash}: hard={cand_hard:.4f}"
                    )
                else:
                    sel_env, sel_n = _build_eval_env(
                        split="valid_seen",
                        env_num=cfg["sel_env_num"],
                        seed=seed,
                    )
                    print(f"    [6/6 EVALUATE] selection items={sel_n}")
                    sel_eval_dir = os.path.join(step_dir, "selection_eval")
                    sel_results = adapter.rollout(sel_env, candidate_skill, sel_eval_dir)
                    selection_rejection_signal = _selection_rejection_signal(sel_results)
                    selection_candidate_context = _selection_eval_context(sel_results)
                    sel_eval_context_cache[cand_hash] = selection_candidate_context
                    if selection_rejection_signal:
                        sel_rejection_signal_cache[cand_hash] = selection_rejection_signal
                    gate_block = find_gate_block(sel_results)
                    if gate_block is not None:
                        block = gate_block.to_dict()
                        _write_gate_block(sel_eval_dir, block)
                        cand_hard = None
                        cand_soft = None
                        cand_gate_score = None
                        step_rec["selection_hard"] = None
                        step_rec["selection_soft"] = None
                        step_rec["gate_metric"] = gate_metric
                        step_rec["candidate_gate_score"] = None
                        step_rec["action"] = f"blocked:{block['blocker']}"
                        step_rec["gate_status"] = "blocked"
                        step_rec["gate_blocker"] = block["blocker"]
                        step_rec["gate_block"] = block
                        print(
                            f"    [6/6 EVALUATE] BLOCKED "
                            f"blocked:{block['blocker']} items={len(block.get('items', []))}"
                        )
                    else:
                        cand_hard, cand_soft = compute_score(sel_results)
                        sel_cache[cand_hash] = (cand_hard, cand_soft)

                if step_rec.get("gate_status") != "blocked":
                    step_rec["selection_hard"] = cand_hard
                    step_rec["selection_soft"] = cand_soft
                    if selection_candidate_context:
                        step_rec["selection_eval_context"] = selection_candidate_context

                    gate = evaluate_gate(
                        candidate_skill=candidate_skill,
                        cand_hard=cand_hard,
                        current_skill=current_skill,
                        current_score=current_score,
                        best_skill=best_skill,
                        best_score=best_score,
                        best_step=best_step,
                        global_step=global_step,
                        cand_soft=cand_soft,
                        metric=gate_metric,
                        mixed_weight=gate_mixed_weight,
                    )
                    cand_gate_score = select_gate_score(
                        cand_hard, cand_soft, gate_metric, gate_mixed_weight,
                    )
                    step_rec["gate_metric"] = gate_metric
                    step_rec["candidate_gate_score"] = cand_gate_score
                    step_rec["action"] = gate.action
                    prev_current = current_score
                    prev_best = best_score
                    current_skill = gate.current_skill
                    current_score = gate.current_score
                    best_skill = gate.best_skill
                    best_score = gate.best_score
                    best_step = gate.best_step
                    if gate.action in {"accept", "accept_new_best"}:
                        current_origin = f"step_{global_step:04d}"
                    if gate.action == "accept_new_best":
                        best_origin = current_origin
                        selection_scores_by_origin[best_origin] = (cand_hard, cand_soft)

                    if gate_metric == "hard":
                        score_label = f"hard={cand_hard:.4f}"
                    elif gate_metric == "soft":
                        score_label = f"soft={cand_soft:.4f}"
                    else:
                        score_label = (
                            f"mixed[w={gate_mixed_weight}]={cand_gate_score:.4f} "
                            f"(hard={cand_hard:.4f} soft={cand_soft:.4f})"
                        )
                    if gate.action == "accept_new_best":
                        print(
                            f"    [6/6 EVALUATE] ACCEPT (new best) "
                            f"{score_label} > prev best {prev_best:.4f}"
                        )
                    elif gate.action == "accept":
                        print(
                            f"    [6/6 EVALUATE] ACCEPT "
                            f"{score_label} > current={prev_current:.4f}"
                        )
                    else:
                        print(
                            f"    [6/6 EVALUATE] REJECT "
                            f"{score_label} <= current={current_score:.4f}"
                        )
                        while True:
                            baseline_scores = sel_cache.get(
                                skill_hash(current_skill),
                                sel_cache.get(skill_hash(skill_init), (None, None)),
                            )
                            gate_rejection = _selection_reject_gate_rejection(
                                history=[step_rec],
                                baseline_scores=baseline_scores,
                                gate_metric=gate_metric,
                                gate_mixed_weight=gate_mixed_weight,
                                rejection_signal=selection_rejection_signal,
                                baseline_context=sel_eval_context_cache.get(skill_hash(current_skill))
                                or sel_eval_context_cache.get(skill_hash(skill_init)),
                                candidate_context=selection_candidate_context,
                                human_feedback_context=_ranked_feedback_context_packet(dataloader),
                            )
                            wrong_artifact_retry = _is_wrong_artifact_rejection(gate_rejection)
                            active_retry_attempts = (
                                wrong_artifact_retry_attempts if wrong_artifact_retry else gate_reject_retry_attempts
                            )
                            active_seen_reasons = (
                                seen_wrong_artifact_rejection_reasons
                                if wrong_artifact_retry
                                else seen_gate_rejection_reasons
                            )
                            active_retry_budget = wrong_artifact_retry_budget if wrong_artifact_retry else gate_reject_retry_budget
                            retry_count = sum(1 for attempt in active_retry_attempts if attempt.get("action") == "retry")
                            gate_rejection = _gate_rejection_with_retry_attempts(
                                gate_rejection,
                                used=retry_count,
                                budget=active_retry_budget,
                            )
                            can_retry_gate_reject, gate_reject_stop_reason = _gate_rejection_retry_decision(
                                gate_rejection,
                                attempt=retry_count,
                                budget=active_retry_budget,
                                seen_reasons=active_seen_reasons,
                                close_gap=None if wrong_artifact_retry else gate_reject_retry_close_gap,
                            )
                            gate_retry_record = {
                                "attempt": retry_count,
                                "candidate_hash": cand_hash,
                                "retry_class": "wrong_artifact_type" if wrong_artifact_retry else "gate_reject",
                                "action": "retry" if can_retry_gate_reject else "stop",
                                "stop_reason": gate_reject_stop_reason,
                                "gate_rejection": gate_rejection,
                            }
                            if wrong_artifact_retry and isinstance(gate_rejection, dict):
                                gate_retry_record["expected_artifact"] = str(
                                    gate_rejection.get("expected_artifact") or "vue-vite bundle"
                                )
                                gate_retry_record["actual_artifact"] = str(
                                    gate_rejection.get("actual_artifact") or "skill markdown/template"
                                )
                            if wrong_artifact_retry:
                                wrong_artifact_retry_attempts.append(gate_retry_record)
                                retry_file_prefix = "wrong_artifact_retry"
                            else:
                                gate_reject_retry_attempts.append(gate_retry_record)
                                retry_file_prefix = "gate_reject_retry"
                            step_rec["gate_rejection"] = gate_rejection
                            if gate_reject_retry_attempts:
                                step_rec["gate_reject_retry_attempts"] = gate_reject_retry_attempts
                            if wrong_artifact_retry_attempts:
                                step_rec["wrong_artifact_retry_attempts"] = wrong_artifact_retry_attempts
                            gate_retry_record_path = os.path.join(
                                step_dir,
                                f"{retry_file_prefix}_{retry_count:02d}.json",
                            )
                            with open(
                                gate_retry_record_path,
                                "w",
                            ) as f:
                                json.dump(gate_retry_record, f, indent=2, ensure_ascii=False)
                            if not can_retry_gate_reject or gate_rejection is None:
                                step_rec["no_candidate_triggers"] = _dedupe_texts(
                                    step_rec.get("no_candidate_triggers", [])
                                    + ["gate_rejected_best_origin_initial_skill", gate_reject_stop_reason]
                                )
                                step_rec["no_candidate_reason"] = "gate_rejected_best_origin_initial_skill"
                                break

                            gate_rejection_signature = _gate_rejection_signature(gate_rejection)
                            if gate_rejection_signature:
                                active_seen_reasons.add(gate_rejection_signature)
                            gate_retry_context = _format_gate_reject_retry_context(
                                gate_rejection,
                                retry_count + 1,
                                active_retry_budget,
                            )
                            gate_retry_context = _join_optimizer_context(
                                gate_retry_context,
                                "" if wrong_artifact_retry else duplicate_gate_retry_context,
                            )
                            gate_retry_suffix = f"_{retry_file_prefix}_{retry_count + 1:02d}"
                            print(
                                f"    [gate retry {retry_count + 1}/{active_retry_budget}] "
                                "rerunning aggregate/select/update with gate-rejection feedback"
                            )

                            retry_optimizer_context = _join_optimizer_context(
                                active_meta_skill,
                                feedback_direct_optimizer_context,
                                noop_retry_context,
                                gate_retry_context,
                            )
                            retry_rewrite_context = _join_optimizer_context(
                                step_buffer_context,
                                noop_retry_context,
                                gate_retry_context,
                            )

                            t_retry_phase = time.time()
                            merged_patch = merge_patches(
                                current_skill,
                                all_failure_patches,
                                all_success_patches,
                                batch_size=merge_bs,
                                verbose=True,
                                workers=cfg["analyst_workers"],
                                update_mode=update_mode,
                                meta_skill_context=retry_optimizer_context,
                            )
                            with open(os.path.join(step_dir, f"merged_patch{gate_retry_suffix}.json"), "w") as f:
                                json.dump(merged_patch, f, ensure_ascii=False, indent=2)
                            merged_items = get_payload_items(merged_patch, update_mode)
                            n_edits_merged = len(merged_items)
                            step_rec["n_edits_merged"] = n_edits_merged
                            step_rec["timing"]["aggregate_s"] = round(
                                step_rec["timing"].get("aggregate_s", 0) + time.time() - t_retry_phase,
                                1,
                            )

                            t_retry_phase = time.time()
                            lr_decision = None
                            if is_full_rewrite_minibatch_mode(update_mode):
                                edit_budget = None
                                ranked_patch = merged_patch
                                ranked_items = merged_items
                                n_edits_ranked = len(ranked_items)
                                step_rec["n_edits_ranked"] = n_edits_ranked
                                step_rec["edit_budget"] = None
                                step_rec["lr_control_mode"] = "none"
                                with open(os.path.join(step_dir, f"ranked_edits{gate_retry_suffix}.json"), "w") as f:
                                    json.dump(ranked_patch, f, ensure_ascii=False, indent=2)
                            else:
                                if lr_control_mode == "autonomous":
                                    lr_decision = decide_autonomous_learning_rate(
                                        skill_content=current_skill,
                                        merged_patch=merged_patch,
                                        update_mode=update_mode,
                                        rollout_hard=agg_hard,
                                        rollout_soft=agg_soft,
                                        rollout_n=total_n,
                                        step_buffer_context=retry_rewrite_context,
                                        meta_skill_context=retry_optimizer_context,
                                    )
                                    edit_budget = int(lr_decision["learning_rate"])
                                    with open(os.path.join(step_dir, f"lr_decision{gate_retry_suffix}.json"), "w") as f:
                                        json.dump(lr_decision, f, ensure_ascii=False, indent=2)
                                    with open(os.path.join(out_root, "lr_history.jsonl"), "a") as f:
                                        f.write(json.dumps({
                                            "step": global_step,
                                            "epoch": epoch,
                                            "gate_reject_retry_attempt": retry_count + 1,
                                            **lr_decision,
                                        }, ensure_ascii=False) + "\n")
                                else:
                                    edit_budget = int(step_rec.get("edit_budget") or cfg["edit_budget"])
                                ranked_patch = rank_and_select(
                                    current_skill,
                                    merged_patch,
                                    max_edits=edit_budget,
                                    update_mode=update_mode,
                                    meta_skill_context=retry_optimizer_context,
                                )
                                with open(os.path.join(step_dir, f"ranked_edits{gate_retry_suffix}.json"), "w") as f:
                                    json.dump(ranked_patch, f, ensure_ascii=False, indent=2)
                                ranked_items = get_payload_items(ranked_patch, update_mode)
                                n_edits_ranked = len(ranked_items)
                                step_rec["n_edits_ranked"] = n_edits_ranked
                                step_rec["edit_budget"] = edit_budget
                                step_rec["lr_control_mode"] = lr_control_mode
                                if lr_decision is not None:
                                    step_rec["lr_decision"] = lr_decision
                            step_rec["timing"]["select_s"] = round(
                                step_rec["timing"].get("select_s", 0) + time.time() - t_retry_phase,
                                1,
                            )
                            step_rec["support_counts"] = [
                                item.get("support_count", 0)
                                for item in ranked_items
                                if isinstance(item, dict)
                            ]

                            t_retry_phase = time.time()
                            rewrite_result = None
                            if update_mode == "rewrite_from_suggestions":
                                rewrite_result = rewrite_skill_from_suggestions(
                                    current_skill,
                                    ranked_patch,
                                    step_buffer_context=retry_rewrite_context,
                                    env=cfg.get("env"),
                                    reasoning_effort=rewrite_reasoning_effort,
                                    max_completion_tokens=rewrite_max_completion_tokens,
                                )
                                if rewrite_result and rewrite_result.get("new_skill"):
                                    candidate_skill = rewrite_result["new_skill"]
                                    apply_report = []
                                    with open(os.path.join(step_dir, f"rewrite_result{gate_retry_suffix}.json"), "w") as f:
                                        json.dump(rewrite_result, f, ensure_ascii=False, indent=2)
                                else:
                                    candidate_skill = current_skill
                                    apply_report = []
                            elif is_full_rewrite_minibatch_mode(update_mode):
                                skill_candidates = get_payload_items(ranked_patch, update_mode)
                                selected_candidate = next(
                                    (
                                        item for item in skill_candidates
                                        if isinstance(item, dict) and str(item.get("new_skill", "")).strip()
                                    ),
                                    None,
                                )
                                if selected_candidate:
                                    candidate_skill = str(selected_candidate["new_skill"]).rstrip() + "\n"
                                    apply_report = []
                                    rewrite_result = {
                                        "reasoning": ranked_patch.get("reasoning", ""),
                                        "change_summary": selected_candidate.get("change_summary", []),
                                        "title": selected_candidate.get("title", ""),
                                        "source_type": selected_candidate.get("source_type", ""),
                                    }
                                    with open(os.path.join(step_dir, f"full_rewrite_result{gate_retry_suffix}.json"), "w") as f:
                                        json.dump(
                                            {
                                                "selected_candidate": selected_candidate,
                                                "merged_patch": ranked_patch,
                                            },
                                            f,
                                            ensure_ascii=False,
                                            indent=2,
                                        )
                                else:
                                    candidate_skill = current_skill
                                    apply_report = []
                            else:
                                candidate_skill, apply_report = apply_patch_with_report(
                                    current_skill,
                                    ranked_patch,
                                )
                            candidate_skill_path = os.path.join(
                                step_dir,
                                f"candidate_skill{gate_retry_suffix}.md",
                            )
                            with open(candidate_skill_path, "w") as f:
                                f.write(candidate_skill)
                            with open(os.path.join(step_dir, "candidate_skill.md"), "w") as f:
                                f.write(candidate_skill)
                            if apply_report:
                                with open(os.path.join(step_dir, f"edit_apply_report{gate_retry_suffix}.json"), "w") as f:
                                    json.dump(apply_report, f, indent=2, ensure_ascii=False)

                            cand_hash = skill_hash(candidate_skill)
                            step_rec["candidate_hash"] = cand_hash
                            step_rec["candidate_skill_len"] = len(candidate_skill)
                            if rewrite_result:
                                step_rec["rewrite_change_summary"] = rewrite_result.get("change_summary", [])
                            step_rec["timing"]["update_s"] = round(
                                step_rec["timing"].get("update_s", 0) + time.time() - t_retry_phase,
                                1,
                            )
                            previous_retry_candidate_hash = str(gate_retry_record.get("candidate_hash") or "").strip()
                            if (
                                not wrong_artifact_retry
                                and previous_retry_candidate_hash
                                and cand_hash == previous_retry_candidate_hash
                            ):
                                gate_retry_record["retry_produced_duplicate_candidate"] = True
                                gate_retry_record["duplicate_of"] = previous_retry_candidate_hash
                                gate_retry_record["duplicate_candidate_hash"] = cand_hash
                                step_rec["retry_produced_duplicate_candidate"] = True
                                step_rec["duplicate_of"] = previous_retry_candidate_hash
                                if gate_rejection_signature:
                                    active_seen_reasons.discard(gate_rejection_signature)
                                if cand_hash in duplicate_gate_retry_hashes:
                                    step_rec["action"] = "reject"
                                    step_rec["no_candidate_reason"] = "gate_rejected_best_origin_initial_skill"
                                    step_rec["no_candidate_triggers"] = _dedupe_texts(
                                        [
                                            "gate_rejected_best_origin_initial_skill",
                                            "repeated_duplicate_candidate",
                                        ]
                                    )
                                    gate_retry_record["action"] = "stop"
                                    gate_retry_record["stop_reason"] = "repeated_duplicate_candidate"
                                    with open(
                                        gate_retry_record_path,
                                        "w",
                                    ) as f:
                                        json.dump(gate_retry_record, f, indent=2, ensure_ascii=False)
                                    break
                                duplicate_gate_retry_hashes.add(cand_hash)
                                duplicate_gate_retry_context = _format_duplicate_gate_retry_context(
                                    duplicate_of=cand_hash,
                                    attempt=retry_count + 1,
                                )
                                with open(
                                    gate_retry_record_path,
                                    "w",
                                ) as f:
                                    json.dump(gate_retry_record, f, indent=2, ensure_ascii=False)
                                continue

                            retry_change_check = _detect_no_meaningful_change(
                                current_skill=current_skill,
                                candidate_skill=candidate_skill,
                                ranked_items=ranked_items,
                                apply_report=apply_report,
                                update_mode=update_mode,
                                dataloader=dataloader,
                                rollout_results=all_rollout_results,
                            )
                            if retry_change_check.reasons:
                                step_rec["action"] = "reject"
                                step_rec["no_candidate_reason"] = "gate_rejected_best_origin_initial_skill"
                                step_rec["no_candidate_triggers"] = _dedupe_texts(
                                    [
                                        "gate_rejected_best_origin_initial_skill",
                                        "gate_retry_no_meaningful_change",
                                    ]
                                    + retry_change_check.reasons
                                )
                                gate_retry_record["action"] = "stop"
                                gate_retry_record["stop_reason"] = "gate_retry_no_meaningful_change"
                                gate_retry_record["noop_reasons"] = retry_change_check.reasons
                                with open(
                                    gate_retry_record_path,
                                    "w",
                                ) as f:
                                    json.dump(gate_retry_record, f, indent=2, ensure_ascii=False)
                                break

                            t_retry_phase = time.time()
                            selection_rejection_signal = None
                            selection_candidate_context = None
                            if cand_hash in sel_cache:
                                cand_hard, cand_soft = sel_cache[cand_hash]
                                selection_rejection_signal = sel_rejection_signal_cache.get(cand_hash)
                                selection_candidate_context = sel_eval_context_cache.get(cand_hash)
                            else:
                                sel_env, sel_n = _build_eval_env(
                                    split="valid_seen",
                                    env_num=cfg["sel_env_num"],
                                    seed=seed,
                                )
                                print(f"    [6/6 EVALUATE] gate retry selection items={sel_n}")
                                sel_eval_dir = os.path.join(
                                    step_dir,
                                    f"selection_eval{gate_retry_suffix}",
                                )
                                sel_results = adapter.rollout(sel_env, candidate_skill, sel_eval_dir)
                                selection_rejection_signal = _selection_rejection_signal(sel_results)
                                selection_candidate_context = _selection_eval_context(sel_results)
                                sel_eval_context_cache[cand_hash] = selection_candidate_context
                                if selection_rejection_signal:
                                    sel_rejection_signal_cache[cand_hash] = selection_rejection_signal
                                gate_block = find_gate_block(sel_results)
                                if gate_block is not None:
                                    block = gate_block.to_dict()
                                    _write_gate_block(sel_eval_dir, block)
                                    cand_hard = None
                                    cand_soft = None
                                    cand_gate_score = None
                                    step_rec["selection_hard"] = None
                                    step_rec["selection_soft"] = None
                                    step_rec["gate_metric"] = gate_metric
                                    step_rec["candidate_gate_score"] = None
                                    step_rec["action"] = "reject"
                                    step_rec["no_candidate_reason"] = "gate_rejected_best_origin_initial_skill"
                                    step_rec["no_candidate_triggers"] = _dedupe_texts(
                                        [
                                            "gate_rejected_best_origin_initial_skill",
                                            f"gate_retry_blocked:{block['blocker']}",
                                        ]
                                    )
                                    step_rec["gate_status"] = "blocked"
                                    step_rec["gate_blocker"] = block["blocker"]
                                    step_rec["gate_block"] = block
                                    gate_retry_record["action"] = "stop"
                                    gate_retry_record["stop_reason"] = f"gate_retry_blocked:{block['blocker']}"
                                    gate_retry_record["gate_block"] = block
                                    with open(
                                        gate_retry_record_path,
                                        "w",
                                    ) as f:
                                        json.dump(gate_retry_record, f, indent=2, ensure_ascii=False)
                                    break
                                cand_hard, cand_soft = compute_score(sel_results)
                                sel_cache[cand_hash] = (cand_hard, cand_soft)

                            step_rec["selection_hard"] = cand_hard
                            step_rec["selection_soft"] = cand_soft
                            if selection_candidate_context:
                                step_rec["selection_eval_context"] = selection_candidate_context
                            gate = evaluate_gate(
                                candidate_skill=candidate_skill,
                                cand_hard=cand_hard,
                                current_skill=current_skill,
                                current_score=current_score,
                                best_skill=best_skill,
                                best_score=best_score,
                                best_step=best_step,
                                global_step=global_step,
                                cand_soft=cand_soft,
                                metric=gate_metric,
                                mixed_weight=gate_mixed_weight,
                            )
                            cand_gate_score = select_gate_score(
                                cand_hard,
                                cand_soft,
                                gate_metric,
                                gate_mixed_weight,
                            )
                            step_rec["gate_metric"] = gate_metric
                            step_rec["candidate_gate_score"] = cand_gate_score
                            step_rec["action"] = gate.action
                            prev_current = current_score
                            prev_best = best_score
                            current_skill = gate.current_skill
                            current_score = gate.current_score
                            best_skill = gate.best_skill
                            best_score = gate.best_score
                            best_step = gate.best_step
                            if gate.action in {"accept", "accept_new_best"}:
                                current_origin = f"step_{global_step:04d}"
                            if gate.action == "accept_new_best":
                                best_origin = current_origin
                                selection_scores_by_origin[best_origin] = (cand_hard, cand_soft)
                            step_rec["timing"]["evaluate_s"] = round(
                                step_rec["timing"].get("evaluate_s", 0) + time.time() - t_retry_phase,
                                1,
                            )
                            if gate.action == "accept_new_best":
                                print(
                                    f"    [6/6 EVALUATE] GATE RETRY ACCEPT (new best) "
                                    f"{cand_gate_score:.4f} > prev best {prev_best:.4f}"
                                )
                                break
                            if gate.action == "reject":
                                step_rec["gate_rejection"] = _selection_reject_gate_rejection(
                                    history=[step_rec],
                                    baseline_scores=baseline_scores,
                                    gate_metric=gate_metric,
                                    gate_mixed_weight=gate_mixed_weight,
                                    retry_used=retry_count + 1,
                                    retry_budget=active_retry_budget,
                                    rejection_signal=selection_rejection_signal,
                                    baseline_context=sel_eval_context_cache.get(skill_hash(current_skill))
                                    or sel_eval_context_cache.get(skill_hash(skill_init)),
                                    candidate_context=selection_candidate_context,
                                    human_feedback_context=_ranked_feedback_context_packet(dataloader),
                                )
                                print(
                                    f"    [6/6 EVALUATE] GATE RETRY REJECT "
                                    f"{cand_gate_score:.4f} <= current={current_score:.4f}"
                                )
                                continue
                            if gate.action == "accept":
                                print(
                                    f"    [6/6 EVALUATE] GATE RETRY ACCEPT "
                                    f"{cand_gate_score:.4f} > current={prev_current:.4f}"
                                )
                                break
                else:
                    cand_gate_score = None

                step_rec["timing"]["evaluate_s"] = round(time.time() - t_phase, 1)

                # ── Step buffer: unified failure patterns + rejected edits ─
                action = step_rec.get("action", "unknown")
                n_total = len(all_rollout_results) or 1
                n_fail = sum(1 for r in all_rollout_results if _is_failed_rollout_result(r))
                failure_patterns = _extract_failure_patterns(
                    all_rollout_results, step_dir,
                )

                buf_entry: dict = {
                    "step": global_step,
                    "action": action,
                    "n_total": n_total,
                    "n_fail": n_fail,
                    "failure_patterns": failure_patterns,
                }
                if step_rec.get("gate_block"):
                    buf_entry["gate_block"] = step_rec["gate_block"]

                # Attach rejected edits when the step was rejected
                if "reject" in action and ranked_patch:
                    rejected_edits = [
                        short_item_summary(item, update_mode)
                        for item in ranked_items
                        if isinstance(item, dict)
                    ]
                    buf_entry["score_before"] = current_score
                    buf_entry["score_after"] = cand_gate_score
                    buf_entry["rejected_edits"] = rejected_edits

                step_buffer.append(buf_entry)

                # Persist step digest for step buffer context
                digest_path = os.path.join(step_dir, "trajectory_digest.json")
                with open(digest_path, "w") as f:
                    json.dump(buf_entry, f, indent=2, ensure_ascii=False)

                # ── Token snapshot ───────────────────────────────────────
                tokens_after = get_token_summary()
                step_tokens: dict = {}
                for stage in tokens_after:
                    if stage == "_total":
                        continue
                    after = tokens_after[stage]
                    before = tokens_before.get(stage, {})
                    step_tokens[stage] = {
                        "calls": after.get("calls", 0) - before.get("calls", 0),
                        "prompt_tokens": after.get("prompt_tokens", 0)
                        - before.get("prompt_tokens", 0),
                        "completion_tokens": after.get("completion_tokens", 0)
                        - before.get("completion_tokens", 0),
                    }
                step_rec["tokens"] = step_tokens

                # ── Save state ───────────────────────────────────────────
                step_rec["current_score"] = current_score
                step_rec["best_score"] = best_score
                step_rec["best_step"] = best_step
                step_rec["current_origin"] = current_origin
                step_rec["best_origin"] = best_origin
                step_rec["skill_len"] = len(current_skill)
                step_rec["wall_time_s"] = round(time.time() - step_t0, 1)

                _save_skill(out_root, global_step, current_skill)
                with open(os.path.join(out_root, "best_skill.md"), "w") as f:
                    f.write(best_skill)
                history.append(step_rec)
                _save_history(out_root, history)
                _persist_runtime_state(global_step)
                with open(os.path.join(step_dir, "step_record.json"), "w") as f:
                    json.dump(step_rec, f, indent=2, ensure_ascii=False)

                timing = step_rec["timing"]
                print(
                    f"\n  [STEP {global_step} done] "
                    f"epoch={epoch} action={step_rec['action']} "
                    f"current={current_score:.4f} best={best_score:.4f} "
                    f"dt={step_rec['wall_time_s']}s\n"
                    f"    timing: rollout={timing.get('rollout_s',0)}s "
                    f"reflect={timing.get('reflect_s',0)}s "
                    f"aggregate={timing.get('aggregate_s',0)}s "
                    f"select={timing.get('select_s',0)}s "
                    f"evaluate={timing.get('evaluate_s',0)}s"
                )

            epoch_last_step_skill = current_skill
            epoch_comparison_pairs: list[dict] | None = None

            # ── SLOW UPDATE (end of epoch) ──────────────────────────────
            use_slow = cfg.get("use_slow_update", False)
            if use_slow:
                slow_dir = os.path.join(out_root, "slow_update", f"epoch_{epoch:02d}")
                slow_done_path = os.path.join(slow_dir, "slow_result.json")

                if os.path.exists(slow_done_path):
                    # Resume support
                    print(
                        f"\n  [SLOW UPDATE epoch {epoch}] "
                        f"resumed — already done"
                    )
                    with open(slow_done_path) as f:
                        slow_saved = json.load(f)
                    if isinstance(slow_saved.get("gate_block"), dict):
                        run_gate_blocks.append(slow_saved["gate_block"])
                    if slow_saved.get("selection_hard") is not None:
                        selection_scores_by_origin[f"slow_update_epoch_{epoch:02d}"] = (
                            slow_saved.get("selection_hard"),
                            slow_saved.get("selection_soft"),
                        )
                    comparison_path = os.path.join(slow_dir, "comparison_pairs.json")
                    if os.path.exists(comparison_path):
                        try:
                            with open(comparison_path) as f:
                                epoch_comparison_pairs = json.load(f)
                        except Exception:
                            epoch_comparison_pairs = None
                    if (
                        slow_saved.get("slow_update_content")
                        and epoch >= 2
                    ):
                        action = slow_saved.get("action")
                        if slow_gate_with_selection:
                            # Gated mode (follow SkillReflection): re-apply the
                            # guidance to current_skill only when it was accepted.
                            if action in {"accept", "accept_new_best"}:
                                current_skill = replace_slow_update_field(
                                    current_skill,
                                    slow_saved["slow_update_content"],
                                )
                        elif action in {
                            "accept", "accept_new_best", "force_accept",
                        }:
                            # Force-accept mode: re-apply to both current & best.
                            current_skill = replace_slow_update_field(
                                current_skill, slow_saved["slow_update_content"],
                            )
                            best_skill = replace_slow_update_field(
                                best_skill, slow_saved["slow_update_content"],
                            )
                elif epoch == 1:
                    # Epoch 1: inject empty placeholder
                    os.makedirs(slow_dir, exist_ok=True)
                    current_skill = inject_empty_slow_update_field(current_skill)
                    current_origin = f"slow_update_placeholder_epoch_{epoch:02d}"
                    _save_skill(out_root, global_step, current_skill)
                    with open(os.path.join(out_root, "best_skill.md"), "w") as f:
                        f.write(best_skill if best_score > current_score else current_skill)
                    with open(slow_done_path, "w") as f:
                        json.dump({"action": "inject_placeholder", "epoch": epoch}, f, indent=2)
                    _persist_runtime_state(global_step)
                    print(
                        f"\n  [SLOW UPDATE epoch {epoch}] "
                        f"injected empty placeholder"
                    )
                else:
                    # Epoch 2+: longitudinal comparison
                    os.makedirs(slow_dir, exist_ok=True)
                    print(
                        f"\n  {'='*60}\n"
                        f"  SLOW UPDATE — Epoch {epoch} "
                        f"(comparing epoch {epoch-1} vs {epoch})\n"
                        f"  {'='*60}"
                    )

                    # 1. Get skill from last step of previous epoch
                    prev_epoch_records = [
                        h for h in history if h.get("epoch") == epoch - 1
                    ]
                    prev_epoch_last_step = prev_epoch_records[-1]["step"]
                    prev_skill = _load_skill(out_root, prev_epoch_last_step)

                    # 2. Sample items from train set
                    slow_n = cfg.get("slow_update_samples", 20)
                    slow_seed = seed + epoch * 2000
                    if dataloader is not None:
                        slow_batch = dataloader.build_train_batch(
                            batch_size=slow_n,
                            seed=slow_seed,
                            out_root=out_root,
                        )
                        slow_env = adapter.build_env_from_batch(
                            slow_batch, out_root=out_root,
                        )
                    else:
                        slow_env = adapter.build_train_env(
                            batch_size=slow_n,
                            seed=slow_seed,
                            out_root=out_root,
                        )
                    slow_items = list(slow_env) if hasattr(slow_env, "__iter__") else slow_env
                    print(f"    [slow update] sampled {len(slow_items)} train items (seed={slow_seed})")

                    # 3. Rollout with both skills
                    t_slow = time.time()
                    prev_rollout_dir = os.path.join(slow_dir, "rollout_prev")
                    curr_rollout_dir = os.path.join(slow_dir, "rollout_curr")
                    results_prev = adapter.rollout(slow_env, prev_skill, prev_rollout_dir)
                    results_curr = adapter.rollout(slow_env, current_skill, curr_rollout_dir)

                    prev_hard, _ = compute_score(results_prev)
                    curr_hard, _ = compute_score(results_curr)
                    print(
                        f"    [slow update] prev epoch hard={prev_hard:.4f}  "
                        f"curr epoch hard={curr_hard:.4f}"
                    )

                    # 4. Build and save structured comparison pairs
                    comparison_pairs, all_comparison_pairs = _build_longitudinal_pairs(
                        adapter=adapter,
                        dataloader=dataloader,
                        prev_skill=prev_skill,
                        curr_skill=current_skill,
                        initial_items=slow_items,
                        initial_prev_results=results_prev,
                        initial_curr_results=results_curr,
                        prev_rollout_dir=prev_rollout_dir,
                        curr_rollout_dir=curr_rollout_dir,
                        policy=longitudinal_pair_policy,
                        target_n=slow_n,
                        seed=slow_seed,
                        out_root=out_root,
                    )
                    epoch_comparison_pairs = comparison_pairs
                    if all_comparison_pairs is not comparison_pairs:
                        save_comparison_pairs(
                            all_comparison_pairs,
                            os.path.join(slow_dir, "comparison_pairs_all.json"),
                        )
                    save_comparison_pairs(
                        comparison_pairs,
                        os.path.join(slow_dir, "comparison_pairs.json"),
                    )
                    n_regressed = sum(1 for p in comparison_pairs if p["category"] == "regressed")
                    n_improved = sum(1 for p in comparison_pairs if p["category"] == "improved")
                    n_persist = sum(1 for p in comparison_pairs if p["category"] == "persistent_fail")
                    n_stable = sum(1 for p in comparison_pairs if p["category"] == "stable_success")
                    print(
                        f"    [slow update] comparison: "
                        f"regressed={n_regressed} improved={n_improved} "
                        f"persistent_fail={n_persist} stable_success={n_stable} "
                        f"policy={longitudinal_pair_policy} "
                        f"kept={len(comparison_pairs)}/{len(all_comparison_pairs)}"
                    )

                    # 5. Extract previous slow update guidance for reflection
                    existing_guidance = extract_slow_update_field(current_skill)

                    # 6. Optimizer analysis (with reflection on previous guidance)
                    slow_result = run_slow_update(
                        current_skill,
                        results_prev,
                        results_curr,
                        slow_items,
                        prev_skill=prev_skill,
                        prev_slow_update_content=existing_guidance,
                        prev_rollout_dir=prev_rollout_dir,
                        curr_rollout_dir=curr_rollout_dir,
                        comparison_pairs=comparison_pairs,
                    )
                    slow_time = round(time.time() - t_slow, 1)

                    if slow_result and slow_result.get("slow_update_content"):
                        slow_candidate = replace_slow_update_field(
                            current_skill, slow_result["slow_update_content"],
                        )
                        slow_candidate_hash = skill_hash(slow_candidate)
                        with open(os.path.join(slow_dir, "candidate_skill.md"), "w") as f:
                            f.write(slow_candidate)
                        slow_result["time_s"] = slow_time
                        slow_result["prev_hard"] = prev_hard
                        slow_result["curr_hard"] = curr_hard
                        slow_result["candidate_hash"] = slow_candidate_hash
                        slow_result["update_origin"] = "slow_update_momentum"
                        slow_result["update_target"] = (
                            "Address longitudinal regressions and persistent failures "
                            "observed across adjacent epochs."
                        )

                        # Slow update acceptance — two modes selected via
                        # `optimizer.slow_update_gate_with_selection`.
                        if slow_gate_with_selection:
                            # ── Gated mode (follow SkillReflection) ──────────
                            # Evaluate the slow-update candidate on the
                            # selection set and accept/reject via the same
                            # validation gate used for step-level updates.
                            if slow_candidate_hash in sel_cache:
                                slow_sel_hard, slow_sel_soft = sel_cache[
                                    slow_candidate_hash
                                ]
                                print(
                                    f"    [slow gate] cache hit: "
                                    f"hard={slow_sel_hard:.4f}"
                                )
                            else:
                                sel_env, sel_n = _build_eval_env(
                                    split="valid_seen",
                                    env_num=cfg["sel_env_num"],
                                    seed=seed,
                                )
                                print(f"    [slow gate] selection items={sel_n}")
                                slow_eval_dir = os.path.join(
                                    slow_dir, "selection_eval",
                                )
                                slow_eval_results = adapter.rollout(
                                    sel_env, slow_candidate, slow_eval_dir,
                                )
                                slow_block = find_gate_block(slow_eval_results)
                                if slow_block is not None:
                                    block = slow_block.to_dict()
                                    _write_gate_block(slow_eval_dir, block)
                                    slow_result["selection_hard"] = None
                                    slow_result["selection_soft"] = None
                                    slow_result["action"] = f"blocked:{block['blocker']}"
                                    slow_result["gate_status"] = "blocked"
                                    slow_result["gate_blocker"] = block["blocker"]
                                    slow_result["gate_block"] = block
                                    run_gate_blocks.append(block)
                                    print(
                                        f"    [slow gate] BLOCKED "
                                        f"blocked:{block['blocker']} items={len(block.get('items', []))}"
                                    )
                                else:
                                    slow_sel_hard, slow_sel_soft = compute_score(
                                        slow_eval_results
                                    )
                                    sel_cache[slow_candidate_hash] = (
                                        slow_sel_hard, slow_sel_soft,
                                    )

                            if slow_result.get("gate_status") != "blocked":
                                slow_gate = evaluate_gate(
                                    candidate_skill=slow_candidate,
                                    cand_hard=slow_sel_hard,
                                    current_skill=current_skill,
                                    current_score=current_score,
                                    best_skill=best_skill,
                                    best_score=best_score,
                                    best_step=best_step,
                                    global_step=global_step,
                                    cand_soft=slow_sel_soft,
                                    metric=gate_metric,
                                    mixed_weight=gate_mixed_weight,
                                )
                                slow_result["selection_hard"] = slow_sel_hard
                                slow_result["selection_soft"] = slow_sel_soft
                                slow_result["action"] = slow_gate.action
                                prev_current = current_score
                                prev_best = best_score
                                current_skill = slow_gate.current_skill
                                current_score = slow_gate.current_score
                                best_skill = slow_gate.best_skill
                                best_score = slow_gate.best_score
                                best_step = slow_gate.best_step
                                if slow_gate.action in {"accept", "accept_new_best"}:
                                    current_origin = (
                                        f"slow_update_epoch_{epoch:02d}"
                                    )
                                if slow_gate.action == "accept_new_best":
                                    best_origin = current_origin
                                    selection_scores_by_origin[best_origin] = (
                                        slow_sel_hard,
                                        slow_sel_soft,
                                    )
                                    print(
                                        f"    [slow gate] ACCEPT (new best) "
                                        f"hard={slow_sel_hard:.4f} > "
                                        f"prev best {prev_best:.4f}"
                                    )
                                elif slow_gate.action == "accept":
                                    print(
                                        f"    [slow gate] ACCEPT "
                                        f"hard={slow_sel_hard:.4f} > "
                                        f"current={prev_current:.4f}"
                                    )
                                else:
                                    print(
                                        f"    [slow gate] REJECT "
                                        f"hard={slow_sel_hard:.4f} <= "
                                        f"current={current_score:.4f}"
                                    )
                            else:
                                pass
                            print(
                                f"    [slow update] guidance written "
                                f"({len(slow_result['slow_update_content'])} "
                                f"chars), {slow_time}s"
                            )
                        else:
                            # ── Force-accept mode (default) ──────────────────
                            # The epoch-level longitudinal guidance is injected
                            # into both current_skill and best_skill
                            # unconditionally — it must not be gated by
                            # step-level selection scores.
                            slow_content = slow_result["slow_update_content"]
                            current_skill = replace_slow_update_field(
                                current_skill, slow_content,
                            )
                            best_skill = replace_slow_update_field(
                                best_skill, slow_content,
                            )
                            # Update caches so downstream steps use the
                            # slow-update-injected skill for hashing.
                            slow_candidate_hash = skill_hash(current_skill)
                            sel_cache[slow_candidate_hash] = (current_score, 0.0)

                            slow_result["action"] = "force_accept"
                            current_origin = f"slow_update_epoch_{epoch:02d}"

                            print(
                                f"    [slow update] force-injected into "
                                f"current & best "
                                f"({len(slow_content)} chars), "
                                f"{slow_time}s"
                            )
                    else:
                        slow_result = slow_result or {}
                        slow_result["action"] = "no_content"
                        slow_result["time_s"] = slow_time
                        print(
                            f"    [slow update] no guidance produced, "
                            f"{slow_time}s"
                        )

                    # 5. Save
                    with open(slow_done_path, "w") as f:
                        json.dump(slow_result, f, indent=2, ensure_ascii=False)
                    _save_skill(out_root, global_step, current_skill)
                    with open(os.path.join(out_root, "best_skill.md"), "w") as f:
                        f.write(best_skill)
                    _persist_runtime_state(global_step)

                    print(
                        f"\n  [SLOW UPDATE epoch {epoch} done] "
                        f"current={current_score:.4f} best={best_score:.4f}"
                    )

            # ── META SKILL (end of epoch, optimizer-side memory) ─────────
            use_meta_skill = cfg.get("use_meta_skill", False)
            if use_meta_skill:
                meta_skill_dir = os.path.join(out_root, "meta_skill", f"epoch_{epoch:02d}")
                meta_skill_done_path = os.path.join(meta_skill_dir, "meta_skill_result.json")
                os.makedirs(meta_skill_dir, exist_ok=True)

                if os.path.exists(meta_skill_done_path):
                    print(f"\n  [META SKILL epoch {epoch}] resumed — already done")
                elif epoch == 1:
                    with open(meta_skill_done_path, "w") as f:
                        json.dump(
                            {"action": "skip_first_epoch", "epoch": epoch},
                            f, indent=2, ensure_ascii=False,
                        )
                    print(f"\n  [META SKILL epoch {epoch}] skipped — first epoch")
                else:
                    print(
                        f"\n  {'='*60}\n"
                        f"  META SKILL — Epoch {epoch} "
                        f"(optimizer memory from epoch {epoch-1} vs {epoch})\n"
                        f"  {'='*60}"
                    )

                    prev_epoch_records = [h for h in history if h.get("epoch") == epoch - 1]
                    prev_epoch_last_step = prev_epoch_records[-1]["step"]
                    prev_skill = _load_skill(out_root, prev_epoch_last_step)
                    prev_meta_skill = _load_meta_skill_content(out_root, epoch - 1)

                    if epoch_comparison_pairs is None:
                        meta_n = cfg.get("slow_update_samples", 20)
                        meta_seed = seed + epoch * 2000
                        if dataloader is not None:
                            meta_batch = dataloader.build_train_batch(
                                batch_size=meta_n,
                                seed=meta_seed,
                                out_root=out_root,
                            )
                            meta_env = adapter.build_env_from_batch(
                                meta_batch, out_root=out_root,
                            )
                        else:
                            meta_env = adapter.build_train_env(
                                batch_size=meta_n,
                                seed=meta_seed,
                                out_root=out_root,
                            )
                        meta_items = list(meta_env) if hasattr(meta_env, "__iter__") else meta_env
                        prev_rollout_dir = os.path.join(meta_skill_dir, "rollout_prev")
                        curr_rollout_dir = os.path.join(meta_skill_dir, "rollout_curr")
                        results_prev = adapter.rollout(meta_env, prev_skill, prev_rollout_dir)
                        results_curr = adapter.rollout(meta_env, epoch_last_step_skill, curr_rollout_dir)
                        epoch_comparison_pairs, all_meta_comparison_pairs = _build_longitudinal_pairs(
                            adapter=adapter,
                            dataloader=dataloader,
                            prev_skill=prev_skill,
                            curr_skill=epoch_last_step_skill,
                            initial_items=meta_items,
                            initial_prev_results=results_prev,
                            initial_curr_results=results_curr,
                            prev_rollout_dir=prev_rollout_dir,
                            curr_rollout_dir=curr_rollout_dir,
                            policy=longitudinal_pair_policy,
                            target_n=meta_n,
                            seed=meta_seed,
                            out_root=out_root,
                        )
                        if all_meta_comparison_pairs is not epoch_comparison_pairs:
                            save_comparison_pairs(
                                all_meta_comparison_pairs,
                                os.path.join(meta_skill_dir, "comparison_pairs_all.json"),
                            )
                        save_comparison_pairs(
                            epoch_comparison_pairs,
                            os.path.join(meta_skill_dir, "comparison_pairs.json"),
                        )
                        meta_counts = _pair_category_counts(epoch_comparison_pairs)
                        print(
                            f"    [meta skill] comparison: "
                            f"regressed={meta_counts.get('regressed', 0)} "
                            f"improved={meta_counts.get('improved', 0)} "
                            f"persistent_fail={meta_counts.get('persistent_fail', 0)} "
                            f"stable_success={meta_counts.get('stable_success', 0)} "
                            f"policy={longitudinal_pair_policy} "
                            f"kept={len(epoch_comparison_pairs)}/{len(all_meta_comparison_pairs)}"
                        )

                    t_meta_skill = time.time()
                    meta_skill_result = run_meta_skill(
                        prev_skill=prev_skill,
                        curr_skill=epoch_last_step_skill,
                        comparison_pairs=epoch_comparison_pairs or [],
                        prev_meta_skill_content=prev_meta_skill,
                    )
                    meta_skill_time = round(time.time() - t_meta_skill, 1)

                    if meta_skill_result and meta_skill_result.get("meta_skill_content"):
                        meta_skill_result["time_s"] = meta_skill_time
                        meta_skill_result["action"] = "write_meta_skill"
                        print(
                            f"    [meta skill] memory written "
                            f"({len(meta_skill_result['meta_skill_content'])} chars), "
                            f"{meta_skill_time}s"
                        )
                    else:
                        meta_skill_result = meta_skill_result or {}
                        meta_skill_result["time_s"] = meta_skill_time
                        meta_skill_result["action"] = "no_content"
                        print(f"    [meta skill] no memory produced, {meta_skill_time}s")

                    with open(meta_skill_done_path, "w") as f:
                        json.dump(meta_skill_result, f, indent=2, ensure_ascii=False)

        # ── Save best skill ──────────────────────────────────────────────
        with open(os.path.join(out_root, "best_skill.md"), "w") as f:
            f.write(best_skill)
        _persist_runtime_state(global_step)
        print(
            f"\n  [done] best skill from step {best_step}, "
            f"score={best_score:.4f}"
        )

        baseline_selection_scores = sel_cache.get(skill_hash(skill_init), (None, None))
        selection_reject_gate_rejection = None
        skip_final_test_reason = ""
        if _should_skip_final_test_after_selection_reject(
            history=history,
            best_origin=best_origin,
            best_skill=best_skill,
            skill_init=skill_init,
        ):
            selection_reject_gate_rejection = next(
                (
                    record.get("gate_rejection")
                    for record in reversed(history)
                    if isinstance(record.get("gate_rejection"), dict)
                ),
                None,
            )
            if selection_reject_gate_rejection is None:
                selection_reject_gate_rejection = _selection_reject_gate_rejection(
                    history=history,
                    baseline_scores=baseline_selection_scores,
                    gate_metric=gate_metric,
                    gate_mixed_weight=gate_mixed_weight,
                    retry_used=sum(
                        1
                        for record in history
                        for attempt in record.get("gate_reject_retry_attempts", [])
                        if isinstance(attempt, dict) and attempt.get("action") == "retry"
                    ),
                    retry_budget=max(0, int(cfg.get("gate_reject_retry_budget", 3) or 0)),
                    human_feedback_context=_ranked_feedback_context_packet(dataloader),
                )
            if selection_reject_gate_rejection is not None:
                skip_final_test_reason = "selection_gate_rejected_candidate"

        # ── Final test evaluation (valid_unseen) ─────────────────────────
        baseline_test_hard = None
        baseline_test_soft = None
        test_hard = None
        test_soft = None

        if cfg["eval_test"] and skip_final_test_reason:
            print(
                "\n  [skip final test] selection gate rejected the candidate; "
                "final test eval is skipped by default."
            )
        elif cfg["eval_test"]:
            task_types = adapter.get_task_types()

            # Baseline: S_0 on test set (valid_unseen)
            print(f"\n{'='*60}")
            print("  BASELINE TEST — evaluate initial skill on Test set (valid_unseen)")
            print(f"{'='*60}")
            test_env, test_n = _build_eval_env(
                split="valid_unseen",
                env_num=cfg["test_env_num"],
                seed=seed,
            )
            print(f"  Test items: {test_n}")
            baseline_test_dir = os.path.join(out_root, "test_eval_baseline")
            baseline_test_results = adapter.rollout(test_env, skill_init, baseline_test_dir)
            baseline_test_hard, baseline_test_soft = compute_score(baseline_test_results)
            baseline_buckets = _compute_task_type_buckets(baseline_test_results, task_types)
            print("\n  === Baseline Test Results (S_0) ===")
            for task_type in task_types + ["overall"]:
                b = baseline_buckets.get(task_type, {"total": 0, "hard": 0})
                t = max(b["total"], 1)
                print(
                    f"    {task_type:<40s}: "
                    f"hard={b['hard']}/{b['total']}={b['hard']/t:.4f}"
                )
            with open(os.path.join(baseline_test_dir, "summary.json"), "w") as f:
                json.dump(
                    {
                        k: {
                            "total": b["total"],
                            "hard_acc": b["hard"] / max(b["total"], 1),
                        }
                        for k, b in baseline_buckets.items()
                    },
                    f, indent=2, ensure_ascii=False,
                )

            # Best skill on test set
            print(f"\n{'='*60}")
            print("  BEST SKILL TEST — evaluate best skill on Test set (valid_unseen)")
            print(f"{'='*60}")
            test_env2, test_n2 = _build_eval_env(
                split="valid_unseen",
                env_num=cfg["test_env_num"],
                seed=seed,
            )
            print(f"  Test items: {test_n2}")
            test_dir = os.path.join(out_root, "test_eval")
            test_results = adapter.rollout(test_env2, best_skill, test_dir)
            test_hard, test_soft = compute_score(test_results)
            best_buckets = _compute_task_type_buckets(test_results, task_types)
            print("\n  === Best Skill Test Results ===")
            for task_type in task_types + ["overall"]:
                b = best_buckets.get(task_type, {"total": 0, "hard": 0})
                t = max(b["total"], 1)
                print(
                    f"    {task_type:<40s}: "
                    f"hard={b['hard']}/{b['total']}={b['hard']/t:.4f}"
                )
            with open(os.path.join(test_dir, "summary.json"), "w") as f:
                json.dump(
                    {
                        k: {
                            "total": b["total"],
                            "hard_acc": b["hard"] / max(b["total"], 1),
                        }
                        for k, b in best_buckets.items()
                    },
                    f, indent=2, ensure_ascii=False,
                )

            # Comparison
            delta_hard = (test_hard or 0) - (baseline_test_hard or 0)
            print("\n  === Improvement (best vs baseline) ===")
            print(
                f"    hard: {baseline_test_hard:.4f} -> {test_hard:.4f}  "
                f"(delta={delta_hard:+.4f})"
            )

        # ── Global summary ───────────────────────────────────────────────
        total_wall = time.time() - t_loop_start
        n_accept = sum(1 for h in history if "accept" in h.get("action", ""))
        n_reject = sum(1 for h in history if h.get("action") == "reject")
        noop_retry_attempts = [
            attempt
            for h in history
            for attempt in h.get("noop_retry_attempts", [])
            if isinstance(attempt, dict)
        ]
        gate_reject_retry_attempts = [
            attempt
            for h in history
            for attempt in h.get("gate_reject_retry_attempts", [])
            if isinstance(attempt, dict)
        ]
        wrong_artifact_retry_attempts = [
            attempt
            for h in history
            for attempt in h.get("wrong_artifact_retry_attempts", [])
            if isinstance(attempt, dict)
        ]
        final_has_candidate = n_accept > 0 or best_origin != "initial_skill"
        no_candidate_triggers = [] if final_has_candidate else _dedupe_texts([
            trigger
            for h in history
            for trigger in h.get("no_candidate_triggers", [])
            if isinstance(trigger, str)
        ])
        no_candidate_failure = next(
            (h.get("failure") for h in history if isinstance(h.get("failure"), dict)),
            {},
        ) if not final_has_candidate else {}
        no_candidate_optimizer_hint = next(
            (str(h.get("optimizer_hint") or "").strip() for h in history if str(h.get("optimizer_hint") or "").strip()),
            "",
        ) if not final_has_candidate else ""
        no_candidate_feedback_retry_hints = next(
            (h.get("feedback_retry_hints") for h in history if isinstance(h.get("feedback_retry_hints"), dict)),
            {},
        ) if not final_has_candidate else {}
        gate_blockers = [
            h["gate_block"]
            for h in history
            if isinstance(h.get("gate_block"), dict)
        ] + run_gate_blocks
        n_block = len(gate_blockers)
        n_skip = sum(1 for h in history if _is_skipped_step(h.get("action", "")))

        token_summary = get_token_summary()

        # Epoch-level statistics
        epoch_stats = []
        for e in range(1, num_epochs + 1):
            epoch_records = [h for h in history if h.get("epoch") == e]
            if epoch_records:
                epoch_stats.append({
                    "epoch": e,
                    "steps": [h["step"] for h in epoch_records],
                    "accepts": sum(1 for h in epoch_records if "accept" in h.get("action", "")),
                    "rejects": sum(1 for h in epoch_records if h.get("action") == "reject"),
                    "skips": sum(1 for h in epoch_records if _is_skipped_step(h.get("action", ""))),
                    "best_score_at_epoch_end": epoch_records[-1].get("best_score", 0.0),
                    "current_score_at_epoch_end": epoch_records[-1].get("current_score", 0.0),
                })

        best_selection_hard, best_selection_soft = _best_selection_scores(
            history=history,
            best_step=best_step,
            best_origin=best_origin,
            baseline_scores=baseline_selection_scores,
            selection_scores_by_origin=selection_scores_by_origin,
            gate_metric=gate_metric,
            best_score=best_score,
        )
        summary = {
            "version": "skillopt-0.1.0",
            "config": _redact_cfg(cfg),
            "gate_status": "blocked" if gate_blockers else "passed",
            "gate_blocker": gate_blockers[0]["blocker"] if gate_blockers else "",
            "gate_blockers": gate_blockers,
            "promotable": not gate_blockers,
            "baseline_selection_hard": baseline_selection_scores[0],
            "baseline_selection_soft": baseline_selection_scores[1],
            "best_selection_hard": best_selection_hard,
            "best_selection_soft": best_selection_soft,
            "best_step": best_step,
            "current_origin": current_origin,
            "best_origin": best_origin,
            "total_steps": len(history),
            "total_accepts": n_accept,
            "total_rejects": n_reject,
            "total_blocks": n_block,
            "total_skips": n_skip,
            "noop_retry_attempts": noop_retry_attempts,
            "gate_reject_retry_attempts": gate_reject_retry_attempts,
            "wrong_artifact_retry_attempts": wrong_artifact_retry_attempts,
            "no_candidate_triggers": no_candidate_triggers,
            "no_candidate_reason": no_candidate_triggers[0] if no_candidate_triggers else "",
            "failure": no_candidate_failure,
            "optimizer_hint": no_candidate_optimizer_hint,
            "feedback_retry_hints": no_candidate_feedback_retry_hints,
            "gate_rejection": selection_reject_gate_rejection,
            "final_test_skipped_reason": skip_final_test_reason,
            "epoch_stats": epoch_stats,
            "baseline_test_hard": baseline_test_hard,
            "baseline_test_soft": baseline_test_soft,
            "test_hard": test_hard,
            "test_soft": test_soft,
            "test_delta_hard": (
                (test_hard or 0) - (baseline_test_hard or 0)
                if test_hard is not None
                else None
            ),
            "total_wall_time_s": round(total_wall, 1),
            "token_summary": token_summary,
        }
        with open(os.path.join(out_root, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*60}")
        print("  Final Summary")
        print(f"{'='*60}")
        print(
            f"  steps={len(history)} accept={n_accept} "
            f"reject={n_reject} skip={n_skip}"
        )
        print(f"  best_score={best_score:.4f} (step {best_step})  wall={total_wall:.0f}s")
        if epoch_stats:
            for es in epoch_stats:
                print(
                    f"    epoch {es['epoch']}: accept={es['accepts']} reject={es['rejects']} "
                    f"best={es['best_score_at_epoch_end']:.4f}"
                )
        if test_hard is not None:
            print(f"  test_hard={test_hard:.4f} test_soft={test_soft:.4f}")
        if token_summary.get("_total"):
            t = token_summary["_total"]
            print(
                f"  total tokens: {t['total_tokens']:,} "
                f"(prompt={t['prompt_tokens']:,} "
                f"completion={t['completion_tokens']:,} "
                f"calls={t['calls']})"
            )

        return summary
