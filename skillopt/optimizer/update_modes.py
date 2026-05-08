"""Helpers for switching between patch edits and rewrite-from-suggestions."""
from __future__ import annotations

from typing import Any

PATCH_MODE = "patch"
REWRITE_MODE = "rewrite_from_suggestions"
FULL_REWRITE_MINIBATCH_MODE = "full_rewrite_minibatch"


def normalize_update_mode(mode: str | None) -> str:
    raw = str(mode or PATCH_MODE).strip().lower()
    aliases = {
        "patch": PATCH_MODE,
        "edits": PATCH_MODE,
        "rewrite": REWRITE_MODE,
        "rewrite_from_suggestions": REWRITE_MODE,
        "suggestions": REWRITE_MODE,
        "rewrite_suggestions": REWRITE_MODE,
        "full_rewrite": FULL_REWRITE_MINIBATCH_MODE,
        "full_rewrite_minibatch": FULL_REWRITE_MINIBATCH_MODE,
        "minibatch_full_rewrite": FULL_REWRITE_MINIBATCH_MODE,
        "skill_rewrite_minibatch": FULL_REWRITE_MINIBATCH_MODE,
    }
    return aliases.get(raw, PATCH_MODE)


def is_rewrite_mode(mode: str | None) -> bool:
    return normalize_update_mode(mode) == REWRITE_MODE


def is_full_rewrite_minibatch_mode(mode: str | None) -> bool:
    return normalize_update_mode(mode) == FULL_REWRITE_MINIBATCH_MODE


def payload_key(mode: str | None) -> str:
    if is_full_rewrite_minibatch_mode(mode):
        return "skill_candidates"
    return "revise_suggestions" if is_rewrite_mode(mode) else "edits"


def payload_label(mode: str | None, *, singular: bool = False, title: bool = False) -> str:
    if is_full_rewrite_minibatch_mode(mode):
        word = "skill candidate" if singular else "skill candidates"
    elif is_rewrite_mode(mode):
        word = "suggestion" if singular else "suggestions"
    else:
        word = "edit" if singular else "edits"
    return word.title() if title else word


def get_payload_items(container: dict | None, mode: str | None) -> list[dict]:
    if not isinstance(container, dict):
        return []
    items = container.get(payload_key(mode), [])
    return items if isinstance(items, list) else []


def set_payload_items(container: dict, items: list[dict], mode: str | None) -> dict:
    container[payload_key(mode)] = items
    return container


def truncate_payload(container: dict, max_items: int, mode: str | None) -> dict:
    if max_items < 0:
        return container
    items = get_payload_items(container, mode)
    if len(items) > max_items:
        set_payload_items(container, items[:max_items], mode)
    return container


def describe_item(item: dict, mode: str | None, *, max_chars: int = 240) -> str:
    if not isinstance(item, dict):
        return ""
    if is_full_rewrite_minibatch_mode(mode):
        parts = [
            f"title={item.get('title', '')!r}",
            f"change_summary={item.get('change_summary', [])!r}",
        ]
        if item.get("source_type"):
            parts.append(f"source={item.get('source_type')}")
        if item.get("support_count") is not None:
            parts.append(f"support={item.get('support_count')}")
        new_skill = str(item.get("new_skill", "")).strip()
        if new_skill:
            parts.append(f"new_skill_preview={new_skill[:120]!r}")
        text = "  ".join(parts)
    elif is_rewrite_mode(mode):
        parts = [
            f"type={item.get('type', '?')}",
            f"title={item.get('title', '')!r}",
            f"instruction={item.get('instruction', '')!r}",
        ]
        if item.get("priority_hint"):
            parts.append(f"priority={item.get('priority_hint')}")
        if item.get("support_count") is not None:
            parts.append(f"support={item.get('support_count')}")
        text = "  ".join(parts)
    else:
        op = item.get("op", "?")
        target = item.get("target", "")
        content = item.get("content", "")
        parts = [f"op={op}"]
        if target:
            parts.append(f"target={target!r}")
        if content:
            parts.append(f"content={content!r}")
        if item.get("support_count") is not None:
            parts.append(f"support={item.get('support_count')}")
        text = "  ".join(parts)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def short_item_summary(item: dict, mode: str | None, *, max_chars: int = 200) -> dict[str, Any]:
    if is_full_rewrite_minibatch_mode(mode):
        return {
            "title": str(item.get("title", ""))[:max_chars],
            "change_summary": [
                str(x)[:max_chars] for x in item.get("change_summary", [])[:3]
            ] if isinstance(item.get("change_summary"), list) else [],
            "source_type": item.get("source_type", ""),
        }
    if is_rewrite_mode(mode):
        return {
            "type": item.get("type", "?"),
            "title": str(item.get("title", ""))[:max_chars],
            "instruction": str(item.get("instruction", ""))[:max_chars],
        }
    return {
        "op": item.get("op", "?"),
        "content": str(item.get("content", ""))[:max_chars],
        "target": item.get("target", ""),
    }
