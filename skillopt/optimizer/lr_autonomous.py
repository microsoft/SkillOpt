"""Optimizer-driven autonomous update-size decisions."""
from __future__ import annotations

import json
import re
from typing import Any

from skillopt.model import chat_optimizer
from skillopt.optimizer.meta_skill import format_meta_skill_context
from skillopt.optimizer.update_modes import describe_item, get_payload_items, payload_label
from skillopt.prompts import load_prompt
from skillopt.utils import extract_json


def _coerce_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float) and value.is_integer():
        return max(0, int(value))
    text = str(value or "").strip()
    if not text:
        return None
    match = re.search(r"-?\d+", text)
    if not match:
        return None
    return max(0, int(match.group(0)))


def decide_autonomous_learning_rate(
    *,
    skill_content: str,
    merged_patch: dict,
    update_mode: str,
    rollout_hard: float,
    rollout_soft: float,
    rollout_n: int,
    step_buffer_context: str = "",
    meta_skill_context: str = "",
) -> dict:
    """Ask the optimizer to choose the number of update items for this step.

    The prompt intentionally avoids default budgets, candidate budget lists, or
    scheduler history. The only hard post-processing is validity: the returned
    integer is clamped to the available item count.
    """
    items = get_payload_items(merged_patch, update_mode)
    available = len(items)
    item_lines = [
        f"[{idx}] {describe_item(item, update_mode)}"
        for idx, item in enumerate(items)
    ]
    user = (
        f"## Current Skill\n{skill_content}\n\n"
        f"## Current Step Evidence\n"
        f"rollout_n={rollout_n}\n"
        f"rollout_hard={rollout_hard:.6f}\n"
        f"rollout_soft={rollout_soft:.6f}\n"
        f"proposed_update_items={available}\n"
        f"update_item_type={payload_label(update_mode)}\n\n"
        f"## Proposed Update Items\n"
        + "\n".join(item_lines)
        + "\n\nDecide how many proposed update items should be applied now."
    )
    if step_buffer_context.strip():
        user += f"\n\n## Previous Steps in This Epoch\n{step_buffer_context}"
    optimizer_ctx = format_meta_skill_context(meta_skill_context)
    if optimizer_ctx:
        user = f"{optimizer_ctx}\n\n{user}"

    response = ""
    parsed: dict | None = None
    decision: int | None = None
    try:
        response, _ = chat_optimizer(
            system=load_prompt("lr_autonomous"),
            user=user,
            max_completion_tokens=16384,
            retries=3,
            stage="lr_autonomous",
        )
        parsed = extract_json(response)
        if parsed:
            decision = _coerce_nonnegative_int(parsed.get("learning_rate"))
    except Exception as exc:  # noqa: BLE001
        parsed = {"error": str(exc)}

    fallback = False
    if decision is None:
        decision = 0
        fallback = True

    chosen = min(decision, available)
    record = {
        "learning_rate": chosen,
        "raw_learning_rate": decision,
        "available_update_items": available,
        "clamped": chosen != decision,
        "fallback": fallback,
        "reasoning": (parsed or {}).get("reasoning", ""),
        "confidence": (parsed or {}).get("confidence", ""),
        "risk_notes": (parsed or {}).get("risk_notes", []),
        "raw_response": response,
    }
    if parsed and "error" in parsed:
        record["error"] = parsed["error"]
    return record
