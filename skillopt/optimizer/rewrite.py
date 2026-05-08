"""Teacher-driven full skill rewrite from selected revise_suggestions."""
from __future__ import annotations

import json

from skillopt.model import chat_teacher
from skillopt.prompts import load_prompt
from skillopt.optimizer.update_modes import get_payload_items
from skillopt.utils import extract_json


def rewrite_skill_from_suggestions(
    skill_content: str,
    patch: dict,
    *,
    system_prompt: str | None = None,
    step_buffer_context: str = "",
    env: str | None = None,
    reasoning_effort: str | None = "high",
    max_completion_tokens: int = 64000,
) -> dict | None:
    suggestions = get_payload_items(patch, "rewrite_from_suggestions")
    if not suggestions:
        return None

    user = (
        f"## Current Skill\n{skill_content}\n\n"
        f"## Selected Revise Suggestions ({len(suggestions)} total)\n"
        f"{json.dumps(suggestions, ensure_ascii=False, indent=2)}\n\n"
    )
    if step_buffer_context.strip():
        user += f"## Previous Steps in This Epoch\n{step_buffer_context}\n\n"
    user += (
        "Rewrite the full skill document so it integrates the selected suggestions. "
        "Return the complete new skill in `new_skill`."
    )

    actual_system = system_prompt if system_prompt is not None else load_prompt(
        "rewrite_skill", env=env,
    )

    try:
        response, _ = chat_teacher(
            system=actual_system,
            user=user,
            max_completion_tokens=max_completion_tokens,
            retries=3,
            stage="rewrite",
            reasoning_effort=reasoning_effort,
        )
        result = extract_json(response)
        if result and str(result.get("new_skill", "")).strip():
            result["new_skill"] = str(result["new_skill"]).rstrip() + "\n"
            if "change_summary" not in result or not isinstance(result["change_summary"], list):
                result["change_summary"] = []
            return result
    except Exception:  # noqa: BLE001
        return None
    return None
