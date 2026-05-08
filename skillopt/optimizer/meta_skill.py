"""Teacher-side meta skill memory for cross-epoch optimization guidance.

This module maintains a compact teacher-facing memory distilled from
adjacent-epoch skill comparisons. Unlike ``slow_update``, it does not
modify the student skill document. Instead, it produces guidance meant to
improve future teacher behavior when proposing, merging, and ranking edits.
"""
from __future__ import annotations

import traceback

from skillopt.model import chat_teacher
from skillopt.optimizer.slow_update import format_comparison_text
from skillopt.prompts import load_prompt
from skillopt.utils import extract_json


def format_meta_skill_context(meta_skill_content: str) -> str:
    """Render teacher memory into a prompt-ready context block."""
    content = (meta_skill_content or "").strip()
    if not content:
        return ""
    return (
        "## Teacher Meta Skill\n"
        "This is teacher-side memory distilled from prior epoch transitions in "
        "this environment. Use it to improve how you propose, merge, and rank "
        "skill edits. Prefer it when the current evidence is ambiguous, but do "
        "not force it if the current trajectories clearly contradict it.\n\n"
        f"{content}"
    )


def run_meta_skill(
    prev_skill: str,
    curr_skill: str,
    comparison_pairs: list[dict],
    *,
    prev_meta_skill_content: str = "",
    system_prompt: str | None = None,
) -> dict | None:
    """Produce updated teacher-side meta skill from adjacent epochs."""
    actual_system = system_prompt if system_prompt is not None else load_prompt("meta_skill")

    prev_skill_display = prev_skill
    if len(prev_skill_display) > 6000:
        prev_skill_display = prev_skill_display[:6000] + "\n...[truncated]..."

    curr_skill_display = curr_skill
    if len(curr_skill_display) > 6000:
        curr_skill_display = curr_skill_display[:6000] + "\n...[truncated]..."

    prev_meta_section = (
        prev_meta_skill_content.strip()
        if prev_meta_skill_content and prev_meta_skill_content.strip()
        else "(No previous teacher meta skill — this is the first update.)"
    )

    comparison_text = format_comparison_text(comparison_pairs)
    user = (
        f"## Previous Epoch Last-Step Skill\n{prev_skill_display}\n\n"
        f"## Current Epoch Last-Step Skill\n{curr_skill_display}\n\n"
        f"## Previous Teacher Meta Skill\n"
        f"The following teacher memory was available during the current epoch. "
        f"Reflect on whether it improved or harmed the quality of edits.\n\n"
        f"{prev_meta_section}\n\n"
        f"## Longitudinal Comparison (same tasks, two last-step skills)\n"
        f"{comparison_text}"
    )

    try:
        response, _ = chat_teacher(
            system=actual_system,
            user=user,
            max_completion_tokens=3072,
            retries=3,
            stage="meta_skill",
        )
        result = extract_json(response)
        if result and result.get("meta_skill_content"):
            return {
                "reasoning": str(result.get("reasoning", "")).strip(),
                "meta_skill_content": str(result["meta_skill_content"]).strip(),
            }
    except Exception:  # noqa: BLE001
        traceback.print_exc()

    return None
