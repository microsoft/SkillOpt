"""ReflACT Meta-Reflect — epoch-level skill refinement with momentum.

After each epoch, the meta-reflect stage reviews the epoch's step history
(applied edits + gate scores) and performs high-level skill edits:
merging redundant rules, removing ineffective ones, and distilling
cross-step strategic patterns.

This is analogous to momentum in neural network optimization:
- Fast update (per step): analyst edits fix local issues from current batch
- Slow update (per epoch): meta-reflect refines the skill based on what
  worked and what didn't across the full epoch

The meta-reflect also maintains a ``meta_summary`` — a compact memory
passed between epochs that captures directional insights (which editing
directions are effective, which are not). This is the "momentum buffer".

Public API
----------
- :func:`build_epoch_history`   — format an epoch's step records for meta-reflect
- :func:`run_meta_reflect`      — one teacher call to produce high-level edits + meta_summary
"""
from __future__ import annotations

import json
import os
import traceback

from skillopt.model import chat_teacher
from skillopt.optimizer.update_modes import (
    describe_item,
    get_payload_items,
    normalize_update_mode,
    payload_label,
    truncate_payload,
)
from skillopt.prompts import load_prompt
from skillopt.utils import extract_json


# ── Epoch history formatting ─────────────────────────────────────────────────


def build_epoch_history(
    epoch_step_records: list[dict],
    out_root: str,
    *,
    update_mode: str = "patch",
) -> str:
    """Format an epoch's step records into text for the meta-reflect teacher.

    For each step, includes the exact edits applied (read from
    ``ranked_edits.json``) and the gate evaluation result.

    Parameters
    ----------
    epoch_step_records : list[dict]
        Step record dicts from ``history.json`` belonging to this epoch.
    out_root : str
        Training output root directory (to locate ``ranked_edits.json``).

    Returns
    -------
    str
        Formatted epoch history text.
    """
    update_mode = normalize_update_mode(update_mode)
    parts: list[str] = []
    for rec in epoch_step_records:
        step = rec["step"]
        action = rec.get("action", "unknown")
        gate_score = rec.get("selection_hard", rec.get("current_score", "?"))
        best_score = rec.get("best_score", "?")

        header = (
            f"### Step {step} — "
            f"gate: {gate_score}, {action.upper()}, "
            f"best_so_far: {best_score}"
        )

        # Read the actual applied edits
        ranked_path = os.path.join(
            out_root, "steps", f"step_{step:04d}", "ranked_edits.json",
        )
        edits_text = ""
        if os.path.exists(ranked_path):
            try:
                with open(ranked_path) as f:
                    ranked = json.load(f)
                edits = get_payload_items(ranked, update_mode)
                if edits:
                    lines = [f"Selected {payload_label(update_mode)}:"]
                    for i, edit in enumerate(edits, 1):
                        lines.append(f"  {i}. {describe_item(edit, update_mode, max_chars=220)}")
                    edits_text = "\n".join(lines)
                else:
                    edits_text = f"Selected {payload_label(update_mode)}: (none)"
            except Exception:
                edits_text = f"Selected {payload_label(update_mode)}: (could not read)"
        else:
            # Step may have been skipped
            if "skip" in action:
                edits_text = f"Selected {payload_label(update_mode)}: (skipped)"
            else:
                edits_text = f"Selected {payload_label(update_mode)}: (file not found)"

        parts.append(f"{header}\n{edits_text}")

        # Append trajectory failure digest if available
        digest_path = os.path.join(
            out_root, "steps", f"step_{step:04d}", "trajectory_digest.json",
        )
        if os.path.exists(digest_path):
            try:
                with open(digest_path) as f:
                    digest = json.load(f)
                patterns = digest.get("failure_patterns", [])
                if patterns:
                    n_fail = digest.get("n_fail", "?")
                    n_total = digest.get("n_total", "?")
                    lines = [f"Failure patterns ({n_fail}/{n_total} tasks failed):"]
                    for p in patterns:
                        lines.append(
                            f'  - "{p["pattern"]}" (×{p["count"]})'
                        )
                    parts[-1] += "\n" + "\n".join(lines)
            except Exception:
                pass

    return "\n\n".join(parts)


# ── Meta-reflect teacher call ────────────────────────────────────────────────


def run_meta_reflect(
    skill_content: str,
    epoch_history_text: str,
    prev_meta_summary: str,
    meta_edit_budget: int = 4,
    *,
    system_prompt: str | None = None,
    update_mode: str = "patch",
) -> dict | None:
    """Run one meta-reflect teacher call for an epoch.

    Parameters
    ----------
    skill_content : str
        Current skill document (after the epoch's fast updates).
    epoch_history_text : str
        Formatted epoch history from :func:`build_epoch_history`.
    prev_meta_summary : str
        Meta summary from the previous epoch ("" if first epoch).
    meta_edit_budget : int
        Maximum number of high-level edits.
    system_prompt : str | None
        Custom system prompt. ``None`` = use generic default.

    Returns
    -------
    dict | None
        Conforms to :class:`~skillopt.types.MetaReflectResult`:
        ``"meta_summary"`` (str) and ``"patch"`` (:class:`~skillopt.types.Patch`
        dict), or ``None`` on failure.
    """
    mode = normalize_update_mode(update_mode)
    actual_system = system_prompt if system_prompt is not None else load_prompt(
        "meta_reflect_rewrite" if mode == "rewrite_from_suggestions" else "meta_reflect"
    )

    prev_section = prev_meta_summary.strip() if prev_meta_summary else "(First epoch — no previous summary)"

    user = (
        f"## Previous Meta Summary\n{prev_section}\n\n"
        f"## Current Skill Document\n{skill_content}\n\n"
        f"## {payload_label(mode, title=True)} Budget\n"
        f"Produce at most {meta_edit_budget} high-level {payload_label(mode)}.\n\n"
        f"## This Epoch's Step History\n{epoch_history_text}"
    )

    try:
        response, _ = chat_teacher(
            system=actual_system,
            user=user,
            max_completion_tokens=4096,
            retries=3,
            stage="meta_reflect",
        )
        result = extract_json(response)
        if result and "patch" in result:
            truncate_payload(result["patch"], meta_edit_budget, mode)
            if "meta_summary" not in result:
                result["meta_summary"] = ""
            return result
    except Exception:  # noqa: BLE001
        traceback.print_exc()

    return None
