"""ReflACT Slow Update — epoch-level longitudinal skill refinement.

At the end of each epoch, the slow update compares rollout performance of the
same sample set under the previous epoch's skill vs. the current epoch's skill
(Markov: only adjacent epochs). A optimizer analyzes regressions, improvements,
and persistent failures, then writes a free-form guidance block into a
**protected** section of the skill document. This section cannot be modified by
step-level analyst edits — only the slow update process overwrites it.

Public API
----------
- :func:`inject_empty_slow_update_field` — add empty placeholder (epoch 1)
- :func:`extract_slow_update_field`      — read current content
- :func:`replace_slow_update_field`      — overwrite content
- :func:`has_slow_update_field`          — check if markers are present
- :func:`build_comparison_text`          — format side-by-side rollout results
- :func:`run_slow_update`               — optimizer call to produce guidance
"""
from __future__ import annotations

import json
import os
import traceback

from skillopt.model import chat_optimizer
from skillopt.prompts import load_prompt
from skillopt.utils import extract_json, is_quality_failed_result

# ── Protected field markers ─────────────────────────────────────────────────

SLOW_UPDATE_START = "<!-- SLOW_UPDATE_START -->"
SLOW_UPDATE_END = "<!-- SLOW_UPDATE_END -->"

# ── Field manipulation helpers ──────────────────────────────────────────────


def has_slow_update_field(skill: str) -> bool:
    return SLOW_UPDATE_START in skill and SLOW_UPDATE_END in skill


def inject_empty_slow_update_field(skill: str) -> str:
    if has_slow_update_field(skill):
        return skill
    block = (
        f"\n\n{SLOW_UPDATE_START}\n"
        f"{SLOW_UPDATE_END}\n"
    )
    return skill.rstrip() + block


def extract_slow_update_field(skill: str) -> str:
    start = skill.find(SLOW_UPDATE_START)
    end = skill.find(SLOW_UPDATE_END)
    if start == -1 or end == -1:
        return ""
    inner_start = start + len(SLOW_UPDATE_START)
    return skill[inner_start:end].strip()


def _strip_all_slow_update_fields(skill: str) -> str:
    """Remove every SLOW_UPDATE_START/END pair (and content between) from *skill*."""
    while True:
        start = skill.find(SLOW_UPDATE_START)
        if start == -1:
            break
        end = skill.find(SLOW_UPDATE_END, start)
        if end == -1:
            # Orphan start marker — remove it
            skill = skill[:start] + skill[start + len(SLOW_UPDATE_START):]
            break
        skill = skill[:start] + skill[end + len(SLOW_UPDATE_END):]
    # Clean up stray end markers
    skill = skill.replace(SLOW_UPDATE_END, "")
    # Collapse excess blank lines left behind
    while "\n\n\n" in skill:
        skill = skill.replace("\n\n\n", "\n\n")
    return skill.rstrip()


def replace_slow_update_field(skill: str, new_content: str) -> str:
    # Remove all existing slow update regions first to guarantee exactly one.
    skill = _strip_all_slow_update_fields(skill)
    block = (
        f"\n\n{SLOW_UPDATE_START}\n"
        f"{new_content.strip()}\n"
        f"{SLOW_UPDATE_END}\n"
    )
    return skill + block


# ── Comparison text builder ─────────────────────────────────────────────────


# NOTE: The character limits below (whole-trajectory cap + the per-field caps in
# _read_trajectory and the comparison metadata) only trim the comparison samples
# fed to the slow-update optimizer. They exist to cut token usage and speed up the
# call; they do NOT affect what gets written into the skill. If you need richer
# context for the longitudinal comparison, feel free to raise them.
_MAX_TRAJ_CHARS = 3000


def _clip_text(value, limit: int) -> str:
    if value is None:
        return ""
    return str(value)[:limit]


def _read_trajectory(rollout_dir: str, task_id: str) -> str:
    """Read and format a single trajectory from a rollout directory."""
    conv_path = os.path.join(rollout_dir, "predictions", task_id, "conversation.json")
    if not os.path.exists(conv_path):
        return "(trajectory not available)"
    try:
        with open(conv_path) as f:
            conversation = json.load(f)
    except Exception:
        return "(trajectory read error)"
    if not conversation:
        return "(empty trajectory)"

    lines: list[str] = []
    for entry in conversation:
        if not isinstance(entry, dict):
            continue
        # Per-field caps (cmd/obs/reasoning/etc.) keep each trajectory compact to
        # save tokens / time; raise them if you want fuller step detail.
        if entry.get("type") == "tool_call":
            cmd = _clip_text(entry.get("cmd"), 500)
            obs = _clip_text(entry.get("obs"), 800)
            lines.append(f"[action] {cmd}")
            lines.append(f"[obs]    {obs}")
        elif "action" in entry and "env_feedback" in entry:
            step = entry.get("step", "?")
            reasoning = _clip_text(entry.get("reasoning"), 300)
            action = _clip_text(entry.get("action"), 200)
            feedback = _clip_text(entry.get("env_feedback"), 500)
            if reasoning:
                lines.append(f"[step {step} think] {reasoning}")
            lines.append(f"[step {step} action] {action}")
            lines.append(f"[step {step} obs]    {feedback}")
        elif entry.get("role") == "system":
            msg = _clip_text(entry.get("content"), 1000)
            lines.append(f"[verification] {msg}")
        else:
            msg = _clip_text(entry.get("content"), 500)
            role = entry.get("role", "agent")
            lines.append(f"[{role}] {msg}")

    text = "\n".join(lines)
    if len(text) > _MAX_TRAJ_CHARS:
        half = _MAX_TRAJ_CHARS // 2
        text = text[:half] + "\n...[truncated]...\n" + text[-half:]
    return text


# ── Structured comparison pairs ─────────────────────────────────────────────


def build_comparison_pairs(
    results_prev: list[dict],
    results_curr: list[dict],
    items: list[dict],
    prev_rollout_dir: str = "",
    curr_rollout_dir: str = "",
) -> list[dict]:
    """Build a structured list of per-sample comparison entries.

    Each entry bundles the original item, both rollout results, the change
    category, and both trajectories into one dict — the single source of
    truth for this sample's longitudinal comparison.

    Returns
    -------
    list[dict]
        One dict per sample with keys:
        ``id, task, category, prev, curr, prev_trajectory, curr_trajectory``
    """
    prev_by_id = {str(r["id"]): r for r in results_prev}
    curr_by_id = {str(r["id"]): r for r in results_curr}

    pairs: list[dict] = []
    for item in items:
        tid = str(item.get("id", ""))
        prev = prev_by_id.get(tid, {})
        curr = curr_by_id.get(tid, {})
        prev_ok = bool(prev.get("hard", 0)) and not is_quality_failed_result(prev)
        curr_ok = bool(curr.get("hard", 0)) and not is_quality_failed_result(curr)

        if not prev_ok and curr_ok:
            category = "improved"
        elif prev_ok and not curr_ok:
            category = "regressed"
        elif not prev_ok and not curr_ok:
            category = "persistent_fail"
        else:
            category = "stable_success"

        pairs.append({
            "id": tid,
            "task": item.get("question", item.get("task_description", item.get("instruction", tid))),
            "category": category,
            "prev": {
                "hard": int(prev_ok),
                "quality_status": str(prev.get("quality_status") or ""),
                "soft": float(prev.get("soft", 0.0)),
                "predicted_answer": prev.get("predicted_answer", prev.get("answer", "N/A")),
                "fail_reason": prev.get("fail_reason", ""),
            },
            "curr": {
                "hard": int(curr_ok),
                "quality_status": str(curr.get("quality_status") or ""),
                "soft": float(curr.get("soft", 0.0)),
                "predicted_answer": curr.get("predicted_answer", curr.get("answer", "N/A")),
                "fail_reason": curr.get("fail_reason", ""),
            },
            "prev_trajectory": (
                _read_trajectory(prev_rollout_dir, tid) if prev_rollout_dir else ""
            ),
            "curr_trajectory": (
                _read_trajectory(curr_rollout_dir, tid) if curr_rollout_dir else ""
            ),
        })

    return pairs


def save_comparison_pairs(pairs: list[dict], out_path: str) -> None:
    """Persist comparison pairs to JSON (without trajectory text to save space)."""
    slim = []
    for p in pairs:
        slim.append({
            "id": p["id"],
            "task": p["task"][:300],
            "category": p["category"],
            "prev": p["prev"],
            "curr": p["curr"],
        })
    with open(out_path, "w") as f:
        json.dump(slim, f, ensure_ascii=False, indent=2)


def format_comparison_text(pairs: list[dict]) -> str:
    """Format structured comparison pairs into optimizer-readable text."""
    by_cat: dict[str, list[dict]] = {
        "regressed": [],
        "persistent_fail": [],
        "improved": [],
        "stable_success": [],
    }
    for p in pairs:
        by_cat.setdefault(p["category"], []).append(p)

    total = len(pairs)
    parts = [
        f"## Longitudinal Comparison Summary\n"
        f"Total samples: {total}\n"
        f"- Improved (wrong→right): {len(by_cat['improved'])}\n"
        f"- Regressed (right→wrong): {len(by_cat['regressed'])}\n"
        f"- Persistent failures (wrong→wrong): {len(by_cat['persistent_fail'])}\n"
        f"- Stable successes (right→right): {len(by_cat['stable_success'])}\n"
    ]

    categories = [
        ("regressed", "Regressions (right→wrong) — HIGHEST PRIORITY", True),
        ("persistent_fail", "Persistent Failures (wrong→wrong)", True),
        ("improved", "Improvements (wrong→right)", True),
        ("stable_success", "Stable Successes (right→right)", False),
    ]

    for cat_key, label, show_traj in categories:
        entries = by_cat[cat_key]
        if not entries:
            parts.append(f"### {label}\n(none)\n")
            continue

        lines = [f"### {label}"]
        for e in entries:
            prev = e["prev"]
            curr = e["curr"]
            prev_ok = bool(prev.get("hard", 0)) and not is_quality_failed_result(prev)
            curr_ok = bool(curr.get("hard", 0)) and not is_quality_failed_result(curr)
            lines.append(
                f"\n#### Task {e['id']}: {e['task'][:300]}\n"
                f"- Prev epoch: {'PASS' if prev_ok else 'FAIL'} "
                f"(soft={prev['soft']:.2f}) — answer: {str(prev['predicted_answer'])[:200]}\n"
                f"- Curr epoch: {'PASS' if curr_ok else 'FAIL'} "
                f"(soft={curr['soft']:.2f}) — answer: {str(curr['predicted_answer'])[:200]}"
            )
            if curr.get("fail_reason"):
                lines.append(f"- Curr fail reason: {curr['fail_reason'][:300]}")
            if prev.get("fail_reason") and not prev_ok:
                lines.append(f"- Prev fail reason: {prev['fail_reason'][:300]}")

            if show_traj:
                if e.get("prev_trajectory"):
                    lines.append(
                        f"\n**Previous epoch trajectory:**\n```\n{e['prev_trajectory']}\n```"
                    )
                if e.get("curr_trajectory"):
                    lines.append(
                        f"\n**Current epoch trajectory:**\n```\n{e['curr_trajectory']}\n```"
                    )

        parts.append("\n".join(lines))

    return "\n\n".join(parts)



# ── Optimizer call ────────────────────────────────────────────────────────────


def run_slow_update(
    skill_content: str,
    results_prev: list[dict],
    results_curr: list[dict],
    items: list[dict],
    *,
    prev_skill: str = "",
    prev_slow_update_content: str = "",
    prev_rollout_dir: str = "",
    curr_rollout_dir: str = "",
    comparison_pairs: list[dict] | None = None,
    system_prompt: str | None = None,
) -> dict | None:
    """Run the slow update optimizer call for one epoch boundary.

    Parameters
    ----------
    skill_content : str
        Current epoch's skill (after fast updates).
    results_prev : list[dict]
        Rollout results of the 20 samples under previous epoch's skill.
    results_curr : list[dict]
        Rollout results of the 20 samples under current epoch's skill.
    items : list[dict]
        The 20 sample items used for comparison.
    prev_skill : str
        Previous epoch's skill content.
    prev_slow_update_content : str
        The slow update guidance from the previous epoch (to reflect on).
    prev_rollout_dir : str
        Path to previous epoch rollout output (contains predictions/).
    curr_rollout_dir : str
        Path to current epoch rollout output (contains predictions/).
    system_prompt : str | None
        Custom system prompt override.

    Returns
    -------
    dict | None
        Conforms to :class:`~skillopt.types.SlowUpdateResult`:
        ``{"reasoning": str, "slow_update_content": str}`` or ``None``.
    """
    actual_system = system_prompt if system_prompt is not None else load_prompt("slow_update")

    pairs = comparison_pairs
    if pairs is None:
        pairs = build_comparison_pairs(
            results_prev, results_curr, items,
            prev_rollout_dir=prev_rollout_dir,
            curr_rollout_dir=curr_rollout_dir,
        )
    comparison_text = format_comparison_text(pairs)

    prev_guidance_section = (
        prev_slow_update_content.strip()
        if prev_slow_update_content and prev_slow_update_content.strip()
        else "(No previous guidance — this is the first slow update.)"
    )

    user = (
        f"## Previous Epoch's Skill\n{prev_skill}\n\n"
        f"## Current Epoch's Skill\n{skill_content}\n\n"
        f"## Previous Slow Update Guidance\n"
        f"The following guidance was active during the current epoch. "
        f"Reflect on its effectiveness before writing the new version.\n\n"
        f"{prev_guidance_section}\n\n"
        f"## Longitudinal Comparison (same 20 tasks, two skill versions)\n"
        f"{comparison_text}"
    )

    try:
        response, _ = chat_optimizer(
            system=actual_system,
            user=user,
            max_completion_tokens=16384,
            retries=3,
            stage="slow_update",
        )
        result = extract_json(response)
        if result and result.get("slow_update_content"):
            return {
                "reasoning": str(result.get("reasoning", "")).strip(),
                "slow_update_content": str(result["slow_update_content"]).strip(),
            }
    except Exception:  # noqa: BLE001
        traceback.print_exc()

    return None
