"""ReflACT core Reflect engine -- minibatch trajectory analysis.

Provides environment-agnostic minibatch trajectory analysis: instead of
analyzing each trajectory independently, trajectories are grouped into
minibatches of size M and analyzed together -- analogous to minibatch SGD
vs per-sample SGD in neural network training.

Two-level prompt priority system:

1. **Custom prompt** (adapter returns non-None) -- used as-is.
2. **Generic default prompt** (adapter returns None) -- built-in defaults
   that work for any environment without configuration.

Public API
----------
- :func:`fmt_trajectory`               -- format one conversation into text
- :func:`fmt_minibatch_trajectories`   -- format multiple trajectories for batch analysis
- :func:`run_error_analyst_minibatch`   -- one optimizer call for a group of failures
- :func:`run_success_analyst_minibatch` -- one optimizer call for a group of successes
- :func:`run_minibatch_reflect`         -- full reflect stage dispatcher
"""
from __future__ import annotations

import json
import os
import random
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from skillopt.model import chat_optimizer
from skillopt.optimizer.meta_skill import format_meta_skill_context
from skillopt.optimizer.skill_aware import (
    augment_error_prompt,
    augment_success_prompt,
    extract_appendix_notes,
    get_skill_aware_appendix_source,
    is_skill_aware_enabled,
)
from skillopt.optimizer.update_modes import (
    get_payload_items,
    is_full_rewrite_minibatch_mode,
    normalize_update_mode,
    payload_key,
    payload_label,
    truncate_payload,
)
from skillopt.prompts import load_prompt
from skillopt.utils import extract_json


# ── Trajectory formatting ────────────────────────────────────────────────────


def _clip_text(value, limit: int | None = None) -> str:
    """Render optional trajectory fields. Truncation is disabled: the optimizer
    is given the full content so it can see exactly what the agent saw/did.

    ``limit`` is accepted for backward compatibility but ignored.
    """
    if value is None:
        return ""
    return str(value)


def fmt_trajectory(
    conversation: list[dict],
    max_chars: int | None = None,
) -> str:
    """Format a conversation list into analyst-readable text.

    Accepts two common formats:

    1. Tool-call records:   ``{"type": "tool_call", "cmd": ..., "obs": ...}``
    2. Step records:        ``{"step": N, "action": ..., "env_feedback": ..., "reasoning": ...}``

    Any other dict is rendered via its ``"content"`` key.
    """
    lines: list[str] = []
    for item in conversation:
        if not isinstance(item, dict):
            lines.append(f"[agent] {_clip_text(item)}")
            continue
        if item.get("type") == "tool_call":
            cmd = _clip_text(item.get("cmd"))
            obs = _clip_text(item.get("obs"))
            lines.append(f"[action] {cmd}")
            lines.append(f"[obs]    {obs}")
        elif "action" in item and "env_feedback" in item:
            step = item.get("step", "?")
            reasoning = _clip_text(item.get("reasoning"))
            action = _clip_text(item.get("action"))
            feedback = _clip_text(item.get("env_feedback"))
            if reasoning:
                lines.append(f"[step {step} think] {reasoning}")
            lines.append(f"[step {step} action] {action}")
            lines.append(f"[step {step} obs]    {feedback}")
        elif item.get("role") == "system":
            # Post-execution verification / enrichment info
            msg = _clip_text(item.get("content"))
            lines.append(f"[verification] {msg}")
        else:
            msg = _clip_text(item.get("content"))
            role = item.get("role", "agent")
            lines.append(f"[{role}] {msg}")

    return "\n".join(lines)


# ── Minibatch trajectory formatting ──────────────────────────────────────────


def fmt_minibatch_trajectories(
    items: list[dict],
    prediction_dir: str,
) -> str:
    """Format multiple trajectories for minibatch analyst consumption.

    Each item is a rollout result dict with ``"id"``, ``"task_description"``,
    ``"task_type"``, ``"fail_reason"``, etc.  Reads ``conversation.json``
    for each and formats them together with trajectory headers.

    If available, includes the spreadsheet preview and target system prompt
    so the analyst can see what the agent saw.

    Parameters
    ----------
    items : list[dict]
        Rollout result dicts belonging to one minibatch.
    prediction_dir : str
        Path to ``predictions/`` directory containing per-task
        ``<task_id>/conversation.json`` files.

    Returns
    -------
    str
        Formatted text with all trajectories separated by ``---``.
    """
    parts: list[str] = []
    for idx, item in enumerate(items, 1):
        tid = str(item["id"])
        conv_path = os.path.join(prediction_dir, tid, "conversation.json")
        if not os.path.exists(conv_path):
            continue
        with open(conv_path) as f:
            conversation = json.load(f)
        if not conversation:
            continue

        traj_text = fmt_trajectory(conversation)
        header = (
            f"### Trajectory {idx} (id={tid})\n"
            f"Task: {item.get('task_description', item.get('instruction', ''))}\n"
            f"Task type: {item.get('task_type', item.get('instruction_type', ''))}\n"
        )
        fail_reason = item.get("fail_reason", "")
        if fail_reason:
            header += f"Failure reason: {fail_reason}\n"
        header += f"Steps: {item.get('n_turns', '?')}\n"

        reference_text = str(item.get("reference_text") or "").strip()
        if reference_text:
            header += (
                f"\n#### Hidden Reference\n"
                f"{reference_text}\n"
            )

        # ── Append target context (what the agent saw) ──────────────
        target_prompt = item.get("target_system_prompt", "")
        if not target_prompt:
            prompt_path = os.path.join(prediction_dir, tid, "target_system_prompt.txt")
            if os.path.exists(prompt_path):
                with open(prompt_path) as f:
                    target_prompt = f.read()
        if target_prompt:
            header += (
                f"\n#### Target System Prompt\n"
                f"{target_prompt}\n"
            )

        user_prompt = item.get("target_user_prompt", "")
        if not user_prompt:
            user_prompt_path = os.path.join(prediction_dir, tid, "target_user_prompt.txt")
            if os.path.exists(user_prompt_path):
                with open(user_prompt_path) as f:
                    user_prompt = f.read()
        if user_prompt:
            header += (
                f"\n#### Target User Prompt\n"
                f"{user_prompt}\n"
            )

        if os.environ.get("REFLACT_CODEX_TRACE_TO_OPTIMIZER", "0") == "1":
            codex_trace_summary = item.get("codex_trace_summary", "")
            if not codex_trace_summary:
                codex_trace_summary_path = os.path.join(prediction_dir, tid, "codex_trace_summary.txt")
                if os.path.exists(codex_trace_summary_path):
                    with open(codex_trace_summary_path) as f:
                        codex_trace_summary = f.read()
            if codex_trace_summary:
                header += (
                    f"\n#### Codex Trace Summary\n"
                    f"{codex_trace_summary}\n"
                )

        codex_probe_trace_steps = str(item.get("codex_probe_trace_steps") or "").strip()
        if codex_probe_trace_steps:
            header += (
                f"\n#### Codex Trace Steps\n"
                f"{codex_probe_trace_steps}\n"
            )

        preview = item.get("spreadsheet_preview", "")
        if not preview:
            preview_path = os.path.join(prediction_dir, tid, "spreadsheet_preview.txt")
            if os.path.exists(preview_path):
                with open(preview_path) as f:
                    preview = f.read()
        if preview:
            header += (
                f"\n#### Spreadsheet Preview\n"
                f"{preview}\n"
            )

        parts.append(header + "\n" + traj_text)

    return "\n\n---\n\n".join(parts)


# ── Prompt resolution ───────────────────────────────────────────────────────


def _resolve_prompt(custom: str | None, default_name: str, update_mode: str = "patch") -> str:
    """Return *custom* if provided (non-None), otherwise load from file."""
    if custom is not None:
        return custom
    mode = normalize_update_mode(update_mode)
    actual_name = default_name
    if is_full_rewrite_minibatch_mode(mode):
        full_name = f"{default_name}_full_rewrite"
        try:
            return load_prompt(full_name)
        except FileNotFoundError:
            actual_name = default_name
    elif mode == "rewrite_from_suggestions":
        rewrite_name = f"{default_name}_rewrite"
        try:
            return load_prompt(rewrite_name)
        except FileNotFoundError:
            actual_name = default_name
    return load_prompt(actual_name)


# ── Minibatch analysts ──────────────────────────────────────────────────────


def run_error_analyst_minibatch(
    skill_content: str,
    items: list[dict],
    prediction_dir: str,
    edit_budget: int = 4,
    *,
    system_prompt: str | None = None,
    rejection_context: str = "",
    trajectory_memory_context: str = "",
    step_buffer_context: str = "",
    meta_skill_context: str = "",
    update_mode: str = "patch",
    skill_aware_reflection: bool = False,
) -> dict | None:
    """Analyze a minibatch of failed trajectories in one optimizer call.

    Parameters
    ----------
    skill_content : str
        Current skill document text.
    items : list[dict]
        Rollout result dicts (all should have ``hard=0``).
    prediction_dir : str
        Path to ``predictions/`` directory.
    edit_budget : int
        Maximum number of edits (L).
    system_prompt : str | None
        Custom system prompt. ``None`` = use generic default.
    rejection_context : str
        *Deprecated* — use ``step_buffer_context``.
    trajectory_memory_context : str
        *Deprecated* — use ``step_buffer_context``.
    step_buffer_context : str
        Unified summary of previous steps (failure patterns + rejected edits).

    Returns
    -------
    dict | None
        Patch dict with ``source_type="failure"``, or ``None`` on error.
    """
    mode = normalize_update_mode(update_mode)
    actual_system = _resolve_prompt(system_prompt, "analyst_error", mode)
    # Skill-aware reflection: augment the resolved prompt at runtime so both
    # env-specific and generic analyst prompts get the defect/lapse instruction.
    # When the toggle is off this is a no-op (prompt byte-identical to baseline).
    if skill_aware_reflection and not is_full_rewrite_minibatch_mode(mode):
        actual_system = augment_error_prompt(actual_system)

    trajectories_text = fmt_minibatch_trajectories(items, prediction_dir)
    if not trajectories_text.strip():
        return None

    user = (
        f"## Current Skill\n{skill_content}\n\n"
    )
    if is_full_rewrite_minibatch_mode(mode):
        user += (
            f"## Update Format\n"
            f"Produce one complete replacement skill candidate for this minibatch. "
            f"Do not output edits, patches, or revise suggestions.\n\n"
        )
    else:
        user += (
            f"## {payload_label(mode, title=True)} Budget\n"
            f"Produce at most L={edit_budget} {payload_label(mode)}.\n\n"
        )
    # Unified step buffer context (preferred)
    ctx = step_buffer_context or rejection_context or ""
    if trajectory_memory_context:
        ctx = f"{ctx}\n{trajectory_memory_context}" if ctx else trajectory_memory_context
    if ctx.strip():
        user += f"## Previous Steps in This Epoch\n{ctx}\n\n"
    optimizer_ctx = format_meta_skill_context(meta_skill_context)
    if optimizer_ctx:
        user += optimizer_ctx + "\n\n"
    user += f"## Failed Trajectories ({len(items)} total)\n{trajectories_text}"

    try:
        response, _ = chat_optimizer(
            system=actual_system, user=user,
            max_completion_tokens=64000 if is_full_rewrite_minibatch_mode(mode) else 16384,
            retries=3,
            stage="analyst",
        )
        result = extract_json(response)
        if not result:
            return None
        notes = extract_appendix_notes(result) if skill_aware_reflection else []
        if "patch" in result:
            result["source_type"] = "failure"
            if not is_full_rewrite_minibatch_mode(mode):
                truncate_payload(result["patch"], edit_budget, mode)
            if skill_aware_reflection:
                result["appendix_notes"] = notes
            return result
        # Skill-aware: a batch may legitimately yield ONLY execution-lapse notes
        # (no body edit). Return a no-op patch so the notes still reach the
        # trainer via all_raw_patches; empty edits are dropped from the body
        # pipeline by _normalise_patches, so body behavior is unchanged.
        if skill_aware_reflection and notes:
            return {
                "source_type": "failure",
                "patch": {"reasoning": "execution-lapse only", "edits": []},
                "appendix_notes": notes,
            }
    except Exception:  # noqa: BLE001
        traceback.print_exc()
    return None


def run_success_analyst_minibatch(
    skill_content: str,
    items: list[dict],
    prediction_dir: str,
    edit_budget: int = 4,
    *,
    system_prompt: str | None = None,
    trajectory_memory_context: str = "",
    step_buffer_context: str = "",
    meta_skill_context: str = "",
    update_mode: str = "patch",
    skill_aware_reflection: bool = False,
    emit_appendix_notes: bool = True,
) -> dict | None:
    """Analyze a minibatch of successful trajectories in one optimizer call.

    Parameters
    ----------
    system_prompt : str | None
        Custom system prompt. ``None`` = use generic default.
    trajectory_memory_context : str
        *Deprecated* — use ``step_buffer_context``.
    step_buffer_context : str
        Unified summary of previous steps (failure patterns + rejected edits).

    Returns
    -------
    dict | None
        Patch dict with ``source_type="success"``, or ``None`` on error.
    """
    mode = normalize_update_mode(update_mode)
    actual_system = _resolve_prompt(system_prompt, "analyst_success", mode)
    # Only augment + parse appendix notes on the success side when allowed.
    # failure_only mode (paper-faithful S_app) suppresses success-side notes.
    sa_emit = skill_aware_reflection and emit_appendix_notes
    if sa_emit and not is_full_rewrite_minibatch_mode(mode):
        actual_system = augment_success_prompt(actual_system)

    trajectories_text = fmt_minibatch_trajectories(items, prediction_dir)
    if not trajectories_text.strip():
        return None

    user = (
        f"## Current Skill\n{skill_content}\n\n"
    )
    if is_full_rewrite_minibatch_mode(mode):
        user += (
            f"## Update Format\n"
            f"Produce one complete replacement skill candidate for this minibatch. "
            f"Do not output edits, patches, or revise suggestions.\n\n"
        )
    else:
        user += (
            f"## {payload_label(mode, title=True)} Budget\n"
            f"Produce at most L={edit_budget} {payload_label(mode)}.\n\n"
        )
    ctx = step_buffer_context or trajectory_memory_context or ""
    if ctx.strip():
        user += f"## Previous Steps in This Epoch\n{ctx}\n\n"
    optimizer_ctx = format_meta_skill_context(meta_skill_context)
    if optimizer_ctx:
        user += optimizer_ctx + "\n\n"
    user += f"## Successful Trajectories ({len(items)} total)\n{trajectories_text}"

    try:
        response, _ = chat_optimizer(
            system=actual_system, user=user,
            max_completion_tokens=64000 if is_full_rewrite_minibatch_mode(mode) else 16384,
            retries=3,
            stage="analyst",
        )
        result = extract_json(response)
        if result and "patch" in result:
            result["source_type"] = "success"
            if not is_full_rewrite_minibatch_mode(mode):
                truncate_payload(result["patch"], edit_budget, mode)
            if sa_emit:
                result["appendix_notes"] = extract_appendix_notes(result)
            return result
    except Exception:  # noqa: BLE001
        traceback.print_exc()
    return None


# ── Minibatch reflect dispatcher ────────────────────────────────────────────


def _split_minibatches(items: list, batch_size: int) -> list[list]:
    """Split items into minibatches of at most *batch_size*."""
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def _shuffle_for_minibatch(items: list, seed: int | None) -> list:
    """Return items in minibatch order.

    Uses a deterministic shuffle when a seed is provided so resume runs keep
    the same minibatch composition. Falls back to input order when no seed is
    available.
    """
    ordered = list(items)
    if seed is None:
        return ordered
    random.Random(seed).shuffle(ordered)
    return ordered


def run_minibatch_reflect(
    results: list[dict],
    skill_content: str,
    prediction_dir: str,
    patches_dir: str,
    workers: int,
    failure_only: bool,
    minibatch_size: int = 8,
    edit_budget: int = 4,
    random_seed: int | None = None,
    *,
    error_system: str | None = None,
    success_system: str | None = None,
    rejection_context: str = "",
    trajectory_memory_context: str = "",
    step_buffer_context: str = "",
    meta_skill_context: str = "",
    update_mode: str = "patch",
    skill_aware_reflection: bool | None = None,
    skill_aware_appendix_source: str | None = None,
) -> list[dict | None]:
    """Full minibatch reflect stage: group → parallel optimizer calls → patches.

    Separates failure and success trajectories, splits each into minibatches
    of size M, runs all minibatches in parallel, and saves patch files.

    Parameters
    ----------
    results : list[dict]
        Rollout result dicts; see :class:`~skillopt.types.RolloutResult`.
    skill_content : str
        Current skill document.
    prediction_dir : str
        Path to ``predictions/`` with ``conversation.json`` files.
    patches_dir : str
        Path to save per-minibatch patch JSON files.
    workers : int
        Max parallel optimizer calls.
    failure_only : bool
        If True, skip success trajectories.
    minibatch_size : int
        Trajectories per group (M).
    edit_budget : int
        Max edits per minibatch (L).
    random_seed : int | None
        Optional seed used to shuffle trajectories before minibatch splitting.
    error_system, success_system : str | None
        Optional custom prompts. ``None`` = use generic defaults.

    Returns
    -------
    list[dict | None]
        Patch dicts (with ``source_type`` "failure" or "success").
    """
    # Resolve the skill-aware toggle: explicit kwargs win; otherwise fall back
    # to the process-wide config switch set by the trainer, so the feature is
    # env-independent and adapters need no per-benchmark wiring.
    if skill_aware_reflection is None:
        skill_aware_reflection = is_skill_aware_enabled()
    if skill_aware_appendix_source is None:
        skill_aware_appendix_source = get_skill_aware_appendix_source()

    os.makedirs(patches_dir, exist_ok=True)

    # Separate failure / success
    failures = [r for r in results if not r.get("hard") or float(r.get("hard", 0)) < 1e-9]
    successes = [r for r in results if r.get("hard")] if not failure_only else []

    failures = _shuffle_for_minibatch(failures, random_seed)
    successes = _shuffle_for_minibatch(successes, None if random_seed is None else random_seed + 1)

    # Split into minibatches
    fail_batches = _split_minibatches(failures, minibatch_size)
    succ_batches = _split_minibatches(successes, minibatch_size)

    n_fail_batches = len(fail_batches)
    n_succ_batches = len(succ_batches)
    print(
        f"    [2/6 REFLECT minibatch] "
        f"failure={len(failures)}→{n_fail_batches} groups  "
        f"success={len(successes)}→{n_succ_batches} groups  "
        f"(M={minibatch_size}, L={edit_budget}, workers={workers})"
    )

    raw_patches: list[dict | None] = []

    # Resume support: check for already-done minibatch patches
    pending_fail: list[tuple[int, list[dict]]] = []
    for idx, batch in enumerate(fail_batches):
        path = os.path.join(patches_dir, f"minibatch_fail_{idx:03d}.json")
        if os.path.exists(path):
            with open(path) as f:
                raw_patches.append(json.load(f))
        else:
            pending_fail.append((idx, batch))

    pending_succ: list[tuple[int, list[dict]]] = []
    for idx, batch in enumerate(succ_batches):
        path = os.path.join(patches_dir, f"minibatch_succ_{idx:03d}.json")
        if os.path.exists(path):
            with open(path) as f:
                raw_patches.append(json.load(f))
        else:
            pending_succ.append((idx, batch))

    # ── Worker functions ──────────────────────────────────────────────────
    def _do_fail(idx: int, batch: list[dict]) -> tuple[str, dict | None]:
        patch = run_error_analyst_minibatch(
            skill_content, batch, prediction_dir,
            edit_budget=edit_budget,
            system_prompt=error_system,
            step_buffer_context=step_buffer_context,
            # backward compat fallback
            rejection_context=rejection_context,
            trajectory_memory_context=trajectory_memory_context,
            meta_skill_context=meta_skill_context,
            update_mode=update_mode,
            skill_aware_reflection=skill_aware_reflection,
        )
        return f"minibatch_fail_{idx:03d}", patch

    def _do_succ(idx: int, batch: list[dict]) -> tuple[str, dict | None]:
        patch = run_success_analyst_minibatch(
            skill_content, batch, prediction_dir,
            edit_budget=edit_budget,
            system_prompt=success_system,
            step_buffer_context=step_buffer_context,
            trajectory_memory_context=trajectory_memory_context,
            meta_skill_context=meta_skill_context,
            update_mode=update_mode,
            skill_aware_reflection=skill_aware_reflection,
            emit_appendix_notes=(skill_aware_appendix_source != "failure_only"),
        )
        return f"minibatch_succ_{idx:03d}", patch

    # Run all pending minibatches in parallel
    all_pending = (
        [("fail", idx, batch) for idx, batch in pending_fail]
        + [("succ", idx, batch) for idx, batch in pending_succ]
    )

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {}
        for kind, idx, batch in all_pending:
            if kind == "fail":
                futs[ex.submit(_do_fail, idx, batch)] = (kind, idx, len(batch))
            else:
                futs[ex.submit(_do_succ, idx, batch)] = (kind, idx, len(batch))

        for i, fut in enumerate(as_completed(futs), 1):
            kind, idx, batch_len = futs[fut]
            tag, patch = fut.result()
            if patch:
                path = os.path.join(patches_dir, f"{tag}.json")
                with open(path, "w") as f:
                    json.dump(patch, f, ensure_ascii=False, indent=2)
                raw_patches.append(patch)
            n_edits = len(get_payload_items(patch.get("patch", {}) if patch else {}, update_mode))
            print(
                f"      [analyst] {i}/{len(all_pending)} {tag} "
                f"({batch_len} trajs) → {n_edits} {payload_label(update_mode)}"
            )

    return raw_patches
