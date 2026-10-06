"""Integration hooks binding :class:`SkillMemory` into the ReflACT loop.

Every hook is a no-op when *memory* is ``None``, which is the state of any run
that did not set ``mem0_enabled: true``. That keeps the trainer free of
feature-flag branches: it calls the hooks unconditionally and they decide.

The read/write pair is what makes this a memory rather than a log:

* :func:`hook_pre_reflect` — **reads**. Runs before the Reflect stage and
  appends relevant history to the reflection context, so past runs influence
  the patches produced now.
* :func:`hook_post_reflect` / :func:`hook_post_evaluate` — **write**.

Typical usage inside ``trainer.py``::

    from skillopt.memory.trainer_hooks import (
        maybe_init_mem0, hook_pre_reflect, hook_post_evaluate, hook_post_reflect,
    )

    memory = maybe_init_mem0(cfg)

    # before the Reflect stage:
    step_buffer_context = hook_pre_reflect(memory, current_skill, step_buffer_context)

    # after reflection / after the evaluation gate:
    hook_post_reflect(memory, epoch, step, raw_patches, scores=...)
    hook_post_evaluate(memory, epoch, step, current_skill, gate_score, cfg)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from skillopt.memory.settings import resolve_settings

if TYPE_CHECKING:
    from skillopt.memory.mem0_backend import SkillMemory

# Longest slice of skill text used to build a retrieval query. The query is
# only a similarity probe, so a head slice is enough and keeps the outbound
# request small.
_QUERY_CHARS = 600


# ── Initialisation ────────────────────────────────────────────────────────────

def maybe_init_mem0(cfg: dict) -> "SkillMemory | None":
    """Initialise memory for this run, or return ``None``.

    Returns ``None`` — sending nothing — unless **all** of the following hold:

    1. ``mem0_enabled`` is explicitly true in the config. A ``MEM0_API_KEY``
       present in the environment for some other application is *not*
       sufficient and never has been sufficient since this check existed.
    2. An API key is resolvable (``mem0_api_key`` or ``MEM0_API_KEY``).
    3. The optional ``mem0ai`` dependency is installed.
    """
    settings = resolve_settings(cfg)
    if not settings.enabled:
        return None

    if not settings.api_key:
        print("  [mem0] mem0_enabled is set but no API key was found — memory disabled")
        return None

    try:
        from skillopt.memory.mem0_backend import SkillMemory, mem0_available

        if not mem0_available():
            print("  [mem0] mem0_enabled is set but mem0ai is not installed "
                  "(pip install 'skillopt[mem0]') — memory disabled")
            return None

        memory = SkillMemory.from_settings(settings)
        if memory is not None:
            print(f"  [mem0] memory enabled — namespace={memory.namespace!r} "
                  f"retrieval={'on' if settings.retrieval_enabled else 'off'}")
        return memory
    except Exception as exc:  # noqa: BLE001 - memory must never break training
        print(f"  [mem0] WARNING: could not initialise memory: {exc}")
        return None


# ── Pre-reflect hook (the read side) ──────────────────────────────────────────

def hook_pre_reflect(
    memory: "SkillMemory | None",
    skill_text: str,
    step_buffer_context: str = "",
    query: str | None = None,
) -> str:
    """Return *step_buffer_context* with relevant past memories appended.

    This is the one retrieval call per step. It is bounded by
    ``mem0_timeout_seconds`` inside the backend, and on any failure — timeout,
    network error, empty result — the original context is returned unchanged,
    so reflection proceeds exactly as it would with memory disabled.

    Parameters
    ----------
    memory : SkillMemory | None
        Backend, or ``None`` for a run without memory (returns input unchanged).
    skill_text : str
        Current skill document; its head is used as the similarity probe.
    step_buffer_context : str
        Context already assembled for this step. Retrieved memory is appended
        to it rather than replacing it.
    query : str | None
        Explicit query, overriding the skill-derived one. Used by tests.

    Returns
    -------
    str
        Possibly-augmented context, always safe to pass to ``adapter.reflect``.
    """
    if memory is None:
        return step_buffer_context

    try:
        probe = query if query is not None else (skill_text or "")[:_QUERY_CHARS]
        if not probe.strip():
            return step_buffer_context

        memories = memory.retrieve_relevant_context(probe)
        block = memory.format_retrieved_context(memories)
        if not block:
            return step_buffer_context

        print(f"  [mem0] retrieved {len(memories)} memory record(s) into reflection context")
        if step_buffer_context and step_buffer_context.strip():
            return f"{step_buffer_context}\n\n{block}"
        return block
    except Exception as exc:  # noqa: BLE001 - memory must never break training
        print(f"  [mem0] WARNING: hook_pre_reflect failed: {exc}")
        return step_buffer_context


# ── Post-evaluate hook ────────────────────────────────────────────────────────

def hook_post_evaluate(
    memory: "SkillMemory | None",
    epoch: int,
    step: int,
    skill: str,
    score: float,
    cfg: dict,
) -> None:
    """Store the current skill and its gate score. No-op without memory."""
    if memory is None:
        return
    try:
        meta = {
            "env": cfg.get("env", ""),
            "optimizer_model": cfg.get("optimizer_model", ""),
            "target_model": cfg.get("target_model", ""),
        }
        memory.store_skill_iteration(
            epoch=epoch,
            step=step,
            skill_text=skill,
            score=score,
            metadata=meta,
        )
    except Exception as exc:  # noqa: BLE001 - memory must never break training
        print(f"  [mem0] WARNING: hook_post_evaluate failed: {exc}")


# ── Post-reflect hook ─────────────────────────────────────────────────────────

def hook_post_reflect(
    memory: "SkillMemory | None",
    epoch: int,
    step: int,
    patches: list,
    scores: dict | None = None,
) -> None:
    """Store a summary of this step's patches. No-op without memory."""
    if memory is None:
        return
    try:
        memory.store_reflection(
            epoch=epoch,
            step=step,
            patches=patches,
            scores=scores,
        )
    except Exception as exc:  # noqa: BLE001 - memory must never break training
        print(f"  [mem0] WARNING: hook_post_reflect failed: {exc}")
