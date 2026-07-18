"""Lightweight integration hooks for injecting SkillMemory into ReflACT training.

These hooks are designed to be **non-breaking**: if ``MEM0_API_KEY`` is not
present in the environment, :func:`maybe_init_mem0` returns ``None`` and
every ``hook_*`` function becomes a no-op.

Typical usage inside ``trainer.py``::

    from skillopt.memory.trainer_hooks import (
        maybe_init_mem0,
        hook_post_evaluate,
        hook_post_reflect,
    )

    memory = maybe_init_mem0(cfg)

    # ... inside training loop, after evaluation gate:
    hook_post_evaluate(memory, epoch, step, current_skill, gate_score, cfg)

    # ... after reflection stage:
    hook_post_reflect(memory, epoch, step, raw_patches)
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from skillopt.memory.mem0_backend import SkillMemory


# ── Initialisation ────────────────────────────────────────────────────────────

def maybe_init_mem0(cfg: dict) -> "SkillMemory | None":
    """Attempt to initialise a :class:`~skillopt.memory.SkillMemory` instance.

    Reads ``MEM0_API_KEY`` from the environment. If the key is absent or
    mem0ai is not installed, returns ``None`` (graceful degradation).

    Parameters
    ----------
    cfg : dict
        Flat trainer config dict. The ``config_name`` key (if present) is used
        to label experiment results.

    Returns
    -------
    SkillMemory | None
        A live ``SkillMemory`` instance, or ``None`` if mem0 is unavailable.
    """
    api_key = os.environ.get("MEM0_API_KEY", "")
    if not api_key:
        return None

    try:
        from skillopt.memory.mem0_backend import SkillMemory  # local import to avoid hard dep

        user_id = str(cfg.get("config_name") or cfg.get("env") or "skillopt")
        memory = SkillMemory(api_key=api_key, user_id=user_id)
        print(f"  [mem0] SkillMemory initialised — user_id={user_id!r}")
        return memory
    except Exception as exc:  # pragma: no cover
        print(f"  [mem0] WARNING: could not initialise SkillMemory: {exc}")
        return None


# ── Post-evaluate hook ────────────────────────────────────────────────────────

def hook_post_evaluate(
    memory: "SkillMemory | None",
    epoch: int,
    step: int,
    skill: str,
    score: float,
    cfg: dict,
) -> None:
    """Called after the evaluation gate — stores the current skill + score.

    Parameters
    ----------
    memory : SkillMemory | None
        The memory backend. If ``None``, this function is a no-op.
    epoch : int
        Current training epoch.
    step : int
        Current training step.
    skill : str
        Full text of the current skill document.
    score : float
        Gate score for this skill version.
    cfg : dict
        Flat trainer config (used to extract optional metadata).
    """
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
    except Exception as exc:  # pragma: no cover
        print(f"  [mem0] WARNING: hook_post_evaluate failed: {exc}")


# ── Post-reflect hook ─────────────────────────────────────────────────────────

def hook_post_reflect(
    memory: "SkillMemory | None",
    epoch: int,
    step: int,
    patches: list,
    scores: dict | None = None,
) -> None:
    """Called after the Reflect stage — stores patch metadata.

    Parameters
    ----------
    memory : SkillMemory | None
        The memory backend. If ``None``, this function is a no-op.
    epoch : int
        Current training epoch.
    step : int
        Current training step.
    patches : list
        Raw patches returned by the reflection stage (list of dicts).
    scores : dict | None
        Optional rollout scores from this step (e.g. ``{"hard": 0.7}``).
    """
    if memory is None:
        return
    try:
        memory.store_reflection(
            epoch=epoch,
            step=step,
            patches=list(patches) if patches else [],
            scores=scores,
        )
    except Exception as exc:  # pragma: no cover
        print(f"  [mem0] WARNING: hook_post_reflect failed: {exc}")
