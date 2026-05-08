"""Learning-rate (edit budget) schedulers for ReflACT.

The "learning rate" in ReflACT is the maximum number of skill edits allowed
per optimization step.  A scheduler controls how this budget changes over
the course of training.

Supported modes
---------------
- ``constant``   : Fixed budget throughout training.
- ``linear``     : Linear decay from ``max_lr`` to ``min_lr``.
- ``cosine``     : Cosine annealing from ``max_lr`` to ``min_lr``.
- ``autonomous`` : No limit — the model decides how many edits to make.

Usage::

    scheduler = build_scheduler(cfg)
    for step in range(1, total_steps + 1):
        lr = scheduler.step()       # returns edit budget for this step
        # ... use lr as max_edits ...
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod


class LRScheduler(ABC):
    """Base class for edit-budget schedulers."""

    def __init__(self, max_lr: int, min_lr: int, total_steps: int) -> None:
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.total_steps = total_steps
        self._current_step = 0

    @abstractmethod
    def _compute_lr(self, step: int) -> int:
        """Return the edit budget for the given 1-indexed step."""

    def step(self) -> int:
        """Advance one step and return the edit budget."""
        self._current_step += 1
        return self._compute_lr(self._current_step)

    def get_lr(self, step: int) -> int:
        """Return the edit budget for an arbitrary step (1-indexed)."""
        return self._compute_lr(step)

    def state_dict(self) -> dict:
        return {"current_step": self._current_step}

    def load_state_dict(self, state: dict) -> None:
        self._current_step = state.get("current_step", 0)


class ConstantScheduler(LRScheduler):
    """Fixed edit budget throughout training."""

    def _compute_lr(self, step: int) -> int:
        return self.max_lr


class LinearScheduler(LRScheduler):
    """Linear decay from ``max_lr`` to ``min_lr`` over ``total_steps``."""

    def _compute_lr(self, step: int) -> int:
        if self.total_steps <= 1:
            return self.max_lr
        t = min(step, self.total_steps) / self.total_steps
        lr = self.max_lr + (self.min_lr - self.max_lr) * t
        return max(self.min_lr, round(lr))


class CosineScheduler(LRScheduler):
    """Cosine annealing from ``max_lr`` to ``min_lr`` over ``total_steps``."""

    def _compute_lr(self, step: int) -> int:
        if self.total_steps <= 1:
            return self.max_lr
        t = min(step, self.total_steps) / self.total_steps
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * t))
        return max(self.min_lr, round(lr))


class AutonomousScheduler(LRScheduler):
    """No edit limit — the model decides freely."""

    NO_LIMIT = 999

    def _compute_lr(self, step: int) -> int:
        return self.NO_LIMIT


# ── Factory ──────────────────────────────────────────────────────────────

_REGISTRY: dict[str, type[LRScheduler]] = {
    "constant": ConstantScheduler,
    "linear": LinearScheduler,
    "cosine": CosineScheduler,
    "autonomous": AutonomousScheduler,
}


def build_scheduler(
    mode: str = "constant",
    max_lr: int = 8,
    min_lr: int = 2,
    total_steps: int = 8,
) -> LRScheduler:
    """Build a scheduler from config parameters.

    Parameters
    ----------
    mode : str
        One of ``constant``, ``linear``, ``cosine``, ``autonomous``.
    max_lr : int
        Initial / maximum edit budget.
    min_lr : int
        Minimum edit budget (for decay modes).
    total_steps : int
        Total number of optimization steps in training.
    """
    if mode not in _REGISTRY:
        raise ValueError(
            f"Unknown scheduler mode '{mode}'. Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[mode](max_lr=max_lr, min_lr=min_lr, total_steps=total_steps)
