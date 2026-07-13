"""Tests for skillopt.optimizer.scheduler — edit-budget schedulers.

ReflACT trainers use an edit-budget scheduler at each optimisation step to
control how many skill edits are allowed (analogous to gradient clipping /
learning-rate annealing in neural-network training).

This module has zero LLM dependencies — all behaviour is deterministic pure
math — making it an ideal target for precise unit tests.

Scheduler contract
------------------
Decay schedulers (Linear, Cosine) guarantee:
- First ``step()`` returns ``max_lr``.
- ``total_steps``-th ``step()`` returns ``min_lr``.
- Intermediate values are monotonically non-increasing and clamped to
  ``[min_lr, max_lr]``.
- Beyond ``total_steps`` the value plateaus at ``min_lr``.
"""
from __future__ import annotations

import pytest

from skillopt.optimizer.scheduler import (
    LRScheduler,
    ConstantScheduler,
    LinearScheduler,
    CosineScheduler,
    AutonomousScheduler,
    build_scheduler,
)


# ── ConstantScheduler ────────────────────────────────────────────────────────


class TestConstantScheduler:
    """ConstantScheduler — fixed edit budget regardless of step."""

    def test_always_returns_max_lr(self) -> None:
        s = ConstantScheduler(max_lr=8, min_lr=2, total_steps=10)
        for _ in range(10):
            assert s.step() == 8

    def test_step_advances_internal_counter(self) -> None:
        s = ConstantScheduler(max_lr=5, min_lr=1, total_steps=5)
        assert s._current_step == 0
        s.step()
        assert s._current_step == 1
        s.step()
        assert s._current_step == 2

    def test_get_lr_returns_max_for_arbitrary_step(self) -> None:
        s = ConstantScheduler(max_lr=12, min_lr=1, total_steps=100)
        assert s.get_lr(1) == 12
        assert s.get_lr(50) == 12
        assert s.get_lr(999) == 12

    def test_state_dict_and_load_state_dict_round_trip(self) -> None:
        s = ConstantScheduler(max_lr=8, min_lr=2, total_steps=10)
        for _ in range(3):
            s.step()
        assert s._current_step == 3

        state = s.state_dict()
        s2 = ConstantScheduler(max_lr=8, min_lr=2, total_steps=10)
        s2.load_state_dict(state)
        assert s2._current_step == 3
        # Step after resume lands on the correct step
        assert s2.step() == 8
        assert s2._current_step == 4

    def test_load_state_dict_with_missing_key_defaults_to_zero(self) -> None:
        s = ConstantScheduler(max_lr=8, min_lr=2, total_steps=10)
        s.load_state_dict({})
        assert s._current_step == 0


# ── LinearScheduler ──────────────────────────────────────────────────────────


class TestLinearScheduler:
    """LinearScheduler — linear decay from max_lr to min_lr."""

    def test_first_step_returns_max_lr(self) -> None:
        s = LinearScheduler(max_lr=10, min_lr=2, total_steps=10)
        assert s.step() == 10

    def test_last_step_returns_min_lr(self) -> None:
        s = LinearScheduler(max_lr=10, min_lr=2, total_steps=10)
        for _ in range(9):
            s.step()
        assert s.step() == 2

    def test_all_steps_return_integers(self) -> None:
        s = LinearScheduler(max_lr=8, min_lr=2, total_steps=6)
        for _ in range(6):
            lr = s.step()
            assert isinstance(lr, int)

    def test_total_steps_one_returns_max_lr(self) -> None:
        s = LinearScheduler(max_lr=10, min_lr=2, total_steps=1)
        assert s.step() == 10

    def test_total_steps_zero_returns_max_lr(self) -> None:
        """Degenerate case: 0-step training still gets max_lr on the one call."""
        s = LinearScheduler(max_lr=10, min_lr=2, total_steps=0)
        assert s.step() == 10

    def test_monotonically_non_increasing(self) -> None:
        s = LinearScheduler(max_lr=20, min_lr=2, total_steps=100)
        prev: int = 999
        for _ in range(100):
            lr = s.step()
            assert lr <= prev
            prev = lr

    def test_never_below_min_lr(self) -> None:
        s = LinearScheduler(max_lr=10, min_lr=2, total_steps=10)
        for _ in range(20):  # overshoot
            assert s.step() >= 2

    def test_never_above_max_lr(self) -> None:
        s = LinearScheduler(max_lr=10, min_lr=2, total_steps=10)
        for _ in range(20):
            assert s.step() <= 10

    def test_after_total_steps_stays_at_min_lr(self) -> None:
        s = LinearScheduler(max_lr=8, min_lr=2, total_steps=5)
        for _ in range(5):
            s.step()
        # Steps beyond total_steps should plateau at min_lr
        for _ in range(5):
            assert s.step() == 2

    def test_known_decay_sequence(self) -> None:
        """Linear decay max_lr=10, min_lr=2, total_steps=4.
           t = (step-1)/(total_steps-1) = 0, 1/3, 2/3, 1
           lr = 10 + (2-10)*t = 10 - 8t
           t=0: lr=10, t=1/3: lr≈7.33→7, t=2/3: lr≈4.67→5, t=1: lr=2
        """
        s = LinearScheduler(max_lr=10, min_lr=2, total_steps=4)
        assert s.step() == 10
        assert s.step() == 7
        assert s.step() == 5
        assert s.step() == 2

    def test_step_state_dict_resume_consistent(self) -> None:
        """After resume, the next step value is the same as without resume."""
        s1 = LinearScheduler(max_lr=10, min_lr=2, total_steps=5)
        for _ in range(3):
            s1.step()
        resumed_lr = s1.step()  # step 4

        s2 = LinearScheduler(max_lr=10, min_lr=2, total_steps=5)
        s2.load_state_dict({"current_step": 3})
        assert s2.step() == resumed_lr

    def test_max_lr_equals_min_lr_yields_constant(self) -> None:
        s = LinearScheduler(max_lr=5, min_lr=5, total_steps=10)
        for _ in range(10):
            assert s.step() == 5


# ── CosineScheduler ──────────────────────────────────────────────────────────


class TestCosineScheduler:
    """CosineScheduler — cosine annealing from max_lr to min_lr."""

    def test_first_step_returns_max_lr(self) -> None:
        s = CosineScheduler(max_lr=10, min_lr=2, total_steps=10)
        assert s.step() == 10

    def test_last_step_returns_min_lr(self) -> None:
        s = CosineScheduler(max_lr=10, min_lr=2, total_steps=10)
        for _ in range(9):
            s.step()
        assert s.step() == 2

    def test_all_steps_return_integers(self) -> None:
        s = CosineScheduler(max_lr=8, min_lr=2, total_steps=6)
        for _ in range(6):
            lr = s.step()
            assert isinstance(lr, int)

    def test_total_steps_one_returns_max_lr(self) -> None:
        s = CosineScheduler(max_lr=10, min_lr=2, total_steps=1)
        assert s.step() == 10

    def test_total_steps_zero_returns_max_lr(self) -> None:
        s = CosineScheduler(max_lr=10, min_lr=2, total_steps=0)
        assert s.step() == 10

    def test_monotonically_non_increasing(self) -> None:
        s = CosineScheduler(max_lr=20, min_lr=2, total_steps=100)
        prev: int = 999
        for _ in range(100):
            lr = s.step()
            assert lr <= prev
            prev = lr

    def test_never_below_min_lr(self) -> None:
        s = CosineScheduler(max_lr=10, min_lr=2, total_steps=10)
        for _ in range(20):
            assert s.step() >= 2

    def test_never_above_max_lr(self) -> None:
        s = CosineScheduler(max_lr=10, min_lr=2, total_steps=10)
        for _ in range(20):
            assert s.step() <= 10

    def test_after_total_steps_stays_at_min_lr(self) -> None:
        s = CosineScheduler(max_lr=8, min_lr=2, total_steps=5)
        for _ in range(5):
            s.step()
        for _ in range(5):
            assert s.step() == 2

    def test_midpoint_close_to_mean(self) -> None:
        """At the half-way neighbourhood, cosine is close to (max+min)/2.
           total_steps=100, step=50 → t=49/99≈0.495.
           cos(0.495π)≈0, lr ≈ (20+2)/2 = 11.
        """
        s = CosineScheduler(max_lr=20, min_lr=2, total_steps=100)
        for _ in range(49):
            s.step()
        mid = s.step()  # step 50
        assert mid == 11

    def test_step_state_dict_resume_consistent(self) -> None:
        s1 = CosineScheduler(max_lr=10, min_lr=2, total_steps=5)
        for _ in range(3):
            s1.step()
        resumed_lr = s1.step()

        s2 = CosineScheduler(max_lr=10, min_lr=2, total_steps=5)
        s2.load_state_dict({"current_step": 3})
        assert s2.step() == resumed_lr

    def test_max_lr_equals_min_lr_yields_constant(self) -> None:
        s = CosineScheduler(max_lr=5, min_lr=5, total_steps=10)
        for _ in range(10):
            assert s.step() == 5

    def test_early_steps_near_max(self) -> None:
        """Cosine annealing stays near max_lr early on (cos(0)=1)."""
        s = CosineScheduler(max_lr=100, min_lr=0, total_steps=100)
        # step 1: t=0, cos(0)=1 → lr=100
        assert s.step() == 100
        # step 2: t=1/99≈0.01, cos≈0.9995 → lr≈99.97 → 100
        assert s.step() == 100

    def test_late_steps_near_min(self) -> None:
        """Cosine annealing flattens near min_lr at the end (cos(π)=-1)."""
        s = CosineScheduler(max_lr=100, min_lr=0, total_steps=100)
        for _ in range(99):
            s.step()
        # step 100: t=1, cos(π)=-1 → lr=0
        assert s.step() == 0


# ── AutonomousScheduler ──────────────────────────────────────────────────────


class TestAutonomousScheduler:
    """AutonomousScheduler — no edit limit (model decides freely)."""

    def test_always_returns_no_limit(self) -> None:
        s = AutonomousScheduler(max_lr=8, min_lr=2, total_steps=10)
        for _ in range(20):
            assert s.step() == AutonomousScheduler.NO_LIMIT

    def test_step_advances_counter(self) -> None:
        s = AutonomousScheduler(max_lr=5, min_lr=1, total_steps=5)
        assert s._current_step == 0
        s.step()
        assert s._current_step == 1

    def test_get_lr_returns_no_limit(self) -> None:
        s = AutonomousScheduler(max_lr=5, min_lr=1, total_steps=10)
        assert s.get_lr(1) == AutonomousScheduler.NO_LIMIT
        assert s.get_lr(50) == AutonomousScheduler.NO_LIMIT

    def test_state_dict_round_trip(self) -> None:
        s = AutonomousScheduler(max_lr=5, min_lr=1, total_steps=10)
        for _ in range(4):
            s.step()
        s2 = AutonomousScheduler(max_lr=5, min_lr=1, total_steps=10)
        s2.load_state_dict(s.state_dict())
        assert s2._current_step == 4


# ── build_scheduler factory ──────────────────────────────────────────────────


class TestBuildScheduler:
    """build_scheduler factory — creates the right scheduler from a mode name."""

    def test_constant(self) -> None:
        s = build_scheduler("constant", max_lr=8, min_lr=2, total_steps=10)
        assert isinstance(s, ConstantScheduler)
        assert s.max_lr == 8
        assert s.min_lr == 2
        assert s.total_steps == 10

    def test_linear(self) -> None:
        s = build_scheduler("linear", max_lr=12, min_lr=3, total_steps=20)
        assert isinstance(s, LinearScheduler)

    def test_cosine(self) -> None:
        s = build_scheduler("cosine", max_lr=16, min_lr=4, total_steps=30)
        assert isinstance(s, CosineScheduler)

    def test_autonomous(self) -> None:
        s = build_scheduler("autonomous", max_lr=8, min_lr=2, total_steps=10)
        assert isinstance(s, AutonomousScheduler)

    def test_unknown_mode_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown scheduler mode"):
            build_scheduler("exponential", max_lr=8, min_lr=2, total_steps=10)

    def test_default_mode_is_constant(self) -> None:
        s = build_scheduler(max_lr=8, min_lr=2, total_steps=10)
        assert isinstance(s, ConstantScheduler)


# ── Abstract base class ──────────────────────────────────────────────────────


class TestLRSchedulerBase:
    """LRScheduler — abstract base: cannot be instantiated directly."""

    def test_cannot_instantiate_abstract(self) -> None:
        with pytest.raises(TypeError):
            LRScheduler(max_lr=8, min_lr=2, total_steps=10)  # type: ignore[abstract]

    def test_concrete_subclass_instantiates_fine(self) -> None:
        """ConstantScheduler (and others) work normally."""
        s = ConstantScheduler(max_lr=8, min_lr=2, total_steps=10)
        assert isinstance(s, LRScheduler)
        assert s.step() == 8
