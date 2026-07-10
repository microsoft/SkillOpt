"""Tests for skillopt.evaluation.gate — the validation gate decision function.

The gate is the optimizer's model-selection / early-stopping core: given a
candidate skill's score, it decides whether to accept it as the new current
skill and whether it becomes the new best-so-far. These are pure functions,
so they can be exercised directly without any LLM or rollout.
"""
from __future__ import annotations

import dataclasses

import pytest

from skillopt.evaluation.gate import (
    GateResult,
    evaluate_gate,
    select_gate_score,
)


class TestSelectGateScore:
    """select_gate_score — project (hard, soft) onto a single comparison metric."""

    def test_hard_metric_returns_hard(self) -> None:
        assert select_gate_score(0.8, 0.3, "hard") == 0.8

    def test_soft_metric_returns_soft(self) -> None:
        assert select_gate_score(0.8, 0.3, "soft") == 0.3

    def test_default_metric_is_hard(self) -> None:
        assert select_gate_score(0.42, 0.99) == 0.42

    def test_mixed_metric_default_weight(self) -> None:
        # (1 - 0.5) * 1.0 + 0.5 * 0.0 == 0.5
        assert select_gate_score(1.0, 0.0, "mixed") == pytest.approx(0.5)

    def test_mixed_metric_custom_weight(self) -> None:
        # (1 - 0.25) * 0.8 + 0.25 * 0.4 == 0.7
        assert select_gate_score(0.8, 0.4, "mixed", 0.25) == pytest.approx(0.7)

    def test_mixed_weight_zero_equals_hard(self) -> None:
        assert select_gate_score(0.8, 0.3, "mixed", 0.0) == pytest.approx(0.8)

    def test_mixed_weight_one_equals_soft(self) -> None:
        assert select_gate_score(0.8, 0.3, "mixed", 1.0) == pytest.approx(0.3)

    def test_mixed_weight_clamped_above_one(self) -> None:
        """Out-of-range weight is clamped to 1.0 (→ pure soft)."""
        assert select_gate_score(0.8, 0.3, "mixed", 5.0) == pytest.approx(0.3)

    def test_mixed_weight_clamped_below_zero(self) -> None:
        """Negative weight is clamped to 0.0 (→ pure hard)."""
        assert select_gate_score(0.8, 0.3, "mixed", -2.0) == pytest.approx(0.8)

    def test_returns_float(self) -> None:
        assert isinstance(select_gate_score(1, 0, "hard"), float)

    def test_unknown_metric_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown gate metric"):
            select_gate_score(0.5, 0.5, "rouge")  # type: ignore[arg-type]


class TestEvaluateGateAcceptNewBest:
    """evaluate_gate — candidate beats both current and best."""

    def test_accept_new_best_action_and_state(self) -> None:
        result = evaluate_gate(
            candidate_skill="CAND",
            cand_hard=0.9,
            current_skill="CURR",
            current_score=0.5,
            best_skill="BEST",
            best_score=0.5,
            best_step=3,
            global_step=7,
        )
        assert result.action == "accept_new_best"
        assert result.current_skill == "CAND"
        assert result.current_score == pytest.approx(0.9)
        assert result.best_skill == "CAND"
        assert result.best_score == pytest.approx(0.9)
        assert result.best_step == 7  # updated to the accepting step


class TestEvaluateGateAccept:
    """evaluate_gate — candidate beats current but not best.

    This branch is only reachable when ``current_score < best_score``; it
    advances the current skill without disturbing the best-so-far checkpoint.
    """

    def test_accept_updates_current_only(self) -> None:
        result = evaluate_gate(
            candidate_skill="CAND",
            cand_hard=0.6,
            current_skill="CURR",
            current_score=0.4,
            best_skill="BEST",
            best_score=0.8,
            best_step=2,
            global_step=9,
        )
        assert result.action == "accept"
        assert result.current_skill == "CAND"
        assert result.current_score == pytest.approx(0.6)
        # best-so-far is preserved, including its step
        assert result.best_skill == "BEST"
        assert result.best_score == pytest.approx(0.8)
        assert result.best_step == 2

    def test_tie_with_best_but_above_current_accepts(self) -> None:
        """cand == best (not strictly greater) but > current → accept, not new best."""
        result = evaluate_gate(
            candidate_skill="CAND",
            cand_hard=0.8,
            current_skill="CURR",
            current_score=0.5,
            best_skill="BEST",
            best_score=0.8,
            best_step=1,
            global_step=4,
        )
        assert result.action == "accept"
        assert result.current_skill == "CAND"
        assert result.best_skill == "BEST"
        assert result.best_score == pytest.approx(0.8)
        assert result.best_step == 1


class TestEvaluateGateReject:
    """evaluate_gate — candidate does not beat current."""

    def test_reject_below_current(self) -> None:
        result = evaluate_gate(
            candidate_skill="CAND",
            cand_hard=0.3,
            current_skill="CURR",
            current_score=0.5,
            best_skill="BEST",
            best_score=0.8,
            best_step=2,
            global_step=6,
        )
        assert result.action == "reject"
        assert result.current_skill == "CURR"
        assert result.current_score == pytest.approx(0.5)
        assert result.best_skill == "BEST"
        assert result.best_score == pytest.approx(0.8)
        assert result.best_step == 2

    def test_tie_with_current_rejects(self) -> None:
        """Strict inequality: cand == current is rejected (no lateral moves)."""
        result = evaluate_gate(
            candidate_skill="CAND",
            cand_hard=0.5,
            current_skill="CURR",
            current_score=0.5,
            best_skill="BEST",
            best_score=0.5,
            best_step=0,
            global_step=3,
        )
        assert result.action == "reject"
        assert result.current_skill == "CURR"
        assert result.best_skill == "BEST"


class TestEvaluateGateMetrics:
    """evaluate_gate — non-hard metrics drive the comparison via cand_soft."""

    def test_soft_metric_uses_cand_soft(self) -> None:
        # High hard, low soft: under 'soft' the candidate must be rejected.
        result = evaluate_gate(
            candidate_skill="CAND",
            cand_hard=0.95,
            current_skill="CURR",
            current_score=0.5,
            best_skill="BEST",
            best_score=0.5,
            best_step=0,
            global_step=1,
            cand_soft=0.2,
            metric="soft",
        )
        assert result.action == "reject"

    def test_mixed_metric_uses_weighted_score(self) -> None:
        # mixed w=0.5: (0.5 * 1.0) + (0.5 * 0.6) == 0.8 > current 0.5 and best 0.5
        result = evaluate_gate(
            candidate_skill="CAND",
            cand_hard=1.0,
            current_skill="CURR",
            current_score=0.5,
            best_skill="BEST",
            best_score=0.5,
            best_step=0,
            global_step=2,
            cand_soft=0.6,
            metric="mixed",
            mixed_weight=0.5,
        )
        assert result.action == "accept_new_best"
        assert result.current_score == pytest.approx(0.8)
        assert result.best_score == pytest.approx(0.8)

    def test_default_metric_ignores_soft(self) -> None:
        """Default metric is 'hard'; cand_soft must not affect the decision."""
        result = evaluate_gate(
            candidate_skill="CAND",
            cand_hard=0.9,
            current_skill="CURR",
            current_score=0.5,
            best_skill="BEST",
            best_score=0.5,
            best_step=0,
            global_step=1,
            cand_soft=0.0,
        )
        assert result.action == "accept_new_best"
        assert result.current_score == pytest.approx(0.9)


class TestGateResult:
    """GateResult — immutable outcome dataclass."""

    def test_fields(self) -> None:
        result = GateResult(
            action="accept",
            current_skill="c",
            current_score=0.5,
            best_skill="b",
            best_score=0.9,
            best_step=4,
        )
        assert result.action == "accept"
        assert result.current_skill == "c"
        assert result.current_score == 0.5
        assert result.best_skill == "b"
        assert result.best_score == 0.9
        assert result.best_step == 4

    def test_is_frozen(self) -> None:
        result = GateResult(
            action="reject",
            current_skill="c",
            current_score=0.0,
            best_skill="b",
            best_score=0.0,
            best_step=0,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.current_score = 1.0  # type: ignore[misc]


class TestGateInvariants:
    """Behavioral invariants of the gate over a sequence of steps."""

    def test_current_tracks_best_from_equal_start(self) -> None:
        """When current == best at the start, every acceptance is a new best, so
        the two stay locked together and the 'accept' branch is never taken.

        This documents the trainer's ``s_cur``/``s_best`` usage: they are
        initialized equal and updated only through this gate.
        """
        current_skill, current_score = "S0", 0.2
        best_skill, best_score, best_step = "S0", 0.2, 0
        for step, cand in enumerate([0.1, 0.5, 0.4, 0.7], start=1):
            result = evaluate_gate(
                candidate_skill=f"S{step}",
                cand_hard=cand,
                current_skill=current_skill,
                current_score=current_score,
                best_skill=best_skill,
                best_score=best_score,
                best_step=best_step,
                global_step=step,
            )
            current_skill, current_score = result.current_skill, result.current_score
            best_skill = result.best_skill
            best_score = result.best_score
            best_step = result.best_step
            assert result.action in {"accept_new_best", "reject"}
            assert current_score == best_score
            assert current_skill == best_skill
        assert best_score == pytest.approx(0.7)
        assert best_step == 4
