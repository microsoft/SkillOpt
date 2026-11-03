"""Tests for skillopt_sleep.gate — the vendored validation gate.

Verifies standalone behavior and exact parity with skillopt.evaluation.gate.
"""
from __future__ import annotations

import pytest

import skillopt.evaluation.gate as ref_gate
import skillopt_sleep.gate as sleep_gate


class TestSleepGateSelectScore:
    """select_gate_score in skillopt_sleep.gate."""

    def test_hard_metric_returns_hard(self) -> None:
        assert sleep_gate.select_gate_score(0.8, 0.3, "hard") == 0.8

    def test_soft_metric_returns_soft(self) -> None:
        assert sleep_gate.select_gate_score(0.8, 0.3, "soft") == 0.3

    def test_default_metric_is_hard(self) -> None:
        assert sleep_gate.select_gate_score(0.42, 0.99) == 0.42

    def test_mixed_metric_default_weight(self) -> None:
        assert sleep_gate.select_gate_score(1.0, 0.0, "mixed") == pytest.approx(0.5)

    def test_mixed_metric_custom_weight(self) -> None:
        assert sleep_gate.select_gate_score(0.8, 0.4, "mixed", 0.25) == pytest.approx(0.7)

    def test_mixed_weight_zero_equals_hard(self) -> None:
        assert sleep_gate.select_gate_score(0.8, 0.3, "mixed", 0.0) == pytest.approx(0.8)

    def test_mixed_weight_one_equals_soft(self) -> None:
        assert sleep_gate.select_gate_score(0.8, 0.3, "mixed", 1.0) == pytest.approx(0.3)

    def test_mixed_weight_clamped_above_one(self) -> None:
        assert sleep_gate.select_gate_score(0.8, 0.3, "mixed", 5.0) == pytest.approx(0.3)

    def test_mixed_weight_clamped_below_zero(self) -> None:
        assert sleep_gate.select_gate_score(0.8, 0.3, "mixed", -2.0) == pytest.approx(0.8)

    def test_metric_case_insensitive(self) -> None:
        assert sleep_gate.select_gate_score(0.8, 0.3, "Hard") == 0.8
        assert sleep_gate.select_gate_score(0.8, 0.3, "SOFT") == 0.3
        assert sleep_gate.select_gate_score(1.0, 0.0, "Mixed") == pytest.approx(0.5)

    def test_metric_whitespace_tolerant(self) -> None:
        assert sleep_gate.select_gate_score(0.8, 0.3, "  hard  ") == 0.8
        assert sleep_gate.select_gate_score(0.8, 0.3, "\tmixed\n") == pytest.approx(0.55)

    def test_none_metric_defaults_to_hard(self) -> None:
        assert sleep_gate.select_gate_score(0.8, 0.3, None) == 0.8

    def test_unknown_metric_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown gate metric"):
            sleep_gate.select_gate_score(0.5, 0.5, "rouge")


class TestSleepGateEvaluate:
    """evaluate_gate in skillopt_sleep.gate."""

    def test_accept_new_best(self) -> None:
        result = sleep_gate.evaluate_gate(
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
        assert result.current_score == 0.9
        assert result.best_skill == "CAND"
        assert result.best_score == 0.9
        assert result.best_step == 7

    def test_accept_not_best(self) -> None:
        result = sleep_gate.evaluate_gate(
            candidate_skill="CAND",
            cand_hard=0.7,
            current_skill="CURR",
            current_score=0.5,
            best_skill="BEST",
            best_score=0.8,
            best_step=3,
            global_step=7,
        )
        assert result.action == "accept"
        assert result.current_skill == "CAND"
        assert result.current_score == 0.7
        assert result.best_skill == "BEST"
        assert result.best_score == 0.8
        assert result.best_step == 3

    def test_reject(self) -> None:
        result = sleep_gate.evaluate_gate(
            candidate_skill="CAND",
            cand_hard=0.4,
            current_skill="CURR",
            current_score=0.5,
            best_skill="BEST",
            best_score=0.8,
            best_step=3,
            global_step=7,
        )
        assert result.action == "reject"
        assert result.current_skill == "CURR"
        assert result.current_score == 0.5
        assert result.best_skill == "BEST"
        assert result.best_score == 0.8
        assert result.best_step == 3

    def test_case_and_whitespace_tolerant_metric(self) -> None:
        result = sleep_gate.evaluate_gate(
            candidate_skill="CAND",
            cand_hard=0.6,
            current_skill="CURR",
            current_score=0.7,
            best_skill="BEST",
            best_score=0.7,
            best_step=0,
            global_step=1,
            cand_soft=0.9,
            metric="  Mixed  ",
            mixed_weight=0.5,
        )
        assert result.action == "accept_new_best"
        assert result.current_score == pytest.approx(0.75)


class TestSleepGateBehavioralParity:
    """Ensure skillopt_sleep.gate matches skillopt.evaluation.gate output."""

    @pytest.mark.parametrize(
        "hard,soft,metric,weight",
        [
            (0.8, 0.3, "hard", 0.5),
            (0.8, 0.3, "soft", 0.5),
            (1.0, 0.0, "mixed", 0.5),
            (0.8, 0.4, "mixed", 0.25),
            (0.8, 0.3, "Hard", 0.5),
            (0.8, 0.3, "SOFT", 0.5),
            (1.0, 0.0, "Mixed", 0.5),
            (0.8, 0.3, "  hard  ", 0.5),
            (0.8, 0.3, "\tmixed\n", 0.5),
            (0.8, 0.3, None, 0.5),
            (0.8, 0.3, "mixed", -2.0),
            (0.8, 0.3, "mixed", 5.0),
        ],
    )
    def test_select_gate_score_parity(
        self, hard: float, soft: float, metric: str | None, weight: float
    ) -> None:
        ref_score = ref_gate.select_gate_score(hard, soft, metric, weight)
        sleep_score = sleep_gate.select_gate_score(hard, soft, metric, weight)
        assert sleep_score == pytest.approx(ref_score)

    def test_evaluate_gate_parity(self) -> None:
        for metric in ["hard", "soft", "mixed", "Hard", "  Mixed  "]:
            ref_res = ref_gate.evaluate_gate(
                candidate_skill="CAND",
                cand_hard=0.7,
                current_skill="CURR",
                current_score=0.5,
                best_skill="BEST",
                best_score=0.8,
                best_step=2,
                global_step=5,
                cand_soft=0.9,
                metric=metric,
                mixed_weight=0.4,
            )
            sleep_res = sleep_gate.evaluate_gate(
                candidate_skill="CAND",
                cand_hard=0.7,
                current_skill="CURR",
                current_score=0.5,
                best_skill="BEST",
                best_score=0.8,
                best_step=2,
                global_step=5,
                cand_soft=0.9,
                metric=metric,
                mixed_weight=0.4,
            )
            assert sleep_res.action == ref_res.action
            assert sleep_res.current_skill == ref_res.current_skill
            assert sleep_res.current_score == pytest.approx(ref_res.current_score)
            assert sleep_res.best_skill == ref_res.best_skill
            assert sleep_res.best_score == pytest.approx(ref_res.best_score)
            assert sleep_res.best_step == ref_res.best_step
