"""Tests for semantic density heuristic in the validation gate."""
from __future__ import annotations

import unittest
from skillopt.evaluation.gate import (
    compute_semantic_density,
    select_gate_score,
    evaluate_gate,
)


class TestSemanticDensity(unittest.TestCase):
    """Test suite for semantic density scoring and gating decisions."""

    def test_compute_semantic_density_basic(self) -> None:
        """Verify basic compute_semantic_density behaviour with default words."""
        # 10 words, 2 leading words ("always", "never") -> 0.2 density
        skill = "Always check the inputs and never mix up proxy values."
        density = compute_semantic_density(skill)
        self.assertAlmostEqual(density, 0.2)

        # Empty skill should have 0 density
        self.assertEqual(compute_semantic_density(""), 0.0)
        self.assertEqual(compute_semantic_density("   \n  "), 0.0)

    def test_compute_semantic_density_custom_leading_words(self) -> None:
        """Verify compute_semantic_density with custom leading words."""
        skill = "Check the inputs carefully and resolve the equation."
        leading = ["check", "resolve"]
        # 8 words, 2 custom leading words -> 0.25 density
        density = compute_semantic_density(skill, leading_words=leading)
        self.assertAlmostEqual(density, 0.25)

    def test_compute_semantic_density_with_protected_regions(self) -> None:
        """Verify protected comments are excluded from density calculation."""
        skill = (
            "Always check inputs.\n"
            "<!-- SLOW_UPDATE_START -->\n"
            "This contains many words that should not count towards density "
            "always and never and only.\n"
            "<!-- SLOW_UPDATE_END -->\n"
            "<!-- APPENDIX_START -->\n"
            "More excluded words.\n"
            "<!-- APPENDIX_END -->\n"
        )
        # Without stripping, there would be many more words and a different density.
        # Stripped text: "Always check inputs." -> 3 words, 1 leading word ("always") -> 1/3 density
        density = compute_semantic_density(skill)
        self.assertAlmostEqual(density, 1.0 / 3.0)

    def test_select_gate_score_no_density(self) -> None:
        """Verify select_gate_score without semantic density adjustment."""
        # Default behavior: no semantic density adjustment
        score_hard = select_gate_score(0.8, 0.6, metric="hard")
        self.assertEqual(score_hard, 0.8)

        score_soft = select_gate_score(0.8, 0.6, metric="soft")
        self.assertEqual(score_soft, 0.6)

        score_mixed = select_gate_score(0.8, 0.6, metric="mixed", mixed_weight=0.5)
        self.assertAlmostEqual(score_mixed, 0.7)

    def test_select_gate_score_with_density(self) -> None:
        """Verify select_gate_score with semantic density adjustment."""
        # 10 words, 2 leading words ("always", "never") -> 0.2 density
        skill = "Always check the inputs and never mix up proxy values."
        # bonus: 0.1 (weight) * 0.2 (density) = 0.02
        score = select_gate_score(
            hard=0.8,
            soft=0.6,
            metric="hard",
            skill_content=skill,
            use_semantic_density=True,
            semantic_density_weight=0.1,
        )
        self.assertAlmostEqual(score, 0.82)

    def test_evaluate_gate_with_density_preference(self) -> None:
        """Verify evaluate_gate prefers candidates with higher semantic density."""
        # Baseline/current skill:
        # "Always do this task step by step and be very careful because errors are bad."
        # 15 words, 1 leading ("always") -> 1/15 density = ~0.0667
        current_skill = "Always do this task step by step and be very careful because errors are bad."

        # Candidate skill (shorter/more steerable):
        # "Always verify outputs. Never mix proxy values."
        # 7 words, 3 leading ("always", "verify", "never") -> 3/7 density = ~0.4286
        candidate_skill = "Always verify outputs. Never mix proxy values."

        # Both have same rollout accuracy (hard=0.8, soft=0.8)
        # Baseline/current score: 0.8 + 0.1 * (1/15) = ~0.8067
        current_score = select_gate_score(
            hard=0.8,
            soft=0.8,
            metric="hard",
            skill_content=current_skill,
            use_semantic_density=True,
            semantic_density_weight=0.1,
        )

        # Candidate score: 0.8 + 0.1 * (3/7) = ~0.8429
        # Even though accuracy is equal, the candidate should be accepted due to higher semantic density
        res = evaluate_gate(
            candidate_skill=candidate_skill,
            cand_hard=0.8,
            current_skill=current_skill,
            current_score=current_score,
            best_skill=current_skill,
            best_score=current_score,
            best_step=1,
            global_step=2,
            cand_soft=0.8,
            metric="hard",
            use_semantic_density=True,
            semantic_density_weight=0.1,
        )

        self.assertEqual(res.action, "accept_new_best")
        self.assertEqual(res.current_skill, candidate_skill)
        self.assertAlmostEqual(res.current_score, 0.8 + 0.1 * (3.0 / 7.0))


if __name__ == "__main__":
    unittest.main()
