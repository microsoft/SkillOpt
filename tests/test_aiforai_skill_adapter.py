from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from skillopt_sleep.aiforai.skill_adapter import (
    LEARNED_END,
    LEARNED_START,
    apply_learned_rules,
    current_learned_rules,
    run_aiforai_validators,
)


class AiforaiSkillAdapterTests(unittest.TestCase):
    def test_apply_learned_rules_preserves_handwritten_content(self) -> None:
        base = "---\nname: ai-model-rd-protocol\n---\n\n# Skill\n\nKeep this doctrine.\n"

        updated = apply_learned_rules(base, ["Rule A", "Rule A", " Rule B "])

        self.assertIn("Keep this doctrine.", updated)
        self.assertIn(LEARNED_START, updated)
        self.assertIn(LEARNED_END, updated)
        self.assertEqual(updated.count("- Rule A"), 1)
        self.assertEqual(current_learned_rules(updated), ["Rule A", "Rule B"])

    def test_apply_learned_rules_replaces_only_protected_block(self) -> None:
        doc = apply_learned_rules("# Skill\n\nManual line.\nTrailing note.\n", ["Old", "Legacy"])

        updated = apply_learned_rules(doc, ["New"])

        self.assertIn("Manual line.", updated)
        self.assertIn("Trailing note.", updated)
        self.assertNotIn("- Old", updated)
        self.assertNotIn("- Legacy", updated)
        self.assertIn("- New", updated)
        self.assertEqual(updated.count(LEARNED_START), 1)
        self.assertEqual(updated.count(LEARNED_END), 1)

    def test_run_validators_captures_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)

            result = run_aiforai_validators(str(repo))

            self.assertFalse(result["ok"])
            self.assertIn("quick_validate", result["commands"][0]["name"])
            self.assertFalse(result["commands"][0]["ok"])


if __name__ == "__main__":
    unittest.main()
