from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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

    def test_apply_learned_rules_replaces_block_in_place_when_tail_follows(self) -> None:
        doc = (
            "# Skill\n\n"
            "Manual intro.\n\n"
            f"{LEARNED_START}\n"
            "## Learned AIForAI Rules\n\n"
            "_Old banner._\n\n"
            "- Old rule\n"
            f"{LEARNED_END}\n\n"
            "Handwritten tail.\n"
        )

        updated = apply_learned_rules(doc, ["New rule"])

        self.assertIn("Manual intro.", updated)
        self.assertIn("Handwritten tail.", updated)
        self.assertLess(updated.index("Manual intro."), updated.index(LEARNED_START))
        self.assertLess(updated.index(LEARNED_END), updated.index("Handwritten tail."))
        self.assertNotIn("- Old rule", updated)
        self.assertIn("- New rule", updated)
        self.assertEqual(updated.count(LEARNED_START), 1)
        self.assertEqual(updated.count(LEARNED_END), 1)

    def test_malformed_markers_preserve_handwritten_content_and_are_ignored(self) -> None:
        cases = [
            (
                "misordered_markers",
                "# Skill\n\n"
                "Manual intro.\n"
                f"{LEARNED_END}\n"
                "Handwritten tail.\n"
                f"{LEARNED_START}\n"
                "- Ghost rule\n",
            ),
            (
                "unmatched_start",
                "# Skill\n\n"
                "Manual intro.\n"
                f"{LEARNED_START}\n"
                "- Ghost rule\n"
                "Handwritten tail.\n",
            ),
        ]

        for label, doc in cases:
            with self.subTest(case=label):
                updated = apply_learned_rules(doc, ["Recovered rule"])

                self.assertEqual(current_learned_rules(doc), [])
                self.assertIn("Manual intro.", updated)
                self.assertIn("Handwritten tail.", updated)
                self.assertIn("Ghost rule", updated)
                self.assertIn("- Recovered rule", updated)

    def test_flag_rules_survive_cleaning_and_extraction(self) -> None:
        updated = apply_learned_rules(
            "# Skill\n",
            [" --flag must be set ", "- --flag must be set", "--flag must be set"],
        )

        self.assertIn("- --flag must be set", updated)
        self.assertEqual(current_learned_rules(updated), ["--flag must be set"])

    def test_run_validators_captures_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)

            result = run_aiforai_validators(str(repo))

            self.assertFalse(result["ok"])
            self.assertIn("quick_validate", result["commands"][0]["name"])
            self.assertFalse(result["commands"][0]["ok"])

    def test_run_validators_uses_sys_executable(self) -> None:
        completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="ok\n", stderr="")

        with mock.patch(
            "skillopt_sleep.aiforai.skill_adapter.subprocess.run",
            side_effect=[completed, completed],
        ) as run_mock:
            result = run_aiforai_validators("/repo")

        self.assertTrue(result["ok"])
        self.assertEqual(run_mock.call_count, 2)
        for call in run_mock.call_args_list:
            self.assertEqual(call.args[0][0], sys.executable)

    def test_run_validators_timeout_preserves_partial_output_and_marks_failure(self) -> None:
        timeout_error = subprocess.TimeoutExpired(
            cmd=[sys.executable, "scripts/quick_validate.py", "ai-model-rd-protocol"],
            timeout=5,
            output="partial stdout\n",
            stderr="partial stderr\n",
        )
        completed = subprocess.CompletedProcess(args=[], returncode=0, stdout="ok\n", stderr="")

        with mock.patch(
            "skillopt_sleep.aiforai.skill_adapter.subprocess.run",
            side_effect=[timeout_error, completed],
        ):
            result = run_aiforai_validators("/repo", timeout=5)

        first = result["commands"][0]
        second = result["commands"][1]

        self.assertFalse(result["ok"])
        self.assertFalse(first["ok"])
        self.assertEqual(first["status"], "failed")
        self.assertEqual(first["failure_type"], "timeout")
        self.assertIn("partial stdout", first["output"])
        self.assertIn("partial stderr", first["output"])
        self.assertTrue(second["ok"])


if __name__ == "__main__":
    unittest.main()
