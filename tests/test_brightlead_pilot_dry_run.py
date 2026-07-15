"""BrightLead-local pilot dry-run command coverage."""
from __future__ import annotations

import os
import subprocess
import tempfile
import unittest


REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PILOT_DRY_RUN = os.path.join(REPO, "bin", "brightlead-skillopt-pilot-dry-run")
DISPOSABLE_ADOPTION_TEST = os.path.join(REPO, "bin", "brightlead-skillopt-disposable-adoption-test")


class TestBrightLeadPilotDryRun(unittest.TestCase):
    def test_pilot_dry_run_writes_report_without_adoption(self):
        self.assertTrue(os.path.exists(PILOT_DRY_RUN))
        self.assertTrue(os.access(PILOT_DRY_RUN, os.X_OK))

        env = {**os.environ, "PYTHONNOUSERSITE": "1"}
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                [PILOT_DRY_RUN, tmp],
                cwd=REPO,
                env=env,
                capture_output=True,
                text=True,
                timeout=240,
            )

            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertIn("running preflight", proc.stdout)
            self.assertIn("running reviewed mock dry run", proc.stdout)
            self.assertIn("brightlead-skillopt-pilot-dry-run.md", proc.stdout)

            report_path = os.path.join(tmp, "brightlead-skillopt-pilot-dry-run.md")
            json_path = os.path.join(tmp, "dry-run.json")
            tasks_path = os.path.join(tmp, "reviewed-wrap-answer-tasks.json")
            self.assertTrue(os.path.exists(report_path))
            self.assertTrue(os.path.exists(json_path))
            self.assertTrue(os.path.exists(tasks_path))

            with open(report_path, encoding="utf-8") as f:
                report = f.read()

            self.assertIn("status: PASS", report)
            self.assertIn("baseline: 0.0", report)
            self.assertIn("candidate: 1.0", report)
            self.assertIn("adopted: False", report)
            self.assertIn("staging_dir: ``", report)
            self.assertIn("tasks_reviewed: True", report)
            self.assertIn("n_sessions: 0", report)
            self.assertIn("does not adopt skill edits", report)
            self.assertIn("harvest private transcripts", report)
            self.assertIn("touch WordPress", report)

    def test_brightlead_qa_fixture_writes_repeatable_report(self):
        self.assertTrue(os.path.exists(PILOT_DRY_RUN))

        env = {**os.environ, "PYTHONNOUSERSITE": "1"}
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                [PILOT_DRY_RUN, "--fixture", "brightlead-qa", tmp],
                cwd=REPO,
                env=env,
                capture_output=True,
                text=True,
                timeout=240,
            )

            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertIn("running preflight", proc.stdout)
            self.assertIn("running reviewed mock dry run", proc.stdout)

            report_path = os.path.join(tmp, "brightlead-skillopt-pilot-dry-run.md")
            json_path = os.path.join(tmp, "dry-run.json")
            tasks_path = os.path.join(tmp, "reviewed-brightlead-qa-tasks.json")
            self.assertTrue(os.path.exists(report_path))
            self.assertTrue(os.path.exists(json_path))
            self.assertTrue(os.path.exists(tasks_path))

            with open(report_path, encoding="utf-8") as f:
                report = f.read()

            self.assertIn("status: PASS", report)
            self.assertIn("fixture: `brightlead-qa`", report)
            self.assertIn("baseline: 0.4", report)
            self.assertIn("candidate: 1.0", report)
            self.assertIn("adopted: False", report)
            self.assertIn("staging_dir: ``", report)
            self.assertIn("tasks_reviewed: True", report)
            self.assertIn("n_sessions: 0", report)
            self.assertIn("reviewed-brightlead-qa-tasks.json", report)
            self.assertIn("does not adopt skill edits", report)

    def test_brightlead_known_gap_fixture_writes_repeatable_report(self):
        self.assertTrue(os.path.exists(PILOT_DRY_RUN))

        env = {**os.environ, "PYTHONNOUSERSITE": "1"}
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                [PILOT_DRY_RUN, "--fixture", "brightlead-known-gap", tmp],
                cwd=REPO,
                env=env,
                capture_output=True,
                text=True,
                timeout=240,
            )

            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertIn("running preflight", proc.stdout)
            self.assertIn("running reviewed mock dry run", proc.stdout)

            report_path = os.path.join(tmp, "brightlead-skillopt-pilot-dry-run.md")
            json_path = os.path.join(tmp, "dry-run.json")
            tasks_path = os.path.join(tmp, "reviewed-brightlead-known-gap-tasks.json")
            self.assertTrue(os.path.exists(report_path))
            self.assertTrue(os.path.exists(json_path))
            self.assertTrue(os.path.exists(tasks_path))

            with open(report_path, encoding="utf-8") as f:
                report = f.read()

            self.assertIn("status: PASS", report)
            self.assertIn("fixture: `brightlead-known-gap`", report)
            self.assertIn("baseline: 0.5", report)
            self.assertIn("candidate: 1.0", report)
            self.assertIn("adopted: False", report)
            self.assertIn("staging_dir: ``", report)
            self.assertIn("tasks_reviewed: True", report)
            self.assertIn("n_sessions: 0", report)
            self.assertIn("reviewed-brightlead-known-gap-tasks.json", report)
            self.assertIn("Always include SI units in numeric answers.", report)

    def test_disposable_adoption_test_adopts_only_generated_skill(self):
        self.assertTrue(os.path.exists(DISPOSABLE_ADOPTION_TEST))
        self.assertTrue(os.access(DISPOSABLE_ADOPTION_TEST, os.X_OK))

        env = {**os.environ, "PYTHONNOUSERSITE": "1"}
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                [DISPOSABLE_ADOPTION_TEST, tmp],
                cwd=REPO,
                env=env,
                capture_output=True,
                text=True,
                timeout=240,
            )

            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertIn("running mock auto-adopt on disposable skill", proc.stdout)

            report_path = os.path.join(tmp, "brightlead-skillopt-disposable-adoption-test.md")
            json_path = os.path.join(tmp, "auto-adopt-run.json")
            target_skill = os.path.join(
                tmp, "disposable-project", "skills", "disposable-qa", "SKILL.md"
            )
            self.assertTrue(os.path.exists(report_path))
            self.assertTrue(os.path.exists(json_path))
            self.assertTrue(os.path.exists(target_skill))

            with open(report_path, encoding="utf-8") as f:
                report = f.read()
            with open(target_skill, encoding="utf-8") as f:
                skill = f.read()

            self.assertIn("status: PASS", report)
            self.assertIn("adopted: True", report)
            self.assertIn("candidate: 1.0", report)
            self.assertIn("n_sessions: 0", report)
            self.assertIn("tasks_reviewed: True", report)
            self.assertIn("adopted_rule_present: True", report)
            self.assertIn("disposable-project/.skillopt-sleep/staging", report)
            self.assertIn("does not target any live BrightLead skill", report)
            self.assertIn("Always include SI units in numeric answers.", skill)


if __name__ == "__main__":
    unittest.main()
