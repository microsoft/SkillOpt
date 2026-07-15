"""BrightLead-local pilot dry-run command coverage."""
from __future__ import annotations

import os
import subprocess
import tempfile
import unittest


REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PILOT_DRY_RUN = os.path.join(REPO, "bin", "brightlead-skillopt-pilot-dry-run")


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


if __name__ == "__main__":
    unittest.main()
