"""BrightLead-local preflight command coverage."""
from __future__ import annotations

import os
import subprocess
import tempfile
import unittest


REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PREFLIGHT = os.path.join(REPO, "bin", "brightlead-skillopt-preflight")


class TestBrightLeadPreflight(unittest.TestCase):
    def test_preflight_writes_local_report_with_guardrails(self):
        self.assertTrue(os.path.exists(PREFLIGHT))
        self.assertTrue(os.access(PREFLIGHT, os.X_OK))

        env = {**os.environ, "PYTHONNOUSERSITE": "1"}
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                [PREFLIGHT, tmp],
                cwd=REPO,
                env=env,
                capture_output=True,
                text=True,
                timeout=180,
            )

            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertIn("running smoke check", proc.stdout)
            self.assertIn("brightlead-skillopt-preflight.md", proc.stdout)

            report_path = os.path.join(tmp, "brightlead-skillopt-preflight.md")
            smoke_path = os.path.join(tmp, "smoke.log")
            self.assertTrue(os.path.exists(report_path))
            self.assertTrue(os.path.exists(smoke_path))

            with open(report_path, encoding="utf-8") as f:
                report = f.read()

            self.assertIn("status: PASS", report)
            self.assertIn("smoke_check: PASS", report)
            self.assertIn("github_actions_changes: none", report)
            self.assertIn("does not push", report)
            self.assertIn("dispatch automation", report)
            self.assertIn("touch WordPress", report)


if __name__ == "__main__":
    unittest.main()
