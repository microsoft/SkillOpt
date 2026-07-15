"""BrightLead-local smoke command coverage."""
from __future__ import annotations

import os
import subprocess
import unittest


REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SMOKE = os.path.join(REPO, "bin", "brightlead-skillopt-smoke")


class TestBrightLeadSmoke(unittest.TestCase):
    def test_smoke_script_is_executable_and_passes(self):
        self.assertTrue(os.path.exists(SMOKE))
        self.assertTrue(os.access(SMOKE, os.X_OK))

        env = {**os.environ, "PYTHONNOUSERSITE": "1"}
        proc = subprocess.run(
            [SMOKE],
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )

        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("checking wrapper help", proc.stdout)
        self.assertIn("running BrightLead LOL-010 fixture", proc.stdout)
        self.assertIn("OK", proc.stdout)


if __name__ == "__main__":
    unittest.main()
