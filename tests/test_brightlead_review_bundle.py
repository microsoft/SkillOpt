"""BrightLead-local review bundle command coverage."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest


REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REVIEW_BUNDLE = os.path.join(REPO, "bin", "brightlead-skillopt-review-bundle")


class TestBrightLeadReviewBundle(unittest.TestCase):
    def test_review_bundle_writes_manifest_and_handoff_report(self):
        self.assertTrue(os.path.exists(REVIEW_BUNDLE))
        self.assertTrue(os.access(REVIEW_BUNDLE, os.X_OK))

        env = {**os.environ, "PYTHONNOUSERSITE": "1"}
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                [REVIEW_BUNDLE, tmp],
                cwd=REPO,
                env=env,
                capture_output=True,
                text=True,
                timeout=300,
            )

            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertIn("running pilot dry run", proc.stdout)
            self.assertIn("brightlead-skillopt-review-bundle.md", proc.stdout)

            report_path = os.path.join(tmp, "brightlead-skillopt-review-bundle.md")
            manifest_path = os.path.join(tmp, "manifest.json")
            self.assertTrue(os.path.exists(report_path))
            self.assertTrue(os.path.exists(manifest_path))

            with open(report_path, encoding="utf-8") as f:
                report = f.read()
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)

            self.assertEqual(manifest["status"], "PASS")
            self.assertEqual(manifest["baseline"], 0.0)
            self.assertEqual(manifest["candidate"], 1.0)
            self.assertIs(manifest["accepted"], True)
            self.assertIs(manifest["adopted"], False)
            self.assertEqual(manifest["staging_dir"], "")
            self.assertIs(manifest["tasks_reviewed"], True)
            self.assertTrue(os.path.exists(manifest["dry_run_report"]))
            self.assertTrue(os.path.exists(manifest["preflight_report"]))

            self.assertIn("status: PASS", report)
            self.assertIn("single review packet", report)
            self.assertIn("does not push", report)
            self.assertIn("touch WordPress", report)
            self.assertIn("Review Checklist", report)


if __name__ == "__main__":
    unittest.main()
