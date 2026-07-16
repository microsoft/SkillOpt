"""BrightLead-local sanitized task draft command coverage."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest


REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SANITIZE_TASKS = os.path.join(REPO, "bin", "brightlead-skillopt-sanitize-tasks")
REGRESSION_SUITE = os.path.join(REPO, "bin", "brightlead-skillopt-regression-suite")


class TestBrightLeadSanitizeTasks(unittest.TestCase):
    def test_sanitize_tasks_redacts_and_defaults_to_unreviewed(self):
        self.assertTrue(os.path.exists(SANITIZE_TASKS))
        self.assertTrue(os.access(SANITIZE_TASKS, os.X_OK))

        env = {**os.environ, "PYTHONNOUSERSITE": "1"}
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "snippets.jsonl")
            out = os.path.join(tmp, "tasks.json")
            with open(src, "w", encoding="utf-8") as f:
                f.write(json.dumps({
                    "id": "session:1527106928742764586",
                    "user_prompt": "Check https://private.example.test for david@example.com with token=abc123456789.",
                    "assistant_final": "Post 8175 stayed draft under /home/hugh-brightlead/secret/path.",
                    "expected": "No live write occurred for post 8175.",
                    "tags": ["brightlead qa", "rule:no-live-write"],
                    "session_id": "1527106928742764586",
                }) + "\n")

            proc = subprocess.run(
                [SANITIZE_TASKS, src, out, "--project", tmp, "--target-skill-path", "skills/qa-output/SKILL.md"],
                cwd=REPO,
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
            )

            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertIn("wrote 1 tasks", proc.stdout)
            with open(out, encoding="utf-8") as f:
                payload = json.load(f)

            self.assertEqual(payload["format"], "skillopt_sleep.tasks.v1")
            self.assertIs(payload["reviewed"], False)
            self.assertEqual(payload["target_skill_path"], "skills/qa-output/SKILL.md")
            task_blob = json.dumps(payload["tasks"], sort_keys=True)
            self.assertNotIn("private.example.test", task_blob)
            self.assertNotIn("david@example.com", task_blob)
            self.assertNotIn("abc123456789", task_blob)
            self.assertNotIn("/home/hugh-brightlead", task_blob)
            self.assertNotIn("8175", task_blob)
            self.assertIn("[REDACTED_URL]", task_blob)
            self.assertIn("[REDACTED_EMAIL]", task_blob)
            self.assertIn("post [REDACTED_ID]", task_blob)
            self.assertIn("rule:no-live-write", payload["tasks"][0]["tags"])

    def test_sanitize_tasks_can_mark_reviewed_after_human_review(self):
        env = {**os.environ, "PYTHONNOUSERSITE": "1"}
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "snippets.json")
            out = os.path.join(tmp, "tasks.json")
            with open(src, "w", encoding="utf-8") as f:
                json.dump({"intent": "Return QA status", "reference": "QA PASS"}, f)

            proc = subprocess.run(
                [SANITIZE_TASKS, src, out, "--project", tmp, "--mark-reviewed"],
                cwd=REPO,
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
            )

            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            with open(out, encoding="utf-8") as f:
                payload = json.load(f)
            self.assertIs(payload["reviewed"], True)


class TestBrightLeadRegressionSuite(unittest.TestCase):
    def test_regression_suite_runs_brightlead_guardrail_fixtures(self):
        self.assertTrue(os.path.exists(REGRESSION_SUITE))
        self.assertTrue(os.access(REGRESSION_SUITE, os.X_OK))

        env = {**os.environ, "PYTHONNOUSERSITE": "1"}
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                [REGRESSION_SUITE, tmp],
                cwd=REPO,
                env=env,
                capture_output=True,
                text=True,
                timeout=600,
            )

            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            self.assertIn("brightlead-no-live-write", proc.stdout)
            self.assertIn("brightlead-source-citation", proc.stdout)
            self.assertIn("brightlead-draft-first", proc.stdout)

            report_path = os.path.join(tmp, "brightlead-skillopt-regression-suite.md")
            manifest_path = os.path.join(tmp, "manifest.json")
            self.assertTrue(os.path.exists(report_path))
            self.assertTrue(os.path.exists(manifest_path))
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
            self.assertEqual(manifest["status"], "PASS")
            self.assertEqual(len(manifest["fixtures"]), 5)
            fixtures = {item["fixture"]: item for item in manifest["fixtures"]}
            for name in (
                "brightlead-qa",
                "brightlead-known-gap",
                "brightlead-no-live-write",
                "brightlead-source-citation",
                "brightlead-draft-first",
            ):
                self.assertEqual(fixtures[name]["status"], "PASS")
                self.assertIs(fixtures[name]["adopted"], False)
                self.assertEqual(fixtures[name]["staging_dir"], "")
                self.assertIs(fixtures[name]["tasks_reviewed"], True)


if __name__ == "__main__":
    unittest.main()
