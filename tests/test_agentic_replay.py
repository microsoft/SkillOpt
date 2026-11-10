"""Tests for AgenticReplayBackend and agentic replay slice (Issue #155)."""
import os
import subprocess
import sys
import tempfile
import unittest

from skillopt_sleep.agentic_replay import (
    OUTCOME_FAIL,
    OUTCOME_INCONCLUSIVE,
    OUTCOME_PASS,
    REASON_BUDGET_EXHAUSTED,
    REASON_INFRA,
    REASON_INSUFFICIENT_EVIDENCE,
    REASON_TIMEOUT,
    AgenticReplayBackend,
    AgenticReplayConfig,
    AgenticWorkspace,
    TaskVerifier,
)
from skillopt_sleep.backend import get_backend
from skillopt_sleep.replay import aggregate_scores, replay_one
from skillopt_sleep.types import EditRecord, ReplayResult, TaskRecord


class TestReplayResultOutcomes(unittest.TestCase):
    def test_default_pass_fail_inference(self):
        passed = ReplayResult(id="t1", hard=1.0)
        self.assertEqual(passed.outcome, OUTCOME_PASS)

        failed = ReplayResult(id="t2", hard=0.0)
        self.assertEqual(failed.outcome, OUTCOME_FAIL)

        mixed = ReplayResult(id="t3", hard=0.5)
        self.assertEqual(mixed.outcome, OUTCOME_FAIL)

    def test_explicit_inconclusive_outcome(self):
        res = ReplayResult(
            id="t1",
            hard=0.0,
            outcome=OUTCOME_INCONCLUSIVE,
            inconclusive_reason=REASON_TIMEOUT,
            diff="--- a/file\n+++ b/file\n",
        )
        self.assertEqual(res.outcome, OUTCOME_INCONCLUSIVE)
        self.assertEqual(res.inconclusive_reason, REASON_TIMEOUT)
        self.assertIn("--- a/file", res.diff)

        d = res.to_dict()
        self.assertEqual(d["outcome"], OUTCOME_INCONCLUSIVE)
        self.assertEqual(d["inconclusive_reason"], REASON_TIMEOUT)
        self.assertIn("--- a/file", d["diff"])


class TestAgenticWorkspace(unittest.TestCase):
    def test_workspace_isolation_and_git_baseline(self):
        with AgenticWorkspace(isolate_env=True, isolate_network=True) as ws:
            self.assertTrue(os.path.isdir(ws.workspace_path))
            # Verify git repo was initialized with baseline commit
            git_res = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=ws.workspace_path,
                capture_output=True,
                text=True,
            )
            self.assertEqual(git_res.returncode, 0)
            self.assertTrue(len(git_res.stdout.strip()) >= 7)

            # Check environment scrubbing
            os.environ["TEST_SECRET_KEY"] = "supersecret"
            os.environ["AWS_ACCESS_KEY_ID"] = "AKIAEXAMPLE"
            os.environ["GITHUB_TOKEN"] = "ghp_example"
            env = ws.build_isolated_env()

            self.assertNotIn("TEST_SECRET_KEY", env)
            self.assertNotIn("AWS_ACCESS_KEY_ID", env)
            self.assertNotIn("GITHUB_TOKEN", env)
            self.assertEqual(env.get("HTTP_PROXY"), "http://127.0.0.1:9")
            self.assertTrue(os.path.isdir(env.get("HOME", "")))
            self.assertIn(".sandbox_home", env.get("HOME", ""))

            # Test git diff tracking
            file_path = os.path.join(ws.workspace_path, "sample.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("hello world\n")

            diff = ws.get_diff()
            self.assertIn("sample.txt", diff)
            self.assertIn("+hello world", diff)

        # Verify workspace is cleaned up on exit
        self.assertFalse(os.path.exists(ws.workspace_path))


class TestTaskVerifier(unittest.TestCase):
    def test_verifier_fn(self):
        verifier = TaskVerifier(verifier_fn=lambda path: (True, "all checks passed"))
        with tempfile.TemporaryDirectory() as tmp:
            outcome, hard, soft, rationale, reason = verifier.verify(tmp)
            self.assertEqual(outcome, OUTCOME_PASS)
            self.assertEqual(hard, 1.0)
            self.assertEqual(soft, 1.0)
            self.assertEqual(reason, "")

        failing_verifier = TaskVerifier(verifier_fn=lambda path: (False, "assertion failed"))
        with tempfile.TemporaryDirectory() as tmp:
            outcome, hard, soft, rationale, reason = failing_verifier.verify(tmp)
            self.assertEqual(outcome, OUTCOME_FAIL)
            self.assertEqual(hard, 0.0)
            self.assertEqual(soft, 0.0)

    def test_expected_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            verifier = TaskVerifier(expected_files={"output.json": '"status": "ok"'})
            outcome, hard, _, _, _ = verifier.verify(tmp)
            self.assertEqual(outcome, OUTCOME_FAIL)

            with open(os.path.join(tmp, "output.json"), "w", encoding="utf-8") as f:
                f.write('{"status": "ok", "value": 42}')

            outcome, hard, _, _, _ = verifier.verify(tmp)
            self.assertEqual(outcome, OUTCOME_PASS)
            self.assertEqual(hard, 1.0)

    def test_test_cmd_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Pass command
            verifier = TaskVerifier(test_cmd=[sys.executable, "-c", "exit(0)"])
            outcome, hard, _, _, _ = verifier.verify(tmp)
            self.assertEqual(outcome, OUTCOME_PASS)
            self.assertEqual(hard, 1.0)

            # Deterministic fail command
            failing_verifier = TaskVerifier(test_cmd=[sys.executable, "-c", "exit(1)"])
            outcome, hard, _, _, _ = failing_verifier.verify(tmp)
            self.assertEqual(outcome, OUTCOME_FAIL)
            self.assertEqual(hard, 0.0)

            # Timeout command
            timeout_verifier = TaskVerifier(
                test_cmd=[sys.executable, "-c", "import time; time.sleep(2)"],
                timeout=0.2,
            )
            outcome, hard, _, _, reason = timeout_verifier.verify(tmp)
            self.assertEqual(outcome, OUTCOME_INCONCLUSIVE)
            self.assertEqual(reason, REASON_TIMEOUT)


class TestAgenticReplayBackend(unittest.TestCase):
    def test_successful_agentic_replay(self):
        backend = AgenticReplayBackend()
        task = TaskRecord(id="task_build", project="test", intent="fix build config")

        def agent_runner(workspace_path, env):
            target_file = os.path.join(workspace_path, "build.txt")
            with open(target_file, "w", encoding="utf-8") as f:
                f.write("build=success\n")
            return ["write_file", "run_test"]

        result = backend.replay_agentic_task(
            task,
            skill="use correct build flags",
            memory="",
            expected_files={"build.txt": "build=success"},
            agent_runner=agent_runner,
        )

        self.assertEqual(result.outcome, OUTCOME_PASS)
        self.assertEqual(result.hard, 1.0)
        self.assertEqual(result.soft, 1.0)
        self.assertIn("build.txt", result.diff)
        self.assertEqual(result.tools_called, ["write_file", "run_test"])

    def test_failing_agentic_replay(self):
        backend = AgenticReplayBackend()
        task = TaskRecord(id="task_fail", project="test", intent="break test")

        def agent_runner(workspace_path, env):
            return ["run_tests"]

        result = backend.replay_agentic_task(
            task,
            skill="",
            memory="",
            test_cmd=[sys.executable, "-c", "sys.exit(1)"],
            agent_runner=agent_runner,
        )

        self.assertEqual(result.outcome, OUTCOME_FAIL)
        self.assertEqual(result.hard, 0.0)
        self.assertIn("failed", result.fail_reason.lower())

    def test_insufficient_decidable_actions(self):
        config = AgenticReplayConfig(min_decidable_actions=2)
        backend = AgenticReplayBackend(config=config)
        task = TaskRecord(id="task_thin", project="test", intent="simple query")

        def zero_action_runner(workspace_path, env):
            return []  # 0 actions, below threshold of 2

        result = backend.replay_agentic_task(
            task,
            skill="",
            memory="",
            agent_runner=zero_action_runner,
        )

        self.assertEqual(result.outcome, OUTCOME_INCONCLUSIVE)
        self.assertEqual(result.inconclusive_reason, REASON_INSUFFICIENT_EVIDENCE)

    def test_budget_exhaustion(self):
        config = AgenticReplayConfig(max_replays_per_run=1)
        backend = AgenticReplayBackend(config=config)
        task1 = TaskRecord(id="task_1", project="test", intent="first task")
        task2 = TaskRecord(id="task_2", project="test", intent="second task")

        res1 = backend.replay_agentic_task(task1, skill="", memory="")
        self.assertNotEqual(res1.inconclusive_reason, REASON_BUDGET_EXHAUSTED)

        # Second replay exceeds budget cap of 1
        res2 = backend.replay_agentic_task(task2, skill="", memory="")
        self.assertEqual(res2.outcome, OUTCOME_INCONCLUSIVE)
        self.assertEqual(res2.inconclusive_reason, REASON_BUDGET_EXHAUSTED)

    def test_tiered_reflection_excludes_inconclusive(self):
        backend = AgenticReplayBackend()
        task_real_fail = TaskRecord(id="tf", project="p", intent="real fail")
        task_infra_fail = TaskRecord(id="ti", project="p", intent="infra fail")

        r_real_fail = ReplayResult(id="tf", hard=0.0, outcome=OUTCOME_FAIL)
        r_infra_fail = ReplayResult(
            id="ti",
            hard=0.0,
            outcome=OUTCOME_INCONCLUSIVE,
            inconclusive_reason=REASON_TIMEOUT,
        )

        failures = [(task_real_fail, r_real_fail), (task_infra_fail, r_infra_fail)]
        # Reflect should only process real failures, discarding the infra failure
        edits = backend.reflect(
            failures,
            [],
            skill="baseline",
            memory="",
            edit_budget=2,
            evolve_skill=True,
            evolve_memory=False,
        )
        self.assertIsInstance(edits, list)

    def test_aggregate_scores_exclude_inconclusive(self):
        task1 = TaskRecord(id="t1", project="p", intent="i1")
        task2 = TaskRecord(id="t2", project="p", intent="i2")
        task3 = TaskRecord(id="t3", project="p", intent="i3")

        r1 = ReplayResult(id="t1", hard=1.0, soft=1.0, outcome=OUTCOME_PASS)
        r2 = ReplayResult(id="t2", hard=0.0, soft=0.0, outcome=OUTCOME_FAIL)
        r3 = ReplayResult(
            id="t3",
            hard=0.0,
            soft=0.0,
            outcome=OUTCOME_INCONCLUSIVE,
            inconclusive_reason=REASON_TIMEOUT,
        )

        pairs = [(task1, r1), (task2, r2), (task3, r3)]

        # Standard aggregation: 1 / 3 = 0.3333
        hard_all, soft_all = aggregate_scores(pairs)
        self.assertAlmostEqual(hard_all, 1.0 / 3.0)

        # Excluding inconclusive: 1 / 2 = 0.5
        hard_filtered, soft_filtered = aggregate_scores(pairs, exclude_inconclusive=True)
        self.assertAlmostEqual(hard_filtered, 0.5)

    def test_get_backend_agentic(self):
        be = get_backend("agentic")
        self.assertIsInstance(be, AgenticReplayBackend)
        self.assertEqual(be.name, "agentic")

        be2 = get_backend("agentic_replay")
        self.assertIsInstance(be2, AgenticReplayBackend)


if __name__ == "__main__":
    unittest.main()
