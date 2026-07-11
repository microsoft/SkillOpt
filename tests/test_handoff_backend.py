"""Tests for the handoff backend: session-executed model calls via
prompt/answer files, resumed across engine runs."""
from __future__ import annotations

import json
import os
import re
import tempfile
import unittest

from skillopt_sleep.backend import get_backend
from skillopt_sleep.config import load_config
from skillopt_sleep.cycle import run_sleep_cycle
from skillopt_sleep.handoff_backend import (
    PENDING_SENTINEL_PREFIX,
    HandoffBackend,
    PendingCalls,
)
from skillopt_sleep.mine import assign_splits
from skillopt_sleep.types import TaskRecord

# The rule the simulated executor "learns"; once present in the skill the
# executor answers arithmetic tasks correctly, mirroring MockBackend's
# rule-gated model of reality.
RULE = "Always answer with just the number."


def _tasks():
    ts = [
        TaskRecord(
            id=f"t{i}", project="/p",
            intent=f"What is {i} + {i}?",
            reference_kind="exact", reference=str(i + i),
        )
        for i in range(1, 7)
    ]
    return assign_splits(ts, holdout_fraction=0.34, seed=42)


def _answer_pending(backend: HandoffBackend) -> None:
    """Deterministic stand-in for the interactive session answering PROMPTS.md."""
    for key, item in list(backend.pending.items()):
        prompt = str(item["prompt"])
        if "You are SkillOpt's optimizer" in prompt:
            ans = json.dumps([{
                "op": "add", "content": RULE,
                "rationale": "outputs failed exact match; answer bare numbers",
            }])
        else:
            m = re.search(r"What is (\d+) \+ (\d+)\?", prompt)
            if m and RULE in prompt:
                ans = str(int(m.group(1)) + int(m.group(2)))
            else:
                ans = "cannot say"
        with open(backend.answer_path(key), "w", encoding="utf-8") as f:
            f.write(ans)


class TestHandoffBackendUnit(unittest.TestCase):
    def test_miss_records_pending_and_returns_sentinel(self):
        with tempfile.TemporaryDirectory() as hdir:
            be = HandoffBackend(handoff_dir=hdir)
            out = be._call("some prompt")
            self.assertTrue(out.startswith(PENDING_SENTINEL_PREFIX))
            self.assertEqual(len(be.pending), 1)

    def test_answer_file_resolves_call(self):
        with tempfile.TemporaryDirectory() as hdir:
            be = HandoffBackend(handoff_dir=hdir)
            key = be._call("what is 2+2?").split(":")[-1].rstrip("]")
            with open(be.answer_path(key), "w", encoding="utf-8") as f:
                f.write("4\n")
            fresh = HandoffBackend(handoff_dir=hdir)
            self.assertEqual(fresh._call("what is 2+2?"), "4")
            self.assertEqual(len(fresh.pending), 0)

    def test_dependent_prompt_raises(self):
        with tempfile.TemporaryDirectory() as hdir:
            be = HandoffBackend(handoff_dir=hdir)
            placeholder = be._call("first question")
            with self.assertRaises(PendingCalls) as ctx:
                be._call(f"judge this response: {placeholder}")
            self.assertEqual(len(ctx.exception.pending), 1)

    def test_reflect_retry_of_pending_reply_raises(self):
        with tempfile.TemporaryDirectory() as hdir:
            be = HandoffBackend(handoff_dir=hdir)
            be._call("reflect on failures")
            with self.assertRaises(PendingCalls):
                be._call("reflect on failures\n\nIMPORTANT: "
                         "your previous reply was not valid JSON. Reply with "
                         "ONLY the JSON array, no prose, no markdown fences.")

    def test_flush_writes_prompts_and_pending_json(self):
        with tempfile.TemporaryDirectory() as hdir:
            be = HandoffBackend(handoff_dir=hdir)
            be._call("prompt A")
            be._call("prompt B")
            prompts_path = be.flush_pending()
            self.assertTrue(os.path.exists(prompts_path))
            with open(os.path.join(hdir, "pending.json"), encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual(payload["format"], "skillopt_sleep.handoff.v1")
            self.assertEqual(len(payload["pending"]), 2)
            self.assertEqual(payload["pending"][0]["prompt"], "prompt A")
            with open(prompts_path, encoding="utf-8") as f:
                md = f.read()
            self.assertIn("BEGIN PROMPT", md)
            self.assertIn("prompt B", md)

    def test_get_backend_registration(self):
        with tempfile.TemporaryDirectory() as proj:
            be = get_backend("handoff", project_dir=proj)
            self.assertIsInstance(be, HandoffBackend)
            self.assertTrue(be.handoff_dir.startswith(os.path.realpath(proj))
                            or be.handoff_dir.startswith(proj))


class TestHandoffCycle(unittest.TestCase):
    def test_cycle_converges_over_handoff_rounds(self):
        with tempfile.TemporaryDirectory() as proj, \
                tempfile.TemporaryDirectory() as home:
            hdir = os.path.join(proj, ".skillopt-sleep-handoff")

            def cfg():
                return load_config(
                    invoked_project=proj, projects="invoked",
                    backend="handoff",
                    claude_home=os.path.join(home, ".claude"),
                )

            final = None
            rounds = 0
            for _ in range(8):
                rounds += 1
                backend = HandoffBackend(handoff_dir=hdir)
                try:
                    outcome = run_sleep_cycle(
                        cfg(), seed_tasks=_tasks(), dry_run=True, backend=backend,
                    )
                except PendingCalls:
                    outcome = None
                if backend.pending:
                    backend.flush_pending()
                    _answer_pending(backend)
                    continue
                final = outcome
                break

            self.assertIsNotNone(final, "cycle never completed within 8 rounds")
            self.assertGreater(rounds, 1, "expected at least one handoff round")
            rep = final.report
            self.assertTrue(rep.accepted, f"gate did not accept: {rep.gate_action}")
            self.assertGreater(rep.candidate_score, rep.baseline_score)
            self.assertTrue(any(RULE in e.content for e in rep.edits))


class TestHandoffCli(unittest.TestCase):
    def test_run_exits_3_and_stages_prompts(self):
        from skillopt_sleep.__main__ import main
        from skillopt_sleep.tasks_file import make_tasks_payload, write_tasks_file

        with tempfile.TemporaryDirectory() as proj, \
                tempfile.TemporaryDirectory() as home:
            tasks_path = os.path.join(proj, "tasks.json")
            payload = make_tasks_payload(_tasks(), project=proj)
            payload["reviewed"] = True
            write_tasks_file(tasks_path, payload)

            rc = main([
                "run", "--backend", "handoff",
                "--project", proj,
                "--claude-home", os.path.join(home, ".claude"),
                "--tasks-file", tasks_path,
            ])
            self.assertEqual(rc, 3)
            hdir = os.path.join(proj, ".skillopt-sleep-handoff")
            self.assertTrue(os.path.exists(os.path.join(hdir, "PROMPTS.md")))
            self.assertTrue(os.path.exists(os.path.join(hdir, "pending.json")))

    def test_corrupt_digests_pin_falls_back_to_reharvest(self):
        from skillopt_sleep.__main__ import main

        with tempfile.TemporaryDirectory() as proj, \
                tempfile.TemporaryDirectory() as home:
            hdir = os.path.join(proj, ".skillopt-sleep-handoff")
            os.makedirs(hdir, exist_ok=True)
            with open(os.path.join(hdir, "digests.json"), "w", encoding="utf-8") as f:
                f.write("{not valid json")
            rc = main([
                "run", "--backend", "handoff",
                "--project", proj,
                "--claude-home", os.path.join(home, ".claude"),
            ])
            # must not crash: corrupt pin -> fresh harvest -> no tasks -> 0
            self.assertEqual(rc, 0)

    def test_run_with_no_tasks_exits_0_and_advances_harvest_window(self):
        from skillopt_sleep.__main__ import main
        from skillopt_sleep.config import load_config
        from skillopt_sleep.state import SleepState

        with tempfile.TemporaryDirectory() as proj, \
                tempfile.TemporaryDirectory() as home:
            claude_home = os.path.join(home, ".claude")
            rc = main([
                "run", "--backend", "handoff",
                "--project", proj,
                "--claude-home", claude_home,
            ])
            self.assertEqual(rc, 0)
            # the no-tasks branch must persist last-harvest, otherwise every
            # later run re-scans the same stale window forever
            cfg = load_config(invoked_project=proj, claude_home=claude_home)
            state = SleepState.load(cfg.state_path)
            self.assertIsNotNone(state.last_harvest_for(proj))


if __name__ == "__main__":
    unittest.main()
