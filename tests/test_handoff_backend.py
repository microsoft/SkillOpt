"""Tests for the handoff backend: session-executed model calls via
prompt/answer files, resumed across engine runs."""
from __future__ import annotations

import json
import os
import re
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

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
    def test_handoff_directory_resumes_identified_legacy_source_snapshot(self):
        from skillopt_sleep.__main__ import _handoff_dir_for

        with tempfile.TemporaryDirectory() as proj:
            root = os.path.join(proj, ".skillopt-sleep-handoff")
            os.makedirs(root)
            with open(os.path.join(root, "tasks.json"), "w",
                      encoding="utf-8") as f:
                json.dump({"transcript_source": "codex", "tasks": []}, f)

            codex_dir = _handoff_dir_for({
                "invoked_project": proj,
                "transcript_source": "codex",
            })
            claude_dir = _handoff_dir_for({
                "invoked_project": proj,
                "transcript_source": "claude",
            })

            self.assertEqual(codex_dir, root)
            self.assertNotEqual(claude_dir, root)

    def test_handoff_snapshots_are_isolated_by_transcript_source(self):
        from skillopt_sleep.__main__ import (
            _handoff_dir_for,
            _handoff_mine_and_pin,
        )
        from skillopt_sleep.types import SessionDigest

        with tempfile.TemporaryDirectory() as proj, \
                tempfile.TemporaryDirectory() as home:
            root = os.path.join(proj, "handoff-root")
            calls = []

            def harvest(cfg, *, since_iso, limit):
                source = cfg.get("transcript_source")
                calls.append(source)
                return [SessionDigest(
                    session_id=f"session-{source}", project=proj,
                )]

            def mine(digests, **kwargs):
                return [TaskRecord(
                    id=f"task-{digests[0].session_id}",
                    project=proj,
                    intent="source-specific task",
                )]

            with mock.patch.dict(
                os.environ, {"SKILLOPT_SLEEP_HANDOFF_DIR": root},
            ), mock.patch(
                "skillopt_sleep.__main__.harvest_for_config",
                side_effect=harvest,
            ), mock.patch(
                "skillopt_sleep.__main__.mine", side_effect=mine,
            ):
                handoff_dirs = {}
                for source in ("claude", "codex"):
                    cfg = load_config(
                        invoked_project=proj,
                        projects="invoked",
                        backend="handoff",
                        claude_home=os.path.join(home, source, ".claude"),
                        transcript_source=source,
                        llm_mine=False,
                    )
                    handoff_dir = _handoff_dir_for(cfg)
                    handoff_dirs[source] = handoff_dir
                    backend = HandoffBackend(handoff_dir=handoff_dir)
                    _handoff_mine_and_pin(
                        cfg,
                        SimpleNamespace(json=False),
                        backend,
                        os.path.join(handoff_dir, "tasks.json"),
                        dry=True,
                    )

            self.assertEqual(calls, ["claude", "codex"])
            self.assertEqual(handoff_dirs["claude"], root)
            self.assertNotEqual(handoff_dirs["claude"], handoff_dirs["codex"])
            self.assertNotEqual(
                HandoffBackend(handoff_dir=handoff_dirs["claude"]).answer_path(
                    "same-prompt-key"
                ),
                HandoffBackend(handoff_dir=handoff_dirs["codex"]).answer_path(
                    "same-prompt-key"
                ),
            )
            for source, handoff_dir in handoff_dirs.items():
                with open(os.path.join(handoff_dir, "digests.json"),
                          encoding="utf-8") as f:
                    digests = json.load(f)
                with open(os.path.join(handoff_dir, "tasks.json"),
                          encoding="utf-8") as f:
                    tasks = json.load(f)
                self.assertEqual(digests[0]["session_id"], f"session-{source}")
                self.assertEqual(tasks["transcript_source"], source)

    def test_empty_handoff_digest_pin_does_not_hide_another_source(self):
        from skillopt_sleep.__main__ import (
            _handoff_dir_for,
            _handoff_mine_and_pin,
        )
        from skillopt_sleep.types import SessionDigest

        with tempfile.TemporaryDirectory() as proj, \
                tempfile.TemporaryDirectory() as home:
            root = os.path.join(proj, "handoff-root")
            calls = []

            def harvest(cfg, *, since_iso, limit):
                source = cfg.get("transcript_source")
                calls.append(source)
                if source == "claude":
                    return []
                return [SessionDigest(
                    session_id=f"session-{source}", project=proj,
                )]

            def mine(digests, **kwargs):
                if not digests:
                    return []
                return [TaskRecord(
                    id=f"task-{digests[0].session_id}",
                    project=proj,
                    intent="task from the later source",
                )]

            with mock.patch.dict(
                os.environ, {"SKILLOPT_SLEEP_HANDOFF_DIR": root},
            ), mock.patch(
                "skillopt_sleep.__main__.harvest_for_config",
                side_effect=harvest,
            ), mock.patch(
                "skillopt_sleep.__main__.mine", side_effect=mine,
            ):
                handoff_dirs = {}
                for source in ("claude", "codex"):
                    cfg = load_config(
                        invoked_project=proj,
                        projects="invoked",
                        backend="handoff",
                        claude_home=os.path.join(home, source, ".claude"),
                        transcript_source=source,
                        llm_mine=False,
                    )
                    handoff_dir = _handoff_dir_for(cfg)
                    handoff_dirs[source] = handoff_dir
                    _handoff_mine_and_pin(
                        cfg,
                        SimpleNamespace(json=False),
                        HandoffBackend(handoff_dir=handoff_dir),
                        os.path.join(handoff_dir, "tasks.json"),
                        dry=True,
                    )

            self.assertEqual(calls, ["claude", "codex"])
            self.assertEqual(handoff_dirs["claude"], root)
            with open(os.path.join(handoff_dirs["claude"], "digests.json"),
                      encoding="utf-8") as f:
                self.assertEqual(json.load(f), [])
            with open(os.path.join(handoff_dirs["codex"], "digests.json"),
                      encoding="utf-8") as f:
                codex_digests = json.load(f)
            with open(os.path.join(handoff_dirs["codex"], "tasks.json"),
                      encoding="utf-8") as f:
                codex_tasks = json.load(f)
            self.assertEqual(codex_digests[0]["session_id"], "session-codex")
            self.assertEqual(codex_tasks["transcript_source"], "codex")

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

    def test_completed_run_json_stdout_is_one_parseable_document(self):
        import contextlib
        import io

        from skillopt_sleep.__main__ import main
        from skillopt_sleep.cycle import CycleOutcome
        from skillopt_sleep.tasks_file import make_tasks_payload, write_tasks_file
        from skillopt_sleep.types import SleepReport

        with tempfile.TemporaryDirectory() as proj, \
                tempfile.TemporaryDirectory() as home:
            tasks_path = os.path.join(proj, "tasks.json")
            payload = make_tasks_payload(_tasks(), project=proj)
            payload["reviewed"] = True
            write_tasks_file(tasks_path, payload)
            staging_dir = os.path.join(proj, "finished-staging")
            os.makedirs(staging_dir)
            outcome = CycleOutcome(
                report=SleepReport(
                    night=7,
                    project=proj,
                    n_tasks=len(_tasks()),
                    accepted=True,
                    gate_action="accept_new_best",
                ),
                staging_dir=staging_dir,
                adopted=False,
                adopted_paths=[],
            )
            stdout = io.StringIO()
            stderr = io.StringIO()
            with mock.patch(
                "skillopt_sleep.__main__.run_sleep_cycle",
                return_value=outcome,
            ), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                rc = main([
                    "run", "--backend", "handoff", "--json",
                    "--project", proj,
                    "--claude-home", os.path.join(home, ".claude"),
                    "--tasks-file", tasks_path,
                ])

            self.assertEqual(rc, 0)
            result = json.loads(stdout.getvalue())
            self.assertEqual(result["night"], 7)
            self.assertEqual(result["staging_dir"], staging_dir)
            self.assertIn("archived round data", stderr.getvalue())

    def test_archive_failure_never_prints_a_success_document(self):
        import contextlib
        import io

        from skillopt_sleep.__main__ import main
        from skillopt_sleep.cycle import CycleOutcome
        from skillopt_sleep.tasks_file import make_tasks_payload, write_tasks_file
        from skillopt_sleep.types import SleepReport

        with tempfile.TemporaryDirectory() as proj, \
                tempfile.TemporaryDirectory() as home:
            tasks_path = os.path.join(proj, "tasks.json")
            payload = make_tasks_payload(_tasks(), project=proj)
            payload["reviewed"] = True
            write_tasks_file(tasks_path, payload)
            staging_dir = os.path.join(proj, "finished-staging")
            os.makedirs(staging_dir)
            outcome = CycleOutcome(
                report=SleepReport(night=7, project=proj, accepted=True),
                staging_dir=staging_dir,
                adopted=False,
                adopted_paths=[],
            )
            stdout = io.StringIO()
            with mock.patch(
                "skillopt_sleep.__main__.run_sleep_cycle", return_value=outcome
            ), mock.patch(
                "skillopt_sleep.__main__.os.rename",
                side_effect=OSError("archive denied"),
            ), contextlib.redirect_stdout(stdout):
                rc = main([
                    "run", "--backend", "handoff", "--json",
                    "--project", proj,
                    "--claude-home", os.path.join(home, ".claude"),
                    "--tasks-file", tasks_path,
                ])
            self.assertEqual(rc, 1)
            result = json.loads(stdout.getvalue())
            self.assertEqual(result["ok"], False)
            self.assertEqual(result["error"], "staging_refused")
            self.assertNotIn("night", result)

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
            # later run re-scans the same stale window forever. The cursor is
            # keyed by (project, transcript source) — #312.
            cfg = load_config(invoked_project=proj, claude_home=claude_home)
            state = SleepState.load(cfg.state_path)
            self.assertIsNotNone(
                state.last_harvest_for(proj, cfg.get("transcript_source", ""))
            )


if __name__ == "__main__":
    unittest.main()
