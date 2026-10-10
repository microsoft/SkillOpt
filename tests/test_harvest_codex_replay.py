"""Regression tests for excluding SkillOpt-generated Codex sessions."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

from skillopt_sleep.backend import CodexCliBackend
from skillopt_sleep.harvest_codex import harvest_codex
from skillopt_sleep.types import ReplayResult, TaskRecord


def _write_session(path: str, prompt: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for record in (
            {
                "timestamp": "2026-09-20T00:00:00Z",
                "payload": {"type": "turn_context", "cwd": "/repo"},
            },
            {
                "timestamp": "2026-09-20T00:00:01Z",
                "payload": {"type": "user_message", "message": prompt},
            },
            {
                "timestamp": "2026-09-20T00:00:10Z",
                "payload": {"type": "agent_message", "message": "Completed."},
            },
        ):
            handle.write(json.dumps(record) + "\n")


class TestCodexReplayHarvest(unittest.TestCase):
    def test_actual_backend_prompts_are_excluded(self):
        prompts = []

        def capture(cmd, **kwargs):
            prompts.append(kwargs["input"])
            with open(cmd[cmd.index("-o") + 1], "w", encoding="utf-8") as handle:
                if "You are SkillOpt's optimizer." in kwargs["input"]:
                    handle.write('[{"op": "add", "content": "Check the answer."}]')
                else:
                    handle.write('{"score": 0.5, "reason": "synthetic"}')
            return mock.Mock(returncode=0, stdout="", stderr="")

        backend = CodexCliBackend(codex_path="codex")
        task = TaskRecord(id="t", project="/repo", intent="Answer the question", reference_kind="rubric")
        with mock.patch("skillopt_sleep.backend.subprocess.run", side_effect=capture):
            backend.attempt(task, "skill", "memory")
            backend.judge(task, "answer")
            backend.reflect(
                [(task, ReplayResult(id="t", response="wrong", fail_reason="incorrect"))],
                [], "skill", "memory", edit_budget=1, evolve_skill=True, evolve_memory=False,
            )
            backend.attempt_with_tools(task, "skill", "memory", ["search"])
            # Benchmark/custom templates must carry the same provenance.
            backend.attempt(TaskRecord(id="custom", project="/repo", intent="task", system="custom system"), "", "")

        self.assertEqual(len(prompts), 5)
        for index, prompt in enumerate(prompts):
            with self.subTest(operation=index), tempfile.TemporaryDirectory() as tmp:
                _write_session(os.path.join(tmp, "engine.jsonl"), prompt)
                self.assertEqual(harvest_codex(tmp), [])

    def test_quoted_headings_and_multiturn_user_sessions_are_preserved(self):
        for heading in ("## CURRENT SKILL", "## TASK\n", "## FAILED TASKS", "You are a strict grader"):
            for followup in (False, True):
                with self.subTest(heading=heading, followup=followup), tempfile.TemporaryDirectory() as tmp:
                    path = os.path.join(tmp, "user.jsonl")
                    _write_session(path, f"Please explain the {heading} section")
                    if followup:
                        with open(path, "a", encoding="utf-8") as handle:
                            handle.write(json.dumps({"payload": {"type": "user_message", "message": "perfect, thanks"}}) + "\n")
                    digests = harvest_codex(tmp)
                    self.assertEqual(len(digests), 1)
                    self.assertEqual(digests[0].n_user_turns, 2 if followup else 1)

    def test_skillopt_replay_session_is_excluded(self):
        prompt = (
            "Complete the task. Apply the skill and memory rules EXACTLY, including "
            "any rule about searching before answering.\n\n"
            "# Skill\nlearned rules\n\n# Memory\nprior notes\n\n"
            "# Task\nanswer the task\n\nReturn ONLY the final answer."
        )
        with tempfile.TemporaryDirectory() as tmp:
            _write_session(os.path.join(tmp, "replay.jsonl"), prompt)
            self.assertEqual(harvest_codex(tmp, scope="all"), [])

    def test_real_user_session_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_session(os.path.join(tmp, "real.jsonl"), "Please update the parser.")
            digests = harvest_codex(tmp, scope="all")

        self.assertEqual(len(digests), 1)
        self.assertEqual(digests[0].session_id, "real")


if __name__ == "__main__":
    unittest.main()
