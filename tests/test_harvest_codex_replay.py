"""Regression tests for excluding SkillOpt-generated Codex sessions."""
from __future__ import annotations

import json
import os
import tempfile
import unittest

from skillopt_sleep.harvest_codex import harvest_codex


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
