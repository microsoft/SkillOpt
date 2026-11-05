"""Regression tests for transcript files disappearing during discovery."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

from skillopt_sleep.harvest import harvest
from skillopt_sleep.harvest_codex import harvest_codex


class TestHarvestDiscoveryRace(unittest.TestCase):
    def test_claude_harvest_skips_mtime_race(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "session.jsonl")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "timestamp": "2026-09-20T00:00:00Z",
                    "cwd": "/repo",
                    "message": {"role": "user", "content": "Fix the parser bug"},
                }) + "\n")
                handle.write(json.dumps({
                    "timestamp": "2026-09-20T00:00:10Z",
                    "cwd": "/repo",
                    "message": {"role": "assistant", "content": "The parser is fixed."},
                }) + "\n")

            real_mtime = os.path.getmtime

            def mtime_with_race(candidate):
                if candidate == path:
                    raise FileNotFoundError(candidate)
                return real_mtime(candidate)

            with mock.patch("skillopt_sleep.harvest.os.path.getmtime", side_effect=mtime_with_race):
                digests = harvest(tmp, scope="all")

        self.assertEqual(len(digests), 1)
        self.assertEqual(digests[0].session_id, "session")

    def test_codex_harvest_skips_mtime_race(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "session.jsonl")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "timestamp": "2026-09-20T00:00:00Z",
                    "payload": {
                        "type": "user_message",
                        "message": "Fix the parser bug",
                        "cwd": "/repo",
                    },
                }) + "\n")
                handle.write(json.dumps({
                    "timestamp": "2026-09-20T00:00:01Z",
                    "payload": {
                        "type": "agent_message",
                        "message": "The parser is fixed.",
                    },
                }) + "\n")

            real_mtime = os.path.getmtime

            def mtime_with_race(candidate):
                if candidate == path:
                    raise FileNotFoundError(candidate)
                return real_mtime(candidate)

            with mock.patch("skillopt_sleep.harvest.os.path.getmtime", side_effect=mtime_with_race):
                digests = harvest_codex(tmp, scope="all")

        self.assertEqual(len(digests), 1)
        self.assertEqual(digests[0].session_id, "session")


if __name__ == "__main__":
    unittest.main()
