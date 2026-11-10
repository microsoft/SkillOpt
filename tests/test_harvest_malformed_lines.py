"""Regression tests for transcript lines that are not readable JSON objects."""
from __future__ import annotations

import json
import os
import tempfile
import unittest

from skillopt_sleep.harvest import harvest
from skillopt_sleep.harvest_codex import harvest_codex

# A record whose last character is cut in half, as when a session is still
# being written: the file ends in an incomplete UTF-8 sequence.
_CUT_LINE = json.dumps(
    {"message": {"role": "assistant", "content": "Done ✅"}},
    ensure_ascii=False,
).encode("utf-8")[:-4]

# Valid JSON lines that are not objects.
_NON_OBJECT_LINES = b"null\n[]\n\"text\"\n42\n"

_CLAUDE_RECORDS = [
    {
        "timestamp": "2026-09-20T00:00:00Z",
        "cwd": "/repo",
        "message": {"role": "user", "content": "Fix the parser bug"},
    },
    {
        "timestamp": "2026-09-20T00:00:10Z",
        "cwd": "/repo",
        "message": {"role": "assistant", "content": "The parser is fixed."},
    },
]

_CODEX_RECORDS = [
    {
        "timestamp": "2026-09-20T00:00:00Z",
        "payload": {"type": "user_message", "message": "Fix the parser bug", "cwd": "/repo"},
    },
    {
        "timestamp": "2026-09-20T00:00:01Z",
        "payload": {"type": "agent_message", "message": "The parser is fixed."},
    },
]


def _write(path: str, records: list, *, head: bytes = b"", tail: bytes = b"") -> None:
    with open(path, "wb") as handle:
        handle.write(head)
        for record in records:
            handle.write(json.dumps(record).encode("utf-8") + b"\n")
        handle.write(tail)


class TestHarvestMalformedLines(unittest.TestCase):
    def _harvest(self, harvester, records, **parts):
        with tempfile.TemporaryDirectory() as tmp:
            _write(os.path.join(tmp, "session.jsonl"), records, **parts)
            return harvester(tmp, scope="all")

    def test_claude_harvest_skips_line_cut_mid_character(self):
        digests = self._harvest(harvest, _CLAUDE_RECORDS, tail=_CUT_LINE)
        self.assertEqual(len(digests), 1)
        self.assertEqual(digests[0].user_prompts, ["Fix the parser bug"])
        self.assertEqual(digests[0].n_assistant_turns, 1)

    def test_claude_harvest_skips_non_object_lines(self):
        digests = self._harvest(harvest, _CLAUDE_RECORDS, head=_NON_OBJECT_LINES)
        self.assertEqual(len(digests), 1)
        self.assertEqual(digests[0].user_prompts, ["Fix the parser bug"])

    def test_codex_harvest_skips_unreadable_lines(self):
        digests = self._harvest(
            harvest_codex, _CODEX_RECORDS, head=_NON_OBJECT_LINES, tail=_CUT_LINE,
        )
        self.assertEqual(len(digests), 1)
        self.assertEqual(digests[0].user_prompts, ["Fix the parser bug"])


if __name__ == "__main__":
    unittest.main()
