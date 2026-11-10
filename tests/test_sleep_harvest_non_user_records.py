"""Text the user did not type is not harvested as a user prompt.

Pure-stdlib (unittest), deterministic, no API key, no third-party deps.
Run:  python -m pytest tests/test_sleep_harvest_non_user_records.py
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest

from skillopt_sleep.harvest import digest_transcript

SKILL_BODY = "Base directory for this skill: /x\n\nNever do the wrong thing."


def _user(text, **extra):
    return {"type": "user", "message": {"role": "user", "content": text}, **extra}


def _digest(rows):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "session.jsonl")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(json.dumps(row) for row in rows))
        return digest_transcript(path)


class TestNonUserRecords(unittest.TestCase):
    def test_api_request_blob_is_not_a_user_prompt(self):
        digest = _digest([
            _user("please open a PR"),
            # How the loaded skill is recorded for the user: isMeta, skipped.
            _user([{"type": "text", "text": SKILL_BODY}], isMeta=True),
            # The same text in the raw API payload record: no isMeta.
            {"type": "api-request-blob", "message": {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "ok"},
                {"type": "text", "text": SKILL_BODY}]}},
            _user("thanks"),
        ])

        self.assertEqual(digest.user_prompts, ["please open a PR", "thanks"])
        self.assertEqual(digest.feedback_signals, ["pos:thanks"])

    def test_task_notification_is_not_a_user_prompt(self):
        digest = _digest([
            _user("please open a PR"),
            _user("<task-notification>\n<status>completed</status>\n</task-notification>"),
        ])

        self.assertEqual(digest.user_prompts, ["please open a PR"])

    def test_compaction_summary_is_not_a_user_prompt(self):
        digest = _digest([
            _user("please open a PR"),
            _user("This session is being continued from a previous conversation "
                  "that ran out of context. Never do the wrong thing."),
        ])

        self.assertEqual(digest.user_prompts, ["please open a PR"])
        self.assertEqual(digest.feedback_signals, [])

    def test_records_without_a_type_are_still_harvested(self):
        digest = _digest([
            {"message": {"role": "user", "content": "Fix the parser bug"}},
            {"message": {"role": "assistant", "content": "The parser is fixed."}},
        ])

        self.assertEqual(digest.user_prompts, ["Fix the parser bug"])


if __name__ == "__main__":
    unittest.main()
