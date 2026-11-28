"""The heuristic miner respects the order of feedback within a session.

``heuristic_mine`` documents "negative feedback present and no later positive
-> fail". Before per-turn signals were recorded, any negative signal gave
"fail", so a session where the user said "that's wrong" and then "perfect,
thanks" was mined as a failure.

Pure-stdlib (unittest), deterministic, no API key, no third-party deps.
Run:  python -m pytest tests/test_sleep_feedback_order.py
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest

from skillopt_sleep.harvest import digest_transcript
from skillopt_sleep.mine import heuristic_mine
from skillopt_sleep.types import SessionDigest


def _digest(turns, *, signals=None, by_turn=True):
    """A digest whose user turns carry the given per-turn signals."""
    flat = [s for turn in turns for s in turn]
    return SessionDigest(
        session_id="s1",
        project="/p",
        user_prompts=[f"user request number {i}" for i in range(len(turns))],
        assistant_finals=["done"],
        feedback_signals=flat if signals is None else signals,
        feedback_by_turn=[list(t) for t in turns] if by_turn else [],
        n_user_turns=len(turns),
    )


def _outcome(digest):
    tasks = heuristic_mine([digest])
    assert len(tasks) == 1
    return tasks[0].outcome


class TestOrderedOutcome(unittest.TestCase):
    def test_negative_then_later_approval_is_success(self):
        d = _digest([[], ["neg:wrong"], ["pos:perfect", "pos:thanks"]])
        self.assertEqual(_outcome(d), "success")

    def test_approval_then_later_negative_is_fail(self):
        d = _digest([[], ["pos:thanks"], ["neg:still broken"]])
        self.assertEqual(_outcome(d), "fail")

    def test_negative_and_positive_in_the_last_turn_is_fail(self):
        # "thanks, but it's still broken"
        d = _digest([[], ["neg:still broken", "pos:thanks"]])
        self.assertEqual(_outcome(d), "fail")

    def test_negative_after_approval_after_negative_is_fail(self):
        d = _digest([["neg:wrong"], ["pos:thanks"], ["neg:broken"]])
        self.assertEqual(_outcome(d), "fail")

    def test_only_negative_is_still_fail(self):
        d = _digest([[], ["neg:wrong"]])
        self.assertEqual(_outcome(d), "fail")

    def test_only_positive_is_still_success(self):
        d = _digest([[], ["pos:thanks"]])
        self.assertEqual(_outcome(d), "success")


class TestFallsBackWithoutTrustworthyOrder(unittest.TestCase):
    def test_without_per_turn_signals_any_negative_is_fail(self):
        # Harvesters that do not record order, and digests persisted before
        # the field existed, keep the old rule.
        d = _digest([["neg:wrong"], ["pos:thanks"]], by_turn=False)
        self.assertEqual(_outcome(d), "fail")

    def test_signal_outside_any_turn_disables_ordering(self):
        # A harvester-added signal (e.g. a tool error) has no turn position.
        d = _digest(
            [["neg:wrong"], ["pos:thanks"]],
            signals=["neg:wrong", "pos:thanks", "neg:tool_error"],
        )
        self.assertEqual(_outcome(d), "fail")

    def test_persisted_digest_without_the_field_loads(self):
        old = {
            "session_id": "s1", "project": "/p", "user_prompts": ["Fix the export bug"],
            "feedback_signals": ["neg:wrong", "pos:thanks"], "n_user_turns": 2,
        }
        known = set(SessionDigest.__dataclass_fields__)
        d = SessionDigest(**{k: v for k, v in old.items() if k in known})
        self.assertEqual(d.feedback_by_turn, [])
        self.assertEqual(_outcome(d), "fail")


def _user(text):
    return {"type": "user", "message": {"role": "user", "content": text}}


def _assistant(text):
    return {"type": "assistant", "message": {"role": "assistant", "content": text}}


class TestClaudeHarvesterRecordsTurns(unittest.TestCase):
    def _harvest(self, rows):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "session.jsonl")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("\n".join(json.dumps(row) for row in rows))
            return digest_transcript(path)

    def test_signals_are_grouped_by_user_turn(self):
        digest = self._harvest([
            _user("Export the report as CSV"),
            _assistant("Here is the CSV."),
            _user("The totals column is wrong"),
            _assistant("Fixed the totals."),
            _user("Perfect, thanks"),
        ])

        self.assertEqual(len(digest.feedback_by_turn), 3)
        self.assertEqual(digest.feedback_by_turn[0], [])
        self.assertIn("neg:wrong", digest.feedback_by_turn[1])
        self.assertTrue(all(s.startswith("pos:") for s in digest.feedback_by_turn[2]))
        self.assertEqual(
            [s for turn in digest.feedback_by_turn for s in turn],
            digest.feedback_signals,
        )
        self.assertEqual(_outcome(digest), "success")

    def test_session_left_broken_is_still_a_failure(self):
        digest = self._harvest([
            _user("Export the report as CSV"),
            _assistant("Here is the CSV."),
            _user("Thanks. Hmm, the totals column is still wrong"),
        ])

        self.assertEqual(_outcome(digest), "fail")


if __name__ == "__main__":
    unittest.main()
