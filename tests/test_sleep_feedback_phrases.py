"""Feedback phrases match whole words, not substrings of other words.

The heuristic miner labels a task "fail" when any user turn carries a
negative phrase, so a phrase that fires inside an unrelated word mislabels
the task and sends it into replay as a failure.

Pure-stdlib (unittest), deterministic, no API key, no third-party deps.
Run:  python -m pytest tests/test_sleep_feedback_phrases.py
"""
from __future__ import annotations

import unittest

from skillopt_sleep.harvest import _detect_feedback
from skillopt_sleep.mine import heuristic_mine
from skillopt_sleep.types import SessionDigest


def _neg(text):
    return [s for s in _detect_feedback(text) if s.startswith("neg:")]


def _pos(text):
    return [s for s in _detect_feedback(text) if s.startswith("pos:")]


class TestNoSubstringMatches(unittest.TestCase):
    def test_negative_phrases_do_not_fire_inside_other_words(self):
        for text in (
            "Add a piano, a violin and drums to the demo",  # "no,"
            "Write a script that undoes the last migration",  # "undo"
            "Fix items in the backlog by priority",  # "fix it"
            "Mark the wrongful-termination clause for review",  # "wrong"
        ):
            with self.subTest(text=text):
                self.assertEqual(_neg(text), [])

    def test_positive_phrases_do_not_fire_inside_other_words(self):
        for text in (
            "Add a section on Venice",  # "nice"
            "Rename the prefixed columns",  # "fixed"
            "Pick the greatest common divisor",  # "great"
        ):
            with self.subTest(text=text):
                self.assertEqual(_pos(text), [])

    def test_incorrect_is_not_also_correct(self):
        self.assertEqual(_detect_feedback("That's incorrect"), ["neg:incorrect"])


class TestStillMatchesRealFeedback(unittest.TestCase):
    def test_existing_phrases_still_match(self):
        self.assertIn("neg:still broken", _detect_feedback("this is still broken"))
        self.assertIn("neg:no,", _detect_feedback("No, use the other file"))
        self.assertIn("neg:wrong", _detect_feedback("that output is wrong."))
        self.assertIn("pos:thanks", _detect_feedback("perfect, thanks"))
        self.assertIn("pos:perfect", _detect_feedback("perfect, thanks"))

    def test_curly_apostrophe_is_matched(self):
        self.assertIn("neg:didn't", _detect_feedback("It didn\u2019t work"))
        self.assertIn("neg:doesn't work", _detect_feedback("It doesn\u2019t work"))


class TestPositiveInsideNegative(unittest.TestCase):
    def test_not_fixed_is_not_also_fixed(self):
        signals = _detect_feedback("Still not fixed")
        self.assertIn("neg:not fixed", signals)
        self.assertNotIn("pos:fixed", signals)

    def test_negated_positive_is_not_approval(self):
        for text in ("no thanks", "that's not great", "it isn't correct",
                     "not really perfect", "still not great"):
            with self.subTest(text=text):
                self.assertEqual(_pos(text), [])

    def test_positive_after_unrelated_clause_still_counts(self):
        self.assertIn("pos:thanks",
                      _detect_feedback("Not what I pictured, but thanks"))


class TestMinedOutcome(unittest.TestCase):
    def test_opening_request_with_piano_is_not_a_failure(self):
        prompt = "Add a piano, a violin and drums to the demo"
        digest = SessionDigest(
            session_id="s1", project="/p", git_branch="", started_at="",
            ended_at="", user_prompts=[prompt], assistant_finals=["done"],
            feedback_signals=_detect_feedback(prompt), n_user_turns=1,
        )

        tasks = heuristic_mine([digest])

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].outcome, "unknown")


if __name__ == "__main__":
    unittest.main()
