"""Tests for propagating harvested skill hints into mined tasks (issue #120).

Pure-stdlib (unittest), deterministic, no API key, no third-party deps.
Run:  python -m pytest tests/test_sleep_skill_hint.py
"""
from __future__ import annotations

import unittest
from itertools import permutations, product

from skillopt_sleep.dream import dream_augment, recall_similar
from skillopt_sleep.llm_miner import _mk_task
from skillopt_sleep.mine import dedup_tasks, heuristic_mine, session_skill_hint
from skillopt_sleep.types import SessionDigest, TaskRecord


def _digest(session_id="s1", skills=(), project="/repo/example", prompt="add a login form"):
    return SessionDigest(
        session_id=session_id,
        project=project,
        user_prompts=[prompt],
        assistant_finals=["done"],
        tools_used=["Skill"] if skills else ["Read"],
        skills_used=list(skills),
    )


class TestTaskRecordSkillHint(unittest.TestCase):
    def test_defaults_to_empty_string(self):
        self.assertEqual(TaskRecord(id="t1", project="/repo/example", intent="x").skill_hint, "")

    def test_serialization_roundtrip(self):
        task = TaskRecord(
            id="t1", project="/repo/example", intent="x", skill_hint="example-skill"
        )
        self.assertEqual(task.to_dict()["skill_hint"], "example-skill")
        self.assertEqual(TaskRecord.from_dict(task.to_dict()).skill_hint, "example-skill")

    def test_legacy_payload_without_skill_hint_still_loads(self):
        legacy = {"id": "t1", "project": "/repo/example", "intent": "x", "split": "val"}
        task = TaskRecord.from_dict(legacy)
        self.assertEqual(task.skill_hint, "")
        self.assertEqual(task.split, "val")

    def test_existing_positional_fields_keep_their_slots(self):
        task = TaskRecord(
            "t1", "/repo/example", "x", "context", "system", "attempt",
            "success", "exact", "answer", {}, ["tag"], ["s1"],
            "val", "dream", "source-task",
        )
        self.assertEqual(task.split, "val")
        self.assertEqual(task.origin, "dream")
        self.assertEqual(task.derived_from, "source-task")
        self.assertEqual(task.skill_hint, "")


class TestSessionSkillHint(unittest.TestCase):
    def test_single_skill_is_unambiguous(self):
        self.assertEqual(session_skill_hint(_digest(skills=["example-skill"])), "example-skill")

    def test_no_skill_is_empty(self):
        self.assertEqual(session_skill_hint(_digest()), "")

    def test_several_skills_are_ambiguous(self):
        digest = _digest(skills=["example-skill", "second-skill"])
        self.assertEqual(session_skill_hint(digest), "")

    def test_repeated_single_skill_stays_unambiguous(self):
        digest = _digest(skills=["example-skill", "example-skill"])
        self.assertEqual(session_skill_hint(digest), "example-skill")


class TestHeuristicMineSkillHint(unittest.TestCase):
    def test_hint_propagates_from_session(self):
        tasks = heuristic_mine([_digest(skills=["example-skill"])])
        self.assertEqual([t.skill_hint for t in tasks], ["example-skill"])

    def test_ambiguous_and_absent_hints_stay_in_catch_all(self):
        tasks = heuristic_mine([
            _digest(session_id="s1", skills=["example-skill", "second-skill"]),
            _digest(session_id="s2", prompt="write the release notes"),
        ])
        self.assertEqual(len(tasks), 2)
        self.assertEqual([t.skill_hint for t in tasks], ["", ""])

    def test_dedup_keeps_agreeing_hint_and_clears_conflicting_one(self):
        agreed = dedup_tasks([
            TaskRecord(id="t1", project="/p", intent="x", skill_hint="example-skill",
                       source_sessions=["s1"]),
            TaskRecord(id="t1", project="/p", intent="x", skill_hint="example-skill",
                       source_sessions=["s2"]),
        ])
        self.assertEqual([t.skill_hint for t in agreed], ["example-skill"])

        filled = dedup_tasks([
            TaskRecord(id="t1", project="/p", intent="x", source_sessions=["s1"]),
            TaskRecord(id="t1", project="/p", intent="x", skill_hint="example-skill",
                       source_sessions=["s2"]),
        ])
        self.assertEqual([t.skill_hint for t in filled], ["example-skill"])

        conflicting = dedup_tasks([
            TaskRecord(id="t1", project="/p", intent="x", skill_hint="example-skill",
                       source_sessions=["s1"]),
            TaskRecord(id="t1", project="/p", intent="x", skill_hint="second-skill",
                       source_sessions=["s2"]),
        ])
        self.assertEqual([t.skill_hint for t in conflicting], [""])

    def test_dedup_conflict_is_permutation_independent(self):
        sequences = set(permutations(["skill-a", "skill-b", "skill-a"]))
        sequences.update(permutations(["skill-b", "skill-a", "skill-b"]))
        for hints in sequences:
            with self.subTest(hints=hints):
                tasks = [
                    TaskRecord(id="t1", project="/p", intent="x", skill_hint=hint,
                               source_sessions=[f"s{idx}"])
                    for idx, hint in enumerate(hints)
                ]
                self.assertEqual(dedup_tasks(tasks)[0].skill_hint, "")

    def test_dedup_matches_final_non_empty_hint_set_exhaustively(self):
        for length in range(1, 5):
            for hints in product(("", "skill-a", "skill-b"), repeat=length):
                with self.subTest(hints=hints):
                    distinct = {hint for hint in hints if hint}
                    expected = next(iter(distinct)) if len(distinct) == 1 else ""
                    tasks = [
                        TaskRecord(id="t1", project="/p", intent="x", skill_hint=hint)
                        for hint in hints
                    ]
                    self.assertEqual(dedup_tasks(tasks)[0].skill_hint, expected)

    def test_dedup_ignores_empty_hints_when_resolving_final_set(self):
        cases = [
            (("", "skill-a", ""), "skill-a"),
            (("skill-a", "", "skill-a"), "skill-a"),
            (("", "", ""), ""),
            (("skill-a", "", "skill-b"), ""),
        ]
        for hints, expected in cases:
            with self.subTest(hints=hints):
                tasks = [
                    TaskRecord(id="t1", project="/p", intent="x", skill_hint=hint)
                    for hint in hints
                ]
                self.assertEqual(dedup_tasks(tasks)[0].skill_hint, expected)


class TestDreamAndRecallSkillHint(unittest.TestCase):
    def test_dream_augment_preserves_hint(self):
        source = TaskRecord(
            id="t1", project="/p", intent="add validation", skill_hint="example-skill",
            source_sessions=["session-1"],
        )
        dreamed = dream_augment([source], factor=1)
        self.assertEqual(len(dreamed), 1)
        self.assertEqual(dreamed[0].skill_hint, "example-skill")
        self.assertEqual(dreamed[0].source_sessions, ["session-1"])

    def test_recall_similar_preserves_hint(self):
        new = TaskRecord(id="new", project="/p", intent="add form validation")
        historical = TaskRecord(
            id="old", project="/p", intent="improve form validation",
            skill_hint="example-skill",
            source_sessions=["session-2"],
        )
        recalled = recall_similar([new], [historical], 1)
        self.assertEqual(len(recalled), 1)
        self.assertEqual(recalled[0].skill_hint, "example-skill")
        self.assertEqual(recalled[0].source_sessions, ["session-2"])


class TestLlmMinerSkillHint(unittest.TestCase):
    def test_rule_task_carries_the_hint(self):
        task = _mk_task(
            _digest(skills=["example-skill"]),
            {"intent": "add a login form", "checks": [{"op": "contains", "arg": "form"}]},
            0,
        )
        self.assertIsNotNone(task)
        self.assertEqual(task.reference_kind, "rule")
        self.assertEqual(task.skill_hint, "example-skill")

    def test_rubric_task_carries_the_hint(self):
        task = _mk_task(
            _digest(skills=["example-skill"]),
            {"intent": "add a login form", "rubric": "mentions validation"},
            0,
        )
        self.assertEqual(task.reference_kind, "rubric")
        self.assertEqual(task.skill_hint, "example-skill")

    def test_ambiguous_session_leaves_hint_empty(self):
        task = _mk_task(
            _digest(skills=["example-skill", "second-skill"]),
            {"intent": "add a login form", "rubric": "mentions validation"},
            0,
        )
        self.assertEqual(task.skill_hint, "")


if __name__ == "__main__":
    unittest.main()
