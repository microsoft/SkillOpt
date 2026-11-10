"""val_fraction / test_fraction must actually flow from config to the splits.

The config documents three knobs (`holdout_fraction` as a legacy alias of
`val_fraction`, plus `test_fraction`), and ``assign_splits`` implements all
three -- but the nightly path only ever forwarded ``holdout_fraction``, so
``test_fraction`` was dead config: no untouched test split could exist and no
held-out test score was ever recorded. These tests pin the wiring end to end.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest

from skillopt_sleep.config import DEFAULTS, load_config
from skillopt_sleep.cycle import _resolve_split_fractions, run_sleep_cycle
from skillopt_sleep.mine import assign_splits, mine
from skillopt_sleep.types import TaskRecord


def _mk_tasks(n):
    # Realistic mined tasks: exact-reference judged, unique stable ids.
    return [
        TaskRecord(
            id=f"t{i:03d}", project="/repo/example",
            intent=f"do the recurring thing number {i}",
            reference_kind="exact", reference=f"answer {i}",
        )
        for i in range(n)
    ]


class TestMineForwardsFractions(unittest.TestCase):
    def test_test_fraction_reaches_assign_splits(self):
        tasks = mine(
            [], llm_miner=lambda digests: _mk_tasks(60),
            max_tasks=60, val_fraction=0.2, test_fraction=0.3, seed=7,
        )
        splits = {t.split for t in tasks}
        self.assertIn("test", splits, "test_fraction did not reach assign_splits")
        self.assertIn("val", splits)
        self.assertIn("train", splits)

    def test_default_call_is_two_way_like_before(self):
        tasks = mine([], llm_miner=lambda digests: _mk_tasks(60), max_tasks=60)
        self.assertEqual({t.split for t in tasks} - {"train", "val"}, set(),
                         "defaults must reproduce the legacy two-way split")

    def test_legacy_holdout_alias_still_wins_when_passed(self):
        a = mine([], llm_miner=lambda d: _mk_tasks(60), max_tasks=60,
                 holdout_fraction=0.6, seed=7)
        b = mine([], llm_miner=lambda d: _mk_tasks(60), max_tasks=60,
                 val_fraction=0.6, seed=7)
        self.assertEqual([t.split for t in a], [t.split for t in b])


class TestConfigAliasPrecedence(unittest.TestCase):
    def _cfg(self, **over):
        return load_config(invoked_project="/tmp/x", projects="invoked", **over)

    def test_defaults_resolve_to_legacy_behavior(self):
        val, test = _resolve_split_fractions(self._cfg())
        self.assertEqual(val, DEFAULTS["val_fraction"])
        self.assertEqual(test, DEFAULTS["test_fraction"])

    def test_legacy_config_holdout_only_still_wins(self):
        val, _ = _resolve_split_fractions(self._cfg(holdout_fraction=0.2))
        self.assertEqual(val, 0.2)

    def test_user_val_fraction_beats_stale_alias_default(self):
        val, _ = _resolve_split_fractions(self._cfg(val_fraction=0.5))
        self.assertEqual(val, 0.5)

    def test_explicit_val_fraction_beats_explicit_alias(self):
        val, _ = _resolve_split_fractions(
            self._cfg(val_fraction=0.5, holdout_fraction=0.2))
        self.assertEqual(val, 0.5)

    def test_test_fraction_flows(self):
        _, test = _resolve_split_fractions(self._cfg(test_fraction=0.25))
        self.assertEqual(test, 0.25)

    def test_explicit_zero_val_fraction_is_preserved(self):
        val, _ = _resolve_split_fractions(self._cfg(val_fraction=0.0))
        self.assertEqual(val, 0.0)

    def test_explicit_val_at_default_beats_alias(self):
        val, _ = _resolve_split_fractions(
            self._cfg(val_fraction=0.34, holdout_fraction=0.5))
        self.assertEqual(val, 0.34)

    def test_invalid_fraction_sum_raises(self):
        with self.assertRaises(ValueError):
            _resolve_split_fractions(self._cfg(val_fraction=0.6, test_fraction=0.5))


class TestSplitStabilityAcrossNights(unittest.TestCase):
    def test_appending_tasks_does_not_reassign_existing_test(self):
        first = assign_splits(
            _mk_tasks(4), val_fraction=0.34, test_fraction=0.10, seed=42,
        )
        splits_first = {t.id: t.split for t in first}
        second = assign_splits(
            _mk_tasks(5), val_fraction=0.34, test_fraction=0.10, seed=42,
        )
        splits_second = {t.id: t.split for t in second}
        for task_id, split in splits_first.items():
            self.assertEqual(
                splits_second[task_id], split,
                f"{task_id} changed split when a new task was appended",
            )

    def test_maintainer_repro_seed_42_t1_does_not_reassign_t4(self):
        """Hash splits are id-stable; appending t1 must not move t4's split."""
        base = [TaskRecord(id=f"t{i}", project="/p", intent=f"task {i}") for i in (0, 3, 4)]
        with_t1 = base + [TaskRecord(id="t1", project="/p", intent="task 1")]
        first = assign_splits(list(base), val_fraction=0.34, test_fraction=0.10, seed=42)
        second = assign_splits(list(with_t1), val_fraction=0.34, test_fraction=0.10, seed=42)
        self.assertEqual(next(t for t in second if t.id == "t1").split, "test")
        self.assertEqual(
            next(t for t in first if t.id == "t4").split,
            next(t for t in second if t.id == "t4").split,
        )


class TestRecallSplitHygiene(unittest.TestCase):
    def test_archived_test_tasks_are_never_recalled(self):
        from skillopt_sleep.dream import recall_similar

        new = TaskRecord(id="n1", project="/p", intent="validate login form")
        archived_test = TaskRecord(
            id="same-id", project="/p", intent="validate login form fields",
            split="test",
        )
        recalled = recall_similar([new], [archived_test], k=1)
        self.assertEqual(recalled, [])

    def test_recall_honors_exclude_ids_for_tonights_held_out(self):
        from skillopt_sleep.dream import recall_similar

        held = TaskRecord(
            id="held-out", project="/p", intent="validate login form fields",
            split="train",
        )
        new_train = TaskRecord(
            id="n1", project="/p", intent="validate login form", split="train",
        )
        recalled = recall_similar(
            [new_train], [held], k=1, exclude_ids={"held-out"},
        )
        self.assertEqual(recalled, [])


class TestHeldOutScoreEvidence(unittest.TestCase):
    """A night with test-split tasks writes a write-only held-out score row."""

    def _run_night(self, tasks):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config(
                invoked_project=tmp, projects="invoked", backend="mock",
                state_dir=os.path.join(tmp, "state"),
                claude_home=os.path.join(tmp, ".claude"),
            )
            run_sleep_cycle(cfg, seed_tasks=tasks, dry_run=True)
            ev_dir = os.path.join(cfg.state_dir, "evidence")
            rows = []
            for name in sorted(os.listdir(ev_dir)):
                with open(os.path.join(ev_dir, name), encoding="utf-8") as fh:
                    rows += [json.loads(line) for line in fh if line.strip()]
            return rows

    def _seed(self, with_test):
        tasks = _mk_tasks(6)
        for i, t in enumerate(tasks):
            t.split = "train" if i < 3 else ("val" if i < 5 else
                                             ("test" if with_test else "val"))
        return tasks

    def test_score_row_written_when_test_tasks_exist(self):
        rows = self._run_night(self._seed(with_test=True))
        score_rows = [r for r in rows
                      if r.get("stage") == "test"
                      and r.get("event") == "held_out_score"]
        self.assertEqual(len(score_rows), 1)
        row = score_rows[0]
        self.assertEqual(row["n_test"], 1)
        self.assertIn("hard", row)
        self.assertIn("soft", row)
        self.assertIn("accepted", row)

    def test_no_score_row_without_test_tasks(self):
        rows = self._run_night(self._seed(with_test=False))
        self.assertEqual(
            [r for r in rows if r.get("stage") == "test"], [],
            "no test tasks => no held-out row => bit-for-bit legacy nights",
        )


if __name__ == "__main__":
    unittest.main()


def test_parse_split_ratio_list_and_tuple() -> None:
    from skillopt.datasets.base import _compute_split_counts, _parse_split_ratio
    assert _parse_split_ratio([2, 1, 7]) == (2, 1, 7)
    assert _parse_split_ratio((1, 1, 1)) == (1, 1, 1)
    assert _parse_split_ratio("2:1:7") == (2, 1, 7)
    assert _compute_split_counts(0, (2, 1, 7)) == (0, 0, 0)
    assert _compute_split_counts(-5, (2, 1, 7)) == (0, 0, 0)
