"""Two hardening passes x three approaches for EXC-001 split/recall hygiene.

Pass 1 (unit / adversarial):
  A recall edge vectors
  B assign_splits invariants (docstring claims)
  C fraction boundary validation

Pass 2 (integration / provenance):
  A dream_consolidate recall envelope
  B nightly cycle with preloaded archive + recall_k
  C on-disk config provenance for alias precedence
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

from skillopt_sleep.backend import build_backend
from skillopt_sleep.config import load_config
from skillopt_sleep.cycle import _resolve_split_fractions, run_sleep_cycle
from skillopt_sleep.consolidate import consolidate
from skillopt_sleep.dream import dream_consolidate, recall_similar
from skillopt_sleep.mine import assign_splits
from skillopt_sleep.state import SleepState
from skillopt_sleep.types import TaskRecord


def _task(task_id: str, intent: str, *, split: str = "train", origin: str = "real"):
    return TaskRecord(
        id=task_id,
        project="/repo/example",
        intent=intent,
        reference_kind="exact",
        reference=f"answer-{task_id}",
        split=split,
        origin=origin,
    )


class Pass1ApproachARecallEdgeVectors(unittest.TestCase):
    """Pass 1 / approach A: adversarial recall vectors."""

    def test_legacy_holdout_archived_task_not_recalled(self):
        new = _task("n1", "validate login form")
        archived = _task("old", "validate login form fields", split="holdout")
        self.assertEqual(recall_similar([new], [archived], k=1), [])

    def test_archived_val_not_recalled(self):
        new = _task("n1", "validate login form")
        archived = _task("old", "validate login form fields", split="val")
        self.assertEqual(recall_similar([new], [archived], k=1), [])

    def test_unrelated_archived_train_can_still_be_recalled(self):
        new = _task("n1", "validate login form")
        archived = _task(
            "old",
            "validate login form fields",
            split="replay",
        )
        recalled = recall_similar([new], [archived], k=1)
        self.assertEqual(len(recalled), 1)
        self.assertEqual(recalled[0].split, "train")
        self.assertTrue(recalled[0].id.startswith("recall:"))

    def test_zero_similarity_returns_empty(self):
        new = _task("n1", "alpha beta gamma")
        archived = _task("old", "delta epsilon zeta", split="train")
        self.assertEqual(recall_similar([new], [archived], k=3), [])


class Pass1ApproachBAssignSplitsInvariants(unittest.TestCase):
    """Pass 1 / approach B: docstring counting/ordering claims."""

    def test_dream_tasks_always_train_even_with_high_test_fraction(self):
        real = [_task(f"r{i}", f"real task {i}") for i in range(4)]
        dream = [_task("d0", "dream variant", origin="dream")]
        out = assign_splits(
            real + dream,
            val_fraction=0.34,
            test_fraction=0.10,
            seed=42,
        )
        for t in out:
            if t.origin == "dream":
                self.assertEqual(t.split, "train")

    def test_real_tasks_have_exactly_one_split_label(self):
        tasks = assign_splits(
            [_task(f"t{i}", f"task {i}") for i in range(12)],
            val_fraction=0.34,
            test_fraction=0.10,
            seed=7,
        )
        for t in tasks:
            if t.origin != "dream":
                self.assertIn(t.split, {"train", "val", "test"})

    def test_hash_assigned_test_not_demoted_for_val_top_up(self):
        """val top-up promotes from train only; hash test labels stay test."""
        tasks = assign_splits(
            [_task(f"t{i}", f"task {i}") for i in range(6)],
            val_fraction=0.01,
            test_fraction=0.50,
            seed=42,
        )
        test_ids = {t.id for t in tasks if t.split == "test"}
        again = assign_splits(
            [_task(f"t{i}", f"task {i}") for i in range(7)],
            val_fraction=0.01,
            test_fraction=0.50,
            seed=42,
        )
        for t in again:
            if t.id in test_ids:
                self.assertEqual(t.split, "test")

    def test_val_and_train_pull_from_test_when_empty(self):
        """When all real tasks hash to test, val and train borrow from test to satisfy non-empty guarantees."""
        # Seed 42 with this configuration was observed to put all tasks into test
        tasks = assign_splits(
            [_task(f"t{i}", f"task {i}") for i in range(5)],
            val_fraction=0.10,
            test_fraction=0.80,
            seed=42,
        )
        splits = {t.split for t in tasks}
        self.assertIn("val", splits, "val must not be empty even if it needs to borrow from test")
        self.assertIn("train", splits, "train must not be empty even if it needs to borrow from test")
        self.assertIn("test", splits, "test must still contain remaining tasks")


class Pass1ApproachCFractionBoundaries(unittest.TestCase):
    """Pass 1 / approach C: reject invalid fraction knobs early."""

    def test_assign_splits_rejects_negative_test_fraction(self):
        with self.assertRaises(ValueError):
            assign_splits([_task("t0", "x")], test_fraction=-0.1)

    def test_assign_splits_rejects_fraction_sum_ge_one(self):
        with self.assertRaises(ValueError):
            assign_splits([_task("t0", "x")], val_fraction=0.6, test_fraction=0.5)

    def test_mine_forwards_invalid_fractions_to_assign_splits(self):
        from skillopt_sleep.mine import mine

        with self.assertRaises(ValueError):
            mine([], llm_miner=lambda d: [_task("t0", "x")], test_fraction=1.5)


class Pass2ApproachADreamConsolidateEnvelope(unittest.TestCase):
    """Pass 2 / approach A: recall enlarges train only."""

    def test_recall_rows_are_train_split_only(self):
        backend = build_backend(backend="mock")
        tonight = assign_splits(
            [_task("t0", "validate login form", split="train"),
             _task("t1", "validate login form fields", split="val"),
             _task("t2", "validate login form errors", split="test")],
            val_fraction=0.34,
            test_fraction=0.10,
            seed=42,
        )
        archive = [
            _task("arch-test", "validate login form fields", split="test"),
            _task("arch-train", "validate login form helper", split="train"),
        ]
        seen: list[TaskRecord] = []

        def _capture(backend, tasks, skill, memory, **kwargs):
            seen.extend(tasks)
            return consolidate(backend, tasks, skill, memory, **kwargs)

        with mock.patch("skillopt_sleep.dream.consolidate", side_effect=_capture):
            dream_consolidate(
                backend,
                tonight,
                skill="# skill",
                memory="",
                history_tasks=archive,
                recall_k=2,
                dream_rollouts=1,
                dream_factor=0,
                gate_mode="off",
            )
        recalled = [t for t in seen if t.id.startswith("recall:")]
        self.assertGreater(len(recalled), 0, "expected at least one recalled row")
        for t in recalled:
            self.assertEqual(t.split, "train")
        self.assertFalse(
            any(t.id == "recall:arch-test" for t in recalled),
            "archived test rows must not be recalled",
        )


class Pass2ApproachBCycleArchiveRecall(unittest.TestCase):
    """Pass 2 / approach B: nightly path with archive + recall_k."""

    def test_archived_test_never_recalled_through_cycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = os.path.join(tmp, "state")
            state_path = os.path.join(state_dir, "state.json")
            state = SleepState.load(state_path)
            state.add_to_archive([
                _task(
                    "leak-me",
                    "validate login form fields",
                    split="test",
                ).to_dict(),
            ])
            state.save()

            cfg = load_config(
                invoked_project=tmp,
                projects="invoked",
                backend="mock",
                state_dir=state_dir,
                claude_home=os.path.join(tmp, ".claude"),
                recall_k=2,
                test_fraction=0.0,
            )
            seed = assign_splits(
                [_task("n1", "validate login form", split="train"),
                 _task("n2", "validate login form helper", split="val")],
                val_fraction=0.34,
                test_fraction=0.0,
                seed=42,
            )
            with mock.patch(
                "skillopt_sleep.dream.recall_similar",
                wraps=recall_similar,
            ) as spy:
                run_sleep_cycle(cfg, seed_tasks=seed, dry_run=True)
            self.assertGreater(spy.call_count, 0)
            for _args, kwargs in spy.call_args_list:
                history = _args[1]
                recalled = recall_similar(*_args, **kwargs)
                for row in recalled:
                    self.assertEqual(row.split, "train")
                for row in history:
                    if row.id == "leak-me":
                        self.assertEqual(row.split, "test")
                        self.assertEqual(
                            [r for r in recalled if r.derived_from == "leak-me"],
                            [],
                        )


class Pass2ApproachCConfigFileProvenance(unittest.TestCase):
    """Pass 2 / approach C: user file keys beat alias guessing."""

    def test_on_disk_zero_val_beats_holdout_alias(self):
        with tempfile.TemporaryDirectory() as cfg_dir:
            cfg_path = os.path.join(cfg_dir, "config.json")
            with open(cfg_path, "w", encoding="utf-8") as fh:
                json.dump(
                    {"val_fraction": 0.0, "holdout_fraction": 0.5},
                    fh,
                )
            with mock.patch(
                "skillopt_sleep.config._user_config_path",
                return_value=cfg_path,
            ):
                val, test = _resolve_split_fractions(load_config())
            self.assertEqual(val, 0.0)
            self.assertEqual(test, 0.0)


if __name__ == "__main__":
    unittest.main()
