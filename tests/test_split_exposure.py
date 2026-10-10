"""Cross-night split provenance through the real cycle and persisted archive."""
from __future__ import annotations

import hashlib
import json
from unittest import mock

from skillopt_sleep.backend import build_backend
from skillopt_sleep.config import load_config
from skillopt_sleep.cycle import run_sleep_cycle
from skillopt_sleep.state import SleepState
from skillopt_sleep.types import TaskRecord


def _task(task_id, project, **kwargs):
    return TaskRecord(id=task_id, project=project, intent=f"task {task_id}",
                      reference_kind="exact", reference=f"answer {task_id}", **kwargs)


def _hash_id(split):
    for index in range(1000):
        task_id = f"normal-{index}"
        bucket = int(hashlib.sha256(("42" + task_id).encode()).hexdigest(), 16) % 100
        if ("val" if bucket < 10 else "test" if bucket < 90 else "train") == split:
            return task_id
    raise AssertionError("no matching hash bucket")


def test_fallback_exposure_survives_growing_shrinking_and_archive_eviction(tmp_path):
    project = str(tmp_path)
    cfg = load_config(invoked_project=project, projects="invoked", backend="mock",
                      state_dir=str(tmp_path / "state"), claude_home=str(tmp_path / ".claude"),
                      val_fraction=0.10, test_fraction=0.80, seed=42)
    backend = build_backend(backend="mock")
    first_ids = [f"all-test-{i}" for i in range(1, 6)]
    pools = [first_ids, first_ids + [_hash_id("train"), _hash_id("val")],
             ["all-test-2", "all-test-4", _hash_id("val")]]
    state_path = str(tmp_path / "state" / "state.json")
    for night, ids in enumerate(pools):
        # Only the external harvest/mining input is synthetic. Splitting,
        # consolidation, evidence, archive writes and state reloads are real.
        with mock.patch("skillopt_sleep.cycle.harvest_for_config", return_value=[]), \
             mock.patch("skillopt_sleep.mine.heuristic_mine", return_value=[_task(i, project) for i in ids]), \
             mock.patch("skillopt_sleep.cycle.build_backend", return_value=backend), \
             mock.patch.object(backend, "reflect", wraps=backend.reflect) as reflect:
            outcome = run_sleep_cycle(cfg)
        with open(outcome.staging_dir + "/evidence.jsonl", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle]
        ready = {r["task_id"]: r["split"] for r in rows if r["event"] == "task_ready"}
        assert ready["all-test-2"] == "train", (night, ready)
        assert any(t.id == "all-test-2" for call in reflect.call_args_list
                   for pairs in call.args[:2] for t, _ in pairs)
        held = [r for r in rows if r["event"] == "held_out_score"]
        assert held[0]["n_test"] == sum(s == "test" for s in ready.values())
        archived = SleepState.load(state_path).task_archive()
        assert any(t["id"] == "all-test-2" and t["split"] == "train" for t in archived)

    # Evict the raw task archive, then re-mine a derivative under a test hash.
    state = SleepState.load(state_path)
    state.add_to_archive([_task("unrelated", project).to_dict()], cap=1)
    state.save()
    derivative = _task(_hash_id("test"), project, derived_from="all-test-2", split="test")
    outcome = run_sleep_cycle(cfg, seed_tasks=[derivative, _task("gate", project, split="val")])
    assert derivative.split == "train"
    assert not outcome.report.holdout_leaked


def test_legacy_archive_exposure_is_loaded_before_scoring(tmp_path):
    project = str(tmp_path)
    cfg = load_config(invoked_project=project, projects="invoked", backend="mock",
                      state_dir=str(tmp_path / "state"), claude_home=str(tmp_path / ".claude"))
    state = SleepState.load(str(tmp_path / "state" / "state.json"))
    state.add_to_archive([_task("old", project, split="replay").to_dict()])
    state.save()
    tasks = [_task("old", project, split="test"), _task("gate", project, split="val")]
    run_sleep_cycle(cfg, seed_tasks=tasks)
    assert tasks[0].split == "train"


def test_lineage_survives_archive_eviction_and_is_project_scoped(tmp_path):
    state_path = str(tmp_path / "state.json")
    state = SleepState.load(state_path)
    state.protect_splits([
        _task("child", "/repo", split="test", derived_from="parent"),
        _task("grandchild", "/repo", split="test", derived_from="child"),
    ])
    state.save()
    state = SleepState.load(state_path)
    state.protect_splits([_task("grandchild", "/repo", split="train")])
    state.save()
    state = SleepState.load(state_path)
    tasks = [_task("parent", "/repo", split="test"),
             _task("sibling", "/repo", split="val", derived_from="parent"),
             _task("parent", "/other", split="test")]
    state.protect_splits(tasks)
    assert [t.split for t in tasks] == ["train", "train", "test"]


def test_exposed_only_pool_cannot_be_certified_as_clean_holdout(tmp_path):
    project = str(tmp_path)
    cfg = load_config(invoked_project=project, projects="invoked", backend="mock",
                      state_dir=str(tmp_path / "state"), claude_home=str(tmp_path / ".claude"))
    run_sleep_cycle(cfg, seed_tasks=[_task("a", project), _task("b", project)])
    outcome = run_sleep_cycle(cfg, seed_tasks=[_task("a", project, split="test"),
                                               _task("b", project, split="val")])
    assert outcome.report.holdout_leaked
    assert not outcome.report.accepted
    with open(outcome.staging_dir + "/evidence.jsonl", encoding="utf-8") as handle:
        assert all(json.loads(line)["event"] != "held_out_score" for line in handle)
