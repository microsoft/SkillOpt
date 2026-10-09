"""Recall must never put a held-out task's twin into the training pool.

Task ids hash the project with the intent, and the recall archive is shared by
every project in one state directory. Blocking recall by id alone therefore let
another project's copy of tonight's val task re-enter training under a
different id, while the leak check (also id-based) reported a clean split.
"""
from __future__ import annotations

import os

from skillopt_sleep import dream
from skillopt_sleep.backend import MockBackend
from skillopt_sleep.config import load_config
from skillopt_sleep.consolidate import _split, consolidate, task_content_key
from skillopt_sleep.cycle import run_sleep_cycle
from skillopt_sleep.dream import recall_similar
from skillopt_sleep.types import TaskRecord

VAL_INTENT = "give the final revenue total for the report"


def _task(task_id, project, split, intent, rule="wrap-answer", reference="1200"):
    return TaskRecord(
        id=task_id,
        project=project,
        intent=intent,
        reference_kind="exact",
        reference=reference,
        tags=[f"rule:{rule}"],
        split=split,
        origin="real",
    )


def test_content_key_ignores_id_project_case_and_whitespace() -> None:
    a = _task("a1", "/a", "train", "Give the final  revenue total\nfor the report")
    b = _task("b2", "/b", "val", VAL_INTENT)
    assert task_content_key(a) == task_content_key(b)
    c = _task("c1", "/b", "val", VAL_INTENT)
    c.context_excerpt = "Q3 only"
    assert task_content_key(c) != task_content_key(b)


def test_recall_blocks_a_held_out_twin_archived_under_another_id() -> None:
    tonight_val = _task("b2", "/b", "val", VAL_INTENT)
    tonight_train = _task("b1", "/b", "train", "list the revenue sources for the report")
    history = [
        _task("a1", "/a", "train", VAL_INTENT.upper() + "  "),
        _task("a3", "/a", "train", "list the revenue sources for the quarterly report"),
    ]

    unguarded = recall_similar([tonight_train], history, 5, exclude_ids={"b2"})
    guarded = recall_similar(
        [tonight_train], history, 5, exclude_ids={"b2"}, exclude_tasks=[tonight_val]
    )

    assert [t.derived_from for t in unguarded] == ["a3", "a1"]
    # The twin is blocked; similar, non-identical experience is still recalled.
    assert [t.derived_from for t in guarded] == ["a3"]


def test_split_flags_a_content_twin_of_val_as_a_leak() -> None:
    train = _task("recall:a1", "/a", "train", VAL_INTENT)
    val = _task("b2", "/b", "val", VAL_INTENT)
    other = _task("b1", "/b", "train", "list the revenue sources for the report")

    _train, _val, leaked = _split([other, train, val])
    assert leaked is True
    _train, _val, leaked = _split([other, val])
    assert leaked is False


def test_gate_abstains_when_val_has_a_twin_in_train() -> None:
    tasks = [
        _task("t1", "/b", "train", "list the revenue sources for the report"),
        _task("recall:a1", "/a", "train", VAL_INTENT),
        _task("b2", "/b", "val", VAL_INTENT),
    ]
    result = consolidate(MockBackend(), tasks, "", "", evolve_memory=False)
    assert result.holdout_leaked is True
    assert result.accepted is False


def _night(tmp_path, project, tasks, *, recall_k):
    config = load_config(
        invoked_project=str(project),
        projects="invoked",
        backend="mock",
        state_dir=str(tmp_path / "state"),
        claude_home=str(tmp_path / ".claude"),
        recall_k=recall_k,
        evolve_memory=False,
    )
    return run_sleep_cycle(config, seed_tasks=tasks, dry_run=recall_k > 0)


def test_two_project_cycle_never_trains_on_tonights_val_twin(tmp_path, monkeypatch) -> None:
    """Project A archives a task as train; project B later holds the same request
    out as val. Recall must not hand B's gate its own val answer."""
    project_a = tmp_path / "a"
    project_b = tmp_path / "b"
    project_a.mkdir()
    project_b.mkdir()
    pools: list[list[TaskRecord]] = []
    original = dream.consolidate

    def spy(backend, tasks, *args, **kwargs):
        pools.append(list(tasks))
        return original(backend, tasks, *args, **kwargs)

    monkeypatch.setattr(dream, "consolidate", spy)
    _night(
        tmp_path,
        project_a,
        [
            _task("a1", str(project_a), "train", VAL_INTENT),
            _task("a2", str(project_a), "val", "unrelated chore", "json-only", "{}"),
        ],
        recall_k=0,
    )
    pools.clear()
    outcome = _night(
        tmp_path,
        project_b,
        [
            _task(
                "b1", str(project_b), "train",
                "list the revenue sources for the report", "units-si", "9 kg",
            ),
            _task("b2", str(project_b), "val", VAL_INTENT),
        ],
        recall_k=5,
    )

    assert os.path.exists(tmp_path / "state" / "state.json")
    twins = [
        t for t in pools[0]
        if t.split == "train" and task_content_key(t) == (VAL_INTENT, "")
    ]
    assert twins == []
    # Without the twin, nothing in B's training teaches the val answer.
    assert outcome.report.gate_action != "accept_new_best"
    assert outcome.report.accepted is False
