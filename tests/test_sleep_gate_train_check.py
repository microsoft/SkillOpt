"""Training-side check: a candidate may not be clearly worse on tonight's training tasks."""

from __future__ import annotations

import importlib
import json
import math
import os

import pytest

import skillopt_sleep.cycle as cycle_module
from skillopt_sleep.backend import Backend, DualBackend
from skillopt_sleep.config import DEFAULTS, load_config
from skillopt_sleep.consolidate import (
    TRAIN_CHECK_MIN_NET_REGRESSED,
    consolidate,
    train_check_enabled,
    train_check_margin,
)
from skillopt_sleep.cycle import _render_report_md, run_sleep_cycle
from skillopt_sleep.dream import dream_consolidate
from skillopt_sleep.evidence import EvidenceLog, read_events
from skillopt_sleep.experiments.personas import researcher_persona
from skillopt_sleep.memory import set_learned
from skillopt_sleep.mine import assign_splits
from skillopt_sleep.types import EditRecord, SleepReport, TaskRecord

SKILL_RULE = "Always add the category of the answer in parentheses."
MEMORY_RULE = "Remember to restate the question first."
SKILL = set_learned("# Skill\n", [])
MEMORY = set_learned("# Memory\n", [])


def _task(task_id: str, split: str, **extra) -> TaskRecord:
    return TaskRecord(id=task_id, project="test", intent=f"question {task_id}",
                      split=split, **extra)


class ScriptedBackend(Backend):
    """Each task scores a fixed value per document state.

    ``script`` maps a task's source id to its scores for
    ``(no rule, skill rule, skill and memory rule)``; shorter tuples repeat
    their last value. The memory rule never appears without the skill rule in
    these tests.
    """

    name = "scripted-stub"

    def __init__(self, script: dict[str, tuple[float, ...]], propose_memory: bool = False):
        self.script = script
        self.propose_memory = propose_memory
        self.calls: list[tuple[str, str]] = []       # (task id, phase)

    def attempt(self, task, skill, memory, sample_id: int = 0) -> str:
        self.calls.append((task.id, self.evidence_phase))
        state = 2 if MEMORY_RULE in memory else (1 if SKILL_RULE in skill else 0)
        source = (task.derived_from or task.id).removeprefix("recall:")
        scores = self.script[source]
        return repr(scores[min(state, len(scores) - 1)])

    def judge(self, task, response):
        score = float(response)
        return score, score, "matched" if score >= 1.0 else "did not match"

    def reflect(self, failures, successes, skill, memory, *, edit_budget,
                evolve_skill, evolve_memory):
        if evolve_skill:
            return [EditRecord(target="skill", op="add", content=SKILL_RULE, rationale="r")]
        if evolve_memory and self.propose_memory:
            return [EditRecord(target="memory", op="add", content=MEMORY_RULE, rationale="r")]
        return []


def _run(script, *, train=5, val=("v1",), **kwargs):
    backend = ScriptedBackend(script)
    tasks = [_task(f"t{i}", "train") for i in range(train)] + [_task(v, "val") for v in val]
    kwargs.setdefault("evolve_memory", False)
    result = consolidate(backend, tasks, SKILL, MEMORY, gate_metric="hard", **kwargs)
    return backend, result


def _phases(backend: ScriptedBackend) -> set[str]:
    return {phase for _task_id, phase in backend.calls}


def _check_ran(backend: ScriptedBackend) -> bool:
    return any(phase.startswith("gate_train_check") for phase in _phases(backend))


# A rule that fixes the one validation task and breaks every training task.
HARMFUL = {"v1": (0.0, 1.0), **{f"t{i}": (1.0, 0.0) for i in range(5)}}
# A rule that fixes the validation task and the training tasks.
HELPFUL = {"v1": (0.0, 1.0), **{f"t{i}": (0.0, 1.0) for i in range(5)}}


# ── off by default, and then nothing changes ─────────────────────────────────


def test_default_makes_no_extra_replay_and_writes_the_same_records(tmp_path) -> None:
    backend, result = _run(HARMFUL)
    explicit_backend, explicit = _run(HARMFUL, gate_train_check=False)
    none_backend, none = _run(HARMFUL, gate_train_check=None)

    assert DEFAULTS["gate_train_check"] is False
    assert DEFAULTS["gate_train_margin"] == 0.1
    assert backend.calls == explicit_backend.calls == none_backend.calls
    assert not _check_ran(backend)
    # the validation slice alone accepts the harmful rule
    assert result.accepted is explicit.accepted is none.accepted is True
    for trial in result.gate_trials:
        assert set(trial) == {
            "target", "baseline_score", "candidate_score", "accepted",
            "blocked_by_regression", "task_deltas",
        }

    logged = ScriptedBackend(HARMFUL)
    path = os.path.join(tmp_path, "evidence.jsonl")
    logged.evidence = EvidenceLog(path)
    tasks = [_task(f"t{i}", "train") for i in range(5)] + [_task("v1", "val")]
    consolidate(logged, tasks, SKILL, MEMORY, gate_metric="hard", evolve_memory=False)
    events = read_events(path)
    assert not [e for e in events if e.get("event") == "train_check"]
    (trial,) = [e for e in events if e.get("stage") == "gate" and e.get("event") == "trial"]
    assert "blocked_by_train_check" not in trial


# ── what the check changes ───────────────────────────────────────────────────


def test_a_candidate_that_breaks_its_own_training_tasks_is_rejected() -> None:
    backend, result = _run(HARMFUL, gate_train_check=True)

    assert result.accepted is False
    assert SKILL_RULE not in result.new_skill
    assert [edit.content for edit in result.rejected_edits] == [SKILL_RULE]
    trial = result.gate_trials[0]
    assert trial["candidate_score"] > trial["baseline_score"]     # validation liked it
    assert trial["accepted"] is False
    assert trial["blocked_by_train_check"] is True
    check = trial["train_check"]
    assert (check["n_tasks"], check["margin"]) == (5, 0.1)
    assert (check["n_regressed"], check["n_improved"]) == (5, 0)
    assert check["min_net_regressed"] == TRAIN_CHECK_MIN_NET_REGRESSED == 2
    assert (check["baseline_score"], check["candidate_score"]) == (1.0, 0.0)
    assert check["regressed"] is True and check["blocked"] is True
    # both arms of every training task are kept
    assert [(row["task_id"], row["baseline_score"], row["candidate_score"], row["status"])
            for row in check["task_deltas"]] == [
        (f"t{i}", 1.0, 0.0, "regressed") for i in range(5)
    ]
    assert ("t0", "gate_train_check:skill") in backend.calls
    final = result.gate_trials[-1]
    assert final["target"] == "final"
    assert final["blocked_by_train_check"] is False and final["train_check"] is None
    json.dumps(result.gate_trials, allow_nan=False)


def test_a_candidate_that_helps_its_training_tasks_is_still_accepted() -> None:
    _, result = _run(HELPFUL, gate_train_check=True)

    assert result.accepted is True
    assert SKILL_RULE in result.new_skill
    check = result.gate_trials[0]["train_check"]
    assert (check["baseline_score"], check["candidate_score"]) == (0.0, 1.0)
    assert (check["n_regressed"], check["n_improved"]) == (0, 5)
    assert check["regressed"] is False and check["blocked"] is False
    assert result.gate_trials[0]["blocked_by_train_check"] is False


def test_the_check_never_accepts_what_validation_rejected() -> None:
    # Validation ties, so nothing is accepted and nothing is replayed for the check.
    script = {"v1": (1.0, 1.0), **{f"t{i}": (0.0, 1.0) for i in range(5)}}

    backend, result = _run(script, gate_train_check=True)

    assert result.accepted is False
    assert not _check_ran(backend)
    assert result.gate_trials[0]["train_check"] is None


def test_a_candidate_the_no_regression_rule_blocked_is_not_replayed() -> None:
    script = {"v1": (0.0, 1.0), "v2": (1.0, 0.5), **{f"t{i}": (1.0, 0.0) for i in range(5)}}

    backend, result = _run(script, val=("v1", "v2"), gate_train_check=True,
                           gate_no_regression=True)

    trial = result.gate_trials[0]
    assert trial["blocked_by_regression"] is True
    assert trial["blocked_by_train_check"] is False and trial["train_check"] is None
    assert not _check_ran(backend)


# ── the rejection rule ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("margin", "accepted"),
    [(0.0, False), (0.1, False), (0.39, False), (0.4, True), (0.5, True)],
)
def test_margin_is_the_tolerated_drop_in_mean_training_score(margin, accepted) -> None:
    # Two of five training tasks go from right to wrong: the mean drops by 0.4.
    script = {"v1": (0.0, 1.0), "t0": (1.0, 0.0), "t1": (1.0, 0.0),
              **{f"t{i}": (1.0, 1.0) for i in range(2, 5)}}

    _, result = _run(script, gate_train_check=True, gate_train_margin=margin)

    check = result.gate_trials[0]["train_check"]
    assert check["baseline_score"] == pytest.approx(1.0)
    assert check["candidate_score"] == pytest.approx(0.6)
    assert (check["n_regressed"], check["n_improved"]) == (2, 0)
    assert result.accepted is accepted
    assert check["blocked"] is (not accepted)


@pytest.mark.parametrize("passing_before", [3, 4, 5, 6, 7, 8, 9, 10])
def test_a_drop_of_exactly_the_margin_is_tolerated_at_every_baseline(passing_before) -> None:
    # Ten training tasks; two go from right to wrong, so the mean drops by
    # exactly 0.2. With margin 0.2 that is tolerated whatever the baseline is
    # (0.8 - 0.2 is not 0.6 in floating point).
    script = {"v1": (0.0, 1.0)}
    for i in range(10):
        before = 1.0 if i < passing_before else 0.0
        script[f"t{i}"] = (before, 0.0 if i < 2 else before)

    _, at_margin = _run(script, train=10, gate_train_check=True, gate_train_margin=0.2)
    _, below_margin = _run(script, train=10, gate_train_check=True, gate_train_margin=0.19)

    check = at_margin.gate_trials[0]["train_check"]
    assert check["baseline_score"] == pytest.approx(passing_before / 10)
    assert check["candidate_score"] == pytest.approx((passing_before - 2) / 10)
    assert check["regressed"] is False and check["blocked"] is False
    assert at_margin.accepted is True
    assert below_margin.gate_trials[0]["train_check"]["blocked"] is True


def test_the_margin_is_absolute_not_relative() -> None:
    # Baseline 0.4, candidate 0.2: an absolute drop of 0.2 and a relative drop of half.
    script = {"v1": (0.0, 1.0), "t0": (1.0, 0.0), "t1": (1.0, 0.0),
              "t2": (0.0, 0.0), "t3": (0.0, 0.0), "t4": (0.0, 1.0)}
    _, tolerated = _run(script, gate_train_check=True, gate_train_margin=0.3)

    check = tolerated.gate_trials[0]["train_check"]
    assert check["baseline_score"] == pytest.approx(0.4)
    assert check["candidate_score"] == pytest.approx(0.2)
    assert check["regressed"] is False
    assert tolerated.accepted is True


def test_margin_zero_rejects_any_broad_drop() -> None:
    # Two tasks lose a little: mean drop 0.04, far below the default margin.
    script = {"v1": (0.0, 1.0), "t0": (1.0, 0.9), "t1": (1.0, 0.9),
              **{f"t{i}": (1.0, 1.0) for i in range(2, 5)}}

    _, default = _run(script, gate_train_check=True)
    _, zero = _run(script, gate_train_check=True, gate_train_margin=0)

    assert default.accepted is True
    assert default.gate_trials[0]["train_check"]["regressed"] is False
    assert zero.accepted is False
    assert zero.gate_trials[0]["train_check"]["blocked"] is True


def test_one_task_flipping_can_never_reject() -> None:
    # One of five training tasks goes from right to wrong. The mean drops by
    # 0.2, more than the margin, but a single task is not enough to reject.
    script = {"v1": (0.0, 1.0), "t0": (1.0, 0.0),
              **{f"t{i}": (1.0, 1.0) for i in range(1, 5)}}

    _, result = _run(script, gate_train_check=True, gate_train_margin=0.0)

    check = result.gate_trials[0]["train_check"]
    assert check["candidate_score"] == pytest.approx(0.8)
    assert (check["n_regressed"], check["n_improved"]) == (1, 0)
    assert check["regressed"] is True and check["blocked"] is False
    assert result.gate_trials[0]["blocked_by_train_check"] is False
    assert result.accepted is True


@pytest.mark.parametrize(("better", "blocked"), [(0, True), (1, True), (2, False)])
def test_net_count_of_regressed_tasks_decides(better, blocked) -> None:
    # Three tasks get much worse; ``better`` tasks get slightly better. The
    # mean always drops well past the margin; only the net count changes.
    script = {"v1": (0.0, 1.0), "t0": (1.0, 0.0), "t1": (1.0, 0.0), "t2": (1.0, 0.0)}
    for i in (3, 4):
        script[f"t{i}"] = (0.0, 0.2) if i - 3 < better else (0.0, 0.0)

    _, result = _run(script, gate_train_check=True)

    check = result.gate_trials[0]["train_check"]
    assert (check["n_regressed"], check["n_improved"]) == (3, better)
    assert check["regressed"] is True and check["blocked"] is blocked
    assert result.accepted is (not blocked)


@pytest.mark.parametrize(("train", "blocked"), [(1, False), (2, True), (3, True)])
def test_how_many_training_tasks_it_takes(train, blocked) -> None:
    script = {"v1": (0.0, 1.0), **{f"t{i}": (1.0, 0.0) for i in range(train)}}

    _, result = _run(script, train=train, gate_train_check=True)

    check = result.gate_trials[0]["train_check"]
    assert check["n_tasks"] == train
    assert check["regressed"] is True and check["blocked"] is blocked
    assert result.accepted is (not blocked)


def test_a_malformed_training_score_fails_closed() -> None:
    script = {"v1": (0.0, 1.0), "t0": (1.0, math.nan),
              **{f"t{i}": (1.0, 1.0) for i in range(1, 5)}}

    _, result = _run(script, gate_train_check=True)

    check = result.gate_trials[0]["train_check"]
    assert check["candidate_score"] is None
    assert check["regressed"] is True and check["blocked"] is True
    assert result.accepted is False
    json.dumps(result.gate_trials, allow_nan=False)


def test_a_training_task_missing_from_the_candidate_replay_counts_as_worse(monkeypatch) -> None:
    module = importlib.import_module("skillopt_sleep.consolidate")
    real = module.replay_batch

    def dropping(backend, tasks, skill, memory):
        pairs = real(backend, tasks, skill, memory)
        if backend.evidence_phase.startswith("gate_train_check"):
            pairs = [pair for pair in pairs if pair[0].id not in {"t0", "t1"}]
        return pairs

    monkeypatch.setattr(module, "replay_batch", dropping)

    _, result = _run(HELPFUL | {"t2": (1.0, 0.0), "t3": (1.0, 0.0), "t4": (1.0, 0.0)},
                     gate_train_check=True)

    check = result.gate_trials[0]["train_check"]
    statuses = {row["task_id"]: row["status"] for row in check["task_deltas"]}
    assert statuses["t0"] == statuses["t1"] == "regressed"
    assert check["n_regressed"] == 5
    assert check["blocked"] is True
    assert result.accepted is False


# ── scope ────────────────────────────────────────────────────────────────────


def test_recalled_tasks_are_checked_and_dream_variants_are_not() -> None:
    backend = ScriptedBackend({
        "v1": (0.0, 1.0),
        **{f"t{i}": (1.0, 1.0) for i in range(2)},
        **{f"old{i}": (1.0, 0.0) for i in range(3)},
    })
    tonight = [_task("t0", "train"), _task("t1", "train"), _task("v1", "val")]
    history = [
        TaskRecord(id=f"old{i}", project="test", intent=f"question t{i % 2} again {i}",
                   split="train")
        for i in range(3)
    ]

    result = dream_consolidate(
        backend, tonight, SKILL, MEMORY, history_tasks=history, recall_k=3,
        dream_factor=2, gate_metric="hard", evolve_memory=False,
        gate_train_check=True,
    )

    checked = sorted(task_id for task_id, phase in backend.calls
                     if phase == "gate_train_check:skill")
    assert checked == ["recall:old0", "recall:old1", "recall:old2", "t0", "t1"]
    check = result.gate_trials[0]["train_check"]
    assert check["n_tasks"] == 5
    assert check["baseline_score"] == pytest.approx(1.0)
    assert check["candidate_score"] == pytest.approx(0.4)
    assert result.accepted is False


def test_dream_variants_are_left_out_of_both_arms() -> None:
    # The real training tasks improve; only their dream variants would look
    # worse (a variant's score follows its source task here, so give the
    # variants' sources different scripts through distinct real tasks).
    backend = ScriptedBackend({
        "v1": (0.0, 1.0), "t0": (0.0, 1.0), "t1": (0.0, 1.0),
    })
    tonight = [_task("t0", "train"), _task("t1", "train"), _task("v1", "val")]

    result = dream_consolidate(
        backend, tonight, SKILL, MEMORY, dream_factor=3, gate_metric="hard",
        evolve_memory=False, gate_train_check=True,
    )

    check = result.gate_trials[0]["train_check"]
    assert check["n_tasks"] == 2                       # not 2 real + 6 variants
    assert [row["task_id"] for row in check["task_deltas"]] == ["t0", "t1"]
    assert (check["baseline_score"], check["candidate_score"]) == (0.0, 1.0)
    assert result.accepted is True


def test_gate_off_and_non_disjoint_validation_skip_the_check() -> None:
    off_backend, off = _run(HARMFUL, gate_train_check=True, gate_mode="off")
    leaked_backend, leaked = _run(
        {f"t{i}": (0.0, 1.0) for i in range(5)}, val=(), gate_train_check=True,
    )

    assert not _check_ran(off_backend)
    assert off.gate_action == "greedy_applied"
    assert leaked.holdout_leaked is True
    assert not _check_ran(leaked_backend)


def test_the_check_runs_with_dream_rollouts_and_parallel_workers(monkeypatch) -> None:
    monkeypatch.setenv("SKILLOPT_SLEEP_WORKERS", "4")

    class Contrastive(ScriptedBackend):
        def _call(self, prompt, *, max_tokens=1024):
            return json.dumps([{"op": "add", "content": SKILL_RULE, "rationale": "r"}])

    backend = Contrastive(HARMFUL)
    tasks = [_task(f"t{i}", "train") for i in range(5)] + [_task("v1", "val")]

    result = consolidate(backend, tasks, SKILL, MEMORY, gate_metric="hard",
                         evolve_memory=False, rollouts_k=3, gate_train_check=True)

    assert result.gate_trials[0]["blocked_by_train_check"] is True
    assert result.accepted is False
    checked = sorted(t for t, phase in backend.calls if phase == "gate_train_check:skill")
    assert checked == [f"t{i}" for i in range(5)]


# ── skill then memory ────────────────────────────────────────────────────────


def test_with_a_dual_backend_the_check_replays_on_the_target_only() -> None:
    target, optimizer = ScriptedBackend(HARMFUL), ScriptedBackend(HARMFUL)
    tasks = [_task(f"t{i}", "train") for i in range(5)] + [_task("v1", "val")]

    result = consolidate(DualBackend(target, optimizer), tasks, SKILL, MEMORY,
                         gate_metric="hard", evolve_memory=False, gate_train_check=True)

    assert result.accepted is False
    assert result.gate_trials[0]["blocked_by_train_check"] is True
    assert sorted(task_id for task_id, _phase in target.calls).count("t0") == 2   # baseline, candidate
    assert optimizer.calls == []


def _skill_then_memory(script):
    backend = ScriptedBackend(script, propose_memory=True)
    tasks = [_task(f"t{i}", "train") for i in range(5)] + [_task("v1", "val")]
    result = consolidate(backend, tasks, SKILL, MEMORY, gate_metric="hard",
                         evolve_memory=True, gate_train_check=True)
    return backend, result


def test_memory_candidate_is_compared_with_the_accepted_skill() -> None:
    # The skill rule helps everywhere and is accepted. The memory rule then
    # lifts the validation task further but breaks the training tasks that the
    # accepted skill had just fixed.
    script = {"v1": (0.0, 0.5, 1.0), **{f"t{i}": (0.0, 1.0, 0.0) for i in range(5)}}

    backend, result = _skill_then_memory(script)

    skill_trial, memory_trial, _final = result.gate_trials
    assert skill_trial["accepted"] is True
    assert skill_trial["train_check"]["candidate_score"] == 1.0
    assert memory_trial["target"] == "memory"
    assert memory_trial["blocked_by_train_check"] is True
    assert memory_trial["train_check"]["baseline_score"] == 1.0   # post-skill scores
    assert memory_trial["train_check"]["candidate_score"] == 0.0
    assert SKILL_RULE in result.new_skill
    assert MEMORY_RULE not in result.new_memory
    assert result.accepted is True
    # the baseline arm is the post-skill training replay the night already runs
    assert ("t0", "train_post_skill") in backend.calls
    assert ("t0", "gate_train_check:memory") in backend.calls


def test_a_blocked_skill_does_not_become_the_memory_baseline() -> None:
    # The skill rule breaks training and is blocked. The memory rule (which the
    # stub can only reach with the skill rule absent) is then judged against
    # the original documents' scores, not the blocked candidate's.
    script = {"v1": (0.0, 1.0), **{f"t{i}": (1.0, 0.0) for i in range(5)}}

    class MemoryOnly(ScriptedBackend):
        def attempt(self, task, skill, memory, sample_id: int = 0) -> str:
            self.calls.append((task.id, self.evidence_phase))
            if MEMORY_RULE in memory:
                return repr(1.0)                      # memory rule fixes everything
            return repr(self.script[task.id][1 if SKILL_RULE in skill else 0])

    backend = MemoryOnly(script, propose_memory=True)
    tasks = [_task(f"t{i}", "train") for i in range(5)] + [_task("v1", "val")]
    result = consolidate(backend, tasks, SKILL, MEMORY, gate_metric="hard",
                         evolve_memory=True, gate_train_check=True)

    skill_trial, memory_trial, _final = result.gate_trials
    assert skill_trial["blocked_by_train_check"] is True
    assert memory_trial["target"] == "memory" and memory_trial["accepted"] is True
    assert memory_trial["train_check"]["baseline_score"] == 1.0
    assert memory_trial["train_check"]["candidate_score"] == 1.0
    assert SKILL_RULE not in result.new_skill and MEMORY_RULE in result.new_memory


# ── budget ───────────────────────────────────────────────────────────────────


def test_the_check_is_part_of_the_edit_step_once_the_step_has_started() -> None:
    backend = ScriptedBackend(HARMFUL)
    tasks = [_task(f"t{i}", "train") for i in range(5)] + [_task("v1", "val")]
    checks = iter([False, False, True, True])       # epoch, skill edits, then spent

    result = consolidate(backend, tasks, SKILL, MEMORY, gate_metric="hard",
                         evolve_memory=True, gate_train_check=True,
                         budget_exhausted=lambda: next(checks, True))

    assert result.budget_skipped == ["memory edits", "final validation replay"]
    assert _check_ran(backend)
    assert result.gate_trials[0]["blocked_by_train_check"] is True
    assert result.accepted is False


# ── validation of the settings ───────────────────────────────────────────────


@pytest.mark.parametrize("bad", [-0.01, 1.01, math.nan, math.inf, "0.1", None, True, [0.1]])
def test_an_invalid_margin_stops_before_any_model_call(bad) -> None:
    backend = ScriptedBackend(HARMFUL)
    tasks = [_task(f"t{i}", "train") for i in range(5)] + [_task("v1", "val")]

    with pytest.raises(ValueError, match="gate_train_margin"):
        consolidate(backend, tasks, SKILL, MEMORY, gate_train_check=True,
                    gate_train_margin=bad)
    assert backend.calls == []


@pytest.mark.parametrize("bad", ["true", "false", 1, 0, 0.5, [True]])
def test_a_non_boolean_flag_stops_before_any_model_call(bad) -> None:
    backend = ScriptedBackend(HARMFUL)
    tasks = [_task(f"t{i}", "train") for i in range(5)] + [_task("v1", "val")]

    with pytest.raises(ValueError, match="gate_train_check must be true or false"):
        consolidate(backend, tasks, SKILL, MEMORY, gate_train_check=bad)
    assert backend.calls == []


def test_setting_helpers() -> None:
    assert train_check_enabled(None) is False
    assert train_check_enabled(False) is False
    assert train_check_enabled(True) is True
    assert train_check_margin(0) == 0.0 and train_check_margin(1) == 1.0
    _, result = _run(HELPFUL, gate_train_margin="not a number")   # ignored while off
    assert result.accepted is True


# ── records ──────────────────────────────────────────────────────────────────


def test_evidence_log_keeps_both_arms_of_the_check(tmp_path) -> None:
    backend = ScriptedBackend(HARMFUL)
    path = os.path.join(tmp_path, "evidence.jsonl")
    backend.evidence = EvidenceLog(path)
    tasks = [_task(f"t{i}", "train") for i in range(5)] + [_task("v1", "val")]

    consolidate(backend, tasks, SKILL, MEMORY, gate_metric="hard",
                evolve_memory=False, gate_train_check=True)
    events = read_events(path)

    (check,) = [e for e in events
                if e.get("stage") == "gate" and e.get("event") == "train_check"]
    assert (check["target"], check["n_tasks"], check["blocked"]) == ("skill", 5, True)
    assert (check["baseline_score"], check["candidate_score"]) == (1.0, 0.0)
    assert len(check["task_deltas"]) == 5
    (trial,) = [e for e in events if e.get("stage") == "gate" and e.get("event") == "trial"]
    assert trial["accepted"] is False and trial["blocked_by_train_check"] is True
    replays = [e for e in events if e.get("stage") == "replay"
               and e.get("phase") == "gate_train_check:skill"]
    assert sorted(e["task_id"] for e in replays) == [f"t{i}" for i in range(5)]


def _report_with(check: dict, *, accepted: bool, blocked: bool, cfg) -> str:
    report = SleepReport(night=1, project="p", started_at="t", n_sessions=0, n_tasks=6)
    report.gate_action = "accept_new_best" if accepted else "reject"
    report.accepted = accepted
    report.gate_trials = [{
        "target": "skill", "baseline_score": 0.0, "candidate_score": 1.0,
        "accepted": accepted, "blocked_by_regression": False,
        "blocked_by_train_check": blocked, "train_check": check, "task_deltas": [],
    }]
    return _render_report_md(report, cfg)


def test_report_wording_for_each_outcome(tmp_path) -> None:
    cfg = _cfg(tmp_path, gate_train_check=True)
    base = {"n_tasks": 5, "margin": 0.1, "baseline_score": 1.0}

    blocked = _report_with(
        base | {"candidate_score": 0.0, "n_regressed": 5, "n_improved": 0,
                "regressed": True, "blocked": True},
        accepted=False, blocked=True, cfg=cfg)
    lower = _report_with(
        base | {"candidate_score": 0.8, "n_regressed": 1, "n_improved": 0,
                "regressed": True, "blocked": False},
        accepted=True, blocked=False, cfg=cfg)
    fine = _report_with(
        base | {"candidate_score": 1.0, "n_regressed": 0, "n_improved": 0,
                "regressed": False, "blocked": False},
        accepted=True, blocked=False, cfg=cfg)

    assert "candidate — rejected (training-task regression)" in blocked
    assert ("Training-side check (these tasks are not held out) over 5 training tasks: "
            "1.000 → 0.000 (margin 0.1; 5 worse, 0 better): regressed, candidate rejected."
            in blocked)
    assert "(margin 0.1; 1 worse, 0 better): lower, but not on enough tasks to reject." in lower
    assert "(margin 0.1; 0 worse, 0 better): no regression." in fine
    assert "- training-side check: enabled (margin 0.1)" in fine


def test_report_header_reflects_whether_the_check_can_run(tmp_path) -> None:
    report = SleepReport(night=1, project="p", started_at="t", n_sessions=0, n_tasks=6)

    off = _render_report_md(report, _cfg(tmp_path))
    gate_off = _render_report_md(report, _cfg(tmp_path, gate_train_check=True, gate_mode="off"))
    null_margin = _render_report_md(
        report, _cfg(tmp_path, gate_train_check=True, gate_train_margin=None))

    assert "training-side check" not in off
    assert "training-side check" not in gate_off
    assert "- training-side check: enabled (margin 0.1)" in null_margin


# ── nightly cycle ────────────────────────────────────────────────────────────


def _cfg(tmp_path, **extra):
    project = tmp_path / "project"
    home = tmp_path / "home"
    project.mkdir(exist_ok=True)
    home.mkdir(exist_ok=True)
    return load_config(
        invoked_project=str(project), projects="invoked", backend="mock",
        claude_home=str(home / ".claude"), auto_adopt=False, **extra,
    )


def _spy_on_dream(monkeypatch):
    seen: list[dict] = []
    real = cycle_module.dream_consolidate

    def spy(backend, tasks, skill, memory, **kwargs):
        seen.append(dict(kwargs))
        return real(backend, tasks, skill, memory, **kwargs)

    monkeypatch.setattr(cycle_module, "dream_consolidate", spy)
    return seen


def _night(tmp_path, **cfg_extra):
    tasks = assign_splits(researcher_persona(), holdout_fraction=0.34, seed=42)
    return run_sleep_cycle(_cfg(tmp_path, **cfg_extra), seed_tasks=tasks)


def test_cycle_forwards_the_setting_and_reports_it(tmp_path, monkeypatch) -> None:
    seen = _spy_on_dream(monkeypatch)

    outcome = _night(tmp_path, gate_train_check=True, gate_train_margin=0.2)

    assert [(k["gate_train_check"], k["gate_train_margin"]) for k in seen] == [(True, 0.2)]
    with open(os.path.join(outcome.staging_dir, "report.md"), encoding="utf-8") as fh:
        report = fh.read()
    assert "- training-side check: enabled (margin 0.2)" in report
    events = read_events(os.path.join(outcome.staging_dir, "evidence.jsonl"))
    start = next(e for e in events if e.get("stage") == "cycle" and e.get("event") == "start")
    assert start["config"]["gate_train_check"] is True
    assert start["config"]["gate_train_margin"] == 0.2
    assert outcome.report.accepted is True
    assert ("Training-side check (these tasks are not held out) over 6 training tasks: "
            "0.333 → 1.000 (margin 0.2; 0 worse, 6 better): no regression." in report)
    (check,) = [e for e in events
                if e.get("stage") == "gate" and e.get("event") == "train_check"]
    assert (check["n_tasks"], check["blocked"]) == (6, False)
    assert outcome.report.gate_trials[0]["train_check"]["n_tasks"] == 6


def test_cycle_default_is_off_and_leaves_reports_and_records_alone(tmp_path, monkeypatch) -> None:
    seen = _spy_on_dream(monkeypatch)

    outcome = _night(tmp_path)

    assert [k["gate_train_check"] for k in seen] == [False]
    with open(os.path.join(outcome.staging_dir, "report.md"), encoding="utf-8") as fh:
        report = fh.read()
    assert "training-side check" not in report.lower()
    events = read_events(os.path.join(outcome.staging_dir, "evidence.jsonl"))
    assert not [e for e in events if e.get("event") == "train_check"]
    start = next(e for e in events if e.get("stage") == "cycle" and e.get("event") == "start")
    assert "gate_train_check" not in start["config"]
    assert "gate_train_margin" not in start["config"]
    for trial in outcome.report.gate_trials:
        assert "train_check" not in trial and "blocked_by_train_check" not in trial


def test_cycle_null_margin_means_the_default(tmp_path, monkeypatch) -> None:
    seen = _spy_on_dream(monkeypatch)

    _night(tmp_path, gate_train_check=True, gate_train_margin=None)

    assert [(k["gate_train_check"], k["gate_train_margin"]) for k in seen] == [(True, 0.1)]


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"gate_train_check": True, "gate_train_margin": 7}, "gate_train_margin"),
        ({"gate_train_check": "true"}, "gate_train_check must be true or false"),
        ({"gate_train_check": 1}, "gate_train_check must be true or false"),
    ],
)
def test_cycle_rejects_invalid_settings_before_consolidation(tmp_path, monkeypatch, extra, message) -> None:
    seen = _spy_on_dream(monkeypatch)

    with pytest.raises(ValueError, match=message):
        _night(tmp_path, **extra)
    assert seen == []


def test_cycle_reads_the_settings_from_a_config_file(tmp_path, monkeypatch) -> None:
    seen = _spy_on_dream(monkeypatch)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"gate_train_check": True, "gate_train_margin": 0.25}), encoding="utf-8")
    monkeypatch.setattr("skillopt_sleep.config._user_config_path", lambda: str(config_path))

    _night(tmp_path)

    assert [(k["gate_train_check"], k["gate_train_margin"]) for k in seen] == [(True, 0.25)]
