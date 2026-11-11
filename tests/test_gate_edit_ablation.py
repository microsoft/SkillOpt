"""Leave-one-out edit ablation for accepted gate trials (follow-up to #174)."""

from __future__ import annotations

import importlib
import json
import os
from dataclasses import replace

import pytest

from skillopt_sleep.__main__ import _report_payload
from skillopt_sleep.backend import Backend, MockBackend
from skillopt_sleep.config import DEFAULTS, load_config
from skillopt_sleep.consolidate import run_edit_ablations
from skillopt_sleep.cycle import _render_report_md, run_sleep_cycle
from skillopt_sleep.experiments.personas import programmer_persona, researcher_persona
from skillopt_sleep.memory import set_learned
from skillopt_sleep.mine import assign_splits
from skillopt_sleep.multi_skill import SkillGroup, consolidate_groups
from skillopt_sleep.types import EditRecord, ReplayResult, SleepReport, TaskRecord

RULE_A = "Rule A: answer the question that was asked."
RULE_B = "Rule B: always open with a fixed heading."


def _add(content: str, target: str = "skill") -> EditRecord:
    return EditRecord(target=target, op="add", content=content, rationale="test")


class ScriptedBackend(Backend):
    """Returns fresh copies of a fixed edit list for the evolved target."""

    name = "scripted-stub"

    def __init__(self, edits: list[EditRecord]) -> None:
        self._edits = edits

    def reflect(self, *args, **kwargs):
        target = "skill" if kwargs.get("evolve_skill") else "memory"
        return [
            EditRecord(
                target=edit.target,
                op=edit.op,
                content=edit.content,
                anchor=edit.anchor,
                rationale=edit.rationale,
            )
            for edit in self._edits
            if edit.target == target
        ]


def _tasks() -> list[TaskRecord]:
    return [
        TaskRecord(id="train", project="test", intent="train", split="train"),
        TaskRecord(
            id="essay",
            project="test",
            intent="improve",
            split="val",
            tags=["essay"],
        ),
        TaskRecord(
            id="router",
            project="test",
            intent="hold steady",
            split="val",
            tags=["router"],
        ),
    ]


def _patch_replay(monkeypatch, scorer) -> list[dict]:
    """Score each task from the documents it is replayed under.

    ``scorer(task_id, skill, memory)`` returns the task's score, or ``None`` to
    drop that task's result. Every replay is recorded so a test can count them.
    """
    consolidate_module = importlib.import_module("skillopt_sleep.consolidate")
    calls: list[dict] = []

    def fake_replay_batch(backend, tasks, skill, memory):
        calls.append({
            "task_ids": [task.id for task in tasks],
            "skill": skill,
            "memory": memory,
        })
        pairs = []
        for task in tasks:
            score = scorer(task.id, skill, memory)
            if score is None:
                continue
            pairs.append(
                (
                    task,
                    ReplayResult(
                        id=task.id, hard=score, soft=score, response="reply"
                    ),
                )
            )
        return pairs

    monkeypatch.setattr(consolidate_module, "replay_batch", fake_replay_batch)
    return calls


def _consolidate(edits: list[EditRecord], *, skill_lines=(), **kwargs):
    consolidate_module = importlib.import_module("skillopt_sleep.consolidate")
    kwargs.setdefault("evolve_memory", False)
    return consolidate_module.consolidate(
        ScriptedBackend(edits),
        _tasks(),
        set_learned("# Skill\n", list(skill_lines)),
        set_learned("# Memory\n", []),
        gate_metric="hard",
        **kwargs,
    )


def _val_replays(calls: list[dict]) -> list[dict]:
    return [call for call in calls if call["task_ids"] == ["essay", "router"]]


def _free_rider_scores(task_id: str, skill: str, memory: str) -> float:
    # Rule A fixes the essay task; Rule B changes nothing that is scored.
    if task_id == "essay":
        return 1.0 if RULE_A in skill else 0.0
    if task_id == "router":
        return 0.8
    return 0.0


def _hidden_regression_scores(task_id: str, skill: str, memory: str) -> float:
    # The #174 shape: Rule A lifts one task while Rule B lowers another, and
    # the slice mean still rises (0.40 -> 0.85).
    if task_id == "essay":
        return 1.0 if RULE_A in skill else 0.0
    if task_id == "router":
        return 0.7 if RULE_B in skill else 0.8
    return 0.0


def _skill_trial(result) -> dict:
    return next(trial for trial in result.gate_trials if trial["target"] == "skill")


def test_ablation_setting_defaults_off() -> None:
    assert DEFAULTS["gate_edit_ablation"] is False


def test_default_adds_no_replay_and_no_rows(monkeypatch) -> None:
    default_calls = _patch_replay(monkeypatch, _free_rider_scores)
    default_result = _consolidate([_add(RULE_A), _add(RULE_B)])

    explicit_calls = _patch_replay(monkeypatch, _free_rider_scores)
    explicit_result = _consolidate(
        [_add(RULE_A), _add(RULE_B)], gate_edit_ablation=False
    )

    assert default_result.accepted is True
    # baseline, the skill trial and the final replay: nothing else is scored.
    assert len(_val_replays(default_calls)) == 3
    assert len(default_calls) == len(explicit_calls)
    for trial in default_result.gate_trials + explicit_result.gate_trials:
        assert "edit_ablation" not in trial


def test_ablation_separates_the_edit_that_moved_the_score(monkeypatch) -> None:
    calls = _patch_replay(monkeypatch, _free_rider_scores)

    result = _consolidate([_add(RULE_A), _add(RULE_B)], gate_edit_ablation=True)

    assert result.accepted is True
    rows = _skill_trial(result)["edit_ablation"]
    assert [row["edit_index"] for row in rows] == [0, 1]
    assert [row["content"] for row in rows] == [RULE_A, RULE_B]
    assert [row["observed_change"] for row in rows] == ["higher_with_edit", "no_change"]
    assert all(row["replayed"] is True for row in rows)
    assert all(row["co_dropped"] == [] for row in rows)

    rule_a, rule_b = rows
    assert rule_a["score_with"] == pytest.approx(0.9)
    assert rule_a["score_without"] == pytest.approx(0.4)
    assert rule_a["task_changes"] == [
        {
            "task_id": "essay",
            "tags": ["essay"],
            "score_without": 0.0,
            "score_with": 1.0,
            "status": "improved",
        }
    ]
    assert rule_b["score_with"] == pytest.approx(0.9)
    assert rule_b["score_without"] == pytest.approx(0.9)
    assert rule_b["task_changes"] == []

    # One extra validation replay per edit, each without exactly that edit,
    # and only after the final replay the decision rests on.
    val_replays = _val_replays(calls)
    assert len(val_replays) == 5
    final_replay = val_replays[2]
    assert RULE_A in final_replay["skill"] and RULE_B in final_replay["skill"]
    ablation_skills = [call["skill"] for call in val_replays[3:5]]
    assert RULE_A not in ablation_skills[0] and RULE_B in ablation_skills[0]
    assert RULE_A in ablation_skills[1] and RULE_B not in ablation_skills[1]
    # Each row carries the provenance the edit already had (#365).
    assert all(row["source_task_ids"] == ["train"] for row in rows)
    assert all(row["source_session_ids"] == [] for row in rows)


def test_ablation_names_the_edit_behind_a_hidden_regression(monkeypatch) -> None:
    _patch_replay(monkeypatch, _hidden_regression_scores)

    result = _consolidate([_add(RULE_A), _add(RULE_B)], gate_edit_ablation=True)

    # The aggregate gate still accepts: ablation reports, it does not decide.
    assert result.accepted is True
    trial = _skill_trial(result)
    assert [row["status"] for row in trial["task_deltas"]] == [
        "improved",
        "regressed",
    ]
    rule_a, rule_b = trial["edit_ablation"]
    assert rule_a["observed_change"] == "higher_with_edit"
    assert rule_b["observed_change"] == "lower_with_edit"
    assert rule_b["score_with"] == pytest.approx(0.85)
    assert rule_b["score_without"] == pytest.approx(0.9)
    assert rule_b["task_changes"] == [
        {
            "task_id": "router",
            "tags": ["router"],
            "score_without": 0.8,
            "score_with": 0.7,
            "status": "regressed",
        }
    ]


@pytest.mark.parametrize(
    "scorer", [_free_rider_scores, _hidden_regression_scores]
)
def test_ablation_never_changes_the_decision(monkeypatch, scorer) -> None:
    _patch_replay(monkeypatch, scorer)
    off = _consolidate([_add(RULE_A), _add(RULE_B)])

    _patch_replay(monkeypatch, scorer)
    on = _consolidate([_add(RULE_A), _add(RULE_B)], gate_edit_ablation=True)

    for field in (
        "accepted",
        "gate_action",
        "baseline_score",
        "candidate_score",
        "new_skill",
        "new_memory",
        "applied_edits",
        "rejected_edits",
        "unmatched_edits",
        "holdout_baseline",
        "holdout_candidate",
        "budget_skipped",
    ):
        assert getattr(on, field) == getattr(off, field), field
    assert len(on.gate_trials) == len(off.gate_trials)
    for on_trial, off_trial in zip(on.gate_trials, off.gate_trials):
        without_rows = {
            key: value for key, value in on_trial.items() if key != "edit_ablation"
        }
        assert without_rows == off_trial


def test_ablation_combines_with_the_no_regression_gate(monkeypatch) -> None:
    calls = _patch_replay(monkeypatch, _hidden_regression_scores)

    result = _consolidate(
        [_add(RULE_A), _add(RULE_B)],
        gate_no_regression=True,
        gate_edit_ablation=True,
    )

    # The trial is rejected for the router regression, so there is no accepted
    # candidate to ablate and no extra replay is spent on it.
    assert result.accepted is False
    trial = _skill_trial(result)
    assert trial["blocked_by_regression"] is True
    assert "edit_ablation" not in trial
    assert len(_val_replays(calls)) == 3


def test_single_edit_trial_reuses_the_trial_comparison(monkeypatch) -> None:
    calls = _patch_replay(monkeypatch, _free_rider_scores)

    result = _consolidate([_add(RULE_A)], gate_edit_ablation=True)

    assert result.accepted is True
    (row,) = _skill_trial(result)["edit_ablation"]
    assert row["observed_change"] == "higher_with_edit"
    assert row["replayed"] is False
    assert row["note"] == "scored by the replay that preceded this trial"
    assert row["score_with"] == pytest.approx(0.9)
    assert row["score_without"] == pytest.approx(0.4)
    assert [change["task_id"] for change in row["task_changes"]] == ["essay"]
    assert len(_val_replays(calls)) == 3


def test_edit_that_only_matched_through_another_is_listed(monkeypatch) -> None:
    refined = "Rule A: answer the question that was asked, then stop."

    def scores(task_id: str, skill: str, memory: str) -> float:
        if task_id == "essay":
            return 1.0 if refined in skill else 0.0
        return 0.8 if task_id == "router" else 0.0

    calls = _patch_replay(monkeypatch, scores)
    edits = [
        _add(RULE_A),
        EditRecord(
            target="skill",
            op="replace",
            anchor=RULE_A,
            content=refined,
            rationale="test",
        ),
    ]

    result = _consolidate(edits, gate_edit_ablation=True)

    assert result.accepted is True
    first, second = _skill_trial(result)["edit_ablation"]
    # Without the add, the replace has no anchor: both edits drop together and
    # the pre-trial replay already scored that document.
    assert first["co_dropped"] == [1]
    assert first["replayed"] is False
    assert first["observed_change"] == "higher_with_edit"
    assert first["score_without"] == pytest.approx(0.4)
    # Without the replace, the unrefined rule remains and is replayed.
    assert second["co_dropped"] == []
    assert second["replayed"] is True
    assert second["observed_change"] == "higher_with_edit"
    assert len(_val_replays(calls)) == 4


def test_edit_overridden_inside_the_trial_needs_no_replay(monkeypatch) -> None:
    old_rule = "Old rule: pad every answer."

    def scores(task_id: str, skill: str, memory: str) -> float:
        if task_id == "essay":
            return 0.0 if "Old rule" in skill else 1.0
        return 0.8 if task_id == "router" else 0.0

    calls = _patch_replay(monkeypatch, scores)
    edits = [
        EditRecord(
            target="skill",
            op="replace",
            anchor=old_rule,
            content="Old rule: pad every answer twice.",
            rationale="test",
        ),
        EditRecord(target="skill", op="delete", anchor="Old rule", rationale="test"),
    ]

    result = _consolidate(edits, skill_lines=[old_rule], gate_edit_ablation=True)

    assert result.accepted is True
    replace_row, delete_row = _skill_trial(result)["edit_ablation"]
    # The delete removes the line whether or not the replace ran first.
    assert replace_row["observed_change"] == "no_change"
    assert replace_row["replayed"] is False
    assert replace_row["note"] == "document is the same without this edit"
    assert replace_row["task_changes"] == []
    assert delete_row["observed_change"] == "higher_with_edit"
    assert delete_row["replayed"] is True
    assert len(_val_replays(calls)) == 4


def test_ablation_covers_memory_trials(monkeypatch) -> None:
    def scores(task_id: str, skill: str, memory: str) -> float:
        if task_id == "essay":
            return 1.0 if RULE_A in memory else 0.0
        return 0.8 if task_id == "router" else 0.0

    _patch_replay(monkeypatch, scores)

    result = _consolidate(
        [_add(RULE_A, target="memory"), _add(RULE_B, target="memory")],
        evolve_skill=False,
        evolve_memory=True,
        gate_edit_ablation=True,
    )

    assert result.accepted is True
    trial = next(t for t in result.gate_trials if t["target"] == "memory")
    assert [row["target"] for row in trial["edit_ablation"]] == ["memory", "memory"]
    assert [row["observed_change"] for row in trial["edit_ablation"]] == ["higher_with_edit", "no_change"]
    assert RULE_A in result.new_memory and RULE_B in result.new_memory


def test_spent_budget_leaves_rows_unmeasured(monkeypatch) -> None:
    calls = _patch_replay(monkeypatch, _free_rider_scores)
    checks: list[int] = []

    def budget_exhausted() -> bool:
        # Checked before consolidation and before the skill edits (both still
        # within budget), then before the final replay and before ablation.
        checks.append(1)
        return len(checks) > 2

    result = _consolidate(
        [_add(RULE_A), _add(RULE_B)],
        gate_edit_ablation=True,
        budget_exhausted=budget_exhausted,
    )

    assert result.accepted is True
    rows = _skill_trial(result)["edit_ablation"]
    assert [row["observed_change"] for row in rows] == ["unmeasured", "unmeasured"]
    assert all(row["replayed"] is False for row in rows)
    assert all(row["score_without"] is None for row in rows)
    assert all(row["task_changes"] == [] for row in rows)
    assert all(row["note"] == "not replayed: token budget spent" for row in rows)
    # The skipped step is listed once, not once per edit, and after the
    # steps the decision needed.
    assert result.budget_skipped == ["final validation replay", "edit ablation"]
    assert len(_val_replays(calls)) == 2


@pytest.mark.parametrize("broken", [float("nan"), float("inf"), None])
def test_malformed_ablation_score_is_unmeasured(monkeypatch, broken) -> None:
    def scores(task_id: str, skill: str, memory: str):
        if task_id == "essay":
            return 1.0 if RULE_A in skill else 0.0
        if task_id == "router":
            # Only the replay without Rule B (Rule A alone) is malformed.
            if RULE_A in skill and RULE_B not in skill:
                return broken
            return 0.8
        return 0.0

    _patch_replay(monkeypatch, scores)

    result = _consolidate([_add(RULE_A), _add(RULE_B)], gate_edit_ablation=True)

    assert result.accepted is True
    rule_a, rule_b = _skill_trial(result)["edit_ablation"]
    assert rule_a["observed_change"] == "higher_with_edit"
    assert rule_b["observed_change"] == "unmeasured"
    assert rule_b["replayed"] is True
    assert rule_b["score_without"] is None
    assert rule_b["task_changes"] == []
    assert rule_b["note"] == "a validation score was missing or non-finite"
    json.dumps(result.gate_trials, allow_nan=False)


def test_ablation_is_written_to_the_evidence_log(monkeypatch, tmp_path) -> None:
    from skillopt_sleep import evidence
    from skillopt_sleep.evidence import EvidenceLog

    consolidate_module = importlib.import_module("skillopt_sleep.consolidate")
    _patch_replay(monkeypatch, _free_rider_scores)
    backend = ScriptedBackend([_add(RULE_A), _add(RULE_B)])
    log_path = tmp_path / "evidence.jsonl"
    evidence.attach(backend, EvidenceLog(str(log_path)))

    consolidate_module.consolidate(
        backend,
        _tasks(),
        set_learned("# Skill\n", []),
        "",
        gate_metric="hard",
        evolve_memory=False,
        gate_edit_ablation=True,
    )

    events = [
        event
        for event in evidence.read_events(str(log_path))
        if event["event"] == "edit_ablation"
    ]
    assert len(events) == 1
    assert events[0]["stage"] == "gate"
    assert events[0]["target"] == "skill"
    assert events[0]["n_edits"] == 2
    assert events[0]["n_replays"] == 2
    assert [row["observed_change"] for row in events[0]["rows"]] == ["higher_with_edit", "no_change"]


def _report_with_ablation(content: str = RULE_B) -> SleepReport:
    return SleepReport(
        night=1,
        project="/tmp/project",
        gate_action="accept_new_best",
        accepted=True,
        gate_trials=[
            {
                "target": "skill",
                "baseline_score": 0.4,
                "candidate_score": 0.85,
                "accepted": True,
                "blocked_by_regression": False,
                "task_deltas": [],
                "edit_ablation": [
                    {
                        "edit_index": 0,
                        "target": "skill",
                        "op": "add",
                        "content": RULE_A,
                        "anchor": "",
                        "source_task_ids": ["train-task"],
                        "source_session_ids": ["session-1"],
                        "score_with": 0.85,
                        "score_without": 0.35,
                        "observed_change": "higher_with_edit",
                        "replayed": True,
                        "note": "",
                        "co_dropped": [],
                        "task_changes": [
                            {
                                "task_id": "essay|one",
                                "tags": ["essay"],
                                "score_without": 0.0,
                                "score_with": 1.0,
                                "status": "improved",
                            }
                        ],
                    },
                    {
                        "edit_index": 1,
                        "target": "skill",
                        "op": "add",
                        "content": content,
                        "anchor": "",
                        "source_task_ids": [],
                        "source_session_ids": [],
                        "score_with": 0.85,
                        "score_without": None,
                        "observed_change": "unmeasured",
                        "replayed": False,
                        "note": "not replayed: token budget spent",
                        "co_dropped": [0],
                        "task_changes": [],
                    },
                ],
            }
        ],
    )


def test_report_renders_ablation_rows() -> None:
    markdown = _render_report_md(
        _report_with_ablation(content="x" * 200 + " | tail"),
        load_config(gate_edit_ablation=True),
    )

    assert "- edit ablation: enabled" in markdown
    assert "#### Leave-one-out edit ablation" in markdown
    assert (
        "| Edit | With | Without | Observed (one replay) | Task changes |"
    ) in markdown
    # The caption says what a row is and is not before any row is read.
    assert "_Observed change on one replay, not a causal attribution." in markdown
    assert "selected for scoring high" in markdown
    assert (
        f"| 1. \\[add\\] {RULE_A} | 0.850 | 0.350 | higher with edit "
        "| `essay&#124;one` 0.000 → 1.000 |"
    ) in markdown
    # Long edit text is cut, and the cut text cannot break out of its cell.
    assert f"| 2. \\[add\\] {'x' * 80}… | 0.850 | — " in markdown
    assert "x" * 81 not in markdown
    assert (
        "| unmeasured (not replayed: token budget spent; also removes edit 1) "
        "| — |"
    ) in markdown


def test_default_report_does_not_mention_ablation() -> None:
    report = _report_with_ablation()
    report.gate_trials[0].pop("edit_ablation")

    markdown = _render_report_md(report, load_config())

    assert "ablation" not in markdown.lower()


def test_cli_json_payload_redacts_ablation_rows() -> None:
    synthetic_secret = "api_key=synthetic-example-123456"
    report = _report_with_ablation(content=f"Use {synthetic_secret} for requests.")
    outcome = type("Outcome", (), {"staging_dir": "", "adopted": False})()

    payload = _report_payload(report, outcome)

    serialized = json.dumps(payload)
    rows = payload["gate_trials"][0]["edit_ablation"]
    assert [row["observed_change"] for row in rows] == ["higher_with_edit", "unmeasured"]
    assert synthetic_secret not in serialized
    assert "REDACTED" in serialized


def test_cycle_persists_ablation_setting_and_rows(tmp_path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    config = load_config(
        invoked_project=str(project),
        projects="invoked",
        backend="mock",
        state_dir=str(tmp_path / "state"),
        claude_home=str(tmp_path / ".claude"),
        gate_edit_ablation=True,
        auto_adopt=False,
    )
    tasks = assign_splits(researcher_persona(), holdout_fraction=0.34, seed=42)

    outcome = run_sleep_cycle(
        config,
        seed_tasks=tasks,
        backend=MockBackend(),
    )

    with open(
        os.path.join(outcome.staging_dir, "diagnostics.json"),
        encoding="utf-8",
    ) as handle:
        diagnostics = json.load(handle)
    assert diagnostics["gate_edit_ablation"] is True
    assert diagnostics["gate_trials"] == outcome.report.gate_trials
    for trial in diagnostics["gate_trials"]:
        if trial["target"] != "final" and trial["accepted"]:
            assert trial["edit_ablation"]
            for row in trial["edit_ablation"]:
                assert row["observed_change"] in {
                    "higher_with_edit",
                    "no_change",
                    "lower_with_edit",
                    "unmeasured",
                }
    with open(
        os.path.join(outcome.staging_dir, "report.md"), encoding="utf-8"
    ) as handle:
        assert "- edit ablation: enabled" in handle.read()


def test_fanout_groups_log_their_own_ablation_rows(tmp_path) -> None:
    from dataclasses import replace

    from skillopt_sleep import evidence

    project = tmp_path / "project"
    project.mkdir()
    claude_home = tmp_path / ".claude"
    for name in ("research-skill", "programming-skill"):
        skill_path = claude_home / "skills" / name / "SKILL.md"
        skill_path.parent.mkdir(parents=True)
        skill_path.write_text(f"# {name}\n", encoding="utf-8")
    config = load_config(
        invoked_project=str(project),
        projects="invoked",
        backend="mock",
        state_dir=str(tmp_path / "state"),
        claude_home=str(claude_home),
        managed_skill_name="skillopt-sleep-learned",
        multi_skill_fanout=True,
        gate_edit_ablation=True,
        auto_adopt=False,
    )
    research = assign_splits(researcher_persona(), holdout_fraction=0.34, seed=42)
    programming = assign_splits(programmer_persona(), holdout_fraction=0.34, seed=1)
    tasks = [replace(task, skill_hint="research-skill") for task in research]
    tasks += [
        replace(task, id=f"prog-{task.id}", skill_hint="programming-skill")
        for task in programming
    ]

    outcome = run_sleep_cycle(config, seed_tasks=tasks, backend=MockBackend())

    accepted_groups = [group for group in outcome.report.skill_groups if group.accepted]
    assert len(accepted_groups) == 2
    events = [
        event
        for event in evidence.read_events(
            os.path.join(outcome.staging_dir, "evidence.jsonl")
        )
        if event["event"] == "edit_ablation"
    ]
    # One event for the managed aggregate run, then one per accepted group.
    assert len(events) == 1 + len(accepted_groups)
    assert [event["n_edits"] for event in events[1:]] == [
        group.n_applied_edits for group in accepted_groups
    ]
    for event in events:
        assert len(event["rows"]) == event["n_edits"]


# ── ablation never moves a decision, including when the token budget binds ───

MEMORY_RULE = "Memory rule: keep the router answer stable."
RULE_C = "Rule C: cite the source of every number."
RULE_D = "Rule D: keep answers under ten lines."

_DECISION_FIELDS = (
    "accepted",
    "gate_action",
    "baseline_score",
    "candidate_score",
    "new_skill",
    "new_memory",
    "applied_edits",
    "rejected_edits",
    "unmatched_edits",
    "holdout_baseline",
    "holdout_candidate",
)


def _assert_same_decisions(on, off) -> None:
    for field in _DECISION_FIELDS:
        assert getattr(on, field) == getattr(off, field), field
    assert len(on.gate_trials) == len(off.gate_trials)
    for on_trial, off_trial in zip(on.gate_trials, off.gate_trials):
        without_rows = {
            key: value for key, value in on_trial.items() if key != "edit_ablation"
        }
        assert without_rows == off_trial


def _skill_and_memory_scores(task_id: str, skill: str, memory: str) -> float:
    # Rule A fixes the essay task; the memory rule then lifts the router task.
    if task_id == "essay":
        return 1.0 if RULE_A in skill else 0.0
    if task_id == "router":
        return 0.9 if MEMORY_RULE in memory else 0.8
    return 0.0


def _run_with_token_budget(monkeypatch, *, ablation: bool, budget_replays: int):
    """Two skill edits and one memory edit under a budget of whole replays."""
    tokens_per_replay = 1000
    calls = _patch_replay(monkeypatch, _skill_and_memory_scores)
    result = _consolidate(
        [_add(RULE_A), _add(RULE_B), _add(MEMORY_RULE, target="memory")],
        evolve_memory=True,
        gate_edit_ablation=ablation,
        budget_exhausted=lambda: (
            len(calls) * tokens_per_replay >= budget_replays * tokens_per_replay
        ),
    )
    return result, calls


def test_memory_edit_survives_ablation_under_a_binding_budget(monkeypatch) -> None:
    # A budget of four replays covers the baseline, the train replay, the
    # skill trial and the post-skill train replay. Ablation used to run right
    # after the skill trial and spend what the memory step needed.
    off, off_calls = _run_with_token_budget(
        monkeypatch, ablation=False, budget_replays=4
    )
    on, on_calls = _run_with_token_budget(
        monkeypatch, ablation=True, budget_replays=4
    )

    assert off.accepted is True and MEMORY_RULE in off.new_memory
    assert MEMORY_RULE in on.new_memory
    _assert_same_decisions(on, off)
    assert off.budget_skipped == ["final validation replay"]
    assert on.budget_skipped == ["final validation replay", "edit ablation"]
    # The budget was spent before any ablation replay could start.
    assert on_calls == off_calls
    skill_rows = _skill_trial(on)["edit_ablation"]
    assert [row["observed_change"] for row in skill_rows] == [
        "unmeasured",
        "unmeasured",
    ]
    memory_trial = next(t for t in on.gate_trials if t["target"] == "memory")
    (memory_row,) = memory_trial["edit_ablation"]
    # A one-edit trial needs no replay, so its row survives the spent budget.
    assert memory_row["replayed"] is False
    assert memory_row["observed_change"] == "higher_with_edit"


@pytest.mark.parametrize("budget_replays", range(0, 11))
def test_decisions_match_at_every_token_budget(monkeypatch, budget_replays) -> None:
    off, off_calls = _run_with_token_budget(
        monkeypatch, ablation=False, budget_replays=budget_replays
    )
    on, on_calls = _run_with_token_budget(
        monkeypatch, ablation=True, budget_replays=budget_replays
    )

    _assert_same_decisions(on, off)
    assert [
        step for step in on.budget_skipped if step != "edit ablation"
    ] == off.budget_skipped
    assert on.budget_skipped.count("edit ablation") <= 1
    # Every replay a decision used ran first, in the same order; ablation
    # replays can only follow them.
    assert on_calls[: len(off_calls)] == off_calls


def test_skill_ablation_replays_the_documents_of_its_own_trial(monkeypatch) -> None:
    on, on_calls = _run_with_token_budget(
        monkeypatch, ablation=True, budget_replays=1000
    )

    assert on.accepted is True and MEMORY_RULE in on.new_memory
    ablation_replays = _val_replays(on_calls)[-2:]
    # The skill trial was scored before the memory edit existed, so its
    # ablation is replayed against that same memory, not the final one.
    assert all(MEMORY_RULE not in call["memory"] for call in ablation_replays)
    assert [RULE_A in call["skill"] for call in ablation_replays] == [False, True]
    assert [row["observed_change"] for row in _skill_trial(on)["edit_ablation"]] == [
        "higher_with_edit",
        "no_change",
    ]


def test_budget_is_checked_before_each_ablation_replay(monkeypatch) -> None:
    calls = _patch_replay(monkeypatch, _free_rider_scores)

    # The baseline, train, trial and final replays fit, and so does exactly
    # one ablation replay.
    result = _consolidate(
        [_add(RULE_A), _add(RULE_B), _add(RULE_C), _add(RULE_D)],
        gate_edit_ablation=True,
        budget_exhausted=lambda: len(calls) * 1000 >= 5000,
    )

    assert result.accepted is True
    rows = _skill_trial(result)["edit_ablation"]
    assert [row["replayed"] for row in rows] == [True, False, False, False]
    assert [row["observed_change"] for row in rows] == [
        "higher_with_edit",
        "unmeasured",
        "unmeasured",
        "unmeasured",
    ]
    assert len(calls) == 5
    assert result.budget_skipped == ["edit ablation"]


def test_rejected_epoch_is_not_ablated(monkeypatch) -> None:
    seen = {"candidate": 0}

    def scores(task_id: str, skill: str, memory: str) -> float:
        if task_id == "essay":
            if RULE_A in skill and RULE_B in skill:
                seen["candidate"] += 1
                # The trial replay passes; the fresh final replay does not.
                return 1.0 if seen["candidate"] == 1 else 0.0
            return 0.0
        return 0.8 if task_id == "router" else 0.0

    calls = _patch_replay(monkeypatch, scores)

    result = _consolidate([_add(RULE_A), _add(RULE_B)], gate_edit_ablation=True)

    assert result.accepted is False
    trial = _skill_trial(result)
    assert trial["accepted"] is True
    # No edit was accepted, so there is nothing to describe and nothing is
    # replayed for it.
    assert "edit_ablation" not in trial
    assert len(_val_replays(calls)) == 3
    assert result.budget_skipped == []


def test_queue_hands_the_ablation_to_the_caller(monkeypatch) -> None:
    calls = _patch_replay(monkeypatch, _free_rider_scores)
    queue: list = []

    result = _consolidate(
        [_add(RULE_A), _add(RULE_B)],
        gate_edit_ablation=True,
        edit_ablation_queue=queue,
    )

    # Nothing is replayed or attached until the caller runs the queue.
    assert result.accepted is True
    assert len(queue) == 1
    assert len(_val_replays(calls)) == 3
    assert "edit_ablation" not in _skill_trial(result)

    assert run_edit_ablations(queue) is False

    assert queue == []
    assert len(_val_replays(calls)) == 5
    assert [
        row["observed_change"] for row in _skill_trial(result)["edit_ablation"]
    ] == ["higher_with_edit", "no_change"]
    assert result.budget_skipped == []


def test_queue_reports_a_spent_budget_to_the_caller(monkeypatch) -> None:
    calls = _patch_replay(monkeypatch, _free_rider_scores)
    queue: list = []
    result = _consolidate(
        [_add(RULE_A), _add(RULE_B)],
        gate_edit_ablation=True,
        edit_ablation_queue=queue,
        budget_exhausted=lambda: len(calls) >= 4,
    )

    assert run_edit_ablations(queue) is True

    assert len(calls) == 4
    assert [
        row["observed_change"] for row in _skill_trial(result)["edit_ablation"]
    ] == ["unmeasured", "unmeasured"]
    # The caller owns the report of a deferred skip; the epoch's own list is
    # what it was when its decision was made.
    assert result.budget_skipped == []


def _two_groups() -> list[SkillGroup]:
    return [
        SkillGroup("alpha", set_learned("# Alpha\n", []), _tasks()),
        SkillGroup("beta", set_learned("# Beta\n", []), _tasks()),
    ]


def _is_ablation_replay(call: dict) -> bool:
    if call["task_ids"] != ["essay", "router"]:
        return False
    return (RULE_A in call["skill"]) != (RULE_B in call["skill"])


def _consolidate_two_groups(monkeypatch, *, ablation: bool, budget_replays=None):
    calls = _patch_replay(monkeypatch, _free_rider_scores)
    outcomes = consolidate_groups(
        ScriptedBackend([_add(RULE_A), _add(RULE_B)]),
        _two_groups(),
        "",
        gate_metric="hard",
        gate_edit_ablation=ablation,
        budget_exhausted=(
            None if budget_replays is None
            else lambda: len(calls) >= budget_replays
        ),
    )
    return outcomes, calls


def test_group_ablations_wait_for_every_group(monkeypatch) -> None:
    outcomes, calls = _consolidate_two_groups(monkeypatch, ablation=True)

    assert [outcome.accepted for outcome in outcomes.values()] == [True, True]
    flags = [_is_ablation_replay(call) for call in calls]
    assert sum(flags) == 4
    # Once the first ablation replay starts, no group has anything left to
    # decide: every later replay is an ablation replay too.
    assert all(flags[flags.index(True):])
    for outcome in outcomes.values():
        rows = _skill_trial(outcome.result)["edit_ablation"]
        assert [row["observed_change"] for row in rows] == [
            "higher_with_edit",
            "no_change",
        ]


def test_second_group_keeps_its_budget(monkeypatch) -> None:
    # Seven replays: four for the first group, three for the second, whose
    # final replay is then skipped. The first group's ablation used to take
    # two of them, and the second group's skill edits were skipped.
    off, off_calls = _consolidate_two_groups(
        monkeypatch, ablation=False, budget_replays=7
    )
    on, on_calls = _consolidate_two_groups(
        monkeypatch, ablation=True, budget_replays=7
    )

    assert [outcome.accepted for outcome in off.values()] == [True, True]
    assert [outcome.accepted for outcome in on.values()] == [True, True]
    assert on_calls == off_calls


@pytest.mark.parametrize("budget_replays", range(0, 13))
def test_group_outcomes_match_at_every_token_budget(
    monkeypatch, budget_replays
) -> None:
    off, off_calls = _consolidate_two_groups(
        monkeypatch, ablation=False, budget_replays=budget_replays
    )
    on, on_calls = _consolidate_two_groups(
        monkeypatch, ablation=True, budget_replays=budget_replays
    )

    assert list(on) == list(off)
    for name, expected in off.items():
        outcome = on[name]
        assert outcome.status == expected.status
        # A skipped ablation never shows up as a group cut short on budget.
        assert outcome.reason == expected.reason
        assert outcome.accepted == expected.accepted
        if expected.result is not None:
            _assert_same_decisions(outcome.result, expected.result)
            assert outcome.result.budget_skipped == expected.result.budget_skipped
    assert on_calls[: len(off_calls)] == off_calls


class TokenCountingMock(MockBackend):
    """MockBackend that charges a fixed number of tokens per model call."""

    def __init__(self) -> None:
        super().__init__()
        self.spent = 0

    def attempt(self, *args, **kwargs):
        self.spent += 100
        return super().attempt(*args, **kwargs)

    def judge(self, *args, **kwargs):
        self.spent += 20
        return super().judge(*args, **kwargs)

    def reflect(self, *args, **kwargs):
        self.spent += 300
        return super().reflect(*args, **kwargs)

    def tokens_used(self) -> int:
        return self.spent


def _fanout_night(tmp_path, name: str, *, ablation: bool, max_tokens: int):
    root = tmp_path / name
    project = root / "project"
    project.mkdir(parents=True)
    claude_home = root / ".claude"
    for skill_name in ("research-skill", "programming-skill"):
        skill_path = claude_home / "skills" / skill_name / "SKILL.md"
        skill_path.parent.mkdir(parents=True)
        skill_path.write_text(f"# {skill_name}\n", encoding="utf-8")
    config = load_config(
        invoked_project=str(project),
        projects="invoked",
        backend="mock",
        state_dir=str(root / "state"),
        claude_home=str(claude_home),
        managed_skill_name="skillopt-sleep-learned",
        multi_skill_fanout=True,
        gate_edit_ablation=ablation,
        max_tokens_per_night=max_tokens,
        auto_adopt=False,
    )
    research = assign_splits(researcher_persona(), holdout_fraction=0.34, seed=42)
    programming = assign_splits(programmer_persona(), holdout_fraction=0.34, seed=1)
    tasks = [replace(task, skill_hint="research-skill") for task in research]
    tasks += [
        replace(task, id=f"prog-{task.id}", skill_hint="programming-skill")
        for task in programming
    ]
    return run_sleep_cycle(config, seed_tasks=tasks, backend=TokenCountingMock())


def _night_decisions(outcome) -> dict:
    report = outcome.report
    return {
        "accepted": report.accepted,
        "gate_action": report.gate_action,
        "baseline_score": report.baseline_score,
        "candidate_score": report.candidate_score,
        "edits": [(edit.target, edit.op, edit.content) for edit in report.edits],
        "rejected_edits": [
            (edit.target, edit.op, edit.content) for edit in report.rejected_edits
        ],
        "gate_trials": [
            {key: value for key, value in trial.items() if key != "edit_ablation"}
            for trial in report.gate_trials
        ],
        "skill_groups": [
            (
                group.skill_name,
                group.status,
                group.accepted,
                group.gate_action,
                group.reason,
                group.n_applied_edits,
            )
            for group in report.skill_groups
        ],
    }


def _budget_notes(outcome) -> list[str]:
    return [note for note in outcome.report.notes if "max_tokens_per_night" in note]


def _skipped_steps(outcome) -> list[str]:
    steps: list[str] = []
    for note in _budget_notes(outcome):
        steps += note.split("skipped: ", 1)[1].split(", ")
    return steps


def test_fanout_night_decides_the_same_when_the_budget_runs_out(tmp_path) -> None:
    unlimited = _fanout_night(tmp_path, "unlimited", ablation=False, max_tokens=0)
    spend = unlimited.report.tokens_used
    assert spend > 0
    assert [group.accepted for group in unlimited.report.skill_groups] == [True, True]

    # A cap the night reaches only with its very last call: with ablation off
    # every budget check still passes and both groups are accepted.
    off = _fanout_night(tmp_path, "off", ablation=False, max_tokens=spend)
    on = _fanout_night(tmp_path, "on", ablation=True, max_tokens=spend)

    assert _night_decisions(off) == _night_decisions(unlimited)
    assert _budget_notes(off) == []
    assert _night_decisions(on) == _night_decisions(off)
    # Every decision was final before ablation was considered, and by then
    # the budget was spent, so no ablation replay ran.
    assert on.report.tokens_used == spend
    (note,) = _budget_notes(on)
    assert note.endswith("skipped: edit ablation")
    accepted_trial = next(
        trial for trial in on.report.gate_trials
        if trial["target"] != "final" and trial["accepted"]
    )
    assert accepted_trial["edit_ablation"]


@pytest.mark.parametrize("fraction", [0.2, 0.4, 0.6, 0.8, 0.95])
def test_fanout_night_decisions_match_under_a_tight_cap(tmp_path, fraction) -> None:
    unlimited = _fanout_night(tmp_path, "unlimited", ablation=False, max_tokens=0)
    cap = max(1, int(unlimited.report.tokens_used * fraction))

    off = _fanout_night(tmp_path, "off", ablation=False, max_tokens=cap)
    on = _fanout_night(tmp_path, "on", ablation=True, max_tokens=cap)

    assert _night_decisions(on) == _night_decisions(off)
    # Ablation is never the reason a decision step is skipped: apart from its
    # own entry, both nights skip the same steps.
    assert _skipped_steps(off)
    assert [
        step for step in _skipped_steps(on) if step != "edit ablation"
    ] == _skipped_steps(off)
    assert on.report.tokens_used >= off.report.tokens_used


def test_ablation_spend_is_part_of_tokens_used(tmp_path) -> None:
    off = _fanout_night(tmp_path, "off", ablation=False, max_tokens=0)
    on = _fanout_night(tmp_path, "on", ablation=True, max_tokens=0)

    assert _night_decisions(on) == _night_decisions(off)
    assert _budget_notes(on) == []
    # Ablation replays are ordinary model calls: the report's token count
    # includes them, and it is the count max_tokens_per_night is checked on.
    assert on.report.tokens_used > off.report.tokens_used


def _night_with_test_split(tmp_path, name: str, *, ablation: bool, max_tokens: int):
    root = tmp_path / name
    project = root / "project"
    project.mkdir(parents=True)
    config = load_config(
        invoked_project=str(project),
        projects="invoked",
        backend="mock",
        state_dir=str(root / "state"),
        claude_home=str(root / ".claude"),
        gate_edit_ablation=ablation,
        max_tokens_per_night=max_tokens,
        auto_adopt=False,
    )
    tasks = assign_splits(
        researcher_persona(), val_fraction=0.34, test_fraction=0.2, seed=42
    )
    assert any(task.split == "test" for task in tasks)
    return run_sleep_cycle(config, seed_tasks=tasks, backend=TokenCountingMock())


def _evidence_events(outcome) -> list[tuple[str, str]]:
    from skillopt_sleep import evidence

    return [
        (event["stage"], event["event"])
        for event in evidence.read_events(
            os.path.join(outcome.staging_dir, "evidence.jsonl")
        )
    ]


def test_held_out_test_replay_keeps_its_budget(tmp_path) -> None:
    unlimited = _night_with_test_split(
        tmp_path, "unlimited", ablation=False, max_tokens=0
    )
    spend = unlimited.report.tokens_used
    assert ("test", "held_out_score") in _evidence_events(unlimited)

    # The cap is reached by the held-out test replay itself, the night's last
    # call. Ablation must not get to the budget before it.
    on = _night_with_test_split(tmp_path, "on", ablation=True, max_tokens=spend)

    events = _evidence_events(on)
    assert ("test", "held_out_score") in events
    assert _skipped_steps(on) == ["edit ablation"]
    assert on.report.tokens_used == spend
    # With room to spare, ablation still runs only after the test replay.
    free = _night_with_test_split(tmp_path, "free", ablation=True, max_tokens=0)
    free_events = _evidence_events(free)
    assert free_events.index(("test", "held_out_score")) < free_events.index(
        ("gate", "edit_ablation")
    )
    assert free.report.tokens_used > spend
