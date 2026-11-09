"""Leave-one-out edit ablation for accepted gate trials (follow-up to #174)."""

from __future__ import annotations

import importlib
import json
import os

import pytest

from skillopt_sleep.__main__ import _report_payload
from skillopt_sleep.backend import Backend, MockBackend
from skillopt_sleep.config import DEFAULTS, load_config
from skillopt_sleep.cycle import _render_report_md, run_sleep_cycle
from skillopt_sleep.experiments.personas import programmer_persona, researcher_persona
from skillopt_sleep.memory import set_learned
from skillopt_sleep.mine import assign_splits
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
    assert [row["effect"] for row in rows] == ["positive", "none"]
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

    # One extra validation replay per edit, each without exactly that edit.
    val_replays = _val_replays(calls)
    assert len(val_replays) == 5
    ablation_skills = [call["skill"] for call in val_replays[2:4]]
    assert RULE_A not in ablation_skills[0] and RULE_B in ablation_skills[0]
    assert RULE_A in ablation_skills[1] and RULE_B not in ablation_skills[1]


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
    assert rule_a["effect"] == "positive"
    assert rule_b["effect"] == "negative"
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
    assert row["effect"] == "positive"
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
    assert first["effect"] == "positive"
    assert first["score_without"] == pytest.approx(0.4)
    # Without the replace, the unrefined rule remains and is replayed.
    assert second["co_dropped"] == []
    assert second["replayed"] is True
    assert second["effect"] == "positive"
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
    assert replace_row["effect"] == "none"
    assert replace_row["replayed"] is False
    assert replace_row["note"] == "document is the same without this edit"
    assert replace_row["task_changes"] == []
    assert delete_row["effect"] == "positive"
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
    assert [row["effect"] for row in trial["edit_ablation"]] == ["positive", "none"]
    assert RULE_A in result.new_memory and RULE_B in result.new_memory


def test_spent_budget_leaves_rows_unmeasured(monkeypatch) -> None:
    calls = _patch_replay(monkeypatch, _free_rider_scores)
    checks: list[int] = []

    def budget_exhausted() -> bool:
        # Checked before consolidation and before the skill edits (both still
        # within budget), then before ablation and the final replay.
        checks.append(1)
        return len(checks) > 2

    result = _consolidate(
        [_add(RULE_A), _add(RULE_B)],
        gate_edit_ablation=True,
        budget_exhausted=budget_exhausted,
    )

    assert result.accepted is True
    rows = _skill_trial(result)["edit_ablation"]
    assert [row["effect"] for row in rows] == ["unmeasured", "unmeasured"]
    assert all(row["replayed"] is False for row in rows)
    assert all(row["score_without"] is None for row in rows)
    assert all(row["task_changes"] == [] for row in rows)
    assert all(row["note"] == "not replayed: token budget spent" for row in rows)
    # The skipped step is listed once, not once per edit.
    assert result.budget_skipped == ["edit ablation", "final validation replay"]
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
    assert rule_a["effect"] == "positive"
    assert rule_b["effect"] == "unmeasured"
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
    assert [row["effect"] for row in events[0]["rows"]] == ["positive", "none"]


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
                        "score_with": 0.85,
                        "score_without": 0.35,
                        "effect": "positive",
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
                        "score_with": 0.85,
                        "score_without": None,
                        "effect": "unmeasured",
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
    assert "| Edit | With | Without | Effect | Task changes |" in markdown
    assert (
        f"| 1. \\[add\\] {RULE_A} | 0.850 | 0.350 | positive "
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
    assert [row["effect"] for row in rows] == ["positive", "unmeasured"]
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
                assert row["effect"] in {
                    "positive",
                    "none",
                    "negative",
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

