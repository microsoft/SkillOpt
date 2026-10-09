"""Adversarial dream probes reject brittle candidate rules before adoption."""
from __future__ import annotations

import json
import os
from dataclasses import asdict

import pytest

from skillopt_sleep.adversarial import (
    MAX_ADVERSARIAL_PROBES,
    evaluate_adversarial_probes,
    generate_adversarial_probes,
)
from skillopt_sleep.backend import DualBackend, MockBackend
from skillopt_sleep.config import DEFAULTS, load_config
from skillopt_sleep.consolidate import consolidate
from skillopt_sleep.cycle import run_sleep_cycle
from skillopt_sleep.types import EditRecord, TaskRecord


def _task(
    task_id: str,
    *,
    intent: str = "Please return ok",
    split: str = "train",
    origin: str = "real",
    derived_from: str = "",
) -> TaskRecord:
    return TaskRecord(
        id=task_id,
        project="/project",
        intent=intent,
        context_excerpt="context",
        system="system",
        attempted_solution="prior",
        outcome="fail",
        reference_kind="exact",
        reference="ok",
        judge={"kind": "exact"},
        tags=["fixture"],
        source_sessions=["session-1"],
        split=split,
        origin=origin,
        derived_from=derived_from,
        skill_hint="demo-skill",
    )


class _CandidateBackend(MockBackend):
    """The planted rule works only on the harvested task's exact surface."""

    RULE = "Use the planted literal-only rule."
    HARVESTED_INTENTS = {"Please return ok", "Say ok"}

    def reflect(
        self,
        failures,
        successes,
        skill,
        memory,
        *,
        edit_budget,
        evolve_skill,
        evolve_memory,
    ):
        if self.RULE in f"{skill}\n{memory}":
            return []
        return [
            EditRecord(
                target="skill" if evolve_skill else "memory",
                op="add",
                content=self.RULE,
                rationale="planted brittle candidate",
            )
        ]

    def attempt(self, task, skill, memory, sample_id=0):
        if self.RULE not in f"{skill}\n{memory}":
            return "wrong"
        return task.reference if task.intent in self.HARVESTED_INTENTS else "wrong"


class _RobustCandidateBackend(_CandidateBackend):
    def attempt(self, task, skill, memory, sample_id=0):
        if self.RULE not in f"{skill}\n{memory}":
            return "wrong"
        return task.reference


class _CountingRobustBackend(_RobustCandidateBackend):
    def __init__(self) -> None:
        self.attempt_calls = 0

    def attempt(self, task, skill, memory, sample_id=0):
        self.attempt_calls += 1
        return super().attempt(task, skill, memory, sample_id)


class _NonFiniteProbeBackend(_RobustCandidateBackend):
    def judge(self, task, response):
        if task.origin == "dream":
            return float("nan"), 0.0, "invalid probe score"
        return super().judge(task, response)


class _CountingRoleBackend(_RobustCandidateBackend):
    def __init__(self, name: str) -> None:
        self.name = name
        self.attempt_calls = 0
        self.judge_calls = 0

    def attempt(self, task, skill, memory, sample_id=0):
        self.attempt_calls += 1
        return super().attempt(task, skill, memory, sample_id)

    def judge(self, task, response):
        self.judge_calls += 1
        return super().judge(task, response)


def _candidate_tasks() -> list[TaskRecord]:
    return [_task("train"), _task("val", intent="Say ok", split="val")]


def test_probe_generation_uses_only_real_underived_train_tasks() -> None:
    source = _task("source")
    tasks = [
        source,
        _task("val", split="val"),
        _task("test", split="test"),
        _task("dream", origin="dream", derived_from="source"),
        _task("recall", derived_from="old"),
    ]

    probes = generate_adversarial_probes(tasks, factor=3)

    assert len(probes) == 3
    assert {probe.derived_from for probe in probes} == {"source"}
    assert len({probe.id for probe in probes}) == 3
    assert all(probe.split == "train" and probe.origin == "dream" for probe in probes)
    assert all(probe.intent != source.intent for probe in probes)
    assert all("adversarial" in probe.tags for probe in probes)
    assert all(probe.reference == source.reference for probe in probes)
    assert all(probe.judge == source.judge for probe in probes)
    assert all(probe.system == source.system for probe in probes)
    assert all(probe.source_sessions == source.source_sessions for probe in probes)
    assert probes[0].intent == "Return ok"


def test_probe_generation_is_bounded_and_rejects_ambiguous_factor_types() -> None:
    tasks = [_task(f"task-{index}") for index in range(100)]

    probes = generate_adversarial_probes(tasks, factor=99)

    assert len(probes) == MAX_ADVERSARIAL_PROBES
    with pytest.raises(ValueError, match="must be an integer"):
        generate_adversarial_probes(tasks, factor=True)
    with pytest.raises(ValueError, match="source ids must be unique"):
        generate_adversarial_probes([_task("duplicate"), _task("duplicate")])


def test_probe_report_flags_a_planted_literal_surface_rule() -> None:
    report = evaluate_adversarial_probes(
        _CandidateBackend(),
        [_task("train")],
        _CandidateBackend.RULE,
        "",
        baseline_skill="",
        baseline_memory="",
        factor=2,
    )

    assert report["n_sources"] == 1
    assert report["n_probes"] == 2
    assert report["n_flagged"] == 2
    assert report["brittleness_rate"] == 1.0
    assert report["worst_gap_change"] == -1.0
    assert {row["status"] for row in report["rows"]} == {"brittle"}
    for row in report["rows"]:
        assert row["baseline_source_score"] == 0.0
        assert row["baseline_probe_score"] == 0.0
        assert row["candidate_source_score"] == 1.0
        assert row["candidate_probe_score"] == 0.0
        assert row["gap_change"] == -1.0


def test_probe_report_keeps_a_surface_robust_rule_stable() -> None:
    report = evaluate_adversarial_probes(
        _RobustCandidateBackend(),
        [_task("train")],
        _CandidateBackend.RULE,
        "",
        baseline_skill="",
        baseline_memory="",
        factor=3,
    )

    assert report["n_flagged"] == 0
    assert report["flagged"] is False
    assert report["worst_gap_change"] == 0.0
    assert {row["status"] for row in report["rows"]} == {"stable"}


def test_dual_backend_probes_route_attempts_and_exact_judging_to_target() -> None:
    target = _CountingRoleBackend("target")
    optimizer = _CountingRoleBackend("optimizer")
    dual = DualBackend(target=target, optimizer=optimizer)

    report = evaluate_adversarial_probes(
        dual,
        [_task("train")],
        _CandidateBackend.RULE,
        "",
        baseline_skill="",
        baseline_memory="",
    )

    assert report["flagged"] is False
    # source + one probe, each scored under the baseline AND candidate arms
    assert target.attempt_calls == 4
    assert target.judge_calls == 4
    assert optimizer.attempt_calls == 0
    assert optimizer.judge_calls == 0


def test_non_finite_probe_score_is_json_safe_and_fails_closed() -> None:
    report = evaluate_adversarial_probes(
        _NonFiniteProbeBackend(),
        [_task("train")],
        _CandidateBackend.RULE,
        "",
        baseline_skill="",
        baseline_memory="",
    )

    assert report["flagged"] is True
    assert report["n_invalid"] == 1
    assert report["rows"][0]["candidate_probe_score"] is None
    assert report["rows"][0]["gap_change"] is None
    assert report["rows"][0]["status"] == "invalid"
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("margin", [True, -0.1, 1.1, float("nan"), "0.1"])
def test_probe_margin_rejects_non_finite_or_out_of_range_values(margin) -> None:
    with pytest.raises(ValueError, match="finite number"):
        evaluate_adversarial_probes(
            _RobustCandidateBackend(),
            [_task("train")],
            _CandidateBackend.RULE,
            "",
        baseline_skill="",
        baseline_memory="",
            margin=margin,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"dream_adversarial": True}, "must be an integer"),
        ({"dream_adversarial": 1.5}, "must be an integer"),
        (
            {
                "dream_adversarial": 1,
                "dream_adversarial_blocking": "false",
            },
            "must be a boolean",
        ),
        (
            {"dream_adversarial": 1, "dream_adversarial_margin": float("inf")},
            "finite number",
        ),
        (
            {"dream_adversarial": 1, "dream_adversarial_rollouts": True},
            "rollouts must be an integer",
        ),
        (
            {"dream_adversarial": 1, "dream_adversarial_rollouts": 0},
            "rollouts must be an integer",
        ),
        (
            {"dream_adversarial": 1, "dream_adversarial_rollouts": 99},
            "rollouts must be an integer",
        ),
        (
            {
                "dream_adversarial": 1,
                "dream_adversarial_blocking": True,
                "dream_adversarial_rollouts": 1,
            },
            "blocking requires",
        ),
    ],
)
def test_consolidate_rejects_ambiguous_adversarial_config(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        consolidate(
            _RobustCandidateBackend(),
            _candidate_tasks(),
            "# skill\n",
            "",
            evolve_memory=False,
            **overrides,
        )


def test_advisory_probe_surfaces_brittleness_without_changing_gate_decision() -> None:
    result = consolidate(
        _CandidateBackend(),
        _candidate_tasks(),
        "# skill\n",
        "",
        evolve_memory=False,
        dream_adversarial=2,
        dream_adversarial_blocking=False,
    )

    assert result.accepted is True
    trial = result.gate_trials[0]
    assert trial["accepted"] is True
    assert trial["blocked_by_adversarial"] is False
    assert trial["adversarial_probe"]["flagged"] is True
    assert trial["adversarial_probe"]["blocking"] is False


def test_advisory_probe_is_held_out_result_equivalent_for_robust_candidate() -> None:
    off = consolidate(
        _RobustCandidateBackend(),
        _candidate_tasks(),
        "# skill\n",
        "",
        evolve_memory=False,
    )
    on = consolidate(
        _RobustCandidateBackend(),
        _candidate_tasks(),
        "# skill\n",
        "",
        evolve_memory=False,
        dream_adversarial=3,
        dream_adversarial_blocking=False,
    )

    assert on.accepted == off.accepted is True
    assert on.gate_action == off.gate_action
    assert on.baseline_score == off.baseline_score
    assert on.candidate_score == off.candidate_score
    assert on.new_skill == off.new_skill
    assert on.new_memory == off.new_memory
    assert on.gate_trials[0]["adversarial_probe"]["n_flagged"] == 0


def test_blocking_probe_rejects_candidate_that_passed_the_held_out_gate() -> None:
    result = consolidate(
        _CandidateBackend(),
        _candidate_tasks(),
        "# skill\n",
        "",
        evolve_memory=False,
        dream_adversarial=2,
        dream_adversarial_blocking=True,
        dream_adversarial_rollouts=2,
    )

    assert result.accepted is False
    assert result.applied_edits == []
    assert [edit.content for edit in result.rejected_edits] == [_CandidateBackend.RULE]
    trial = result.gate_trials[0]
    assert trial["candidate_score"] == 1.0
    assert trial["accepted"] is False
    assert trial["blocked_by_adversarial"] is True
    assert trial["adversarial_probe"]["blocked"] is True


def test_blocking_probe_fails_closed_when_no_eligible_variant_exists() -> None:
    result = consolidate(
        _CandidateBackend(),
        [_task("train", intent=""), _task("val", intent="Say ok", split="val")],
        "# skill\n",
        "",
        evolve_memory=False,
        dream_adversarial=1,
        dream_adversarial_blocking=True,
        dream_adversarial_rollouts=2,
    )

    assert result.accepted is False
    probe = result.gate_trials[0]["adversarial_probe"]
    assert probe["conclusive"] is False
    assert probe["blocked"] is True
    assert probe["block_reason"] == "inconclusive_no_probes"


def test_default_off_is_result_identical_to_explicit_zero() -> None:
    default_backend = _CountingRobustBackend()
    explicit_backend = _CountingRobustBackend()
    default = consolidate(
        default_backend,
        _candidate_tasks(),
        "# skill\n",
        "",
        evolve_memory=False,
    )
    explicit = consolidate(
        explicit_backend,
        _candidate_tasks(),
        "# skill\n",
        "",
        evolve_memory=False,
        dream_adversarial=0,
        dream_adversarial_blocking=False,
        dream_adversarial_margin=0.0,
    )

    assert asdict(default) == asdict(explicit)
    assert default_backend.attempt_calls == explicit_backend.attempt_calls == 4
    assert "adversarial_probe" not in default.gate_trials[0]
    assert "blocked_by_adversarial" not in default.gate_trials[0]
    assert DEFAULTS["dream_adversarial"] == 0
    assert DEFAULTS["dream_adversarial_blocking"] is False
    assert DEFAULTS["dream_adversarial_margin"] == 0.0
    assert load_config().get("dream_adversarial") == 0


def test_cycle_persists_advisory_probe_evidence_in_review_artifacts(tmp_path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    config = load_config(
        invoked_project=str(project),
        projects="invoked",
        backend="mock",
        state_dir=str(tmp_path / "state"),
        claude_home=str(tmp_path / ".claude"),
        evolve_memory=False,
        dream_adversarial=2,
        dream_adversarial_blocking=False,
        dream_adversarial_margin=0.015,
        dream_adversarial_rollouts=2,
        auto_adopt=False,
    )

    tasks = _candidate_tasks()
    tasks[0].id = "train|<script>\n# heading"
    outcome = run_sleep_cycle(
        config,
        seed_tasks=tasks,
        backend=_CandidateBackend(),
    )

    with open(
        os.path.join(outcome.staging_dir, "diagnostics.json"),
        encoding="utf-8",
    ) as handle:
        diagnostics = json.load(handle)
    with open(
        os.path.join(outcome.staging_dir, "report.md"),
        encoding="utf-8",
    ) as handle:
        markdown = handle.read()
    assert diagnostics["dream_adversarial"] == 2
    assert diagnostics["dream_adversarial_blocking"] is False
    assert diagnostics["dream_adversarial_margin"] == 0.015
    assert diagnostics["dream_adversarial_rollouts"] == 2
    assert diagnostics["gate_trials"] == outcome.report.gate_trials
    assert diagnostics["gate_trials"][0]["adversarial_probe"]["n_flagged"] == 2
    assert "adversarial dream probes: advisory" in markdown
    assert (
        "Adversarial probes (advisory, baseline-relative, rollouts=2): "
        "2 flagged / 2 total" in markdown
    )
    assert "<script>" not in markdown
    assert "&lt;script&gt;" in markdown


def test_multi_skill_cycle_keeps_blocking_probe_rollouts(tmp_path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    claude_home = tmp_path / ".claude"
    skill_dir = claude_home / "skills" / "demo-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# demo skill\n", encoding="utf-8")
    config = load_config(
        invoked_project=str(project),
        projects="invoked",
        backend="mock",
        state_dir=str(tmp_path / "state"),
        claude_home=str(claude_home),
        managed_skill_name="skillopt-sleep-learned",
        multi_skill_fanout=True,
        evolve_memory=False,
        dream_adversarial=1,
        dream_adversarial_blocking=True,
        dream_adversarial_rollouts=2,
        auto_adopt=False,
    )

    outcome = run_sleep_cycle(
        config,
        seed_tasks=_candidate_tasks(),
        backend=_RobustCandidateBackend(),
    )

    assert len(outcome.report.skill_groups) == 1
    group = outcome.report.skill_groups[0]
    assert group.skill_name == "demo-skill"
    assert group.status == "consolidated"
    assert group.accepted is True


@pytest.mark.parametrize("blocking", [False, True], ids=["advisory", "blocking"])
@pytest.mark.parametrize(
    ("backend_type", "probe_score", "gap_change", "status"),
    [
        (_CandidateBackend, 0.0, -1.0, "brittle"),
        (_RobustCandidateBackend, 1.0, 0.0, "stable"),
    ],
    ids=["brittle", "robust"],
)
def test_cycle_report_renders_baseline_relative_probe_scores(
    tmp_path, blocking, backend_type, probe_score, gap_change, status
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    config = load_config(
        invoked_project=str(project),
        projects="invoked",
        backend="mock",
        state_dir=str(tmp_path / "state"),
        claude_home=str(tmp_path / ".claude"),
        evolve_memory=False,
        dream_adversarial=1,
        dream_adversarial_blocking=blocking,
        dream_adversarial_rollouts=2,
        auto_adopt=False,
    )

    outcome = run_sleep_cycle(
        config, seed_tasks=_candidate_tasks(), backend=backend_type()
    )

    row = outcome.report.gate_trials[0]["adversarial_probe"]["rows"][0]
    assert row["baseline_source_score"] == 0.0
    assert row["baseline_probe_score"] == 0.0
    assert row["candidate_source_score"] == 1.0
    assert row["candidate_probe_score"] == probe_score
    assert row["gap_change"] == gap_change
    assert row["status"] == status
    with open(
        os.path.join(outcome.staging_dir, "report.md"), encoding="utf-8"
    ) as handle:
        markdown = handle.read()
    assert (
        "| Source | Probe | Variant | Baseline source | Baseline probe | "
        "Candidate source | Candidate probe | Gap change | Status |"
    ) in markdown
    assert (
        "| `train` | `train_adversarial_request-frame` | request-frame | "
        f"0.000 | 0.000 | 1.000 | {probe_score:.3f} | {gap_change:.3f} | {status} |"
    ) in markdown


class _EquallySensitiveBackend(_CandidateBackend):
    """Frame sensitivity exists identically with and without the candidate."""

    def attempt(self, task, skill, memory, sample_id=0):
        return task.reference if task.intent in self.HARVESTED_INTENTS else "wrong"


class _ImprovesBothKeepsGapBackend(MockBackend):
    """The candidate improves source AND probe but keeps the pre-existing gap."""

    SCORES = {
        (False, False): 0.4,   # baseline arm, source task
        (False, True): 0.1,    # baseline arm, probe task
        (True, False): 0.9,    # candidate arm, source task
        (True, True): 0.6,     # candidate arm, probe task
    }

    def attempt(self, task, skill, memory, sample_id=0):
        has_rule = _CandidateBackend.RULE in f"{skill}\n{memory}"
        return f"marker:{int(has_rule)}:{int(task.origin == 'dream')}"

    def judge(self, task, response):
        _, has_rule, is_probe = response.split(":")
        value = self.SCORES[(has_rule == "1", is_probe == "1")]
        return value, value, "graded fixture"


class _SingleFluctuationBackend(MockBackend):
    """One probe rollout dips under the candidate; the repeat does not."""

    def attempt(self, task, skill, memory, sample_id=0):
        has_rule = _CandidateBackend.RULE in f"{skill}\n{memory}"
        dips = has_rule and task.origin == "dream" and sample_id == 0
        return f"marker:{int(dips)}"

    def judge(self, task, response):
        value = 0.8 if response.endswith(":1") else 1.0
        return value, value, "graded fixture"


def test_equal_frame_sensitivity_in_baseline_and_candidate_is_not_flagged() -> None:
    report = evaluate_adversarial_probes(
        _EquallySensitiveBackend(),
        [_task("train")],
        _CandidateBackend.RULE,
        "",
        baseline_skill="",
        baseline_memory="",
        factor=2,
        rollouts=2,
    )

    assert report["n_flagged"] == 0
    assert report["flagged"] is False
    for row in report["rows"]:
        assert row["baseline_gap"] == row["candidate_gap"]
        assert row["gap_change"] == 0.0
        assert row["status"] == "stable"


def test_candidate_that_improves_both_but_keeps_the_gap_is_not_flagged() -> None:
    report = evaluate_adversarial_probes(
        _ImprovesBothKeepsGapBackend(),
        [_task("train")],
        _CandidateBackend.RULE,
        "",
        baseline_skill="",
        baseline_memory="",
        factor=2,
        rollouts=2,
    )

    assert report["flagged"] is False
    for row in report["rows"]:
        assert row["candidate_source_score"] > row["baseline_source_score"]
        assert row["candidate_probe_score"] > row["baseline_probe_score"]
        assert row["gap_change"] == 0.0
        assert row["status"] == "stable"


def test_single_rollout_fluctuation_never_flags_under_majority_rule() -> None:
    report = evaluate_adversarial_probes(
        _SingleFluctuationBackend(),
        [_task("train")],
        _CandidateBackend.RULE,
        "",
        baseline_skill="",
        baseline_memory="",
        factor=1,
        rollouts=2,
    )

    row = report["rows"][0]
    assert row["gap_change"] < 0.0
    assert row["worsening_fraction"] == 0.5
    assert row["status"] == "stable"
    assert report["flagged"] is False


def test_evidence_rows_retain_all_four_scores_and_samples() -> None:
    report = evaluate_adversarial_probes(
        _CandidateBackend(),
        [_task("train")],
        _CandidateBackend.RULE,
        "",
        baseline_skill="",
        baseline_memory="",
        factor=1,
        rollouts=2,
    )

    row = report["rows"][0]
    for key in (
        "baseline_source_score",
        "baseline_probe_score",
        "candidate_source_score",
        "candidate_probe_score",
        "baseline_gap",
        "candidate_gap",
        "gap_change",
        "worsening_fraction",
    ):
        assert key in row
    for arm in (
        "baseline_source",
        "baseline_probe",
        "candidate_source",
        "candidate_probe",
    ):
        assert len(row["samples"][arm]) == 2
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("rollouts", [True, 0, -1, 9, 1.5, "2"])
def test_probe_rollouts_rejects_out_of_range_or_ambiguous_values(rollouts) -> None:
    with pytest.raises(ValueError, match="rollouts must be an integer"):
        evaluate_adversarial_probes(
            _RobustCandidateBackend(),
            [_task("train")],
            _CandidateBackend.RULE,
            "",
        baseline_skill="",
        baseline_memory="",
            rollouts=rollouts,
        )


@pytest.mark.parametrize(
    "intent",
    [
        "Can you swim?",
        "Could you ever forgive me?",
        "Would you like some tea?",
        "Can you believe the score?",
        "I want you to know this matters.",
        "I need you to be honest with me.",
    ],
)
def test_polite_frame_never_reframes_ability_permission_or_desire(intent) -> None:
    from skillopt_sleep.adversarial import _strip_polite_frame, _variant_intents

    assert _strip_polite_frame(intent) == ""
    assert all(kind != "request-frame" for kind, _ in _variant_intents(intent))


@pytest.mark.parametrize(
    ("intent", "expected"),
    [
        ("Please add validation to the signup page", "Add validation to the signup page"),
        ("please return ok", "Return ok"),
        ("Could you please add input validation?", "Add input validation."),
        ("can you please rerun the failing job", "Rerun the failing job"),
        ("Please can you swim?", "Can you swim?"),
    ],
)
def test_polite_frame_reframes_only_politeness_marked_requests(intent, expected) -> None:
    from skillopt_sleep.adversarial import _strip_polite_frame

    assert _strip_polite_frame(intent) == expected


# ── Repeated-sample independence on tool and text routes ─────────────────────

_PI_RULE = "Always call the search tool and finish with VERIFIED."
_PI_INTENTS = (
    "Please look up the current release notes before answering",
    "Please check the changelog for the latest breaking change",
    "Please find the deprecation date of the old endpoint",
    "Please confirm which version introduced the retry flag",
)
_PI_CHECKS = {
    "tool": {"op": "tool_called", "arg": "search"},
    "text": {"op": "contains", "arg": "VERIFIED"},
}


class _FakePiProvider:
    """Stands in for the Pi child process, the only faked boundary.

    The miner, reflection, replay, cache, gate, probes, and report all run the
    shipped code. Each call is recorded so tests can count what actually
    reached the provider. The first candidate-probe call of the probe phase
    fails once; every other candidate call succeeds and baseline calls fail.
    """

    GOOD = "TOOL_CALL: search\nVERIFIED"
    BAD = "I answered from memory."

    def __init__(self, backend, check) -> None:
        self.backend = backend
        self.check = check
        self.probe_calls: list[tuple[str, str]] = []
        self.failed_once = False

    def __call__(self, cmd, *, input, **kwargs):
        import subprocess

        prompt = input
        if prompt.startswith("You are mining"):
            intent = next(text for text in _PI_INTENTS if text in prompt)
            out = json.dumps(
                [{"intent": intent, "checks": [self.check], "satisfied": False}]
            )
        elif prompt.startswith("You are SkillOpt's optimizer"):
            current = prompt.split("# Recurring failures", 1)[0]
            out = "[]" if _PI_RULE in current else json.dumps(
                [{"op": "add", "content": _PI_RULE, "rationale": "fixture"}]
            )
        elif prompt.startswith("Complete the following task"):
            candidate = _PI_RULE in prompt
            out = self.GOOD if candidate else self.BAD
            if self.backend.evidence_phase.startswith("adversarial_probe"):
                arm = "candidate" if candidate else "baseline"
                role = (
                    "source" if any(text in prompt for text in _PI_INTENTS) else "probe"
                )
                self.probe_calls.append((arm, role))
                if candidate and role == "probe" and not self.failed_once:
                    self.failed_once = True
                    out = self.BAD
        else:
            out = "{}"
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")


@pytest.mark.parametrize("route", ["tool", "text"])
def test_miner_derived_pi_cycle_obtains_distinct_probe_rollouts(tmp_path, route) -> None:
    """A miner-produced task replayed through the shipped Pi backend must give
    each probe rollout its own provider sample on both routes.

    Before the fix, the inherited tool fallback dropped ``sample_id`` and served
    every rollout from the sample-zero cache entry: one failed candidate-probe
    response counted as three failures and blocked a candidate that the text
    route (the positive control) accepts.
    """
    from collections import Counter
    from unittest import mock

    from skillopt_sleep.backend import PiCliBackend
    from skillopt_sleep.llm_miner import make_llm_miner
    from skillopt_sleep.mine import mine
    from skillopt_sleep.types import SessionDigest

    rollouts = 3
    backend = PiCliBackend()
    provider = _FakePiProvider(backend, _PI_CHECKS[route])
    digests = [
        SessionDigest(
            session_id=f"session-{index}",
            project="/project",
            user_prompts=[intent],
            assistant_finals=["done"],
        )
        for index, intent in enumerate(_PI_INTENTS)
    ]
    project = tmp_path / "project"
    project.mkdir()
    config = load_config(
        invoked_project=str(project),
        projects="invoked",
        backend="pi",
        state_dir=str(tmp_path / "state"),
        claude_home=str(tmp_path / ".claude"),
        evolve_memory=False,
        dream_adversarial=1,
        dream_adversarial_blocking=True,
        dream_adversarial_rollouts=rollouts,
        auto_adopt=False,
    )
    with mock.patch("skillopt_sleep.backend.subprocess.run", side_effect=provider):
        tasks = mine(digests, llm_miner=make_llm_miner(backend), max_tasks=10, seed=42)
        outcome = run_sleep_cycle(config, seed_tasks=tasks, backend=backend)

    assert {task.judge["checks"][0]["op"] for task in tasks} == {
        _PI_CHECKS[route]["op"]
    }
    trial = outcome.report.gate_trials[0]
    probe = trial["adversarial_probe"]
    assert trial["accepted"] is True
    assert trial["blocked_by_adversarial"] is False
    assert probe["n_flagged"] == 0
    n_pairs = probe["n_probes"]
    assert n_pairs == probe["n_sources"] == 3
    # The single provider failure is visible as exactly one failed rollout.
    candidate_probe_samples = [
        score for row in probe["rows"] for score in row["samples"]["candidate_probe"]
    ]
    assert candidate_probe_samples.count(0.0) == 1
    assert provider.failed_once

    # Provider-boundary call counts in the probe phase. Probe prompts are new,
    # so every rollout of every probe reaches the provider in both arms. The
    # candidate's source prompts are also new. Baseline source rollout zero is
    # byte-identical to the training replay under the same documents, so it is
    # served from that cache entry; rollouts one and two still reach Pi.
    calls = Counter(provider.probe_calls)
    assert calls[("candidate", "probe")] == rollouts * n_pairs
    assert calls[("baseline", "probe")] == rollouts * n_pairs
    assert calls[("candidate", "source")] == rollouts * n_pairs
    assert calls[("baseline", "source")] == (rollouts - 1) * n_pairs


def test_inherited_tool_fallback_forwards_sample_id_to_the_attempt_cache() -> None:
    from unittest import mock

    from skillopt_sleep.backend import PiCliBackend
    from skillopt_sleep.replay import replay_one

    backend = PiCliBackend()
    provider = _FakePiProvider(backend, _PI_CHECKS["tool"])
    task = TaskRecord(
        id="tool-task",
        project="/project",
        intent=_PI_INTENTS[0],
        reference_kind="rule",
        judge={"kind": "rule", "checks": [_PI_CHECKS["tool"]]},
    )
    with mock.patch(
        "skillopt_sleep.backend.subprocess.run", side_effect=provider
    ) as run:
        for sample_id in (0, 1, 2, 0, 1):
            result = replay_one(backend, task, _PI_RULE, "", sample_id=sample_id)
            assert result.tools_called == ["search"]
            assert result.hard == 1.0
    # Three distinct samples reach Pi; repeats of a sample id stay cached.
    assert run.call_count == 3
    assert backend.distinct_samples(tools=True) is True
    assert backend.distinct_samples(tools=False) is True


@pytest.mark.parametrize("route", ["tool", "text"])
def test_dream_rollouts_of_a_tool_task_reach_the_provider_as_distinct_samples(
    route,
) -> None:
    """The same dropped ``sample_id`` also collapsed mainline dream rollouts:
    ``multi_rollout`` of a ``tool_called`` task made one Pi call and reported
    K identical scores, so contrastive reflection never saw any spread."""
    import itertools
    import subprocess
    from unittest import mock

    from skillopt_sleep.backend import PiCliBackend
    from skillopt_sleep.rollout import multi_rollout

    counter = itertools.count()

    def stochastic_provider(cmd, *, input, **kwargs):
        # Alternates between calling the tool and answering without it.
        if next(counter) % 2 == 0:
            out = "TOOL_CALL: search\nanswer"
        else:
            out = "answer without the tool"
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    check = (
        {"op": "tool_called", "arg": "search"}
        if route == "tool"
        else {"op": "contains", "arg": "TOOL_CALL"}
    )
    task = TaskRecord(
        id="dream-task",
        project="/project",
        intent="Please look it up",
        reference_kind="rule",
        judge={"kind": "rule", "checks": [check]},
    )
    with mock.patch(
        "skillopt_sleep.backend.subprocess.run", side_effect=stochastic_provider
    ) as run:
        rollout_set = multi_rollout(PiCliBackend(), task, "skill", "", k=4, workers=1)
    assert run.call_count == 4
    assert [attempt.hard for attempt in rollout_set.attempts] == [1.0, 0.0, 1.0, 0.0]


def test_dual_backend_forwards_tool_sample_ids_to_the_target() -> None:
    seen: list[int] = []

    class _Target(MockBackend):
        def attempt(self, task, skill, memory, sample_id=0):
            seen.append(sample_id)
            return "TOOL_CALL: search"

    task = TaskRecord(
        id="tool-task",
        project="/project",
        intent="Please search",
        reference_kind="rule",
        judge={"kind": "rule", "checks": [{"op": "tool_called", "arg": "search"}]},
    )
    from skillopt_sleep.replay import repeated_samples_distinct, replay_one

    dual = DualBackend(_Target(), MockBackend())
    for sample_id in range(3):
        replay_one(dual, task, "", "", sample_id=sample_id)
    assert seen == [0, 1, 2]
    assert repeated_samples_distinct(dual, task) is True


def test_every_shipped_backend_route_claims_distinct_samples() -> None:
    from skillopt_sleep import backend as backend_module

    shipped = [
        backend_module.MockBackend,
        backend_module.PiCliBackend,
        backend_module.ClaudeCliBackend,
        backend_module.OpenCodeCliBackend,
        backend_module.CodexCliBackend,
        backend_module.CopilotCliBackend,
        backend_module.CursorCliBackend,
        backend_module.AzureOpenAIBackend,
        backend_module.AzureResponsesBackend,
    ]
    for cls in shipped:
        # distinct_samples inspects method signatures only, so no CLI, network,
        # or credential set-up is needed to evaluate the route contract.
        instance = cls.__new__(cls)
        assert instance.distinct_samples(tools=False) is True, cls.__name__
        assert instance.distinct_samples(tools=True) is True, cls.__name__


class _LegacyToolBackend(_RobustCandidateBackend):
    """A third-party backend written before repeated rollouts existed."""

    def __init__(self) -> None:
        self.attempt_calls = 0

    def attempt(self, task, skill, memory):  # no sample_id parameter
        self.attempt_calls += 1
        if self.RULE not in f"{skill}\n{memory}":
            return "wrong"
        return task.reference

    def attempt_with_tools(self, task, skill, memory, tools):
        return self.attempt(task, skill, memory), list(tools)


def _tool_tasks() -> list[TaskRecord]:
    tasks = _candidate_tasks()
    for task in tasks:
        task.reference_kind = "rule"
        task.judge = {
            "kind": "rule",
            "checks": [
                {"op": "tool_called", "arg": "search"},
                {"op": "contains", "arg": "ok"},
            ],
        }
    return tasks


def test_route_without_distinct_samples_is_inconclusive_not_brittle() -> None:
    backend = _LegacyToolBackend()
    before = backend.attempt_calls
    report = evaluate_adversarial_probes(
        backend,
        _tool_tasks(),
        _LegacyToolBackend.RULE,
        "",
        baseline_skill="",
        baseline_memory="",
        rollouts=3,
    )
    assert backend.distinct_samples(tools=True) is False
    assert report["n_probes"] == 1
    assert report["n_inconclusive"] == 1
    assert report["n_flagged"] == 0
    assert report["flagged"] is False
    assert report["conclusive"] is False
    assert report["brittleness_rate"] == 0.0
    row = report["rows"][0]
    assert row["status"] == "inconclusive"
    assert row["distinct_samples"] is False
    assert row["inconclusive_reason"] == "repeated_samples_unsupported"
    assert row["gap_change"] is None and row["samples"] == {}
    # Collapsed rollouts are not replayed at all, so they cost nothing.
    assert backend.attempt_calls == before
    json.dumps(report, allow_nan=False)


def test_single_rollout_makes_no_independence_claim_on_a_legacy_route() -> None:
    report = evaluate_adversarial_probes(
        _LegacyToolBackend(),
        _tool_tasks(),
        _LegacyToolBackend.RULE,
        "",
        baseline_skill="",
        baseline_memory="",
        rollouts=1,
    )
    assert report["n_inconclusive"] == 0
    assert report["conclusive"] is True
    assert report["rows"][0]["status"] == "stable"


@pytest.mark.parametrize("blocking", [False, True], ids=["advisory", "blocking"])
def test_legacy_route_inconclusive_evidence_in_the_gate_and_report(
    tmp_path, blocking
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    config = load_config(
        invoked_project=str(project),
        projects="invoked",
        backend="mock",
        state_dir=str(tmp_path / "state"),
        claude_home=str(tmp_path / ".claude"),
        evolve_memory=False,
        dream_adversarial=1,
        dream_adversarial_blocking=blocking,
        dream_adversarial_rollouts=2,
        auto_adopt=False,
    )
    outcome = run_sleep_cycle(
        config, seed_tasks=_tool_tasks(), backend=_LegacyToolBackend()
    )

    trial = outcome.report.gate_trials[0]
    probe = trial["adversarial_probe"]
    assert probe["n_inconclusive"] == 1 and probe["conclusive"] is False
    # Advisory evidence never changes the decision; blocking fails closed with
    # a reason that names the unsupported route rather than a score drop.
    assert trial["blocked_by_adversarial"] is blocking
    assert probe["block_reason"] == (
        "inconclusive_repeated_samples_unsupported" if blocking else ""
    )
    with open(
        os.path.join(outcome.staging_dir, "report.md"), encoding="utf-8"
    ) as handle:
        markdown = handle.read()
    assert (
        "1 probe(s) inconclusive: their replay route cannot produce distinct "
        "repeated samples, so they were not scored and cannot flag the candidate."
    ) in markdown
    assert "No conclusive probe was scored; blocking mode fails closed" in markdown
    assert (
        "| `train` | `train_adversarial_request-frame` | request-frame | "
        "— | — | — | — | — | inconclusive |"
    ) in markdown
