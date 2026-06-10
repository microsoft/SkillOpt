"""Mock replay and gate utilities for AIForAI SkillOpt-Sleep."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from skillopt_sleep.aiforai.types import AiforaiReplayResult, AiforaiTaskRecord


@dataclass(slots=True)
class AiforaiScoreSummary:
    aggregate_hard: float
    aggregate_soft: float
    by_source: dict[str, float] = field(default_factory=dict)
    by_family: dict[str, float] = field(default_factory=dict)
    results: list[AiforaiReplayResult] = field(default_factory=list)


@dataclass(slots=True)
class AiforaiGateDecision:
    accepted: bool
    action: str
    reason: str


def evaluate_tasks(
    tasks: list[AiforaiTaskRecord],
    skill: str,
) -> AiforaiScoreSummary:
    results: list[AiforaiReplayResult] = []
    for task in tasks:
        hard, soft, missing = _score_task(task, skill)
        results.append(
            AiforaiReplayResult(
                task_id=task.id,
                source_agent=task.source_agent,
                task_family=task.task_family,
                hard=hard,
                soft=soft,
                response=skill,
                fail_reason=f"missing: {', '.join(missing)}" if missing else "",
            )
        )
    return _summarize(results)


def propose_mock_rules(
    tasks: list[AiforaiTaskRecord],
    skill: str,
) -> list[str]:
    missing_by_family: dict[str, set[str]] = defaultdict(set)

    for task in tasks:
        missing_by_family[task.task_family].update(_missing_required_checks(task, skill))

    proposed: list[str] = []
    for family in sorted(missing_by_family):
        missing = sorted(missing_by_family[family], key=str.casefold)
        if missing:
            proposed.append(
                f"For {family} tasks, explicitly include: {', '.join(missing)}."
            )
    return proposed


def gate_candidate(
    baseline: AiforaiScoreSummary,
    candidate: AiforaiScoreSummary,
) -> AiforaiGateDecision:
    source_regressions = _slice_regressions(
        baseline.by_source,
        candidate.by_source,
        label="source",
    )
    if source_regressions:
        return AiforaiGateDecision(
            accepted=False,
            action="reject",
            reason=f"candidate regressed source hard-score slices: {source_regressions}",
        )

    family_regressions = _slice_regressions(
        baseline.by_family,
        candidate.by_family,
        label="family",
    )
    if family_regressions:
        return AiforaiGateDecision(
            accepted=False,
            action="reject",
            reason=f"candidate regressed family hard-score slices: {family_regressions}",
        )

    if candidate.aggregate_hard > baseline.aggregate_hard:
        return AiforaiGateDecision(
            accepted=True,
            action="accept",
            reason=(
                "candidate aggregate hard score improved "
                f"({baseline.aggregate_hard:.3f} -> {candidate.aggregate_hard:.3f})"
            ),
        )
    return AiforaiGateDecision(
        accepted=False,
        action="reject",
        reason=(
            "candidate did not improve aggregate hard score "
            f"({baseline.aggregate_hard:.3f} -> {candidate.aggregate_hard:.3f})"
        ),
    )


def _score_task(task: AiforaiTaskRecord, skill: str) -> tuple[float, float, list[str]]:
    required = _required_contains_checks(task)
    if not required:
        return 0.0, 0.0, ["no local checks"]

    missing = _missing_required_checks(task, skill)
    present = len(required) - len(missing)
    soft = present / len(required)
    hard = 1.0 if not missing else 0.0
    return hard, soft, missing


def _required_contains_checks(task: AiforaiTaskRecord) -> list[str]:
    checks = task.judge.get("checks", []) if task.judge else []
    required: list[str] = []
    for check in checks:
        if not isinstance(check, dict):
            continue
        if check.get("op") != "contains":
            continue
        arg = str(check.get("arg", "")).strip()
        if arg:
            required.append(arg)
    return required


def _missing_required_checks(task: AiforaiTaskRecord, skill: str) -> list[str]:
    return [
        item
        for item in _required_contains_checks(task)
        if not _contains_required_text(skill, item)
    ]


def _summarize(results: list[AiforaiReplayResult]) -> AiforaiScoreSummary:
    if not results:
        return AiforaiScoreSummary(aggregate_hard=0.0, aggregate_soft=0.0)

    aggregate_hard = sum(result.hard for result in results) / len(results)
    aggregate_soft = sum(result.soft for result in results) / len(results)
    return AiforaiScoreSummary(
        aggregate_hard=aggregate_hard,
        aggregate_soft=aggregate_soft,
        by_source=_group_mean(results, "source_agent"),
        by_family=_group_mean(results, "task_family"),
        results=results,
    )


def _group_mean(results: list[AiforaiReplayResult], attr: str) -> dict[str, float]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for result in results:
        buckets[str(getattr(result, attr))].append(result.hard)
    return {
        key: sum(values) / len(values)
        for key, values in sorted(buckets.items())
    }


def _contains_required_text(text: str, needle: str) -> bool:
    if not needle:
        return False
    folded_text = text.casefold()
    folded_needle = needle.casefold()
    if folded_needle.isascii():
        pattern = rf"(?<!\w){re.escape(folded_needle)}(?!\w)"
        return re.search(pattern, folded_text) is not None
    return folded_needle in folded_text


def _slice_regressions(
    baseline: dict[str, float],
    candidate: dict[str, float],
    *,
    label: str,
) -> str:
    regressions: list[str] = []
    for key in sorted(set(baseline) & set(candidate)):
        if candidate[key] < baseline[key]:
            regressions.append(
                f"{label}={key} ({baseline[key]:.3f} -> {candidate[key]:.3f})"
            )
    return ", ".join(regressions)
