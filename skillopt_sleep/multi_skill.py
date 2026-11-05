"""SkillOpt-Sleep — consolidate several skill groups in one night.

A night normally consolidates one managed skill. When mined tasks have been
grouped per skill, each group is an independent consolidation: its own tasks, its
own document, its own gate decision. This module drives those runs so that a
group with no tasks, an unusable name, or a failing backend is isolated and
reported instead of aborting the groups around it.

Adoption is unchanged and still explicit: nothing here writes a skill file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from skillopt_sleep.backend import Backend, CursorBackendError
from skillopt_sleep.consolidate import ConsolidationResult, consolidate
from skillopt_sleep.handoff_backend import PendingCalls
from skillopt_sleep.types import SkillGroupReport, TaskRecord

CONSOLIDATED = "consolidated"
SKIPPED = "skipped"
FAILED = "failed"

BUDGET_REACHED = "token budget for the night was reached"


@dataclass
class SkillGroup:
    """One skill's slice of the night: its name, its document, its tasks."""

    skill_name: str
    skill: str = ""
    tasks: List[TaskRecord] = field(default_factory=list)


@dataclass
class GroupConsolidation:
    """Per-group outcome. ``result`` is set only when the group actually ran."""

    skill_name: str
    status: str
    result: Optional[ConsolidationResult] = None
    reason: str = ""
    n_tasks: int = 0

    @property
    def accepted(self) -> bool:
        return bool(self.result and self.result.accepted)


def consolidate_groups(
    backend: Backend,
    groups: Sequence[SkillGroup],
    memory: str = "",
    *,
    consolidate_fn: Callable[..., ConsolidationResult] = consolidate,
    group_kwargs_fn: Optional[Callable[[SkillGroup], Dict[str, object]]] = None,
    budget_exhausted: Optional[Callable[[], bool]] = None,
    **consolidate_kwargs: object,
) -> Dict[str, GroupConsolidation]:
    """Consolidate each group independently, in order, isolating failures.

    Returns one entry per distinct skill name, keyed by that name and ordered
    first-seen — at most one entry per input group, not exactly one. The result
    is keyed by name, so a repeated name structurally cannot carry a second
    outcome: the first group under a name wins and later ones are dropped
    without an entry of their own. Every blank name likewise collapses into the
    single ``""`` entry. Callers needing a decision for every input must pass
    distinct, non-blank names.

    A group is ``skipped`` when it is unusable (blank name, or no tasks) and
    ``failed`` when its own consolidation raised; both leave every other
    group's decision untouched. When ``budget_exhausted`` returns True before a
    usable group starts, that group is ``skipped`` too; a group that already
    started finishes, and the predicate is passed on to ``consolidate_fn``.

    ``memory`` is the shared agent memory and is passed through read-only: group
    runs evolve skills only, so no group can rewrite another group's memory.
    ``group_kwargs_fn`` can add group-scoped inputs such as recalled history;
    its ordinary failures are isolated to that group. ``PendingCalls`` and
    ``CursorBackendError`` from either the factory or consolidator propagate as
    cycle-level pause/fail-closed control flow rather than becoming report rows.
    """
    # This wrapper's contract is stricter than consolidate(): shared memory is
    # always read-only.  Override a caller-supplied value instead of passing a
    # duplicate keyword (which would otherwise turn the group into a failure).
    consolidate_kwargs = dict(consolidate_kwargs)
    consolidate_kwargs["evolve_memory"] = False
    if budget_exhausted is not None:
        consolidate_kwargs["budget_exhausted"] = budget_exhausted
    out: Dict[str, GroupConsolidation] = {}
    for group in groups:
        name = (group.skill_name or "").strip()
        if not name:
            # Carry the task count even though the group is unusable: the row
            # is meant to be that group's own evidence, and reporting 0 tasks
            # for a group that had several misstates why it was dropped.
            out.setdefault("", GroupConsolidation(
                "", SKIPPED, reason="group has no skill name",
                n_tasks=len(group.tasks),
            ))
            continue
        if name in out:
            continue  # first group wins; a repeated name is not a second night
        if not group.tasks:
            out[name] = GroupConsolidation(
                name, SKIPPED, reason="no mined tasks for this skill"
            )
            continue
        if budget_exhausted is not None and budget_exhausted():
            out[name] = GroupConsolidation(
                name, SKIPPED, reason=BUDGET_REACHED, n_tasks=len(group.tasks)
            )
            continue
        try:
            group_kwargs = dict(consolidate_kwargs)
            if group_kwargs_fn is not None:
                group_kwargs.update(group_kwargs_fn(group))
                # A per-group factory cannot weaken the shared-memory invariant.
                group_kwargs["evolve_memory"] = False
            result = consolidate_fn(
                backend, list(group.tasks), group.skill, memory,
                **group_kwargs,
            )
        except (PendingCalls, CursorBackendError):
            # These exceptions are cycle-level control flow, not isolated
            # evidence about one group. Swallowing them can advance/save an
            # incomplete handoff night or weaken Cursor's fail-closed path.
            raise
        except Exception as exc:  # one group's failure must not abort the night
            out[name] = GroupConsolidation(
                name, FAILED, reason=f"{type(exc).__name__}: {exc}"[:300],
                n_tasks=len(group.tasks),
            )
            continue
        skipped_steps = getattr(result, "budget_skipped", None) or []
        reason = (
            f"{BUDGET_REACHED}; skipped {', '.join(skipped_steps)}"
            if skipped_steps
            else ""
        )
        out[name] = GroupConsolidation(
            name, CONSOLIDATED, result=result, reason=reason,
            n_tasks=len(group.tasks),
        )
    return out


def skill_group_reports(
    outcomes: Dict[str, GroupConsolidation],
) -> List[SkillGroupReport]:
    """Build one report row per entry in ``outcomes``, from its evidence only.

    One row per *outcome*, which is at most one per input group: ``outcomes`` is
    keyed by skill name, so groups that collapsed onto a shared key upstream
    have already become a single entry and cannot produce a row each.

    A skipped or failed group reports its reason and keeps zeroed scores rather
    than inheriting another group's numbers, and an accepted group's row is
    unaffected by its neighbours' decisions.
    """
    rows: List[SkillGroupReport] = []
    for name, outcome in outcomes.items():
        row = SkillGroupReport(
            skill_name=name,
            status=outcome.status,
            reason=outcome.reason,
            n_tasks=outcome.n_tasks,
        )
        result = outcome.result
        if result is not None:
            row.accepted = result.accepted
            row.gate_action = result.gate_action
            row.baseline_score = result.baseline_score
            row.candidate_score = result.candidate_score
            row.n_applied_edits = len(result.applied_edits)
            row.n_rejected_edits = len(result.rejected_edits)
        rows.append(row)
    return rows


def accepted_group_skills(outcomes: Dict[str, GroupConsolidation]) -> Dict[str, str]:
    """Skill name -> new document, for groups whose gate accepted an update."""
    return {
        name: outcome.result.new_skill
        for name, outcome in outcomes.items()
        if outcome.accepted and outcome.result is not None
    }
