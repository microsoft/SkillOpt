"""Candidate-level robustness probes for SkillOpt-Sleep.

The nightly gate measures a candidate on held-out tasks, but a candidate can
still be brittle to harmless changes in how a request is framed.  This module
creates bounded, deterministic variants of *real training tasks* and compares
the candidate's score on each source task with its score on the corresponding
variant.

Probes are evidence, not training examples: they never enter validation/test
splits and they never influence reflection.  The caller decides whether a
flag is advisory or blocks a candidate.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Sequence, Tuple

from skillopt_sleep.backend import Backend
from skillopt_sleep.gate import select_gate_score
from skillopt_sleep.replay import replay_one
from skillopt_sleep.types import ReplayResult, TaskRecord

MAX_PROBES_PER_TASK = 3
MAX_ADVERSARIAL_PROBES = 256
MAX_PROBE_ROLLOUTS = 8
# Blocking decisions need repeated rollouts so a single stochastic sample can
# never reject a candidate on its own; advisory runs may use one rollout.
MIN_BLOCKING_ROLLOUTS = 2


def _normalize_split(value: str) -> str:
    return {"replay": "train", "holdout": "val"}.get(value, value)


def _strip_polite_frame(intent: str) -> str:
    """Reframe explicitly politeness-marked requests, and nothing else.

    Semantic-preservation contract: a transformation is emitted only when the
    removed prefix is an unambiguous request marker, so removal cannot change
    what is being asked:

    * a leading ``please`` (a pure politeness marker), and
    * a leading ``can/could/would you please`` (a modal question that the
      politeness marker disambiguates as a request; the trailing question
      mark, if any, becomes a period because the reframed text is the same
      request in imperative form).

    Bare modal questions (``Can you swim?``, ``Would you like tea?``) are
    never reframed: without the politeness marker they may ask about ability,
    permission, or desire, and stripping the modal changes the meaning. The
    same applies to first-person desire framings (``I want you to ...``),
    which earlier revisions stripped and this contract deliberately drops.
    """
    modal_request = r"(?is)^\s*(?:can|could|would)\s+you\s+please\s+"
    plain_please = r"(?is)^\s*please\s+"
    for pattern, reframed_request in ((modal_request, True), (plain_please, False)):
        rewritten, count = re.subn(pattern, "", intent, count=1)
        if not count or not rewritten.strip():
            continue
        rewritten = rewritten.strip()
        if not rewritten[:1].isalpha():
            return ""
        if reframed_request and rewritten.endswith("?"):
            rewritten = rewritten[:-1].rstrip()
            if not rewritten or not rewritten[:1].isalpha():
                return ""
            rewritten += "."
        return rewritten[:1].upper() + rewritten[1:]
    return ""


def _variant_intents(intent: str) -> List[Tuple[str, str]]:
    """Return conservative, deterministic surface variants in priority order."""
    raw = str(intent or "").strip()
    if not raw:
        return []
    variants: List[Tuple[str, str]] = []
    reframed = _strip_polite_frame(raw)
    if reframed and reframed != raw:
        variants.append(("request-frame", reframed))
    variants.extend((
        (
            "section-wrapper",
            f"Task to complete:\n\n{raw}\n\nRespond to the task above.",
        ),
        (
            "delimiter-wrapper",
            f"The request is between the markers.\n<request>\n{raw}\n</request>",
        ),
        (
            "boundary-shift",
            f"Use the following request as the complete instruction:\n---\n{raw}\n---",
        ),
    ))
    out: List[Tuple[str, str]] = []
    seen = {raw}
    for kind, value in variants:
        if value not in seen:
            seen.add(value)
            out.append((kind, value))
    return out


def generate_adversarial_probes(
    tasks: Sequence[TaskRecord],
    *,
    factor: int = 1,
) -> List[TaskRecord]:
    """Create bounded semantic-preserving probes from real TRAIN tasks only.

    ``factor`` is the maximum variants per source task and is capped at three.
    Synthetic, recalled, validation, and test records are excluded so probes
    cannot recycle held-out material or amplify already-derived tasks.
    """
    if isinstance(factor, bool) or not isinstance(factor, int):
        raise ValueError("adversarial probe factor must be an integer")
    per_task = max(0, min(factor, MAX_PROBES_PER_TASK))
    if per_task == 0:
        return []
    out: List[TaskRecord] = []
    source_ids: set[str] = set()
    for task in tasks:
        if len(out) >= MAX_ADVERSARIAL_PROBES:
            break
        if _normalize_split(task.split) != "train" or task.origin != "real":
            continue
        if task.derived_from or "recall" in (task.tags or []):
            continue
        if task.id in source_ids:
            raise ValueError(
                f"adversarial probe source ids must be unique: {task.id!r}"
            )
        source_ids.add(task.id)
        for kind, intent in _variant_intents(task.intent)[:per_task]:
            if len(out) >= MAX_ADVERSARIAL_PROBES:
                break
            out.append(TaskRecord(
                id=f"{task.id}_adversarial_{kind}",
                project=task.project,
                intent=intent,
                context_excerpt=task.context_excerpt,
                system=task.system,
                attempted_solution=task.attempted_solution,
                outcome=task.outcome,
                reference_kind=task.reference_kind,
                reference=task.reference,
                judge=dict(task.judge),
                tags=list(task.tags) + ["dream", "adversarial", f"probe:{kind}"],
                source_sessions=list(task.source_sessions),
                split="train",
                origin="dream",
                derived_from=task.id,
                skill_hint=task.skill_hint,
            ))
    return out


def _score(result: ReplayResult, metric: str, mixed_weight: float) -> float | None:
    value = select_gate_score(result.hard, result.soft, metric, mixed_weight)
    return value if math.isfinite(value) else None


def _rollout_scores(
    backend: Backend,
    task: TaskRecord,
    skill: str,
    memory: str,
    *,
    metric: str,
    mixed_weight: float,
    rollouts: int,
) -> List[float | None]:
    """Score one task ``rollouts`` times under one document pair.

    Each rollout uses a distinct ``sample_id`` so caching backends produce
    genuinely repeated samples instead of collapsing to one response.
    """
    scores: List[float | None] = []
    for sample_id in range(rollouts):
        result = replay_one(backend, task, skill, memory, sample_id=sample_id)
        scores.append(_score(result, metric, mixed_weight))
    return scores


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def evaluate_adversarial_probes(
    backend: Backend,
    tasks: Sequence[TaskRecord],
    skill: str,
    memory: str,
    *,
    baseline_skill: str,
    baseline_memory: str,
    factor: int = 1,
    metric: str = "mixed",
    mixed_weight: float = 0.5,
    margin: float = 0.0,
    rollouts: int = 1,
) -> Dict[str, Any]:
    """Score identical source/probe pairs under the BASELINE and the CANDIDATE
    documents and report candidate-introduced brittleness.

    Decision rule (documented so the evidence is auditable):

    * every task in every arm is replayed ``rollouts`` times and the arm's
      score for that task is the MEAN of those rollouts;
    * per row, ``gap = probe_score - source_score`` is computed for both the
      baseline arm and the candidate arm, and
      ``gap_change = candidate_gap - baseline_gap``;
    * a row is brittle only when ``gap_change < -margin`` AND the per-index
      paired worsening holds in a strict majority of rollout indices;
    * any non-finite score in any arm marks the row invalid, which fails
      closed: invalid rows count as flagged.

    Frame sensitivity already present under the baseline documents therefore
    never flags a candidate; only the change the candidate introduces does.
    All four aggregated scores and the per-rollout samples are retained per
    row so the decision can be audited from the evidence alone. The total
    replay cost is ``rollouts * 2 * (n_sources + n_probes)``.
    """
    if isinstance(margin, bool) or not isinstance(margin, (int, float)):
        raise ValueError("adversarial probe margin must be a finite number in [0, 1]")
    numeric_margin = float(margin)
    if not math.isfinite(numeric_margin) or not 0.0 <= numeric_margin <= 1.0:
        raise ValueError("adversarial probe margin must be a finite number in [0, 1]")
    if isinstance(rollouts, bool) or not isinstance(rollouts, int):
        raise ValueError(
            f"adversarial probe rollouts must be an integer in [1, {MAX_PROBE_ROLLOUTS}]"
        )
    if not 1 <= rollouts <= MAX_PROBE_ROLLOUTS:
        raise ValueError(
            f"adversarial probe rollouts must be an integer in [1, {MAX_PROBE_ROLLOUTS}]"
        )

    probes = generate_adversarial_probes(tasks, factor=factor)
    source_by_id = {
        task.id: task
        for task in tasks
        if _normalize_split(task.split) == "train"
        and task.origin == "real"
        and not task.derived_from
        and "recall" not in (task.tags or [])
    }
    source_ids = list(dict.fromkeys(probe.derived_from for probe in probes))
    arms = {
        "baseline": (baseline_skill, baseline_memory),
        "candidate": (skill, memory),
    }
    source_scores: Dict[str, Dict[str, List[float | None]]] = {}
    probe_scores: Dict[str, Dict[str, List[float | None]]] = {}
    for arm, (arm_skill, arm_memory) in arms.items():
        source_scores[arm] = {
            source_id: _rollout_scores(
                backend, source_by_id[source_id], arm_skill, arm_memory,
                metric=metric, mixed_weight=mixed_weight, rollouts=rollouts,
            )
            for source_id in source_ids
        }
        probe_scores[arm] = {
            probe.id: _rollout_scores(
                backend, probe, arm_skill, arm_memory,
                metric=metric, mixed_weight=mixed_weight, rollouts=rollouts,
            )
            for probe in probes
        }

    rows: List[Dict[str, Any]] = []
    flagged = 0
    invalid = 0
    gap_changes: List[float] = []
    for probe in probes:
        samples = {
            "baseline_source": source_scores["baseline"][probe.derived_from],
            "baseline_probe": probe_scores["baseline"][probe.id],
            "candidate_source": source_scores["candidate"][probe.derived_from],
            "candidate_probe": probe_scores["candidate"][probe.id],
        }
        valid = all(
            score is not None for scores in samples.values() for score in scores
        )
        if valid:
            means = {name: _mean(scores) for name, scores in samples.items()}
            baseline_gap = means["baseline_probe"] - means["baseline_source"]
            candidate_gap = means["candidate_probe"] - means["candidate_source"]
            gap_change = candidate_gap - baseline_gap
            worsened = sum(
                1
                for index in range(rollouts)
                if (
                    samples["candidate_probe"][index]
                    - samples["candidate_source"][index]
                )
                - (
                    samples["baseline_probe"][index]
                    - samples["baseline_source"][index]
                )
                < 0.0
            )
            worsening_fraction = worsened / rollouts
            majority_worsened = worsened * 2 > rollouts
            is_brittle = gap_change < -numeric_margin and majority_worsened
            gap_changes.append(gap_change)
        else:
            means = {name: None for name in samples}
            baseline_gap = candidate_gap = gap_change = None
            worsening_fraction = None
            is_brittle = True
        if is_brittle:
            flagged += 1
        if not valid:
            invalid += 1
        kind = next(
            (tag.removeprefix("probe:") for tag in probe.tags if tag.startswith("probe:")),
            "unknown",
        )
        rows.append({
            "source_task_id": probe.derived_from,
            "probe_task_id": probe.id,
            "probe_kind": kind,
            "baseline_source_score": means["baseline_source"],
            "baseline_probe_score": means["baseline_probe"],
            "candidate_source_score": means["candidate_source"],
            "candidate_probe_score": means["candidate_probe"],
            "baseline_gap": baseline_gap,
            "candidate_gap": candidate_gap,
            "gap_change": gap_change,
            "worsening_fraction": worsening_fraction,
            "samples": {name: list(scores) for name, scores in samples.items()},
            "status": "invalid" if not valid else ("brittle" if is_brittle else "stable"),
        })

    n = len(rows)
    return {
        "enabled": True,
        "factor": max(0, min(factor, MAX_PROBES_PER_TASK)),
        "margin": numeric_margin,
        "rollouts": rollouts,
        "n_sources": len(source_ids),
        "n_probes": n,
        "n_flagged": flagged,
        "n_invalid": invalid,
        "brittleness_rate": (flagged / n) if n else 0.0,
        "worst_gap_change": min(gap_changes) if gap_changes else None,
        "conclusive": n > 0,
        "flagged": flagged > 0,
        "rows": rows,
    }
