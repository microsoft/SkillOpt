"""SkillOpt-Sleep — Stage 4: consolidate (one SkillOpt epoch).

This is the core that makes nightly evolution *safe*: it proposes bounded
edits from replayed failures, applies them to a candidate skill/memory, then
**gates** the candidate on a held-out slice of the user's own tasks. Only a
candidate that strictly improves the held-out score is accepted — the SkillOpt
validation gate, vendored self-contained in ``skillopt_sleep.gate``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple

from skillopt_sleep.adversarial import (
    MAX_PROBE_ROLLOUTS,
    MIN_BLOCKING_ROLLOUTS,
    evaluate_adversarial_probes,
)
from skillopt_sleep.backend import Backend

# Self-contained validation gate (vendored from SkillOpt; zero dependency on the
# research package, so this open-source tool stays decoupled from the paper code).
from skillopt_sleep.gate import evaluate_gate, select_gate_score
from skillopt_sleep.memory import apply_edits_detailed
from skillopt_sleep.replay import aggregate_scores, replay_batch
from skillopt_sleep.types import EditRecord, ReplayResult, TaskRecord

_HAVE_REPO_GATE = True


def _finite_score(value: float) -> float | None:
    """Return a JSON-safe score, preserving finite values only."""
    return value if math.isfinite(value) else None


@dataclass
class ConsolidationResult:
    accepted: bool
    gate_action: str
    baseline_score: float
    candidate_score: float
    new_skill: str
    new_memory: str
    applied_edits: List[EditRecord]
    rejected_edits: List[EditRecord]
    holdout_baseline: float
    holdout_candidate: float
    # ── observability (so a 0.0->0.0 night is self-diagnosing, not a black box) ──
    # Edits that changed nothing (anchor not found, or a duplicate add). They are
    # neither applied nor gate-rejected, so without this list they would vanish
    # from the report and the night would look like the optimizer produced less
    # than it did.
    unmatched_edits: List[EditRecord] = field(default_factory=list)
    holdout_detail: List[dict] = field(default_factory=list)  # per val task: hard/soft/resp/why
    reflect_raw: str = ""        # the optimizer's last raw reply (empty => reflect produced nothing)
    call_error: str = ""         # backend's last call error (timeout/auth/empty)
    # True when the gate's val slice was not disjoint from the tasks reflect
    # saw, so its comparison cannot detect overfitting. A night in this state
    # stages edits but never certifies them.
    holdout_leaked: bool = False
    # Each gated candidate carries the task-level comparison that produced its
    # decision. This includes rejected intermediate skill/memory trials and the
    # fresh final replay, so aggregate changes never hide an individual task.
    # Keep new optional fields last to preserve positional construction.
    gate_trials: List[dict] = field(default_factory=list)


def _split(tasks: List[TaskRecord]) -> Tuple[List[TaskRecord], List[TaskRecord], bool]:
    """Return ``(train_tasks, val_tasks, holdout_leaked)``.

    train drives reflect; val gates updates. test is held out entirely from
    consolidation and is scored by the caller. Accepts legacy split names
    (replay->train, holdout->val) for robustness.

    ``holdout_leaked`` is True when val is not disjoint from train — i.e. the
    gate would score the very tasks the edits were derived from. A single mined
    task that carries a train/val (or legacy) split always lands here; a lone
    ``test`` task instead yields empty train/val and no gate, so it is not
    flagged as leaked. Such a non-disjoint comparison cannot detect overfitting,
    so the caller must not treat it as validation.
    """
    def _norm(s: str) -> str:
        return {"replay": "train", "holdout": "val"}.get(s, s)

    train = [t for t in tasks if _norm(t.split) == "train"]
    val = [t for t in tasks if _norm(t.split) == "val"]
    leaked = False
    # Be robust if a split is empty: fall back so a night still does something,
    # but never silently use test as train or val. An all-test batch therefore
    # returns empty train/val (caller scores test separately; gate is a no-op).
    if not val:
        # Prefer train as the gate reference; otherwise any non-test tasks.
        # Do not fall back to the full task list (that would leak held-out test).
        val = train or [t for t in tasks if _norm(t.split) != "test"]
        leaked = bool(val)
    if not train:
        train = val
        leaked = leaked or bool(train)
    if not leaked and train and val:
        train_ids = {t.id for t in train}
        if any(t.id in train_ids for t in val):
            leaked = True
    return train, val, leaked


def _holdout_detail(pairs: List[Tuple[TaskRecord, ReplayResult]]) -> List[dict]:
    """Per-task held-out evidence so a 0.0 night explains itself: was the
    response empty (backend call failed) or non-empty-but-failing-checks
    (judge too strict / edit didn't help)? The two need opposite fixes."""
    out: List[dict] = []
    for t, r in pairs:
        resp = r.response or ""
        out.append({
            "id": t.id,
            "reference_kind": t.reference_kind,
            "hard": r.hard,
            "soft": r.soft,
            "response_len": len(resp),
            "response_head": resp[:200],
            "why": (r.fail_reason or r.judge_rationale or "")[:200],
        })
    return out


def _task_deltas(
    tasks: List[TaskRecord],
    baseline_pairs: List[Tuple[TaskRecord, ReplayResult]],
    candidate_pairs: List[Tuple[TaskRecord, ReplayResult]],
    metric: str,
    mixed_weight: float,
) -> List[dict]:
    """Compare candidate scores with the matching baseline validation tasks."""
    baseline_by_id = {task.id: result for task, result in baseline_pairs}
    candidate_by_id = {task.id: result for task, result in candidate_pairs}
    out: List[dict] = []
    for task in tasks:
        baseline = baseline_by_id.get(task.id)
        candidate = candidate_by_id.get(task.id)
        baseline_score = (
            select_gate_score(baseline.hard, baseline.soft, metric, mixed_weight)
            if baseline is not None
            else None
        )
        candidate_score = (
            select_gate_score(candidate.hard, candidate.soft, metric, mixed_weight)
            if candidate is not None
            else None
        )
        if baseline_score is None or candidate_score is None:
            out.append({
                "task_id": task.id,
                "tags": list(task.tags or []),
                "baseline_score": baseline_score,
                "candidate_score": candidate_score,
                "status": "regressed",
                "scores_are_finite": False,
            })
            continue
        scores_are_finite = math.isfinite(baseline_score) and math.isfinite(
            candidate_score
        )
        if not scores_are_finite:
            # The aggregate gate already rejects NaN, but +inf could otherwise
            # look like an improvement. Strict mode treats any malformed task
            # comparison as a regression and therefore fails closed.
            status = "regressed"
        elif candidate_score > baseline_score:
            status = "improved"
        elif candidate_score < baseline_score:
            status = "regressed"
        else:
            status = "unchanged"
        out.append({
            "task_id": task.id,
            "tags": list(task.tags or []),
            "baseline_score": _finite_score(baseline_score),
            "candidate_score": _finite_score(candidate_score),
            "status": status,
            "scores_are_finite": scores_are_finite,
        })
    return out


def consolidate(
    backend: Backend,
    tasks: List[TaskRecord],
    skill: str,
    memory: str,
    *,
    edit_budget: int = 4,
    gate_metric: str = "mixed",
    gate_mixed_weight: float = 0.5,
    gate_no_regression: bool = False,
    dream_adversarial: int = 0,
    dream_adversarial_blocking: bool = False,
    dream_adversarial_margin: float = 0.0,
    dream_adversarial_rollouts: int = 1,
    gate_mode: str = "on",       # "on" (hard/soft per gate_metric) | "off" (greedy)
    rollouts_k: int = 1,         # >1 => multi-rollout contrastive reflection
    evolve_skill: bool = True,
    evolve_memory: bool = True,
    night: int = 1,
) -> ConsolidationResult:
    """Run one consolidation epoch: reflect -> bounded edit -> gate.

    train tasks drive reflect; val tasks gate the update (test is held out by the
    caller). With ``gate_mode='off'`` edits are accepted greedily (no val-improve
    requirement) — the user opts out of hard filtering — but val scores are still
    recorded so the report shows whether quality moved.

    Skill and memory are evolved in sequence (skill first if both enabled).
    """
    if isinstance(dream_adversarial, bool) or not isinstance(
        dream_adversarial, int
    ):
        raise ValueError("dream_adversarial must be an integer")
    if dream_adversarial > 0:
        if not isinstance(dream_adversarial_blocking, bool):
            raise ValueError("dream_adversarial_blocking must be a boolean")
        if isinstance(dream_adversarial_margin, bool) or not isinstance(
            dream_adversarial_margin, (int, float)
        ):
            raise ValueError(
                "dream_adversarial_margin must be a finite number in [0, 1]"
            )
        margin = float(dream_adversarial_margin)
        if not math.isfinite(margin) or not 0.0 <= margin <= 1.0:
            raise ValueError(
                "dream_adversarial_margin must be a finite number in [0, 1]"
            )
        if isinstance(dream_adversarial_rollouts, bool) or not isinstance(
            dream_adversarial_rollouts, int
        ):
            raise ValueError(
                "dream_adversarial_rollouts must be an integer in "
                f"[1, {MAX_PROBE_ROLLOUTS}]"
            )
        if not 1 <= dream_adversarial_rollouts <= MAX_PROBE_ROLLOUTS:
            raise ValueError(
                "dream_adversarial_rollouts must be an integer in "
                f"[1, {MAX_PROBE_ROLLOUTS}]"
            )
        if (
            dream_adversarial_blocking
            and dream_adversarial_rollouts < MIN_BLOCKING_ROLLOUTS
        ):
            raise ValueError(
                "dream_adversarial_blocking requires "
                f"dream_adversarial_rollouts >= {MIN_BLOCKING_ROLLOUTS} so a "
                "single stochastic sample can never reject a candidate; keep "
                "the probes advisory for single-rollout runs"
            )
    from skillopt_sleep import evidence as evlog
    ev = evlog.get(backend)
    train_tasks, val_tasks, holdout_leaked = _split(tasks)
    gate_off = str(gate_mode).strip().lower() in {"off", "none", "false", "greedy"}
    holdout_detail: List[dict] = []

    # ── baseline on the VAL slice (the gate reference) ────────────────────
    # When the gate is OFF the user has opted out of holding out a validation set
    # (the daily-use design): we accept edits greedily and judge quality only on
    # the real test set, scored by the caller. So we SKIP all val scoring — it is
    # both wasted cost and contrary to the "no val set required" design.
    if gate_off:
        base_hard, base_soft = 0.0, 0.0
        base_pairs: List[Tuple[TaskRecord, ReplayResult]] = []
    else:
        evlog.set_phase(backend, "baseline_val")
        base_pairs = replay_batch(backend, val_tasks, skill, memory)
        base_hard, base_soft = aggregate_scores(base_pairs)
        holdout_detail = _holdout_detail(base_pairs)
    base_score = select_gate_score(base_hard, base_soft, gate_metric, gate_mixed_weight)
    if ev is not None:
        ev.log("gate", "baseline", gate_mode=("off" if gate_off else "on"),
               n_train=len(train_tasks), n_val=len(val_tasks),
               hard=base_hard, soft=base_soft, score=base_score,
               metric=gate_metric, mixed_weight=gate_mixed_weight)

    # ── reflect over TRAIN-split failures/successes ───────────────────────
    evlog.set_phase(backend, "train")
    train_pairs = replay_batch(backend, train_tasks, skill, memory)
    failures = [(t, r) for (t, r) in train_pairs if r.hard < 1.0]
    successes = [(t, r) for (t, r) in train_pairs if r.hard >= 1.0]

    cand_skill, cand_memory = skill, memory
    all_applied: List[EditRecord] = []
    all_rejected: List[EditRecord] = []
    all_unmatched: List[EditRecord] = []
    gate_trials: List[dict] = []
    current_pairs = base_pairs

    def _edits_payload(edits: List[EditRecord]) -> List[dict]:
        return [{"op": e.op, "content": e.content, "anchor": e.anchor,
                 "rationale": e.rationale} for e in edits]

    def _gate_apply(doc: str, edits: List[EditRecord], which: str) -> str:
        nonlocal cand_skill, cand_memory, base_score, current_pairs
        nonlocal all_applied, all_rejected
        if ev is not None:
            ev.log("reflect", "edits_returned", target=which,
                   n_edits=len(edits), edits=_edits_payload(edits))
        if not edits:
            return doc
        new_doc, applied, unmatched = apply_edits_detailed(doc, edits)
        if unmatched:
            all_unmatched.extend(unmatched)
            if ev is not None:
                ev.log("reflect", "edits_unmatched", target=which,
                       n_edits=len(unmatched), edits=_edits_payload(unmatched))
        if not applied:
            return doc
        trial_skill = new_doc if which == "skill" else cand_skill
        trial_memory = new_doc if which == "memory" else cand_memory

        def _probe_candidate() -> tuple[dict | None, bool]:
            if dream_adversarial <= 0:
                return None, False
            evlog.set_phase(backend, f"adversarial_probe:{which}")
            probe = evaluate_adversarial_probes(
                backend,
                train_tasks,
                trial_skill,
                trial_memory,
                baseline_skill=cand_skill,
                baseline_memory=cand_memory,
                factor=dream_adversarial,
                metric=gate_metric,
                mixed_weight=gate_mixed_weight,
                margin=dream_adversarial_margin,
                rollouts=dream_adversarial_rollouts,
            )
            blocked = bool(
                dream_adversarial_blocking
                and (probe["flagged"] or not probe["conclusive"])
            )
            probe["blocking"] = bool(dream_adversarial_blocking)
            probe["blocked"] = blocked
            probe["block_reason"] = (
                "inconclusive_no_probes"
                if blocked and not probe["conclusive"]
                else ("brittle_score_drop" if blocked else "")
            )
            if ev is not None:
                ev.log(
                    "adversarial",
                    "candidate_probe",
                    target=which,
                    **probe,
                )
            return probe, blocked

        # gate OFF: accept greedily with NO val scoring (the daily-use path)
        if gate_off:
            probe, blocked_by_adversarial = _probe_candidate()
            accepted = not blocked_by_adversarial
            if accepted:
                all_applied.extend(applied)
            else:
                all_rejected.extend(applied)
            if probe is not None:
                gate_trials.append({
                    "target": which,
                    "baseline_score": None,
                    "candidate_score": None,
                    "accepted": accepted,
                    "blocked_by_regression": False,
                    "blocked_by_adversarial": blocked_by_adversarial,
                    "task_deltas": [],
                    "adversarial_probe": probe,
                })
            if ev is not None:
                ev.log("gate", "trial", target=which, mode="greedy",
                       accepted=accepted,
                       blocked_by_adversarial=blocked_by_adversarial,
                       n_edits=len(applied))
            return new_doc if accepted else doc
        # gate ON: score the candidate on the VAL slice, keep only if it improves
        evlog.set_phase(backend, f"gate_trial:{which}")
        pairs = replay_batch(backend, val_tasks, trial_skill, trial_memory)
        h, s = aggregate_scores(pairs)
        cand_score = select_gate_score(h, s, gate_metric, gate_mixed_weight)
        task_deltas = _task_deltas(
            val_tasks, current_pairs, pairs, gate_metric, gate_mixed_weight
        )
        blocked_by_regression = bool(
            gate_no_regression
            and any(row["status"] == "regressed" for row in task_deltas)
        )
        trial_base_score = base_score
        gate_improved = cand_score > base_score and not blocked_by_regression
        probe = None
        blocked_by_adversarial = False
        if gate_improved:
            probe, blocked_by_adversarial = _probe_candidate()
        improved = gate_improved and not blocked_by_adversarial
        trial_record = {
            "target": which,
            "baseline_score": _finite_score(trial_base_score),
            "candidate_score": _finite_score(cand_score),
            "accepted": improved,
            "blocked_by_regression": blocked_by_regression,
            "task_deltas": task_deltas,
        }
        if dream_adversarial > 0:
            trial_record["blocked_by_adversarial"] = blocked_by_adversarial
            trial_record["adversarial_probe"] = probe
        gate_trials.append(trial_record)
        if ev is not None:
            trial_evidence = {
                "target": which,
                "mode": "gated",
                "baseline_score": trial_base_score,
                "cand_hard": h,
                "cand_soft": s,
                "cand_score": cand_score,
                "accepted": improved,
                "blocked_by_regression": blocked_by_regression,
                "task_deltas": task_deltas,
                "n_edits": len(applied),
            }
            if dream_adversarial > 0:
                trial_evidence["blocked_by_adversarial"] = blocked_by_adversarial
            ev.log("gate", "trial", **trial_evidence)
        if improved:
            base_score = cand_score
            current_pairs = pairs
            all_applied.extend(applied)
            return new_doc
        all_rejected.extend(applied)
        return doc

    if evolve_skill:
        if rollouts_k > 1:
            # multi-rollout contrastive reflection: run each train task K times
            # and distill a rule from the good-vs-bad contrast (the imagination signal).
            # Parallelize across tasks (each multi_rollout also parallelizes its K
            # attempts). This dream phase is the dominant cost; serial execution
            # times out on real backends. Cap total in-flight at the worker env.
            import os
            from concurrent.futures import ThreadPoolExecutor

            from skillopt_sleep.rollout import contrastive_reflect, multi_rollout
            try:
                _w = int(os.environ.get("SKILLOPT_SLEEP_WORKERS", "1"))
            except ValueError:
                _w = 1
            if _w > 1 and len(train_tasks) > 1:
                # split the worker budget between task-parallelism and per-task K
                task_workers = max(1, min(len(train_tasks), _w))
                per_task = max(1, _w // task_workers)
                with ThreadPoolExecutor(max_workers=task_workers) as ex:
                    sets = list(ex.map(
                        lambda t: multi_rollout(backend, t, cand_skill, cand_memory,
                                                k=rollouts_k, workers=per_task),
                        train_tasks))
            else:
                sets = [multi_rollout(backend, t, cand_skill, cand_memory,
                                      k=rollouts_k, workers=1)
                        for t in train_tasks]
            edits = contrastive_reflect(
                backend, sets, cand_skill, cand_memory,
                edit_budget=edit_budget, target="skill",
                gate_metric=gate_metric,
                gate_mixed_weight=gate_mixed_weight,
            )
            # fall back to single-shot reflect if contrast yielded nothing
            if not edits:
                edits = backend.reflect(
                    failures, successes, cand_skill, cand_memory,
                    edit_budget=edit_budget, evolve_skill=True, evolve_memory=False,
                )
        else:
            edits = backend.reflect(
                failures, successes, cand_skill, cand_memory,
                edit_budget=edit_budget, evolve_skill=True, evolve_memory=False,
            )
        cand_skill = _gate_apply(cand_skill, edits, "skill")

    if evolve_memory:
        # re-evaluate failures under the (possibly improved) skill
        evlog.set_phase(backend, "train_post_skill")
        train_pairs2 = replay_batch(backend, train_tasks, cand_skill, cand_memory)
        failures2 = [(t, r) for (t, r) in train_pairs2 if r.hard < 1.0]
        successes2 = [(t, r) for (t, r) in train_pairs2 if r.hard >= 1.0]
        edits_m = backend.reflect(
            failures2, successes2, cand_skill, cand_memory,
            edit_budget=edit_budget, evolve_skill=False, evolve_memory=True,
        )
        cand_memory = _gate_apply(cand_memory, edits_m, "memory")

    # ── final decision ────────────────────────────────────────────────────
    if gate_off:
        # greedy mode: no val scoring at all. Keep whatever edits we applied; the
        # caller measures real quality on the test set. We report holdout_candidate
        # as 0.0 (val intentionally not computed in this variant).
        final_hard, final_soft = 0.0, 0.0
        final_score = 0.0
        accepted = bool(all_applied)
        action = "greedy_applied" if all_applied else "greedy_noop"
        base_gate_score = 0.0
    else:
        # scored on the VAL slice (the gate reference)
        evlog.set_phase(backend, "final_val")
        final_pairs = replay_batch(backend, val_tasks, cand_skill, cand_memory)
        final_hard, final_soft = aggregate_scores(final_pairs)
        final_score = select_gate_score(final_hard, final_soft, gate_metric, gate_mixed_weight)
        base_gate_score = select_gate_score(base_hard, base_soft, gate_metric, gate_mixed_weight)
        final_task_deltas = _task_deltas(
            val_tasks, base_pairs, final_pairs, gate_metric, gate_mixed_weight
        )
        final_blocked_by_regression = bool(
            gate_no_regression
            and any(row["status"] == "regressed" for row in final_task_deltas)
        )
        if _HAVE_REPO_GATE:
            gate = evaluate_gate(
                candidate_skill=cand_skill,
                cand_hard=final_hard,
                current_skill=skill,
                current_score=base_gate_score,
                best_skill=skill,
                best_score=base_gate_score,
                best_step=night - 1,
                global_step=night,
                cand_soft=final_soft,
                metric=gate_metric,
                mixed_weight=gate_mixed_weight,
            )
            action = gate.action
            accepted = (
                bool(all_applied)
                and final_score > base_gate_score
                and not final_blocked_by_regression
            )
        else:
            action = "accept" if final_score > base_gate_score else "reject"
            accepted = (
                bool(all_applied)
                and final_score > base_gate_score
                and not final_blocked_by_regression
            )
        # The gate scores documents, not edit bookkeeping: when every proposed
        # edit was dropped during the per-target trials, `all_applied` is empty
        # and nothing changed, yet the score comparison can still yield an
        # accept-flavoured action. Reporting that as "accept_new_best" while
        # `accepted` is False makes the headline contradict the outcome.
        if not accepted and action in {"accept", "accept_new_best"}:
            action = "reject"
        # The gate cannot certify an improvement it measured on the same tasks
        # the edits were derived from -- that comparison cannot detect
        # overfitting, which is exactly how a reward hack scores 1.0. Abstain
        # rather than report a verdict the evidence does not support. Edits are
        # still staged, so a human can review them.
        if accepted and holdout_leaked:
            action = "reject_unverified"
            accepted = False
        # A per-target trial can improve and tentatively apply an edit, while a
        # later fresh final replay regresses. The returned documents already
        # roll back in that case; keep the edit bookkeeping/report consistent
        # by moving those tentative edits into the rejected set as well.
        if not accepted and all_applied:
            for edit in all_applied:
                if edit not in all_rejected:
                    all_rejected.append(edit)
            all_applied = []
        gate_trials.append({
            "target": "final",
            "baseline_score": _finite_score(base_gate_score),
            "candidate_score": _finite_score(final_score),
            "accepted": accepted,
            "blocked_by_regression": final_blocked_by_regression,
            "task_deltas": final_task_deltas,
        })

    if ev is not None:
        w = max(0.0, min(1.0, float(gate_mixed_weight)))
        if gate_metric == "mixed":
            formula = (
                f"score = (1-{w})*hard + {w}*soft; "
                f"baseline = (1-{w})*{base_hard:.3f} + {w}*{base_soft:.3f} = {base_gate_score:.3f}; "
                f"candidate = (1-{w})*{final_hard:.3f} + {w}*{final_soft:.3f} = {final_score:.3f}"
            )
        else:
            formula = (
                f"score = {gate_metric}; baseline = {base_gate_score:.3f}; "
                f"candidate = {final_score:.3f}"
            )
        ev.log("gate", "decision", action=action, accepted=accepted,
               baseline_score=base_gate_score, candidate_score=final_score,
               baseline_hard=base_hard, baseline_soft=base_soft,
               candidate_hard=final_hard, candidate_soft=final_soft,
               metric=gate_metric, mixed_weight=gate_mixed_weight,
               blocked_by_regression=(
                   final_blocked_by_regression if not gate_off else False
               ),
               task_deltas=(final_task_deltas if not gate_off else []),
               formula=formula, n_applied=len(all_applied),
               n_rejected=len(all_rejected),
               n_unmatched=len(all_unmatched), night=night)

    return ConsolidationResult(
        accepted=accepted,
        gate_action=action,
        baseline_score=base_gate_score,
        candidate_score=final_score,
        new_skill=cand_skill if accepted else skill,
        new_memory=cand_memory if accepted else memory,
        applied_edits=all_applied,
        rejected_edits=all_rejected,
        unmatched_edits=all_unmatched,
        holdout_baseline=base_hard,
        holdout_candidate=final_hard,
        holdout_detail=holdout_detail,
        gate_trials=gate_trials,
        reflect_raw=getattr(backend, "last_reflect_raw", "") or "",
        call_error=getattr(backend, "last_call_error", "") or "",
        holdout_leaked=holdout_leaked,
    )
