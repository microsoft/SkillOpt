"""Validation gate — accept / reject candidate skills.

Analogous to validation-based early stopping and model selection in neural
network training: compares the candidate's score against the current and
best scores, then returns an accept/reject decision.

The trainer owns side-effects (cache lookup, rollout, printing, state
mutation).  This module is the pure decision function.

Metric selection
----------------
Three gate metrics are supported:

* ``"hard"`` (default, backward-compatible):
  Compare candidate vs current/best using *hard* exact-match accuracy.
* ``"soft"``:
  Compare using *soft* per-item score (F1 / partial credit / etc.).
  Use this when a small held-out selection set has too few items for
  hard accuracy to be sensitive to incremental skill improvements.
* ``"mixed"``:
  Compare using a weighted average ``(1 - w) * hard + w * soft``.
  ``w`` is configurable via ``mixed_weight`` (default ``0.5``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

GateAction = Literal["accept_new_best", "accept", "reject"]
GateMetric = Literal["hard", "soft", "mixed"]


@dataclass(frozen=True)
class GateResult:
    """Immutable outcome of the validation gate."""

    action: GateAction
    current_skill: str
    current_score: float
    best_skill: str
    best_score: float
    best_step: int


@dataclass(frozen=True)
class GateBlock:
    """Structured reason a validation gate cannot make a decision."""

    blocker: str
    items: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {"blocker": self.blocker, "items": self.items}


def find_gate_block(results: list[object]) -> GateBlock | None:
    """Return a block report when required gate results are not scored."""
    items = [_blocking_item(result) for result in results if _is_gate_unscored(result)]
    if not items:
        return None
    return GateBlock(blocker=_primary_blocker(items), items=items)


def require_scored_gate_results(results: list[object]) -> None:
    """Raise if a gate would be comparing incomplete evaluation results."""
    block = find_gate_block(results)
    if block is not None:
        raise ValueError(f"blocked:{block.blocker}")


def select_gate_score(
    hard: float,
    soft: float,
    metric: GateMetric = "hard",
    mixed_weight: float = 0.5,
) -> float:
    """Project (hard, soft) onto a single comparison metric.

    Parameters
    ----------
    hard, soft
        Aggregate hard / soft scores from a rollout batch (both 0..1).
    metric
        Which metric to compare on.
    mixed_weight
        For ``"mixed"``: weight given to ``soft``. Must be in ``[0, 1]``.
        Ignored for ``"hard"`` / ``"soft"``.
    """
    if metric == "hard":
        return float(hard)
    if metric == "soft":
        return float(soft)
    if metric == "mixed":
        w = max(0.0, min(1.0, float(mixed_weight)))
        return (1.0 - w) * float(hard) + w * float(soft)
    raise ValueError(
        f"unknown gate metric {metric!r}; expected 'hard', 'soft', or 'mixed'"
    )


def _is_gate_unscored(result: object) -> bool:
    status = str(_result_field(result, "score_status", "") or "").strip().lower()
    return status == "unscored" or _result_field(result, "hard", None) is None or _result_field(result, "soft", None) is None


def _blocking_item(result: object) -> dict[str, Any]:
    blocker = _result_blocker(result)
    item = {
        "id": str(_result_field(result, "id", "unknown")),
        "blocker": blocker,
        "score_status": str(_result_field(result, "score_status", "unscored") or "unscored"),
        "target_status": str(_result_field(result, "target_status", "") or ""),
        "evaluator_status": str(_result_field(result, "evaluator_status", "") or ""),
        "target_trace_path": str(_result_field(result, "target_trace_path", "") or ""),
        "evaluator_trace_path": str(_result_field(result, "evaluator_trace_path", "") or ""),
        "fail_reason": str(_result_field(result, "fail_reason", "") or ""),
    }
    return {key: value for key, value in item.items() if value != ""}


def _result_blocker(result: object) -> str:
    explicit = str(_result_field(result, "blocker", "") or "").strip()
    if explicit:
        return explicit
    target_status = str(_result_field(result, "target_status", "") or "").strip().lower()
    evaluator_status = str(_result_field(result, "evaluator_status", "") or "").strip().lower()
    if target_status == "failed":
        return "target_rollout_failed"
    if evaluator_status == "not_run":
        return "evaluator_not_run"
    if evaluator_status == "failed":
        return "evaluator_failed"
    if _result_field(result, "hard", None) is None or _result_field(result, "soft", None) is None:
        return "invalid_evaluator_score"
    return "unscored"


def _primary_blocker(items: list[dict[str, Any]]) -> str:
    priority = [
        "target_rollout_failed",
        "evaluator_not_run",
        "evaluator_failed",
        "invalid_evaluator_score",
    ]
    blockers = {str(item.get("blocker") or "") for item in items}
    for blocker in priority:
        if blocker in blockers:
            return blocker
    return next((blocker for blocker in blockers if blocker), "unscored")


def _result_field(result: object, key: str, default: Any) -> Any:
    if hasattr(result, key):
        return getattr(result, key)
    if isinstance(result, dict):
        return result.get(key, default)
    return default


def evaluate_gate(
    candidate_skill: str,
    cand_hard: float,
    current_skill: str,
    current_score: float,
    best_skill: str,
    best_score: float,
    best_step: int,
    global_step: int,
    *,
    cand_soft: float = 0.0,
    metric: GateMetric = "hard",
    mixed_weight: float = 0.5,
) -> GateResult:
    """Pure gate decision: compare candidate score to current/best.

    Parameters
    ----------
    candidate_skill
        The candidate skill content being evaluated.
    cand_hard, cand_soft
        Aggregate hard / soft scores of the candidate on the selection set.
    current_skill, current_score
        The currently-active skill and its *metric-space* score.
    best_skill, best_score, best_step
        The best-so-far skill, its *metric-space* score, and the step
        at which it was accepted.
    global_step
        Current global training step (recorded if a new best is accepted).
    cand_soft
        Soft score of the candidate; only consulted when ``metric != "hard"``.
        Defaults to ``0.0`` for backward compatibility with callers that
        previously passed only ``cand_hard``.
    metric
        Which metric to compare on. Defaults to ``"hard"`` to preserve
        the original gate behavior.
    mixed_weight
        Weight on ``soft`` when ``metric == "mixed"``.

    Returns
    -------
    GateResult
        Updated state; the caller decides what to do with it (print,
        mutate trainer state, log, etc.).
    """
    cand_score = select_gate_score(cand_hard, cand_soft, metric, mixed_weight)

    if cand_score > current_score:
        if cand_score > best_score:
            return GateResult(
                action="accept_new_best",
                current_skill=candidate_skill,
                current_score=cand_score,
                best_skill=candidate_skill,
                best_score=cand_score,
                best_step=global_step,
            )
        return GateResult(
            action="accept",
            current_skill=candidate_skill,
            current_score=cand_score,
            best_skill=best_skill,
            best_score=best_score,
            best_step=best_step,
        )
    return GateResult(
        action="reject",
        current_skill=current_skill,
        current_score=current_score,
        best_skill=best_skill,
        best_score=best_score,
        best_step=best_step,
    )
