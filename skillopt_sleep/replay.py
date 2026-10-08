"""SkillOpt-Sleep — Stage 3: replay.

Re-run mined TaskRecords offline under a given (skill, memory) and score
them, producing the (hard, soft) signal SkillOpt's gate consumes.

Single-shot text replay by default. Tasks whose rule judge requires a tool
call (gbrain's `tool_called`) use the backend's tool-aware path. Backends that
cannot enforce that execution boundary fail explicitly rather than scoring a
self-reported tool call.
"""
from __future__ import annotations

from typing import List, Tuple

from skillopt_sleep.backend import Backend
from skillopt_sleep.types import ReplayResult, TaskRecord


def _required_tools(task: TaskRecord) -> List[str]:
    """Tool names a rule judge requires (op == 'tool_called')."""
    if task.reference_kind != "rule" or not task.judge:
        return []
    tools = []
    for c in task.judge.get("checks", []) or []:
        if isinstance(c, dict) and c.get("op") == "tool_called" and c.get("arg"):
            tools.append(str(c["arg"]))
    return tools


def replay_one(backend: Backend, task: TaskRecord, skill: str, memory: str,
               sample_id: int = 0) -> ReplayResult:
    """``sample_id`` distinguishes repeated dream rollouts of the same
    (task, skill, memory) in the attempt cache — without it all K rollouts
    collapse to one cached response and the contrastive signal is always 0."""
    import time
    tools = _required_tools(task)
    tools_called: List[str] = []
    t0 = time.time()
    # Backends that only expose the older tokens_used() contract (no per-call
    # token_delta) need a before/after difference on the same thread to report
    # the call-local cost; snapshot the total before the attempt for them.
    token_delta_fn = getattr(backend, "token_delta", None)
    tokens_before = None if token_delta_fn is not None else backend.tokens_used()
    if tools:
        response, tools_called = backend.attempt_with_tools(task, skill, memory, tools)
    else:
        response = backend.attempt(task, skill, memory, sample_id=sample_id)
    latency_ms = (time.time() - t0) * 1000.0
    # Call-local token accounting: prefer the backend's per-call delta. For
    # CliBackend - and for a DualBackend whose target has one - that delta lives
    # in thread-local storage and is exact, including a known zero for a cache
    # hit, so no text-length estimate is substituted. A backend that only
    # implements the older tokens_used() contract has no per-call delta, so use
    # its before/after difference. That difference is only call-local while the
    # backend is used sequentially: the counter is shared, so an overlapping
    # worker's spend lands in it too. Fall back to a length estimate only when
    # the backend reports no tracking (zero) at all. See replay_batch() for which
    # combinations are safe to share across workers.
    if token_delta_fn is not None:
        tokens = token_delta_fn()
    else:
        tokens = max(0, backend.tokens_used() - (tokens_before or 0))
        # A backend without token_delta() that still changed tokens_used()
        # reports that real difference above. Only an unchanged total means the
        # backend does not track tokens (e.g. a mock), so approximate then.
        if tokens == 0:
            tokens = (len(skill) + len(memory) + len(task.intent) + len(response)) // 4

    # rule judges may need the detected tool calls; score locally when possible
    if task.reference_kind == "rule" and task.judge:
        from skillopt_sleep.judges import score_rule_judge_with_feedback
        hard, soft, rationale, optimizer_feedback = score_rule_judge_with_feedback(
            task.judge, response, tools_called
        )
    else:
        hard, soft, rationale = backend.judge(task, response)
        # Backend judge rationales are audit evidence and may contain provider
        # or grader implementation details. Keep the optimizer channel generic
        # for non-rule judges unless a future typed safe-feedback API exists.
        optimizer_feedback = (
            "The response did not satisfy the task's evaluation criteria."
            if hard < 1.0
            else ""
        )

    ev = getattr(backend, "evidence", None)
    if ev is not None:
        # One scored-attempt record per (phase, task): the task->score link of
        # the evidentiary chain, with the failing checks named in `why`.
        ev.log("replay", "result",
               phase=getattr(backend, "evidence_phase", ""),
               task_id=task.id, split=task.split, origin=task.origin,
               reference_kind=task.reference_kind, sample_id=sample_id,
               hard=float(hard), soft=float(soft), why=rationale or "",
               response_head=(response or "")[:400], tools_called=tools_called,
               tokens=int(tokens), latency_ms=round(latency_ms, 1))

    return ReplayResult(
        id=task.id,
        hard=float(hard),
        soft=float(soft),
        response=response,
        fail_reason="" if hard >= 1.0 else (rationale or "below threshold"),
        task_type=(task.tags[0] if task.tags else "task"),
        judge_rationale=rationale,
        tools_called=tools_called,
        tokens=int(tokens),
        latency_ms=round(latency_ms, 1),
        optimizer_feedback=(optimizer_feedback if hard < 1.0 else ""),
    )


import os
from concurrent.futures import ThreadPoolExecutor


def replay_batch(
    backend: Backend,
    tasks: List[TaskRecord],
    skill: str,
    memory: str,
    *,
    workers: int = 0,
) -> List[Tuple[TaskRecord, ReplayResult]]:
    """Replay tasks, optionally in parallel.

    Real backends are network-bound, so a thread pool gives a large speedup on
    big test sets (like the research harness's --workers). ``workers`` defaults
    to env SKILLOPT_SLEEP_WORKERS or 1 (sequential). Mock stays sequential
    (deterministic) unless asked otherwise.

    Every worker shares the one ``backend`` passed in, so per-call token
    accounting has to be call-local. Which combinations support that:

    * ``CliBackend`` and subclasses - per-call delta in thread-local storage,
      safe to share across workers.
    * ``DualBackend`` whose target defines ``token_delta()`` - reports that
      target's thread-local delta, safe to share across workers.
    * a bare backend with only the older ``tokens_used()`` contract - sequential
      only: ``replay_one`` differences a counter the other workers are also
      spending, so an overlapping worker's tokens are counted into this task.
    * ``DualBackend`` whose target only has ``tokens_used()`` - sequential only,
      for that reason plus a baseline that lives on the shared DualBackend.

    Only a thread-local per-call delta is safe to share. Everything built on a
    cumulative total - the bare legacy backend as much as the DualBackend
    fallback - over-counts when workers overlap: a 37-token attempt reported 74
    with two workers, because one worker's spend landed in the other's
    difference. Over-counting is the failure direction, so token and cost
    budgets see more spend than actually happened. Both shapes are supported
    sequentially and covered by tests there; neither is claimed here. Use
    ``workers=1``, or a backend that reports a per-call delta, when the
    accounting has to be exact.
    """
    if workers <= 0:
        workers = int(os.environ.get("SKILLOPT_SLEEP_WORKERS", "1") or "1")
    if workers <= 1 or len(tasks) <= 1:
        return [(t, replay_one(backend, t, skill, memory)) for t in tasks]
    results: List = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=min(workers, len(tasks))) as ex:
        futs = {ex.submit(replay_one, backend, t, skill, memory): i
                for i, t in enumerate(tasks)}
        for fut in futs:
            i = futs[fut]
            results[i] = (tasks[i], fut.result())
    return results


def aggregate_scores(pairs: List[Tuple[TaskRecord, ReplayResult]]) -> Tuple[float, float]:
    if not pairs:
        return 0.0, 0.0
    hard = sum(r.hard for _t, r in pairs) / len(pairs)
    soft = sum(r.soft for _t, r in pairs) / len(pairs)
    return hard, soft


def aggregate_cost(pairs: List[Tuple[TaskRecord, ReplayResult]]) -> Tuple[float, float]:
    """Mean (tokens, latency_ms) per task — the cost objectives."""
    if not pairs:
        return 0.0, 0.0
    tok = sum(r.tokens for _t, r in pairs) / len(pairs)
    lat = sum(r.latency_ms for _t, r in pairs) / len(pairs)
    return tok, lat


def multi_objective_reward(
    pairs: List[Tuple[TaskRecord, ReplayResult]],
    *,
    w_acc: float = 1.0,
    w_tokens: float = 0.0,
    w_latency: float = 0.0,
    token_ref: float = 2000.0,
    latency_ref_ms: float = 15000.0,
) -> float:
    """Weighted reward = accuracy↑, tokens↓, latency↓.

    Cost terms are normalized against a reference and clamped to [0,1], so a
    response at/under the reference cost contributes ~1.0 and an expensive one
    less. Weights let the user trade off (default = accuracy only, backward
    compatible).
    """
    if not pairs:
        return 0.0
    acc, _soft = aggregate_scores(pairs)
    tok, lat = aggregate_cost(pairs)
    tok_score = max(0.0, 1.0 - tok / max(1.0, token_ref)) if token_ref else 0.0
    lat_score = max(0.0, 1.0 - lat / max(1.0, latency_ref_ms)) if latency_ref_ms else 0.0
    total_w = w_acc + w_tokens + w_latency
    if total_w <= 0:
        return acc
    return (w_acc * acc + w_tokens * tok_score + w_latency * lat_score) / total_w
