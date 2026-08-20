"""SkillOpt-Sleep — LLM-backed task miner.

The heuristic miner (mine.py) produces TaskRecords without a checkable
reference, so real harvested transcripts can't show measurable lift. This
module uses an optimizer backend to turn session digests into TaskRecords
WITH a checkable rubric judge — the missing piece for real-data improvement.

For each recurring intent it extracts:
  * a clean, generalized `intent` (the reusable task, stripped of one-off specifics)
  * a `rubric` (what a good answer must satisfy). A rubric, when present, is
    always stored as the reference and scored by the backend's judge() -- it
    resists the reward-hacking that literal/format checks invite. Programmatic
    `contains`/`regex`/`section_present`/`section_contains`/`tool_called` checks
    are kept only as a fallback reference for the intents where the miner
    supplies no rubric.
  * a preference signal (was the user satisfied?) to weight failures

It is deliberately conservative: it only emits a task when it can name a
concrete, checkable success criterion, so the gate has real signal. Tasks it
can't make checkable are dropped (logged), not faked.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List

from skillopt_sleep import prompts as prompt_registry
from skillopt_sleep.backend import Backend, _extract_json
from skillopt_sleep.judges import char_bound
from skillopt_sleep.mine import session_skill_hint
from skillopt_sleep.types import SessionDigest, TaskRecord


def _digest_to_prompt(d: SessionDigest) -> str:
    # Template lives in the central prompt registry (skillopt_sleep.prompts)
    # so the dashboard can display and override it live.
    prompts = "\n".join(f"  - {p[:240]}" for p in d.user_prompts[:6]) or "  (none)"
    final = (d.assistant_finals[-1][:400] if d.assistant_finals else "(none)")
    return prompt_registry.render("miner", {
        "__PROJECT__": d.project or "(unknown)",
        "__PROMPTS__": prompts,
        "__FINAL__": final,
        "__FEEDBACK__": ", ".join(d.feedback_signals[:6]) or "(none)",
    })


def _mk_task(d: SessionDigest, obj: Dict[str, Any], idx: int) -> TaskRecord | None:
    intent = str(obj.get("intent", "")).strip()
    if len(intent) < 8:
        return None
    checks = obj.get("checks") or []
    # A non-string rubric (e.g. a JSON null) must not become the literal "None"
    # and outrank the checks; only a real string counts as a rubric.
    rubric_raw = obj.get("rubric")
    rubric = rubric_raw.strip() if isinstance(rubric_raw, str) else ""
    satisfied = bool(obj.get("satisfied", False))

    # Keep only well-formed checks: the scorer runs these verbatim during
    # replay, so a max_chars/min_chars with a non-integer arg (or an arg-less
    # op that needs one) would crash or fail forever. Drop those here, and keep
    # the accepted shapes aligned with validate_checks() so a mined tasks file
    # never fails validation later (it rejects bools and negative bounds).
    _needs_str_arg = {
        "section_present", "section_contains", "regex", "contains",
        "not_contains", "tool_called",
    }
    # Trimming a regex would change what it matches (leading/trailing spaces are
    # significant in a pattern), so only substring/tool/heading args are stripped.
    _strip_arg = _needs_str_arg - {"regex"}
    clean_checks = []
    for c in checks:
        if not isinstance(c, dict):
            continue
        op = c.get("op")
        arg = c.get("arg")
        if op in _needs_str_arg:
            # Store the stripped value: stray whitespace would otherwise become
            # part of the required substring / tool name.
            if isinstance(arg, str) and arg.strip():
                clean_checks.append(
                    {"op": op, "arg": arg.strip() if op in _strip_arg else arg}
                )
        elif op in {"max_chars", "min_chars"}:
            # Shared parser with validate_checks() so the two cannot drift:
            # rejects bools, non-integral floats and inf/nan (OverflowError).
            try:
                bound = char_bound(arg)
            except (OverflowError, TypeError, ValueError):
                continue
            if bound < 0:
                continue
            clean_checks.append({"op": op, "arg": bound})
        elif op == "no_refusal":
            clean_checks.append({"op": op, "arg": None})

    import hashlib
    tid = "llm_" + hashlib.sha256((d.project + intent).encode()).hexdigest()[:12]

    judge = {"kind": "rule", "checks": clean_checks}
    # The optimizer edits skill text that is prepended to the model's context,
    # so ANY check for a literal string in the response can be satisfied by
    # instructing the model to emit that string -- observed twice in practice,
    # first with section_present and then with contains. Only semantic grading
    # resists that, so a rubric always wins when the miner supplies one.
    # Rule judges remain for imported gbrain-style benchmarks, which carry
    # checks but no rubric.
    if rubric:
        return TaskRecord(
            id=tid, project=d.project, intent=intent,
            reference_kind="rubric", reference=rubric,
            outcome="success" if satisfied else "fail",
            tags=["mined:llm"], source_sessions=[d.session_id],
            skill_hint=session_skill_hint(d),
        )
    if clean_checks:
        return TaskRecord(
            id=tid, project=d.project, intent=intent,
            reference_kind="rule", judge=judge,
            outcome="success" if satisfied else "fail",
            tags=["mined:llm"], source_sessions=[d.session_id],
            skill_hint=session_skill_hint(d),
        )
    return None  # not checkable -> drop


def make_llm_miner(
    backend: Backend,
    *,
    max_sessions: int = 20,
    max_tasks: int = 40,
) -> Callable[[List[SessionDigest]], List[TaskRecord]]:
    """Return an llm_miner(digests) -> list[TaskRecord] bound to a backend."""

    def _miner(digests: List[SessionDigest]) -> List[TaskRecord]:
        ev = getattr(backend, "evidence", None)
        out: List[TaskRecord] = []
        for d in digests[:max_sessions]:
            if not d.user_prompts:
                continue
            prompt = _digest_to_prompt(d)
            raw = backend._call(prompt, max_tokens=800)  # type: ignore[attr-defined]
            arr = _extract_json(raw, "array")
            candidates = arr if isinstance(arr, list) else []
            made: List[TaskRecord] = []
            dropped = 0
            full = False
            for i, obj in enumerate(candidates[:3]):
                if isinstance(obj, dict):
                    t = _mk_task(d, obj, i)
                    if t is not None:
                        made.append(t)
                        out.append(t)
                    else:
                        dropped += 1  # not checkable -> dropped, and now logged
                if len(out) >= max_tasks:
                    full = True
                    break
            if ev is not None:
                # The transcript->task link of the evidentiary chain: what this
                # session was, exactly what the miner was asked, exactly what it
                # replied, and which TaskRecords (with checks) came out of it.
                ev.log("mine", "miner_exchange", session_id=d.session_id,
                       project=d.project, prompt=prompt, raw_reply=raw,
                       parse_ok=isinstance(arr, list), n_candidates=len(candidates),
                       n_tasks=len(made), n_dropped_uncheckable=dropped)
                for t in made:
                    ev.log("mine", "task_mined", task_id=t.id,
                           session_id=d.session_id, intent=t.intent,
                           reference_kind=t.reference_kind,
                           checks=(t.judge or {}).get("checks", []),
                           rubric=t.reference, outcome=t.outcome)
            if full:
                return out
        return out

    return _miner
