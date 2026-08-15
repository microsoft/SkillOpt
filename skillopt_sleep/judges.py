"""SkillOpt-Sleep — rule-based judges (gbrain-evals compatible).

Implements the programmatic check operators used by gbrain-evals'
skillopt-v1 benchmark so we can score skill outputs locally, with NO judge
API call:

  * section_present <name>   — a markdown heading containing <name> exists
  * regex <pattern>          — the pattern matches the response
  * max_chars <n>            — response length <= n
  * min_chars <n>            — response length >= n
  * contains <text>          — substring present (case-insensitive)
  * not_contains <text>      — substring absent (case-insensitive)
  * no_refusal               — the response is not a bare refusal/abstention
  * tool_called <name>       — a tool with <name> was invoked (needs a tool loop;
                               in single-shot replay we approximate via an
                               explicit "TOOL_CALL: <name>" marker the agent emits)

Ops divide into two families, and the distinction is load-bearing:

  * *shape* ops (section_present, max_chars, min_chars) constrain how an answer
    is formatted. They are trivially satisfiable by an optimizer — adding a
    heading scores 1.0 without the answer improving at all.
  * *outcome* ops (contains, not_contains, no_refusal, regex, tool_called)
    constrain what the answer actually does.

A judge built only from shape ops is a formatting checker, not a grader; see
:func:`is_shape_only`. Callers should prefer an outcome-based judge (rubric or
outcome ops) so the gate cannot be won by reformatting.

A task whose judge is {"kind": "rule", "checks": [...]} passes (hard=1.0) iff
ALL checks pass; soft = fraction of checks passed. This mirrors gbrain's
all-checks-must-pass rule scoring and gives the gate a smooth signal.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


def _section_present(response: str, name: str) -> bool:
    # a markdown heading line (#, ##, ...) or bold line that contains `name`
    pat = re.compile(
        r"(?im)^\s{0,3}(#{1,6}\s*.*%s|\*\*.*%s.*\*\*\s*:?)\s*$" % (re.escape(name), re.escape(name))
    )
    if pat.search(response or ""):
        return True
    # also accept "Name:" style label at line start
    label = re.compile(r"(?im)^\s*%s\s*:" % re.escape(name))
    return bool(label.search(response or ""))


_REFUSAL_PREFIXES = (
    "cannot complete",
    "i cannot",
    "i can't",
    "i'm unable",
    "i am unable",
    "unable to complete",
    "sorry, i can't",
    "sorry, i cannot",
    "no can do",
)


def _is_refusal(response: str) -> bool:
    """Detect a bare refusal: an abstention with no substantive work reported.

    A refusal that still explains what was searched and what is missing is a
    useful answer, so only short responses whose opening is an abstention count.
    """
    text = (response or "").strip()
    if not text:
        return True
    # Strip leading markdown markers -- blockquote (>), list bullets (-, *),
    # numbered items (1. / 1)), emphasis and headings -- BEFORE bounding the
    # head, so a refusal cannot hide behind >160 marker characters.
    content = re.sub(r"^(?:[>\-*_#\s]|\d+[.)])+", "", text)
    head = content[:160].lower()
    if not any(head.startswith(p) for p in _REFUSAL_PREFIXES):
        return False
    # A long response that opens with an abstention still did the work of
    # explaining why; only terse dead-ends are refusals.
    return len(content) < 600


def _check(
    op: str,
    arg: Any,
    response: str,
    tools_called: List[str],
    *,
    allow_tool_marker: bool,
) -> Tuple[bool, str]:
    """Evaluate one check.

    Returns ``(passed, problem)``. ``problem`` is non-empty only when the check
    itself is malformed (e.g. an unparseable regex) rather than simply unmet —
    the two need opposite fixes, so they must not look alike in the rationale.
    """
    r = response or ""
    if op == "section_present":
        return _section_present(r, str(arg)), ""
    if op == "regex":
        try:
            return bool(re.search(str(arg), r)), ""
        except re.error as exc:
            # A malformed pattern can never match, so it would fail every
            # rollout forever and read exactly like a model that never
            # complies. Surface it instead of hiding it behind a False.
            return False, f"invalid regex ({exc})"
    if op == "max_chars":
        return len(r) <= int(arg), ""
    if op == "min_chars":
        return len(r) >= int(arg), ""
    if op == "contains":
        return str(arg).lower() in r.lower(), ""
    if op == "not_contains":
        return str(arg).lower() not in r.lower(), ""
    if op == "no_refusal":
        return not _is_refusal(r), ""
    if op == "tool_called":
        name = str(arg).lower()
        if any(name == t.lower() for t in tools_called):
            return True, ""
        if allow_tool_marker:
            # Preserve the legacy self-reported marker when no execution data
            # is available.
            return bool(re.search(r"(?i)\btool_call\s*:\s*%s\b" % re.escape(name), r)), ""
        return False, ""
    # unknown op: do not block
    return True, ""


KNOWN_OPS = frozenset({
    "section_present", "regex", "max_chars", "min_chars", "contains",
    "not_contains", "no_refusal", "tool_called",
})

# Ops that only constrain formatting. An optimizer satisfies these by editing
# the output template, which is why a judge made solely of them is gameable.
SHAPE_OPS = frozenset({"section_present", "max_chars", "min_chars"})


def is_shape_only(judge: Any) -> bool:
    """True when every check in ``judge`` merely constrains formatting.

    Such a judge cannot distinguish a better answer from a reformatted one, so
    callers should prefer outcome grading (a rubric) instead of trusting it.
    """
    if not isinstance(judge, dict):
        return False
    checks = judge.get("checks", []) or []
    if not isinstance(checks, list) or not checks:
        return False
    ops = [c.get("op") for c in checks if isinstance(c, dict)]
    if not ops or len(ops) != len(checks):
        return False
    return all(op in SHAPE_OPS for op in ops)


def char_bound(arg: Any) -> int:
    """Parse a ``max_chars``/``min_chars`` argument, or raise ``ValueError``.

    Shared by :func:`validate_checks` and the LLM miner so the accepted shapes
    cannot drift apart: a miner that emits a bound the validator later rejects
    produces a tasks file that fails its own validation. ``bool`` is refused
    (it is an ``int`` subclass, so ``int(True)`` would silently become 1) and a
    non-integral float is refused rather than truncated.
    """
    if isinstance(arg, bool):
        raise ValueError("bool is not a char bound")
    if isinstance(arg, int):
        return arg
    if isinstance(arg, float):
        if not arg.is_integer():
            raise ValueError("non-integral float is not a char bound")
        return int(arg)  # may raise OverflowError for inf/nan
    if isinstance(arg, str) and re.fullmatch(r"[+-]?\d+", arg.strip()):
        return int(arg.strip())
    raise ValueError("not a char bound")


def validate_checks(judge: Any) -> Tuple[List[str], List[str]]:
    """Return ``(errors, warnings)`` for a rule judge's checks.

    An *error* means the check can never behave as written — a regex that does
    not compile always scores 0.0, which is indistinguishable from a model that
    never complies. A *warning* means the check is accepted but toothless, e.g.
    an unknown op, which :func:`_check` deliberately lets pass.
    """
    errors: List[str] = []
    warnings: List[str] = []
    if judge is not None and not isinstance(judge, dict):
        return [f"judge must be an object, got {type(judge).__name__}"], warnings
    checks = (judge or {}).get("checks", []) or []
    if not isinstance(checks, list):
        return [f"judge 'checks' must be an array, got {type(checks).__name__}"], warnings
    for i, c in enumerate(checks):
        if not isinstance(c, dict):
            errors.append(f"check #{i} is not an object")
            continue
        op = c.get("op", "")
        arg = c.get("arg")
        if not isinstance(op, str):
            errors.append(
                f"check #{i} op must be a string, got {type(op).__name__}"
            )
            continue
        if op in {"regex", "section_present", "contains", "tool_called", "not_contains"} and (
            arg is None or not str(arg).strip()
        ):
            errors.append(f"check #{i} {op} needs a non-empty arg")
            continue
        if op == "regex":
            try:
                re.compile(str(arg))
            except re.error as exc:
                errors.append(f"check #{i} regex does not compile ({exc}): {arg!r}")
        elif op in {"max_chars", "min_chars"}:
            try:
                bound = char_bound(arg)
            except (OverflowError, TypeError, ValueError):
                errors.append(f"check #{i} {op} needs an integer arg, got {arg!r}")
            else:
                if bound < 0:
                    errors.append(f"check #{i} {op} cannot be negative, got {bound}")
                elif op == "min_chars" and bound == 0:
                    warnings.append(f"check #{i} min_chars=0 always passes")
        elif op not in KNOWN_OPS:
            warnings.append(f"check #{i} has unknown op {op!r} — it always passes")
    if not errors and is_shape_only(judge):
        warnings.append(
            "judge is shape-only (formatting checks); it can be satisfied by "
            "reformatting rather than by a better answer"
        )
    return errors, warnings


def score_rule_judge(
    judge: Dict[str, Any],
    response: str,
    tools_called: List[str] | None = None,
) -> Tuple[float, float, str]:
    """Return (hard, soft, rationale) for a gbrain-style rule judge."""
    checks = (judge or {}).get("checks", []) or []
    if not checks:
        return 0.0, 0.0, "no checks"
    # Providing tools_called means execution evidence is available, even when
    # the list is empty. Only callers that omit it may use the legacy marker.
    allow_tool_marker = tools_called is None
    tools_called = tools_called or []
    passed = 0
    failed_desc: List[str] = []
    for c in checks:
        ok, problem = _check(
            c.get("op", ""),
            c.get("arg"),
            response,
            tools_called,
            allow_tool_marker=allow_tool_marker,
        )
        if ok:
            passed += 1
        else:
            desc = f"{c.get('op')}={c.get('arg')}"
            if problem:
                desc += f" [{problem}]"
            failed_desc.append(desc)
    soft = passed / len(checks)
    hard = 1.0 if passed == len(checks) else 0.0
    rationale = "all checks passed" if hard else "failed: " + ", ".join(failed_desc)
    return hard, soft, rationale
