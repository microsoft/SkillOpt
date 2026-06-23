"""JSON extraction helpers for LLM responses."""
from __future__ import annotations

import json
import re
import warnings


def _top_level_brace_objects(text: str) -> list[str]:
    """Return every balanced *top-level* ``{...}`` span in ``text``.

    Fully string/escape aware: braces inside quoted strings are ignored both
    when scanning for an object start AND while tracking depth inside one, so a
    ``{`` that appears in prose (e.g. ``'set it to {x}'``) is never mistaken for
    the start of a JSON object. Used to detect ambiguity: when a response carries
    more than one top-level object we must not let a repair pass silently pick
    one — it may pick the wrong (discarded) edit, strictly worse than None.
    """
    spans: list[str] = []
    i, n = 0, len(text)
    outer_in_str = False
    outer_esc = False
    while i < n:
        ch = text[i]
        # Skip over braces that live *inside* a quoted string before any object
        # has started — otherwise a `{` in prose like '"set it to {x}"' is wrongly
        # treated as an object start, and the repair pass below turns non-JSON
        # prose into a bogus dict (strictly worse than returning None).
        if outer_in_str:
            if outer_esc:
                outer_esc = False
            elif ch == "\\":
                outer_esc = True
            elif ch == '"':
                outer_in_str = False
            i += 1
            continue
        if ch == '"':
            outer_in_str = True
            i += 1
            continue
        if ch != "{":
            i += 1
            continue
        depth = 0
        in_str = False
        esc = False
        start = i
        while i < n:
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    spans.append(text[start:i + 1])
                    i += 1
                    break
            i += 1
        else:
            break  # unterminated final object
    return spans


def _looks_json_like(span: str) -> bool:
    """Heuristic: does ``span`` look like an intended JSON object (vs. prose)?

    A genuine JSON object's first non-space character after ``{`` is either ``"``
    (a string key) or ``}`` (an empty object). Prose pseudo-objects that the
    repair pass would otherwise fabricate into bogus dicts — ``{op: delete}``,
    ``{x: 1}`` quoted in single quotes or backticks, etc. — start with a bare
    word and are rejected. This complements the string-aware scan, which only
    skips *double*-quoted prose; single-quoted / backticked / unquoted prose
    braces are caught here instead. Legitimate repair targets (trailing commas,
    unescaped quotes inside string values) all begin with ``"`` and pass.
    """
    inner = span.strip()
    if not (inner.startswith("{") and inner.endswith("}")):
        return False
    after_brace = inner[1:].lstrip()
    return after_brace[:1] in ('"', '}')


def extract_json(text: str) -> dict | None:
    """Extract a JSON object from LLM response text.

    Tries ```json fences first, then bare {...} patterns.
    """
    m = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    # Tolerant fallback for non-OpenAI backends (Claude/Qwen, …) whose free-form
    # JSON strict json.loads rejects — unescaped ASCII quotes inside CJK string
    # values, trailing commas, etc. Repair so the analyst's edits aren't silently
    # dropped, but ONLY a single unambiguous object: never feed the greedy `{.*}`
    # span or the raw text, or json_repair would quietly return one of several
    # objects (empirically the wrong/last one) — strictly worse than None, which
    # the caller can detect and retry/skip.
    #
    # Pick the candidate FIRST, before importing json_repair, so the optional
    # dependency only matters (and only warns) when there is genuinely a single
    # malformed object we could have repaired. Ordinary no-JSON / prose replies
    # have no candidate and return None silently.
    candidate = None
    fenced = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
    if fenced and len(_top_level_brace_objects(fenced.group(1))) == 1:
        candidate = fenced.group(1)
    else:
        objs = _top_level_brace_objects(text)
        if len(objs) == 1:
            candidate = objs[0]
        # 0 or >1 top-level objects → too ambiguous to repair safely → None
    if not candidate:
        return None
    # Final guard: only repair spans that actually look like an intended JSON
    # object. Prose pseudo-objects in single quotes / backticks / bare text
    # (e.g. `{op: delete}`) reach here because the scan only skips double-quoted
    # prose; repairing them would fabricate a wrong dict (worse than None).
    if not _looks_json_like(candidate):
        return None
    try:
        from json_repair import repair_json
    except ModuleNotFoundError:
        warnings.warn(
            "json_repair not installed; malformed-JSON recovery disabled — "
            "a non-OpenAI analyst edit may be silently dropped. pip install json_repair",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    try:
        repaired = repair_json(candidate, return_objects=True)
        if isinstance(repaired, dict) and repaired:
            return repaired
    except Exception:  # noqa: BLE001 — repair is best-effort
        pass
    return None


def extract_json_array(text: str) -> list | None:
    """Extract a JSON array from LLM response text."""
    m = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None
