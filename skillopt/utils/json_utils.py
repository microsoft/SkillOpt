"""JSON extraction helpers for LLM responses."""
from __future__ import annotations

import json
import re
import warnings


def _top_level_brace_objects(text: str) -> list[str]:
    """Return every balanced *top-level* ``{...}`` span in ``text``.

    String/escape aware, so braces inside string values are not miscounted.
    Used to detect ambiguity: when a response carries more than one top-level
    object we must not let a repair pass silently pick one — it may pick the
    wrong (discarded) edit, which is strictly worse than returning None.
    """
    spans: list[str] = []
    i, n = 0, len(text)
    while i < n:
        if text[i] != "{":
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
    try:
        from json_repair import repair_json
    except ModuleNotFoundError:
        warnings.warn(
            "json_repair not installed; malformed-JSON recovery disabled — "
            "non-OpenAI analyst edits may be silently dropped. pip install json_repair",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    candidate = None
    fenced = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
    if fenced and len(_top_level_brace_objects(fenced.group(1))) == 1:
        candidate = fenced.group(1)
    else:
        objs = _top_level_brace_objects(text)
        if len(objs) == 1:
            candidate = objs[0]
        # 0 or >1 top-level objects → too ambiguous to repair safely → None
    if candidate:
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
