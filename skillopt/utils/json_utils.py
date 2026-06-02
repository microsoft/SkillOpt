"""JSON extraction helpers for LLM responses."""
from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Any

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_DECODER = json.JSONDecoder()


def _candidate_blocks(text: str) -> Iterator[str]:
    """Yield fenced blocks before scanning the full text."""
    for match in _JSON_FENCE_RE.finditer(text):
        yield match.group(1).strip()
    yield text


def _raw_decode_at_offsets(text: str, start_chars: str) -> Iterator[Any]:
    """Decode the first valid JSON value that starts at any requested char."""
    for idx, char in enumerate(text):
        if char not in start_chars:
            continue
        try:
            value, _end = _DECODER.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        yield value


def _extract_typed_json(text: str, expected_type: type, start_chars: str) -> Any | None:
    if not isinstance(text, str) or not text:
        return None
    for block in _candidate_blocks(text):
        for value in _raw_decode_at_offsets(block, start_chars):
            if isinstance(value, expected_type):
                return value
    return None


def extract_json(text: str) -> dict | None:
    """Extract the first JSON object from LLM response text.

    Fenced JSON blocks are preferred. Bare text is scanned with
    ``JSONDecoder.raw_decode`` instead of greedy regexes, so multiple adjacent
    objects, nested objects, and arrays next to prose are handled safely.
    """
    return _extract_typed_json(text, dict, "{")


def extract_json_array(text: str) -> list | None:
    """Extract the first JSON array from LLM response text."""
    return _extract_typed_json(text, list, "[")
