"""JSON extraction helpers for LLM responses."""
from __future__ import annotations

import json
import re


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
