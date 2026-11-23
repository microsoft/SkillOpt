"""Parse the ALFWorld action from a model response.

The rollout records this action and the environment executes it, so both use
this one function.
"""
from __future__ import annotations

import re

# The body may not contain another "<action>", so a tag mentioned in the
# reasoning before the real action is not captured (as in the SearchQA evaluator).
_ACTION_RE = re.compile(r"<action>((?:(?!<action>).)*?)</action>", re.DOTALL | re.IGNORECASE)


def extract_action(text: str) -> str | None:
    """Return the last ``<action>...</action>`` body, stripped, or ``None``."""
    matches = _ACTION_RE.findall(text or "")
    if not matches:
        return None
    return matches[-1].strip()
