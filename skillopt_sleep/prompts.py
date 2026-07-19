"""SkillOpt-Sleep — central prompt registry with live user overrides.

Every LLM-facing prompt template the sleep cycle uses (miner / attempt /
judge / reflect) lives here, in one place, instead of being scattered as
inline literals. Two consequences:

  1. **Auditability** — the dashboard (and any human) can display exactly
     what instructions each agent role receives, per stage.
  2. **Live tuning** — a user override file (``prompts.json`` in the state
     dir, or ``SKILLOPT_SLEEP_PROMPTS_PATH``) replaces any template without
     touching code. The file's mtime is checked on every read, so an edit
     made while a cycle is running takes effect on the very next call.

Placeholders use the ``__NAME__`` convention (simple ``str.replace``, no
``str.format``) because the templates themselves contain JSON braces.

The default texts are byte-for-byte the prompts previously inlined in
``backend.py`` / ``llm_miner.py``, so behavior is unchanged unless the user
overrides a template.
"""
from __future__ import annotations

import json
import os
import threading
from typing import Dict, List, Optional

from skillopt_sleep.config import HOME_STATE_DIR

# ── default templates ─────────────────────────────────────────────────────────

_MINER = """You are mining a user's past AI-assistant sessions to find RECURRING tasks
worth optimizing a skill for. From the session below, extract 0-3 reusable tasks.

A good task is something the user asks for repeatedly or had to correct, where a
GENERAL rule would help next time (formatting, structure, tool-use, conventions).
Skip one-off or purely exploratory requests.

For each task return:
  - "intent": the reusable request, generalized (no one-off specifics)
  - "checks": a list of programmatic success checks a grader can run on a future
     answer. Each check is one of:
        {"op":"section_present","arg":"<heading text>"}
        {"op":"regex","arg":"<python regex the answer must match>"}
        {"op":"contains","arg":"<substring the answer must contain>"}
        {"op":"max_chars","arg":<int>}
     Only include checks you are confident a GOOD answer must satisfy.
  - "rubric": a one-sentence description of what a good answer looks like
  - "satisfied": true/false — did the user seem satisfied with the assistant's answer?

Return ONLY a JSON array (possibly empty). No prose.

# Session
project: __PROJECT__
user prompts:
__PROMPTS__
assistant final (last):
__FINAL__
feedback signals: __FEEDBACK__
"""

_ATTEMPT = (
    "Complete the following task for the user. Follow the skill and memory "
    "guidance below, including any output-format and length requirements. "
    "When a 'Learned preferences' rule sets an explicit limit (e.g. a length "
    "cap), prefer that rule over more general advice it refines.\n\n"
    "# Skill\n__SKILL__\n\n# Memory\n__MEMORY__\n\n"
    "# Task\n__INTENT__\n\n__CONTEXT__\n\n"
    "Return ONLY the final answer text, nothing else."
)

_JUDGE = (
    "Score how well the response satisfies the rubric, 0..1. "
    'Return ONLY JSON {"score": <0..1>, "reason": "..."}.\n\n'
    "# Rubric\n__RUBRIC__\n\n# Response\n__RESPONSE__"
)

_REFLECT = (
    "You are SkillOpt's optimizer. The agent keeps failing the recurring "
    "tasks below. Propose at most __EDIT_BUDGET__ bounded edits to the "
    "__TARGET__ document so it stops failing. Each edit MUST be a short, "
    "GENERAL, reusable rule or preference (never task-specific, never an "
    "answer to a single task). If exact failing criteria are listed, your "
    "edits MUST make future outputs satisfy every one of them.\n"
    "BE CONCRETE: quote the exact threshold, section name, or format from "
    "the criteria verbatim in your rule (e.g. write 'keep the entire "
    "response under 1200 characters', NOT 'respect length limits'). Vague "
    "rules do not change behavior; specific numeric/structural rules do.\n"
    "IMPORTANT: your edits are APPENDED to a 'Learned preferences' block; "
    "you CANNOT delete the existing instructions above. If the current "
    "__TARGET__ text conflicts with a criterion (e.g. it says 'be exhaustive' "
    "but outputs must be under a character limit), write an explicit, "
    "forceful OVERRIDE rule stating it supersedes the conflicting "
    "instruction, and put the hard requirement first.\n"
    "HARD CONSTRAINT: every rule you write MUST be consistent with the "
    "'Task output contract' below (if shown). NEVER propose a rule that "
    "changes the required output format/language, tells the agent to ask "
    "the user a question, or otherwise violates that contract — such a "
    "rule scores ZERO because the evaluator cannot honor it.\n"
    'Return ONLY a JSON array: '
    '[{"op":"add|replace|delete","content":"<rule>","anchor":"<text to replace/delete, optional>","rationale":"<why>"}].\n\n'
    "# Current __TARGET__\n__CUR_DOC__\n"
    "__GUARD__"
    "__CRITERIA__\n"
    "__PREFS__\n\n"
    "# Recurring failures\n__FAILURES__"
)

# name -> {text, stage, role, description, placeholders}
DEFAULTS: Dict[str, Dict] = {
    "miner": {
        "text": _MINER,
        "stage": "mine",
        "role": "optimizer",
        "description": "Turns one harvested session digest into 0-3 checkable TaskRecords.",
        "placeholders": ["__PROJECT__", "__PROMPTS__", "__FINAL__", "__FEEDBACK__"],
    },
    "attempt": {
        "text": _ATTEMPT,
        "stage": "replay",
        "role": "target",
        "description": "The clean-context rollout: solve a mined task given only skill+memory.",
        "placeholders": ["__SKILL__", "__MEMORY__", "__INTENT__", "__CONTEXT__"],
    },
    "judge": {
        "text": _JUDGE,
        "stage": "replay",
        "role": "optimizer",
        "description": "Rubric grading for tasks with no programmatic checks (0..1 JSON score).",
        "placeholders": ["__RUBRIC__", "__RESPONSE__"],
    },
    "reflect": {
        "text": _REFLECT,
        "stage": "reflect",
        "role": "optimizer",
        "description": "Proposes bounded skill/memory edits from the recurring failures.",
        "placeholders": [
            "__EDIT_BUDGET__", "__TARGET__", "__CUR_DOC__", "__GUARD__",
            "__CRITERIA__", "__PREFS__", "__FAILURES__",
        ],
    },
}


# ── override file (mtime-cached; edits take effect on the next call) ──────────

_lock = threading.Lock()
_cache: Dict[str, object] = {"path": None, "mtime": None, "data": {}}


def overrides_path() -> str:
    return os.environ.get("SKILLOPT_SLEEP_PROMPTS_PATH", "") or os.path.join(
        HOME_STATE_DIR, "prompts.json"
    )


def load_overrides() -> Dict[str, str]:
    """Return {name: replacement_text}, re-reading the file iff it changed."""
    path = overrides_path()
    with _lock:
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            _cache.update(path=path, mtime=None, data={})
            return {}
        if _cache["path"] == path and _cache["mtime"] == mtime:
            return dict(_cache["data"])  # type: ignore[arg-type]
        try:
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
            data = {
                k: v for k, v in raw.items()
                if k in DEFAULTS and isinstance(v, str) and v.strip()
            }
        except Exception:
            data = {}
        _cache.update(path=path, mtime=mtime, data=data)
        return dict(data)


def save_overrides(overrides: Dict[str, Optional[str]]) -> Dict[str, str]:
    """Merge ``overrides`` into the override file. A None/empty value removes
    that override (reverting the template to its default). Returns the new
    effective override map."""
    path = overrides_path()
    current = load_overrides()
    for k, v in overrides.items():
        if k not in DEFAULTS:
            continue
        if v is None or not str(v).strip():
            current.pop(k, None)
        else:
            current[k] = str(v)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(current, f, ensure_ascii=False, indent=2)
    with _lock:
        _cache.update(path=None, mtime=None, data={})  # force re-read
    return current


def get_prompt(name: str) -> str:
    """Effective template text for ``name`` (override if present, else default)."""
    ov = load_overrides()
    if name in ov:
        return ov[name]
    return DEFAULTS[name]["text"]


def is_overridden(name: str) -> bool:
    return name in load_overrides()


def render(name: str, mapping: Dict[str, str]) -> str:
    """Substitute ``__NAME__`` placeholders via str.replace (format-safe)."""
    text = get_prompt(name)
    for k, v in mapping.items():
        text = text.replace(k, v)
    return text


def describe() -> List[Dict]:
    """Registry snapshot for the dashboard: defaults + active overrides."""
    ov = load_overrides()
    out = []
    for name, meta in DEFAULTS.items():
        out.append({
            "name": name,
            "stage": meta["stage"],
            "role": meta["role"],
            "description": meta["description"],
            "placeholders": meta["placeholders"],
            "default": meta["text"],
            "override": ov.get(name),
            "effective": ov.get(name) or meta["text"],
        })
    return out
