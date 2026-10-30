"""SkillOpt-Sleep Kimi Code session harvesting.

Reads Kimi Code session wires (``~/.kimi-code/sessions/wd_<workspace>/session_<id>/agents/main/wire.jsonl``)
and normalizes them into ``SessionDigest`` records without copying tool
arguments or raw tool outputs.

Wire format notes (protocol >= 1.5):
  * User and assistant messages appear under two event envelopes:
      - ``context.append_message`` — message at the event top level
      - ``agent.message.appended`` — message nested at ``message.message``,
        with origin metadata on the wrapper's ``meta.origin``
  * Genuine user input has ``role == "user"`` and ``origin.kind == "user"``;
    system reminders and injections carry other origin kinds.
  * Assistant content blocks include ``think`` blocks; only ``text`` blocks
    are harvested.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from skillopt_sleep.harvest import (
    _detect_feedback,
    _is_agent_session,
    _is_meta_prompt,
    _project_matches,
    _safe_mtime,
)
from skillopt_sleep.staging import _SECRET_PATTERNS
from skillopt_sleep.types import SessionDigest

# Feedback phrases for Chinese-language sessions. The shared detector in
# harvest.py is English-only; Kimi Code sessions are predominantly Chinese.
_POS_FEEDBACK_ZH = ("谢谢", "感谢", "可以", "对了", "完美", "好用", "没问题", "搞定", "成了")
_NEG_FEEDBACK_ZH = ("不对", "错了", "不行", "重来", "搞错", "不是这个", "回退", "撤销", "还是不行")

_SYSTEM_REMINDER_RE = re.compile(r"^\s*<system-reminder>.*?</system-reminder>\s*$", re.S)


def _detect_feedback_zh(text: str) -> List[str]:
    sig: List[str] = []
    for ph in _NEG_FEEDBACK_ZH:
        if ph in text:
            sig.append("neg:" + ph)
    for ph in _POS_FEEDBACK_ZH:
        if ph in text:
            sig.append("pos:" + ph)
    return sig


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text" and b.get("text"):
                parts.append(str(b["text"]))
        return "\n".join(parts)
    return ""


def _tool_names(tool_calls: Any) -> Tuple[List[str], List[str]]:
    """Return (tool_names, skill_names) from a wire message's toolCalls."""
    tools: List[str] = []
    skills: List[str] = []
    if not isinstance(tool_calls, list):
        return tools, skills
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        name = call.get("name")
        if isinstance(name, str) and name:
            tools.append(re.sub(r"[^A-Za-z0-9_.:-]+", "_", name)[:80])
            if name == "Skill":
                args = call.get("args") or call.get("input") or {}
                if isinstance(args, dict):
                    skill = args.get("skill") or args.get("skill_name")
                    if isinstance(skill, str) and skill.strip():
                        skills.append(skill.strip())
    return tools, skills


def _ms_to_iso(ms: Any) -> str:
    if not isinstance(ms, (int, float)) or ms <= 0:
        return ""
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _sanitize(text: str) -> str:
    out = text.replace("\x00", "").strip()
    if not out:
        return ""
    for pattern, replacement in _SECRET_PATTERNS:
        out = pattern.sub(replacement, out)
    return out


def _workspace_slug(session_dir: str) -> str:
    """`.../wd_358_7051f95cba3a/session_x` -> `358`."""
    wd = os.path.basename(os.path.dirname(session_dir))
    m = re.match(r"wd_(.+)_[0-9a-f]{12}$", wd)
    return m.group(1) if m else wd


def _kimi_project_matches(slug: str, scope: Any, invoked: str) -> bool:
    """Match a Kimi workspace slug against the scope/invoked project.

    Kimi wires do not record the session cwd, so matching is name-based: the
    workspace slug (e.g. ``358``, ``pte``) is compared against the path
    segments of the invoked project. ``scope == "all"`` admits everything;
    a list scope matches any entry the same way.
    """
    if scope == "all":
        return True
    invoked_list = list(scope) if isinstance(scope, (list, tuple)) else [invoked]
    slug_l = slug.lower()
    for inv in invoked_list:
        if not inv:
            continue
        # defensive: a real abs-path project (not a workspace slug) uses the
        # shared path matcher; a bare slug must never be abspath'ed against cwd
        if os.path.isabs(slug) and _project_matches(slug, "invoked", str(inv)):
            return True
        for seg in re.split(r"[\\/]+", str(inv)):
            seg = re.sub(r"[^A-Za-z0-9]+", "", seg).lower()
            if len(seg) >= 3 and seg in slug_l:
                return True
        # invoked basename with non-ascii stripped can still match the slug
        base = re.sub(r"[^A-Za-z0-9]+", "", os.path.basename(str(inv))).lower()
        if len(base) >= 3 and base in slug_l:
            return True
    return False


def digest_kimi_session(path: str, project_hint: str = "") -> Optional[SessionDigest]:
    """Build a ``SessionDigest`` from one Kimi Code ``wire.jsonl``."""
    session_dir = os.path.dirname(os.path.dirname(os.path.dirname(path)))
    session_id = os.path.basename(session_dir)
    slug = _workspace_slug(session_dir)

    started = ""
    ended = ""
    user_prompts: List[str] = []
    assistant_finals: List[str] = []
    tools: List[str] = []
    skills: List[str] = []
    feedback: List[str] = []
    n_user = 0
    n_asst = 0

    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                if not isinstance(e, dict):
                    continue
                etype = e.get("type")
                if etype not in ("context.append_message", "agent.message.appended"):
                    continue
                ts = e.get("time") or e.get("created_at")
                iso = _ms_to_iso(ts)
                if iso:
                    if not started:
                        started = iso
                    ended = iso

                wrapper = e.get("message") or {}
                if etype == "agent.message.appended":
                    meta_origin = (wrapper.get("meta") or {}).get("origin")
                    msg = wrapper.get("message") or {}
                else:
                    meta_origin = None
                    msg = wrapper
                role = msg.get("role")
                origin_kind = (msg.get("origin") or meta_origin or {}).get("kind")

                call_tools, call_skills = _tool_names(msg.get("toolCalls"))
                tools.extend(call_tools)
                skills.extend(call_skills)

                text = _sanitize(_text_from_content(msg.get("content")))
                if not text:
                    continue
                if role == "user":
                    # only genuine user input; reminders/injections are skipped
                    if origin_kind != "user":
                        continue
                    if _SYSTEM_REMINDER_RE.match(text):
                        continue
                    if _is_meta_prompt(text):
                        continue
                    n_user += 1
                    user_prompts.append(text)
                    feedback.extend(_detect_feedback(text))
                    feedback.extend(_detect_feedback_zh(text))
                elif role == "assistant":
                    n_asst += 1
                    assistant_finals.append(text)
    except OSError:
        return None

    if project_hint and not _kimi_project_matches(slug, "invoked", project_hint):
        return None
    if n_user == 0 and n_asst == 0:
        return None

    return SessionDigest(
        session_id=session_id,
        project=slug,
        started_at=started,
        ended_at=ended,
        user_prompts=user_prompts,
        assistant_finals=assistant_finals[-5:],
        tools_used=list(dict.fromkeys(tools)),
        skills_used=list(dict.fromkeys(skills)),
        files_touched=[],
        feedback_signals=feedback,
        n_user_turns=n_user,
        n_assistant_turns=n_asst,
        raw_path=path,
    )


# Sentinels of the engine's own replay sessions. The kimicode backend spawns
# `kimi --prompt` per attempt; those CLI runs persist wires into the session
# store, and without this filter the next night's harvest eats them (#noise).
_REPLAY_MARKERS = (
    "Complete the following task for the user. Follow the skill and memory",
    "[skillopt-sleep:",
)


def _is_kimi_replay(digest: SessionDigest) -> bool:
    if not digest.user_prompts:
        return True  # no user intent at all (programmatic session)
    first = digest.user_prompts[0]
    return any(m in first for m in _REPLAY_MARKERS)


def harvest_kimi(
    sessions_dir: str,
    *,
    scope: Any = "all",
    invoked_project: str = "",
    since_iso: Optional[str] = None,
    limit: int = 0,
) -> List[SessionDigest]:
    """Walk ``~/.kimi-code/sessions`` and return matching digests."""
    digests: List[SessionDigest] = []
    if not os.path.isdir(sessions_dir):
        return digests

    paths: List[str] = []
    for root, _dirs, files in os.walk(sessions_dir):
        # only the main agent transcript; sub-agent wires are not user tasks
        if os.path.basename(root) != "main":
            continue
        if "wire.jsonl" in files:
            paths.append(os.path.join(root, "wire.jsonl"))
    paths.sort(key=_safe_mtime, reverse=True)

    project_hint = invoked_project if scope == "invoked" else ""
    for path in paths:
        digest = digest_kimi_session(path, project_hint=project_hint)
        if digest is None:
            continue
        if not _kimi_project_matches(digest.project or "", scope, invoked_project):
            continue
        if _is_agent_session(digest) or _is_kimi_replay(digest):
            continue
        if since_iso and digest.ended_at and digest.ended_at < since_iso:
            continue
        digests.append(digest)
        if limit and len(digests) >= limit:
            break
    return digests
