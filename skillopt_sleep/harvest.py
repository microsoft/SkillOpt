"""SkillOpt-Sleep — Stage 1: harvest.

Read the user's local Claude Code records (read-only) and normalize them
into :class:`SessionDigest` objects.

Sources (verified schema):
  * ~/.claude/history.jsonl        — one JSON/line:
        {"display": <prompt text>, "pastedContents": {...},
         "timestamp": <epoch ms>, "project": <abs path>}
  * ~/.claude/projects/<slug>/<sessionId>.jsonl — one record/line; the
    records we care about have type "user"/"assistant" and carry:
        message{role, content}, cwd, gitBranch, timestamp, sessionId, version

This module performs NO writes and NO network calls.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from skillopt_sleep.types import SessionDigest


# Heuristic phrases that signal the user (dis)approving of prior output.
# English-only by default. Users whose sessions are in another language can add
# their own phrases via the SKILLOPT_SLEEP_NEG_FEEDBACK / _POS_FEEDBACK env vars
# (comma-separated), so the capability is extensible without hardcoding locales.
_NEGATIVE_FEEDBACK = (
    "still broken", "still not", "still wrong", "doesn't work", "does not work",
    "not working", "that's wrong", "thats wrong", "incorrect", "wrong",
    "no,", "nope", "fix it", "didn't", "did not", "broken", "error again",
    "still failing", "still fails", "not fixed", "revert", "undo",
)
_POSITIVE_FEEDBACK = (
    "thanks", "thank you", "perfect", "great", "works now", "fixed",
    "that works", "lgtm", "looks good", "nice", "awesome", "correct",
)


def _extra_phrases(env_var: str) -> tuple:
    raw = os.environ.get(env_var, "")
    return tuple(p.strip().lower() for p in raw.split(",") if p.strip())


_NEGATIVE_FEEDBACK = _NEGATIVE_FEEDBACK + _extra_phrases("SKILLOPT_SLEEP_NEG_FEEDBACK")
_POSITIVE_FEEDBACK = _POSITIVE_FEEDBACK + _extra_phrases("SKILLOPT_SLEEP_POS_FEEDBACK")


def _iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue
    except (FileNotFoundError, IsADirectoryError, PermissionError):
        return


def _safe_mtime(path: str) -> float:
    """Return a sortable mtime without failing on a concurrently removed file."""
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def _text_from_content(content: Any) -> str:
    """Flatten a message.content (str or list of blocks) into text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for b in content:
            if isinstance(b, dict):
                if b.get("type") == "text" and b.get("text"):
                    parts.append(str(b["text"]))
        return "\n".join(parts)
    return ""


def _tool_names_from_content(content: Any) -> List[str]:
    names: List[str] = []
    if isinstance(content, list):
        for b in content:
            if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name"):
                names.append(str(b["name"]))
    return names


def _skill_names_from_content(content: Any) -> List[str]:
    """Extract skill targets from Claude ``Skill`` tool-use blocks.

    Only well-formed invocations count: the block type must be ``tool_use``,
    its name exactly ``Skill``, and ``input.skill`` a non-blank string. The
    name is returned whitespace-trimmed but otherwise verbatim; no other tool
    input and no tool output is read.
    """
    names: List[str] = []
    if not isinstance(content, list):
        return names
    for b in content:
        if not isinstance(b, dict):
            continue
        if b.get("type") != "tool_use" or b.get("name") != "Skill":
            continue
        args = b.get("input")
        if not isinstance(args, dict):
            continue
        skill = args.get("skill")
        if not isinstance(skill, str):
            continue
        skill = skill.strip()
        if skill:
            names.append(skill)
    return names


def _detect_feedback(text: str) -> List[str]:
    low = text.lower()
    sig: List[str] = []
    for ph in _NEGATIVE_FEEDBACK:
        if ph in low:
            sig.append("neg:" + ph)
    for ph in _POSITIVE_FEEDBACK:
        if ph in low:
            sig.append("pos:" + ph)
    return sig


# ── Issue #318: host wrappers around user text ─────────────────────────
#
# The Claude Code desktop app delivers a pasted block as the user message,
# wrapped in ``<pasted_content id="…">…</pasted_content id="…">``. The brief
# the user pasted is typed by them and is often the whole point of the
# session, so the wrapper is unwrapped instead of dropped — but only while it
# leads the message, because the same tag pair inside a prompt the user wrote
# is their content (#321 review: four literal-example cases).
_PASTED_CONTENT_LEAD_RE = re.compile(
    r"<pasted_content\b[^>]*>(.*?)</pasted_content[^>]*>",
    re.IGNORECASE | re.DOTALL,
)

# Host-injected status notes. A message that is only a reminder is host noise,
# and a reminder *prefixed* to a real request must not ride into the mined
# intent as text the user never wrote. The match is anchored by the caller:
# only the block that leads a message is removed.
_SYSTEM_REMINDER_RE = re.compile(
    r"<system-reminder\b[^>]*>.*?</system-reminder\s*>", re.IGNORECASE | re.DOTALL
)

# Tags the host injects as a whole user message: expanded slash-command
# bodies, local-command output, shell-mode input and editor selections. The
# list is closed deliberately — a wrapper this list does not know about is
# treated as user text, so such a message keeps its request instead of losing
# it. <system-reminder> is listed as well as stripped, so a variant carrying
# attributes is still recognised as host noise.
_HOST_META_TAGS = (
    "command-name", "command-message", "command-args",
    "local-command-stdout", "local-command-caveat",
    "system-reminder",
    "bash-input", "bash-stdout", "bash-stderr",
    "ide_opened_file", "ide_selection",
    # VS Code Copilot Chat brackets the machine context of every request
    # (<environment_info>…</environment_info>); harvesting it leaks local
    # environment detail and was never a user prompt.
    "environment_info",
)


def _strip_host_wrappers(text: str) -> str:
    """Unwrap a host's leading pasted block and drop its own leading notes.

    Both are positioned, not textual. A ``<pasted_content>`` wrapper is
    unwrapped only while it *leads* the message, which is how the host
    prepends it — and a ``<system-reminder>`` block only while it leads as
    well. Neither rule reaches inside the text the user wrote: a request that
    names the tag pair literally (an inline-code sample, a fenced XML
    example, a fixture with attributes) keeps its tags, so the harvested
    request and the mined intent stay what they typed. Stripping every
    occurrence changed both.
    """
    if "<" not in text:
        return text
    out = text
    while True:
        stripped = out.lstrip()
        pasted = _PASTED_CONTENT_LEAD_RE.match(stripped)
        if pasted is not None:
            out = pasted.group(1) + stripped[pasted.end():]
            continue
        reminder = _SYSTEM_REMINDER_RE.match(stripped)
        if reminder is not None:
            out = stripped[reminder.end():]
            continue
        return out


def _is_meta_prompt(text: str) -> bool:
    """Skip slash-commands / system noise that aren't real user intents."""
    t = _strip_host_wrappers(text).strip()
    if not t:
        return True
    if t.startswith("/") and len(t.split()) <= 3:
        return True
    if t.startswith("[Pasted text") or t.startswith("Caveat:"):
        return True
    # Expanded slash-command prompts (Claude Code injects the full command
    # body as a user message, wrapped in <command-message>/<command-name>
    # tags or rendered as "# /<name> — ..." headers). Not a user intent.
    if "<command-message>" in t[:200] or "<command-name>" in t[:200]:
        return True
    if t.startswith("# /"):
        return True
    # A message that is nothing but one host tag block (`<system-reminder>`,
    # `<local-command-stdout>`, …) carries no user intent. The tag list is
    # closed on purpose: text that merely arrives wrapped in angle brackets
    # is user content, and dropping it loses the session's real request
    # (issue #318: the desktop app delivers a pasted brief that way).
    if t.startswith("<") and t.endswith(">"):
        head = t[1:].split(">", 1)[0].split(None, 1)
        return bool(head) and head[0].lower() in _HOST_META_TAGS
    return False


# ── Issue #62: filter headless replay sessions ─────────────────────────

# Prompt markers generated by the engine's own headless `claude -p` calls
# (judge, reflect, attempt). If the sole user prompt in a single-turn
# session matches any of these, the session is engine-generated, not a
# real user task.
_REPLAY_PROMPT_MARKERS = (
    "## CURRENT SKILL",
    "## FAILED TASKS",
    "## SUCCESSFUL TASKS",
    "## OUTPUT FORMAT",
    "You are a strict grader",
    "Score the response 0.0-1.0",
    "You are SkillOpt-Sleep",
    "You are an expert success-pattern analyst",
    "You are an expert failure-pattern analyst",
    "## TASK\n",
    "## SKILL\n",
    # Engine rollouts render the skill under this heading before the task.
    "## Skill\n",
)


# Sessions written by OTHER tools' sub-agents (memory observers, critic
# sub-agents, plugin self-invocations). These are multi-turn, so the
# single-turn heuristic in _is_headless_replay never catches them. If the
# FIRST user prompt matches any marker, the whole session is
# machine-generated, not a real user task.
_AGENT_SESSION_MARKERS = (
    "You are a Claude-Mem",              # claude-mem observer agent
    "Hello memory agent",                # claude-mem observer continuation
    "You are driving **SkillOpt-Sleep**",  # this plugin's own command body
) + tuple(
    # users can add markers for their own tools' agent prompts
    p.strip()
    for p in os.environ.get("SKILLOPT_SLEEP_AGENT_MARKERS", "").split(",")
    if p.strip()
)


def _is_agent_session(digest: "SessionDigest") -> bool:
    """Detect transcripts written by other tools' sub-agents (see markers)."""
    if not digest.user_prompts:
        return False
    first = digest.user_prompts[0]
    return any(marker in first for marker in _AGENT_SESSION_MARKERS)


def _is_headless_replay(digest: "SessionDigest") -> bool:
    """Detect sessions created by the engine's own headless replay calls.

    Heuristics (conservatively applied):
    1. Session has exactly 1 user turn AND
    2. The sole prompt matches engine-generated patterns (grader/reflect),
       OR the session lasted < 3 seconds (programmatic, not interactive).
    Multi-turn sessions are always kept (interactive by definition).
    """
    if digest.n_user_turns > 1:
        return False
    if digest.n_user_turns == 0:
        # Every user turn was filtered as host noise or engine input, so there
        # is no user intent to mine. Such a record must not be admitted: it
        # would take a slot under --max-sessions and yield no task, displacing
        # a real session (review of #318). A pasted brief is not this case —
        # it now survives the filter as a user turn.
        return True
    prompt = digest.user_prompts[0] if digest.user_prompts else ""
    for marker in _REPLAY_PROMPT_MARKERS:
        if marker in prompt:
            return True
    # Sub-3-second single-turn sessions with short prompts are almost
    # certainly programmatic (engine grader/judge calls).  We require the
    # prompt to also be short (<200 chars) to avoid false-positives on
    # real one-shot questions that Claude happens to answer quickly.
    if digest.started_at and digest.ended_at and len(prompt) < 200:
        try:
            fmt = "%Y-%m-%dT%H:%M:%S"
            start = datetime.strptime(digest.started_at[:19], fmt)
            end = datetime.strptime(digest.ended_at[:19], fmt)
            if (end - start).total_seconds() < 3:
                return True
        except (ValueError, TypeError):
            pass
    return False


def digest_transcript(path: str) -> Optional[SessionDigest]:
    """Build a SessionDigest from one ``<sessionId>.jsonl`` transcript."""
    session_id = os.path.splitext(os.path.basename(path))[0]
    project = ""
    git_branch = ""
    started = ""
    ended = ""
    user_prompts: List[str] = []
    assistant_finals: List[str] = []
    tools: List[str] = []
    skills: List[str] = []
    files: List[str] = []
    feedback: List[str] = []
    n_user = 0
    n_asst = 0

    for rec in _iter_jsonl(path):
        rtype = rec.get("type")
        ts = rec.get("timestamp")
        if isinstance(ts, str) and ts:
            if not started:
                started = ts
            ended = ts
        if rec.get("cwd") and not project:
            project = str(rec.get("cwd"))
        if rec.get("gitBranch") and not git_branch:
            git_branch = str(rec.get("gitBranch"))
        if rtype == "file-history-snapshot":
            snap = rec.get("snapshot") or rec.get("files") or {}
            if isinstance(snap, dict):
                files.extend([str(k) for k in list(snap.keys())[:20]])
        msg = rec.get("message")
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if role == "user":
            # Claude Code marks text it injects on the user's behalf with
            # isMeta: a loaded skill's SKILL.md body, messages relayed from
            # other sessions, usage-limit notices. The user typed none of it.
            # A body that names a known agent session is still kept, so
            # _is_agent_session can drop the whole session as before.
            # Host wrappers are removed before the meta test: a pasted brief
            # (#318) is user content, and a <system-reminder> prefix must not
            # ride into the intent (see _strip_host_wrappers).
            text = _strip_host_wrappers(_text_from_content(content))
            if rec.get("isMeta") is True and not any(
                marker in text for marker in _AGENT_SESSION_MARKERS
            ):
                continue
            if text and not _is_meta_prompt(text):
                n_user += 1
                user_prompts.append(text.strip())
                feedback.extend(_detect_feedback(text))
        elif role == "assistant":
            n_asst += 1
            tools.extend(_tool_names_from_content(content))
            skills.extend(_skill_names_from_content(content))
            text = _text_from_content(content)
            if text.strip():
                assistant_finals.append(text.strip())

    if n_user == 0 and n_asst == 0:
        return None

    # de-dup tools/files preserving order
    def _dedup(xs: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in xs:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    return SessionDigest(
        session_id=session_id,
        project=project,
        git_branch=git_branch,
        started_at=started,
        ended_at=ended,
        user_prompts=user_prompts,
        assistant_finals=assistant_finals[-5:],  # last few finals are the useful ones
        tools_used=_dedup(tools),
        skills_used=_dedup(skills),
        files_touched=_dedup(files),
        feedback_signals=feedback,
        n_user_turns=n_user,
        n_assistant_turns=n_asst,
        raw_path=path,
    )


def _git_root(path: str) -> str:
    """Nearest ancestor of `path` holding a .git entry, else "".

    A .git *file* (worktree/submodule) counts too, so `os.path.exists` is used
    rather than isdir.
    """
    cur = os.path.abspath(path)
    while True:
        if os.path.exists(os.path.join(cur, ".git")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            return ""
        cur = parent


def _ancestor_in_scope(ancestor: str, invoked: str) -> bool:
    """True when a session rooted at `ancestor` may belong to `invoked`.

    A session started higher up than the invoked project is in scope only when
    it is still inside the invoked project's git root, so invoking from
    `repo/sub` keeps the sessions started at `repo`. $HOME and the filesystem
    root are never in scope: they are shared by every project, so admitting
    them is what previously pulled every $HOME-rooted session into every
    project below it. Without a git root the ancestor walk stops there.
    """
    home = os.path.abspath(os.path.expanduser("~"))
    if ancestor in (home, os.path.abspath(os.sep)):
        return False
    root = _git_root(invoked)
    if root:
        return ancestor == root or ancestor.startswith(root + os.sep)
    return True


def _project_matches(project: str, scope: Any, invoked: str) -> bool:
    if scope == "all":
        return True
    if isinstance(scope, (list, tuple)):
        return any(os.path.abspath(project) == os.path.abspath(p) for p in scope)
    # "invoked": match the invoked project (or a subdir of it)
    if not invoked:
        return True
    a = os.path.abspath(project)
    b = os.path.abspath(invoked)
    if a == b or a.startswith(b + os.sep):
        return True
    if not b.startswith(a + os.sep):
        return False
    return _ancestor_in_scope(a, b)


def harvest(
    transcripts_dir: str,
    *,
    scope: Any = "all",
    invoked_project: str = "",
    since_iso: Optional[str] = None,
    limit: int = 0,
) -> List[SessionDigest]:
    """Walk ~/.claude/projects and return digests matching scope/time.

    Parameters
    ----------
    transcripts_dir : str    ~/.claude/projects
    scope : "all" | "invoked" | list[path]
    invoked_project : str    used when scope == "invoked"
    since_iso : str|None      ISO8601; only sessions ending after this are kept
    limit : int               cap number of digests (0 = no cap)
    """
    digests: List[SessionDigest] = []
    if not os.path.isdir(transcripts_dir):
        return digests

    paths: List[str] = []
    for root, _dirs, files in os.walk(transcripts_dir):
        # Sub-agent sidechain transcripts (<session>/subagents/agent-*.jsonl)
        # are Claude-authored prompts, not user tasks — never harvest them.
        if os.path.basename(root) == "subagents":
            continue
        for fn in files:
            if fn.endswith(".jsonl") and not fn.startswith("agent-"):
                paths.append(os.path.join(root, fn))
    # newest first by mtime
    paths.sort(key=_safe_mtime, reverse=True)

    for p in paths:
        d = digest_transcript(p)
        if d is None:
            continue
        if _is_headless_replay(d):
            continue  # Issue #62: skip engine's own headless replay sessions
        if _is_agent_session(d):
            continue  # skip other tools' sub-agent transcripts (claude-mem etc.)
        if not _project_matches(d.project or "", scope, invoked_project):
            continue
        if since_iso and d.ended_at and d.ended_at < since_iso:
            # Note: files are sorted by mtime but we compare the embedded
            # ended_at timestamp — mtime can diverge (copy/touch), so we
            # cannot break here; we must continue to check all files.
            continue
        digests.append(d)
        if limit and len(digests) >= limit:
            break
    return digests
