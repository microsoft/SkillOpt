#!/usr/bin/env python3
"""Convert Devin IDE local data into Claude Code-format JSONL transcripts.

Devin (Cognition) does not persist agent conversation transcripts to disk in a
format the sleep engine understands.  This script bridges that gap by synthesising
JSONL files from every locally available source:

  1. **Devin transcripts** (~/.local/share/devin/cli/transcripts/*.json)
     Native ATIF-v1.7 format — source:"user" / source:"agent" messages
     converted directly to user/assistant JSONL turns.

  2. **agentmemory** (~/.agentmemory/standalone.json)
     Memories saved by the `agentmemory` MCP server — each memory's title
     becomes a synthetic user prompt; its content becomes the assistant reply.

  3. **Skill files** (.devin/skills/*/SKILL.md)
     Each skill description is converted to a session where the user asked
     "use the <skill> skill" and the assistant described how to apply it.

Output layout (mirrors ~/.claude/projects/<slug>/<sessionId>.jsonl):
    <out_dir>/projects/<slug>/<session_id>.jsonl

Workspace auto-detection order:
  1. ``SKILLOPT_DEVIN_WORKSPACES`` env var — colon-separated abs paths
  2. Devin registry: ``~/.config/Devin/User/workspaceStorage/*/workspace.json``
  4. Working directory fallback

Usage (standalone):
    python harvest_devin.py [--out-dir PATH] [--workspaces PATH ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import unquote, urlparse

# ── cross-platform path resolution (Linux + Windows + macOS) ──────────────────
#
# Devin is a VS Code-family app, so its user-data dir moves with the OS:
# Linux ~/.config/<App>, Windows %APPDATA%\<App>, macOS
# ~/Library/Application Support/<App>.  Resolve all candidates and let callers
# keep whichever actually exists.

def _app_data_roots(app: str) -> List[str]:
    """User-data dir candidates for a VS Code-family app, current OS first."""
    home = os.path.expanduser("~")
    roots: List[str] = []
    if os.name == "nt":
        appdata = os.environ.get("APPDATA") or os.path.join(home, "AppData", "Roaming")
        roots.append(os.path.join(appdata, app))
    elif sys.platform == "darwin":
        roots.append(os.path.join(home, "Library", "Application Support", app))
    # XDG / Linux (also a sensible fallback everywhere)
    xdg = os.environ.get("XDG_CONFIG_HOME") or os.path.join(home, ".config")
    roots.append(os.path.join(xdg, app))
    # de-dupe, preserve order
    return list(dict.fromkeys(roots))


def _devin_transcript_candidates() -> List[str]:
    """Where the Devin CLI may store ATIF transcripts, per OS."""
    home = os.path.expanduser("~")
    cands: List[str] = []
    if os.name == "nt":
        for base in (os.environ.get("LOCALAPPDATA"), os.environ.get("APPDATA")):
            if base:
                cands.append(os.path.join(base, "devin", "cli", "transcripts"))
    elif sys.platform == "darwin":
        cands.append(os.path.join(home, "Library", "Application Support",
                                  "devin", "cli", "transcripts"))
    cands.append(os.path.join(home, ".local", "share", "devin", "cli", "transcripts"))
    return list(dict.fromkeys(cands))


def _first_existing(paths: List[str]) -> str:
    """First path that exists, else the first candidate (for nice messaging)."""
    for p in paths:
        if os.path.exists(p):
            return p
    return paths[0] if paths else ""


def _uri_to_path(folder: str) -> str:
    """Convert a VS Code ``file://`` workspace URI to a local path, cross-platform.

    Linux:   file:///home/u/proj      -> /home/u/proj
    Windows: file:///c%3A/Users/u/p   -> c:/Users/u/p
    """
    if not folder.startswith("file://"):
        return folder
    path = unquote(urlparse(folder).path)
    # Windows drive paths come through as '/C:/...' — strip the leading slash.
    if os.name == "nt" and re.match(r"^/[A-Za-z]:", path):
        path = path[1:]
    return path

# ── workspace auto-detection ─────────────────────────────────────────────────

def _workspaces_from_registry(storage_root: str) -> List[tuple]:
    """Read VS Code-style workspaceStorage to get (mtime, path) pairs."""
    results: List[tuple] = []
    if not os.path.isdir(storage_root):
        return results
    for entry in os.scandir(storage_root):
        ws_json = os.path.join(entry.path, "workspace.json")
        if not os.path.isfile(ws_json):
            continue
        try:
            with open(ws_json, encoding="utf-8") as f:
                data = json.load(f)
            folder = _uri_to_path(data.get("folder", ""))
            if folder and os.path.isdir(folder):
                results.append((os.path.getmtime(ws_json), folder))
        except Exception:
            continue
    return results


def _detect_workspaces() -> List[str]:
    """Return known workspace paths (Devin registry), newest first."""
    env_val = os.environ.get("SKILLOPT_DEVIN_WORKSPACES", "")
    if env_val:
        # os.pathsep so Windows 'C:\a;C:\b' splits correctly (not on the drive colon)
        return [p for p in env_val.split(os.pathsep) if p and os.path.isdir(p)]

    registries: List[str] = [
        os.path.join(r, "User", "workspaceStorage")
        for r in _app_data_roots("Devin")
    ]

    seen: set = set()
    results: List[tuple] = []
    for registry in registries:
        for mtime, folder in _workspaces_from_registry(registry):
            if folder not in seen:
                seen.add(folder)
                results.append((mtime, folder))
    results.sort(reverse=True)
    paths = [p for _, p in results]
    return paths if paths else [os.getcwd()]

# ── helpers ───────────────────────────────────────────────────────────────────

def _slug(path: str) -> str:
    """SHA-256 of abs-path, first 16 hex chars — matches Claude Code's scheme."""
    return hashlib.sha256(os.path.abspath(path).encode()).hexdigest()[:16]


def _iso(epoch_ms: Optional[float] = None) -> str:
    dt = (datetime.fromtimestamp(epoch_ms / 1000.0, tz=timezone.utc)
          if epoch_ms is not None else datetime.now(tz=timezone.utc))
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _write_session(
    out_dir: str, project: str, session_id: str,
    user_prompts: List[str], assistant_replies: List[str],
    timestamp_base_ms: float,
    task_key: Optional[str] = None,
) -> None:
    slug = _slug(project)
    session_dir = os.path.join(out_dir, "projects", slug)
    os.makedirs(session_dir, exist_ok=True)
    out_path = os.path.join(session_dir, f"{session_id}.jsonl")
    ts = timestamp_base_ms
    with open(out_path, "w", encoding="utf-8") as f:
        for user_text, asst_text in zip(user_prompts, assistant_replies):
            user_rec = {
                "type": "user",
                "message": {"role": "user", "content": user_text},
                "cwd": project,
                "timestamp": _iso(ts),
                "sessionId": session_id,
                "version": "1.0",
            }
            if task_key:
                # grouping key so the miner can collapse repeats into one recurring task
                user_rec["taskKey"] = task_key
            f.write(json.dumps(user_rec, ensure_ascii=False) + "\n")
            # space the reply >=5s after the prompt so a single-turn session
            # isn't misclassified as a <3s headless replay and dropped by the
            # engine's harvest filter (skillopt_sleep Issue #62).
            ts += 5000
            f.write(json.dumps({
                "type": "assistant",
                "message": {"role": "assistant", "content": asst_text},
                "timestamp": _iso(ts),
                "sessionId": session_id,
                "version": "1.0",
            }, ensure_ascii=False) + "\n")
            ts += 2000


def _append_history(out_dir: str, display: str, project: str, timestamp_ms: float) -> None:
    record = {"display": display, "timestamp": timestamp_ms, "project": project}
    with open(os.path.join(out_dir, "history.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _infer_project(text: str, workspaces: List[str]) -> str:
    for ws in workspaces:
        if os.path.basename(ws.rstrip("/")).lower() in text.lower():
            return ws
    return workspaces[0] if workspaces else os.getcwd()

# ── task identity + outcome extraction (fuel for the validation gate) ─────────
#
# SkillOpt's gate only works "where tasks recur and have a checkable correctness
# signal."  These helpers add the two things a raw transcript lacks:
#   * a stable taskKey so repeats collapse into one recurring task, and
#   * an outcome envelope (success + verifier + re-runnable reference) so the
#     held-out replay has something to score against.

_LANG_HINTS = [
    ("java",   r"(java|spring|maven|\bmvn\b|gradle|\.java\b|lombok)"),
    ("python", r"(python|pytest|\bpip\b|\.py\b|django|flask)"),
    ("ts",     r"(typescript|\.tsx?\b|\bnpm\b|jest|node)"),
    ("js",     r"(javascript|\.jsx?\b)"),
    ("sql",    r"(\bsql\b|select\s|mariadb|mysql|postgres|\.sql\b)"),
    ("go",     r"(golang|\bgo test\b|\.go\b)"),
    ("rust",   r"(rust|cargo|\.rs\b)"),
]
_INTENT_HINTS = [
    ("fix",       r"(fix|bug|error|fail|npe|exception|broken|crash)"),
    ("implement", r"(implement|add|create|build|introduce|support)"),
    ("refactor",  r"(refactor|clean ?up|rename|extract|simplify)"),
    ("test",      r"(test|coverage|assert)"),
    ("review",    r"(review|audit|inspect)"),
    ("optimize",  r"(optimi[sz]e|perf|speed up|slow)"),
    ("explain",   r"(explain|understand|what does|how does)"),
]
_STOPWORDS = {"please", "this", "that", "with", "from", "into", "should",
              "would", "code", "using", "the", "have"}


def _normalize_task_key(text: str, project: str) -> str:
    """Stable '<lang>:<intent>:<target>' grouping key for a task."""
    low = text.lower()
    lang = next((n for n, pat in _LANG_HINTS if re.search(pat, low)), "general")
    intent = next((n for n, pat in _INTENT_HINTS if re.search(pat, low)), "task")
    # target: prefer a CamelCase identifier, then a filename, then first real word
    m = re.search(r"\b([A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+)\b", text)  # CamelCase
    if not m:
        m = re.search(r"\b([\w-]+\.\w+)\b", text)                     # filename.ext
    if m:
        target = m.group(1)
    else:
        # first content word that isn't a stopword or an intent verb (e.g. "implement")
        target = next((w for w in re.findall(r"[a-zA-Z]{4,}", low)
                       if w not in _STOPWORDS
                       and not any(re.search(pat, w) for _, pat in _INTENT_HINTS)),
                      "general")
    target = re.sub(r"[^a-zA-Z0-9]+", "-", target).strip("-").lower()[:40] or "general"
    return f"{lang}:{intent}:{target}"


_PASS_PAT = re.compile(
    r"(build success|all tests? pass(?:ed)?|\b\d+ passed\b|\b0 failed\b|"
    r"tests? pass(?:ed)?|✓|no errors)", re.IGNORECASE)
_FAIL_PAT = re.compile(
    r"(build failure|tests? failed|\b[1-9]\d* failed\b|error:|traceback|"
    r"assertion ?error)", re.IGNORECASE)  # note: "0 failed" must NOT match
_CMD_PAT = re.compile(
    r"((?:rtk\s+)?(?:mvn|gradle|pytest|npm(?:\s+run)?\s+test|yarn\s+test|"
    r"go\s+test|cargo\s+test)[^\n`]*)", re.IGNORECASE)


def _detect_outcome(messages: List[str]) -> Optional[Dict[str, Any]]:
    """Best-effort checkable signal from agent messages. None ⇒ no hard signal."""
    blob = "\n".join(m for m in messages if m)
    pass_hit, fail_hit = _PASS_PAT.search(blob), _FAIL_PAT.search(blob)
    if not pass_hit and not fail_hit:
        return None
    verifier = "tests" if re.search(r"test|pytest", blob, re.IGNORECASE) else "build"
    out: Dict[str, Any] = {
        "success": bool(pass_hit) and not fail_hit,
        "verifier": verifier,
        "evidence": (pass_hit or fail_hit).group(0).strip(),
    }
    cmd = _CMD_PAT.search(blob)
    if cmd:
        # keep only the command itself, dropping any "-> result" / ": output" tail
        repro = re.split(r"\s*(?:->|→|:|,)\s*", cmd.group(1))[0].strip()
        out["reference"] = {"repro": repro}
    return out


def _build_rubric(user_prompt: str) -> List[str]:
    """Derive checkable criteria from the task so a judge has something to score."""
    crit: List[str] = []
    ids = re.findall(r"\b([A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+|[\w-]+\.\w+)\b", user_prompt)
    for i in dict.fromkeys(ids):           # dedupe, preserve order
        crit.append(f"Addresses {i}")
    intent = _normalize_task_key(user_prompt, "").split(":")[1]
    crit.append({
        "fix":       "Resolves the reported defect without introducing new errors",
        "implement": "Implements the requested behavior end to end",
        "refactor":  "Preserves behavior while improving structure",
        "test":      "Adds or fixes tests that actually exercise the change",
        "optimize":  "Improves performance without changing results",
    }.get(intent, "Satisfies the user's stated request"))
    crit.append("Response is concrete and actionable, not a restatement of the task")
    return crit[:5]


def _judge_rubric_fallback(user_prompt: str) -> Dict[str, Any]:
    """When no hard signal exists, attach a rubric and mark the task for judge
    scoring. success=None tells the gate to defer/judge rather than trust it.
    The actual scoring is done by judge.py (or the engine) at replay time."""
    return {
        "success": None,
        "verifier": "judge",
        "rubric": _build_rubric(user_prompt or ""),
    }


def _write_outcome(out_dir: str, session_id: str, task_key: str, project: str,
                   ts_ms: float, outcome: Dict[str, Any]) -> None:
    rec = {"type": "outcome", "sessionId": session_id, "taskKey": task_key,
           "project": project, "timestamp": _iso(ts_ms), **outcome}
    with open(os.path.join(out_dir, "outcomes.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# ── source 1: Devin ATIF-v1.7 transcripts ────────────────────────────────────

def harvest_devin_transcripts(
    transcripts_dir: str, out_dir: str, workspaces: List[str]
) -> int:
    """Convert Devin CLI ATIF-v1.7 transcripts to Claude Code JSONL."""
    if not os.path.isdir(transcripts_dir):
        return 0
    written = 0
    for entry in os.scandir(transcripts_dir):
        if not entry.name.endswith(".json"):
            continue
        try:
            with open(entry.path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        if data.get("schema_version", "").startswith("ATIF"):
            pass  # Devin native format
        else:
            continue
        session_id = data.get("session_id") or entry.name[:-5]
        steps = data.get("steps") or []
        user_prompts: List[str] = []
        agent_replies: List[str] = []
        project = ""
        ts_base: Optional[float] = None
        for step in steps:
            src = step.get("source", "")
            msg = str(step.get("message") or "").strip()
            if not msg or src == "system":
                continue
            if src == "user":
                user_prompts.append(msg)
                if not project:
                    project = _infer_project(msg, workspaces)
            elif src == "agent":
                agent_replies.append(msg)
            if ts_base is None:
                raw_ts = step.get("timestamp", "")
                if raw_ts:
                    try:
                        from datetime import datetime as _dt
                        ts_base = _dt.fromisoformat(
                            raw_ts.replace("Z", "+00:00")
                        ).timestamp() * 1000
                    except Exception:
                        pass
        if not user_prompts:
            continue
        if not project:
            project = workspaces[0] if workspaces else os.getcwd()
        if ts_base is None:
            ts_base = datetime.now(tz=timezone.utc).timestamp() * 1000
        # Identity + outcome: what makes this trajectory replayable & gradeable.
        task_key = _normalize_task_key(user_prompts[0], project)
        outcome = _detect_outcome(agent_replies) or _judge_rubric_fallback(user_prompts[0])
        # Pair turns; pad shorter list
        n = max(len(user_prompts), len(agent_replies))
        user_prompts += [""] * (n - len(user_prompts))
        agent_replies += [""] * (n - len(agent_replies))
        sid = f"devin_{session_id}"
        _write_session(
            out_dir, project, sid,
            user_prompts=[p for p in user_prompts if p],
            assistant_replies=[r if r else "[no reply recorded]" for r, p in
                               zip(agent_replies, user_prompts) if p],
            timestamp_base_ms=ts_base,
            task_key=task_key,
        )
        _write_outcome(out_dir, sid, task_key, project, ts_base, outcome)
        _append_history(
            out_dir,
            display=(user_prompts[0] or session_id)[:120],
            project=project,
            timestamp_ms=ts_base,
        )
        written += 1
    return written


# ── source 2: agentmemory ─────────────────────────────────────────────────────

def harvest_agentmemory(agentmemory_path: str, out_dir: str,
                        workspaces: List[str]) -> int:
    if not os.path.isfile(agentmemory_path):
        return 0
    with open(agentmemory_path, encoding="utf-8") as f:
        data = json.load(f)
    memories: Dict[str, Any] = data.get("mem:memories", {})
    written = 0
    base_ts = datetime.now(tz=timezone.utc).timestamp() * 1000 - len(memories) * 60_000
    for i, (mem_id, mem) in enumerate(memories.items()):
        title = str(mem.get("title", "")).strip()
        content = str(mem.get("content", "")).strip()
        if not title or not content:
            continue
        project = _infer_project(title + " " + content, workspaces)
        ts = base_ts + i * 60_000
        _write_session(out_dir, project, mem_id,
                       user_prompts=[title],
                       assistant_replies=[content],
                       timestamp_base_ms=ts)
        _append_history(out_dir, display=title[:120], project=project, timestamp_ms=ts)
        written += 1
    return written

# ── source 3: skill files (.devin/skills) ─────────────────────────────────────

def harvest_skills(workspaces: List[str], out_dir: str) -> int:
    written = 0
    seen_ids: set = set()
    for ws in workspaces:
        skills_root = os.path.join(ws, ".devin", "skills")
        if not os.path.isdir(skills_root):
            continue
        for skill_dir in os.scandir(skills_root):
            if not skill_dir.is_dir():
                continue
            skill_md = os.path.join(skill_dir.path, "SKILL.md")
            if not os.path.isfile(skill_md):
                continue
            sid = f"skill_{skill_dir.name}"
            if sid in seen_ids:
                continue
            seen_ids.add(sid)
            with open(skill_md, encoding="utf-8") as f:
                raw = f.read()
            body = re.sub(r"^---.*?---\s*", "", raw, flags=re.DOTALL).strip()
            if not body:
                continue
            first_line = body.split("\n")[0].lstrip("# ").strip()
            user_ask = f"Please use the {skill_dir.name} skill: {first_line}"
            ts = datetime.now(tz=timezone.utc).timestamp() * 1000 - 3_600_000
            _write_session(out_dir, ws, sid,
                           user_prompts=[user_ask],
                           assistant_replies=[body[:1200]],
                           timestamp_base_ms=ts)
            _append_history(out_dir, display=user_ask[:120], project=ws, timestamp_ms=ts)
            written += 1
    return written

# ── main ─────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate SkillOpt-Sleep transcripts from Devin local data"
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.expanduser("~/.skillopt-sleep-devin"),
        help="Output claude_home dir (default: ~/.skillopt-sleep-devin)",
    )
    parser.add_argument(
        "--agentmemory",
        default=os.path.expanduser("~/.agentmemory/standalone.json"),
        help="Path to agentmemory standalone.json",
    )
    parser.add_argument(
        "--devin-transcripts",
        default=_first_existing(_devin_transcript_candidates()),
        help="Devin CLI ATIF transcripts directory (default: per-OS auto-detect)",
    )
    parser.add_argument(
        "--workspaces", nargs="*",
        help="Workspace paths (default: auto-detect from Devin registry)",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "projects"), exist_ok=True)

    workspaces = args.workspaces or _detect_workspaces()
    workspaces = [ws for ws in workspaces if os.path.isdir(ws)]
    if not workspaces:
        workspaces = [os.getcwd()]

    total = 0
    devin_transcripts = os.path.expanduser(args.devin_transcripts)
    n = harvest_devin_transcripts(devin_transcripts, out_dir, workspaces)
    if not args.quiet:
        print(f"[harvest_devin] devin        : {n} sessions")
    total += n

    n = harvest_agentmemory(args.agentmemory, out_dir, workspaces)
    if not args.quiet:
        print(f"[harvest_devin] agentmemory  : {n} sessions")
    total += n

    n = harvest_skills(workspaces, out_dir)
    if not args.quiet:
        print(f"[harvest_devin] skill files  : {n} sessions")
    total += n

    if not args.quiet:
        print(f"[harvest_devin] total        : {total} synthetic sessions → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
