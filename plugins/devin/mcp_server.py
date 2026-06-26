#!/usr/bin/env python3
"""SkillOpt-Sleep — Devin MCP server (stdio, stdlib-only).

Exposes the sleep engine as MCP tools so Devin (Cognition) can drive it. No
third-party deps: speaks JSON-RPC 2.0 over stdio with just the handful of MCP
methods clients need. Same `sleep_*` interface and engine flags as
`plugins/copilot`, plus a Devin-specific harvest step.

Before each data-reading action this server runs `harvest_devin.py` to convert
locally available Devin data (ATIF-v1.7 transcripts, agentmemory memories, and
.devin skill files) into the Claude Code-compatible JSONL the engine consumes,
writing it under SKILLOPT_DEVIN_CLAUDE_HOME and pointing the engine there with
`--claude-home`. After `sleep_adopt` the evolved skill is synced back into the
workspace's `.devin/skills/`.

Tools: sleep_status, sleep_dry_run, sleep_run, sleep_adopt, sleep_harvest,
sleep_schedule, sleep_unschedule. Each shells out to
`python -m skillopt_sleep <action> ...`. Configure Devin to launch:
  python plugins/devin/mcp_server.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

# expanduser wraps the whole value so a "~/..." env var is expanded too (not
# just a default) — otherwise a literal ~ dir gets created.
REPO_ROOT = os.path.expanduser(
    os.environ.get("SKILLOPT_SLEEP_REPO")
    or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)
PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
CLAUDE_HOME = os.path.expanduser(
    os.environ.get("SKILLOPT_DEVIN_CLAUDE_HOME", "~/.skillopt-sleep-devin")
)
MANAGED_SKILL_NAME = os.environ.get("SKILLOPT_MANAGED_SKILL", "skillopt-sleep-learned")
PROTOCOL_VERSION = "2024-11-05"

TOOLS = [
    {"name": "sleep_status", "action": "status",
     "description": "Show how many SkillOpt-Sleep nights have run and the latest staged proposal."},
    {"name": "sleep_dry_run", "action": "dry-run",
     "description": "Preview a sleep cycle (harvest+mine+replay) without staging anything."},
    {"name": "sleep_run", "action": "run",
     "description": "Run a full sleep cycle; stages a reviewed proposal. Nothing live changes until adopt."},
    {"name": "sleep_adopt", "action": "adopt",
     "description": "Apply the latest staged proposal to the managed SKILL.md and sync it into .devin/skills/."},
    {"name": "sleep_harvest", "action": "harvest",
     "description": "Debug: list the recurring tasks mined from recent Devin sessions."},
    {"name": "sleep_schedule", "action": "schedule",
     "description": "Install a nightly cron entry to run the sleep cycle automatically."},
    {"name": "sleep_unschedule", "action": "unschedule",
     "description": "Remove the nightly cron entry for a project."},
]
_BY_NAME = {t["name"]: t for t in TOOLS}

_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "project": {"type": "string",
                    "description": "Project dir to evolve (default: cwd)."},
        "backend": {"type": "string", "enum": ["mock", "claude", "codex", "copilot"],
                    "description": "mock = no API spend (default); claude/codex/copilot = real."},
        "scope": {"type": "string", "enum": ["invoked", "all"],
                  "description": "Harvest scope (default: invoked project only)."},
        "source": {"type": "string", "enum": ["claude", "codex", "auto"],
                   "description": "Transcript source (default: claude)."},
        "model": {"type": "string",
                  "description": "Backend-specific model override."},
        "tasks_file": {"type": "string",
                       "description": "Path to reviewed TaskRecord JSON (skips harvest)."},
        "target_skill_path": {"type": "string",
                              "description": "Explicit SKILL.md path to evolve/stage/adopt."},
        "progress": {"type": "boolean",
                     "description": "Print phase progress to stderr."},
        "max_sessions": {"type": "integer",
                         "description": "Cap harvested sessions per run."},
        "max_tasks": {"type": "integer",
                      "description": "Cap mined tasks per run."},
        "lookback_hours": {"type": "integer",
                           "description": "Harvest window in hours (default: 72)."},
        "auto_adopt": {"type": "boolean",
                       "description": "Auto-adopt if gate passes (default: false)."},
        "json": {"type": "boolean",
                 "description": "Return machine-readable JSON output."},
        "edit_budget": {"type": "integer",
                        "description": "Max bounded edits per night (default: 4)."},
        "hour": {"type": "integer",
                 "description": "Hour for schedule (0-23, default: 3)."},
        "minute": {"type": "integer",
                   "description": "Minute for schedule (0-59, default: 17)."},
    },
    "additionalProperties": False,
}

# actions that read harvested Devin data (schedule/unschedule/adopt don't)
_HARVEST_ACTIONS = {"status", "dry-run", "run", "harvest"}


def _run_harvest() -> str:
    """Convert local Devin data into the JSONL the engine reads, under CLAUDE_HOME."""
    harvester = os.path.join(PLUGIN_DIR, "harvest_devin.py")
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    try:
        proc = subprocess.run(
            [sys.executable, harvester, "--out-dir", CLAUDE_HOME],
            capture_output=True, text=True, timeout=60, env=env,
        )
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        return out + (("\n[harvest stderr]\n" + err) if err else "")
    except Exception as exc:
        return f"[harvest_devin] warning: {exc}"


def _sync_skill(project: str) -> str:
    """After adopt, copy the evolved skill into the workspace's .devin/skills/."""
    src = os.path.join(CLAUDE_HOME, "skills", MANAGED_SKILL_NAME, "SKILL.md")
    if not (os.path.isfile(src) and project and os.path.isdir(project)):
        return ""
    dot_root = os.path.join(project, ".devin")
    if not os.path.isdir(dot_root):
        return ""
    dst_dir = os.path.join(dot_root, "skills", MANAGED_SKILL_NAME)
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, "SKILL.md")
    shutil.copy2(src, dst)
    return f"\n[sleep] synced evolved skill → {dst}"


def _run_engine(action: str, args: dict) -> str:
    harvest_out = _run_harvest() if action in _HARVEST_ACTIONS else ""

    py = sys.executable or "python3"
    cmd = [py, "-m", "skillopt_sleep", action, "--claude-home", CLAUDE_HOME]
    # Devin transcripts are converted to the Claude format, so default source=claude
    if not args.get("source"):
        cmd += ["--source", "claude"]
    # String-valued flags
    for flag, key in [
        ("--project", "project"), ("--backend", "backend"),
        ("--scope", "scope"), ("--source", "source"),
        ("--model", "model"), ("--tasks-file", "tasks_file"),
        ("--target-skill-path", "target_skill_path"),
    ]:
        val = args.get(key)
        if val:
            cmd += [flag, str(val)]
    # Integer-valued flags
    for flag, key in [
        ("--max-sessions", "max_sessions"), ("--max-tasks", "max_tasks"),
        ("--lookback-hours", "lookback_hours"), ("--edit-budget", "edit_budget"),
        ("--hour", "hour"), ("--minute", "minute"),
    ]:
        val = args.get(key)
        if val is not None:
            cmd += [flag, str(int(val))]
    # Boolean flags
    for flag, key in [
        ("--progress", "progress"), ("--auto-adopt", "auto_adopt"), ("--json", "json"),
    ]:
        if args.get(key):
            cmd.append(flag)

    env = dict(os.environ)
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    try:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True,
                              text=True, timeout=3600, env=env)
    except Exception as e:
        return f"[harvest]\n{harvest_out}\n[error] failed to run engine: {e}"
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    result = (f"[harvest]\n{harvest_out}\n\n" if harvest_out else "") + f"[engine]\n{out}"
    if err:
        result += f"\n[stderr]\n{err}"
    if action == "adopt":
        result += _sync_skill(args.get("project") or os.getcwd())
    return result


def _result(id_, result):
    return {"jsonrpc": "2.0", "id": id_, "result": result}


def _error(id_, code, message):
    return {"jsonrpc": "2.0", "id": id_, "error": {"code": code, "message": message}}


def handle(req: dict):
    method = req.get("method")
    id_ = req.get("id")
    if method == "initialize":
        return _result(id_, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "skillopt-sleep", "version": "0.1.0"},
        })
    if method in ("notifications/initialized", "initialized"):
        return None
    if method == "tools/list":
        return _result(id_, {"tools": [
            {"name": t["name"], "description": t["description"], "inputSchema": _TOOL_SCHEMA}
            for t in TOOLS
        ]})
    if method == "tools/call":
        params = req.get("params") or {}
        name = params.get("name")
        tool = _BY_NAME.get(name)
        if not tool:
            return _error(id_, -32602, f"unknown tool: {name}")
        text = _run_engine(tool["action"], params.get("arguments") or {})
        return _result(id_, {"content": [{"type": "text", "text": text}]})
    if method == "ping":
        return _result(id_, {})
    return _error(id_, -32601, f"method not found: {method}")


def main() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:
            continue
        resp = handle(req)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
