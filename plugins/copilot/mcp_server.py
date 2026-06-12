#!/usr/bin/env python3
"""SkillOpt-Sleep — minimal MCP server (stdio, stdlib-only).

Exposes the sleep engine as MCP tools so any MCP-capable client (GitHub Copilot
CLI / VS Code, Claude Desktop, etc.) can drive it. No third-party deps: speaks
JSON-RPC 2.0 over stdio with just the handful of MCP methods clients need.

Tools exposed:
  - sleep_status   : how many nights have run + the latest staged proposal
  - sleep_dry_run  : harvest+mine+replay, report only (no staging)
  - sleep_run      : full cycle, stages a proposal (nothing live changes)
  - sleep_adopt    : apply the latest staged proposal (with backup)
  - sleep_harvest  : debug — list mined recurring tasks

Each tool shells out to `python -m skillopt_sleep <action> ...` and returns its
stdout. Configure your client to launch:  python plugins/copilot/mcp_server.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

REPO_ROOT = os.environ.get("SKILLOPT_SLEEP_REPO") or os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
PROTOCOL_VERSION = "2024-11-05"

TOOLS = [
    {"name": "sleep_status", "action": "status",
     "description": "Show how many SkillOpt-Sleep nights have run and the latest staged proposal."},
    {"name": "sleep_dry_run", "action": "dry-run",
     "description": "Preview a sleep cycle (harvest+mine+replay) without staging anything."},
    {"name": "sleep_run", "action": "run",
     "description": "Run a full sleep cycle; stages a reviewed proposal. Nothing live changes until adopt."},
    {"name": "sleep_adopt", "action": "adopt",
     "description": "Apply the latest staged proposal to CLAUDE.md/SKILL.md (backs up first)."},
    {"name": "sleep_harvest", "action": "harvest",
     "description": "Debug: list the recurring tasks mined from recent sessions."},
]
_BY_NAME = {t["name"]: t for t in TOOLS}

_TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "project": {"type": "string", "description": "Project dir to evolve (default: cwd)."},
        "backend": {"type": "string", "enum": ["mock", "claude", "codex", "copilot"],
                     "description": "mock = no API spend (default); claude/codex/copilot = real."},
        "scope": {"type": "string", "enum": ["invoked", "all"]},
    },
    "additionalProperties": False,
}


def _run_engine(action: str, args: dict) -> str:
    py = sys.executable or "python3"
    cmd = [py, "-m", "skillopt_sleep", action]
    if args.get("project"):
        cmd += ["--project", str(args["project"])]
    if args.get("backend"):
        cmd += ["--backend", str(args["backend"])]
    if args.get("scope"):
        cmd += ["--scope", str(args["scope"])]
    try:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=3600)
    except Exception as e:  # noqa: BLE001
        return f"[error] failed to run engine: {e}"
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    return out + (("\n[stderr]\n" + err) if err else "")


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
        return None  # notification, no response
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
