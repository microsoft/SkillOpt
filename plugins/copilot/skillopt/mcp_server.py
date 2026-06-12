#!/usr/bin/env python3
"""SkillOpt (research engine) — minimal MCP server (stdio, stdlib-only).

Exposes the core SkillOpt skill-optimization engine as MCP tools so any
MCP-capable client (GitHub Copilot CLI / VS Code, Claude Desktop, etc.) can
drive it. No third-party deps: speaks JSON-RPC 2.0 over stdio with just the
handful of MCP methods clients need.

This is the companion to the SkillOpt-Sleep MCP server (``../mcp_server.py``).
Where Sleep evolves a *local agent* from past sessions, this server drives the
*research* training/eval loops from this repo (``scripts/train.py`` /
``scripts/eval_only.py``) against the benchmark configs in ``configs/``.

Tools exposed:
  - skillopt_list_configs : discover the benchmark YAML configs you can use
  - skillopt_train        : run a reflective skill-optimization (training) loop
  - skillopt_eval         : evaluate a single skill on a dataset (no training)

``skillopt_train`` and ``skillopt_eval`` shell out to the repo's entry-point
scripts and stream back their stdout/stderr. Configure your client to launch:
  python plugins/copilot/skillopt/mcp_server.py
"""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys

# Repo root: three levels up from plugins/copilot/skillopt/mcp_server.py
REPO_ROOT = os.environ.get("SKILLOPT_REPO") or os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
PROTOCOL_VERSION = "2024-11-05"

# Training/eval runs are long; give the engine plenty of headroom.
RUN_TIMEOUT_SECONDS = int(os.environ.get("SKILLOPT_RUN_TIMEOUT", "21600"))  # 6h


def _list_configs() -> str:
    """List the benchmark configs available under configs/ (filesystem only)."""
    pattern = os.path.join(REPO_ROOT, "configs", "**", "*.yaml")
    paths = sorted(glob.glob(pattern, recursive=True))
    if not paths:
        return f"[no configs found under {os.path.join(REPO_ROOT, 'configs')}]"
    rels = [os.path.relpath(p, REPO_ROOT).replace(os.sep, "/") for p in paths]
    lines = ["Available SkillOpt configs (pass as `config`):", ""]
    lines += [f"  - {r}" for r in rels]
    return "\n".join(lines)


def _run_script(script_rel: str, args: dict, *, required: tuple[str, ...] = ()) -> str:
    """Shell out to a repo entry-point script, mapping args -> --flags."""
    for key in required:
        if not args.get(key):
            return f"[error] missing required argument: {key}"

    py = sys.executable or "python3"
    cmd = [py, os.path.join("scripts", script_rel)]

    # Ordered flags that the train/eval scripts accept directly.
    flag_args = (
        "config", "skill", "split", "env", "backend",
        "optimizer_model", "target_model", "out_root",
        "num_epochs", "batch_size", "seed", "use_gate",
    )
    for key in flag_args:
        val = args.get(key)
        if val is None or val == "":
            continue
        cmd += [f"--{key}", str(val)]

    # cfg-options: arbitrary KEY=VALUE YAML overrides (nargs="+").
    cfg_options = args.get("cfg_options")
    if cfg_options:
        if isinstance(cfg_options, str):
            cfg_options = cfg_options.split()
        cmd += ["--cfg-options", *[str(x) for x in cfg_options]]

    # extra_args: raw passthrough for any other train/eval flag.
    extra = args.get("extra_args")
    if extra:
        if isinstance(extra, str):
            extra = extra.split()
        cmd += [str(x) for x in extra]

    try:
        proc = subprocess.run(
            cmd, cwd=REPO_ROOT, capture_output=True, text=True,
            timeout=RUN_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return f"[error] run exceeded {RUN_TIMEOUT_SECONDS}s timeout: {' '.join(cmd)}"
    except Exception as e:  # noqa: BLE001
        return f"[error] failed to run script: {e}"
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    body = out + (("\n[stderr]\n" + err) if err else "")
    return body or f"[done] exit code {proc.returncode}, no output"


TOOLS = [
    {
        "name": "skillopt_list_configs",
        "description": "List the benchmark YAML configs under configs/ that can be passed as `config` to train/eval.",
    },
    {
        "name": "skillopt_train",
        "description": "Run a SkillOpt reflective skill-optimization (training) loop on a benchmark config. Long-running; uses your model backend/budget.",
    },
    {
        "name": "skillopt_eval",
        "description": "Evaluate a single skill markdown file on a dataset without training (scripts/eval_only.py).",
    },
]
_BY_NAME = {t["name"]: t for t in TOOLS}

_NO_ARGS_SCHEMA = {"type": "object", "properties": {}, "additionalProperties": False}

_COMMON_PROPS = {
    "config": {"type": "string",
               "description": "Path to a benchmark YAML config (e.g. configs/searchqa/default.yaml). See skillopt_list_configs."},
    "env": {"type": "string", "description": "Override the environment/adapter name (e.g. searchqa, alfworld)."},
    "backend": {"type": "string", "description": "Model backend (e.g. openai, claude, codex, qwen)."},
    "optimizer_model": {"type": "string", "description": "Model used for reflection/skill rewriting (the optimizer)."},
    "target_model": {"type": "string", "description": "Model used to execute tasks (the target)."},
    "out_root": {"type": "string", "description": "Output directory root for run artifacts."},
    "cfg_options": {"type": "string", "description": "Space-separated YAML overrides, e.g. 'seed=123 batch_size=40'."},
    "extra_args": {"type": "string", "description": "Raw passthrough flags for the underlying script, e.g. '--workers 8 --max_turns 30'."},
}

_TRAIN_SCHEMA = {
    "type": "object",
    "properties": {
        **_COMMON_PROPS,
        "num_epochs": {"type": "integer", "description": "Number of optimization epochs."},
        "batch_size": {"type": "integer", "description": "Tasks per optimization step."},
        "seed": {"type": "integer", "description": "Random seed."},
        "use_gate": {"type": "string", "enum": ["true", "false"],
                     "description": "Whether to keep the held-out validation gate on (default on)."},
    },
    "required": ["config"],
    "additionalProperties": False,
}

_EVAL_SCHEMA = {
    "type": "object",
    "properties": {
        **_COMMON_PROPS,
        "skill": {"type": "string", "description": "Path to the skill markdown file to evaluate."},
        "split": {"type": "string", "description": "Dataset split to evaluate (default: all)."},
    },
    "required": ["config", "skill"],
    "additionalProperties": False,
}

_SCHEMA_BY_NAME = {
    "skillopt_list_configs": _NO_ARGS_SCHEMA,
    "skillopt_train": _TRAIN_SCHEMA,
    "skillopt_eval": _EVAL_SCHEMA,
}


def _result(id_, result):
    return {"jsonrpc": "2.0", "id": id_, "result": result}


def _error(id_, code, message):
    return {"jsonrpc": "2.0", "id": id_, "error": {"code": code, "message": message}}


def _dispatch(name: str, args: dict) -> str:
    if name == "skillopt_list_configs":
        return _list_configs()
    if name == "skillopt_train":
        return _run_script("train.py", args, required=("config",))
    if name == "skillopt_eval":
        return _run_script("eval_only.py", args, required=("config", "skill"))
    return f"[error] unknown tool: {name}"


def handle(req: dict):
    method = req.get("method")
    id_ = req.get("id")
    if method == "initialize":
        return _result(id_, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "skillopt", "version": "0.1.0"},
        })
    if method in ("notifications/initialized", "initialized"):
        return None  # notification, no response
    if method == "tools/list":
        return _result(id_, {"tools": [
            {"name": t["name"], "description": t["description"],
             "inputSchema": _SCHEMA_BY_NAME[t["name"]]}
            for t in TOOLS
        ]})
    if method == "tools/call":
        params = req.get("params") or {}
        name = params.get("name")
        if name not in _BY_NAME:
            return _error(id_, -32602, f"unknown tool: {name}")
        text = _dispatch(name, params.get("arguments") or {})
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
