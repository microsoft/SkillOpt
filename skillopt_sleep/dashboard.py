"""SkillOpt-Sleep — local control-panel dashboard.

A zero-dependency (stdlib ``http.server``) web UI over one project's sleep
pipeline. It is arranged to mirror the actual data flow —

    transcripts -> harvest -> mine -> split -> replay -> reflect -> gate
                -> stage -> adopt

— and for every stage shows: which agent role runs it (target / optimizer /
pure code), which model that role resolves to, the exact prompt template it
receives (editable, live), and the selected night's evidence events for that
stage (from ``evidence.jsonl``). Config changes are written to the user
config file and apply from the next run; prompt overrides apply to the very
next model call (the registry re-reads its override file on mtime change).

Serves on 127.0.0.1 only.

    python -m skillopt_sleep dashboard [--project DIR] [--port N]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

from skillopt_sleep import prompts as prompt_registry
from skillopt_sleep.config import DEFAULTS, HOME_STATE_DIR, load_config
from skillopt_sleep.evidence import read_events
from skillopt_sleep.staging import adopt as adopt_staging
from skillopt_sleep.staging import staging_root

_HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")

# Config keys the dashboard may write (safety allowlist: everything else in
# the user config file is preserved untouched).
_EDITABLE_KEYS = {
    "backend", "model",
    "optimizer_backend", "optimizer_model", "target_backend", "target_model",
    "azure_endpoint",
    "gate_mode", "gate_metric", "gate_mixed_weight",
    "edit_budget", "holdout_fraction", "lookback_hours",
    "max_tasks_per_night", "max_sessions_per_night", "max_tokens_per_night",
    "dream_rollouts", "dream_factor", "recall_k",
    "evolve_skill", "evolve_memory", "llm_mine", "target_skill_path",
    "preferences", "evidence_log", "evidence_max_chars", "auto_adopt",
    "transcript_source",
}


def _read_json(path: str) -> Optional[Any]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _read_text(path: str, limit: int = 200_000) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read(limit)
    except Exception:
        return ""


def _user_config_file() -> str:
    return os.path.join(HOME_STATE_DIR, "config.json")


def _write_config(updates: Dict[str, Any]) -> Dict[str, Any]:
    path = _user_config_file()
    current = _read_json(path) or {}
    for k, v in updates.items():
        if k not in _EDITABLE_KEYS:
            continue
        if v is None or v == "":
            # empty resets the key to the built-in default
            current.pop(k, None)
        else:
            current[k] = v
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(current, f, ensure_ascii=False, indent=2)
    return current


def _list_nights(project: str) -> List[Dict[str, Any]]:
    root = staging_root(project)
    out: List[Dict[str, Any]] = []
    if not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root), reverse=True):
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
        report = _read_json(os.path.join(d, "report.json")) or {}
        entry = {
            "ts": name,
            "night": report.get("night"),
            "accepted": report.get("accepted"),
            "gate_action": report.get("gate_action", ""),
            "baseline": report.get("baseline_score"),
            "candidate": report.get("candidate_score"),
            "n_tasks": report.get("n_tasks"),
            "n_sessions": report.get("n_sessions"),
            "tokens_used": report.get("tokens_used"),
            "has_report": bool(report),
            "has_evidence": os.path.exists(os.path.join(d, "evidence.jsonl")),
            "has_manifest": os.path.exists(os.path.join(d, "manifest.json")),
            "adopted": os.path.isdir(os.path.join(d, "backup")),
        }
        out.append(entry)
    return out


class _RunState:
    """At most one pipeline subprocess at a time, log tailed to a file."""

    def __init__(self) -> None:
        self.proc: Optional[subprocess.Popen] = None
        self.log_path = ""
        self.mode = ""
        self.lock = threading.Lock()

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, project: str, dry_run: bool) -> Dict[str, Any]:
        with self.lock:
            if self.running():
                return {"ok": False, "error": "a run is already in progress"}
            cfg = load_config(invoked_project=project)
            self.log_path = os.path.join(cfg.state_dir, "dashboard-run.log")
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            self.mode = "dry-run" if dry_run else "run"
            cmd = [sys.executable, "-m", "skillopt_sleep", self.mode,
                   "--project", project, "--progress"]
            log = open(self.log_path, "w", encoding="utf-8")
            no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
            self.proc = subprocess.Popen(
                cmd, stdout=log, stderr=subprocess.STDOUT,
                creationflags=no_window, cwd=project or None,
            )
            return {"ok": True, "mode": self.mode}

    def status(self) -> Dict[str, Any]:
        tail = ""
        if self.log_path:
            text = _read_text(self.log_path)
            tail = text[-6000:]
        rc = None
        if self.proc is not None:
            rc = self.proc.poll()
        return {"running": self.running(), "returncode": rc,
                "mode": self.mode, "tail": tail}


class DashboardHandler(BaseHTTPRequestHandler):
    project: str = ""
    run_state: _RunState

    # ── plumbing ──────────────────────────────────────────────────────────
    def log_message(self, fmt: str, *args: Any) -> None:  # quiet server
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: Any, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _body(self) -> Dict[str, Any]:
        try:
            n = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(n) if n else b"{}"
            obj = json.loads(raw.decode("utf-8") or "{}")
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    # ── GET ───────────────────────────────────────────────────────────────
    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        path = self.path.split("?", 1)[0]
        if path in {"/", "/index.html"}:
            html = _read_text(_HTML_PATH, limit=5_000_000)
            self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/api/overview":
            cfg = load_config(invoked_project=self.project)
            effective = {k: cfg.get(k) for k in sorted(_EDITABLE_KEYS)}
            self._json({
                "project": self.project,
                "config": effective,
                "defaults": {k: DEFAULTS.get(k) for k in sorted(_EDITABLE_KEYS)},
                "config_path": _user_config_file(),
                "prompts": prompt_registry.describe(),
                "prompts_path": prompt_registry.overrides_path(),
                "nights": _list_nights(self.project),
            })
            return
        if path.startswith("/api/night/"):
            ts = os.path.basename(path[len("/api/night/"):])
            d = os.path.join(staging_root(self.project), ts)
            if not os.path.isdir(d):
                self._json({"error": "unknown night"}, 404)
                return
            self._json({
                "ts": ts,
                "dir": d,
                "report": _read_json(os.path.join(d, "report.json")),
                "manifest": _read_json(os.path.join(d, "manifest.json")),
                "diagnostics": _read_json(os.path.join(d, "diagnostics.json")),
                "report_md": _read_text(os.path.join(d, "report.md")),
                "proposed_skill": _read_text(os.path.join(d, "proposed_SKILL.md")),
                "proposed_memory": _read_text(os.path.join(d, "proposed_CLAUDE.md")),
                "evidence": read_events(os.path.join(d, "evidence.jsonl")),
                "adopted": os.path.isdir(os.path.join(d, "backup")),
            })
            return
        if path == "/api/run/status":
            self._json(self.run_state.status())
            return
        self._json({"error": "not found"}, 404)

    # ── POST ──────────────────────────────────────────────────────────────
    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        body = self._body()
        if path == "/api/config":
            updates = body.get("updates") or {}
            if not isinstance(updates, dict):
                self._json({"error": "updates must be an object"}, 400)
                return
            saved = _write_config(updates)
            cfg = load_config(invoked_project=self.project)
            self._json({"ok": True, "saved": saved,
                        "config": {k: cfg.get(k) for k in sorted(_EDITABLE_KEYS)}})
            return
        if path == "/api/prompts":
            updates = body.get("updates") or {}
            if not isinstance(updates, dict):
                self._json({"error": "updates must be an object"}, 400)
                return
            prompt_registry.save_overrides(updates)
            self._json({"ok": True, "prompts": prompt_registry.describe()})
            return
        if path == "/api/run":
            self._json(self.run_state.start(self.project, bool(body.get("dry_run"))))
            return
        if path == "/api/adopt":
            ts = os.path.basename(str(body.get("ts", "")))
            d = os.path.join(staging_root(self.project), ts)
            if not os.path.isdir(d):
                self._json({"error": "unknown night"}, 404)
                return
            try:
                updated = adopt_staging(d)
            except Exception as exc:  # surface, don't crash the server
                self._json({"ok": False, "error": str(exc)}, 500)
                return
            self._json({"ok": True, "updated": updated})
            return
        self._json({"error": "not found"}, 404)


def serve(project: str = "", port: int = 8321, open_browser: bool = True) -> int:
    project = os.path.abspath(project or os.getcwd())
    handler = DashboardHandler
    handler.project = project
    handler.run_state = _RunState()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    print(f"[sleep] dashboard for {project}\n[sleep] serving {url}  (Ctrl+C to stop)")
    if open_browser:
        try:
            import webbrowser
            threading.Timer(0.4, webbrowser.open, args=(url,)).start()
        except Exception:
            pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0
