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

Binding to loopback is not by itself an authorization boundary: any page in
the user's browser can send cross-origin requests to 127.0.0.1, and a hostile
name that resolves to loopback (DNS rebinding) can make them look same-origin.
Since these endpoints start real runs, adopt artifacts, and rewrite config and
prompts, every request is checked before it reaches a handler — see
``_authorize``. No CORS headers are ever emitted, so a foreign origin cannot
read responses either.
"""
from __future__ import annotations

import hmac
import json
import os
import secrets
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote

from skillopt_sleep import prompts as prompt_registry
from skillopt_sleep.config import DEFAULTS, HOME_STATE_DIR, load_config
from skillopt_sleep.evidence import read_events
from skillopt_sleep.staging import adopt as adopt_staging
from skillopt_sleep.staging import staging_root

_HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")

# The served HTML carries the capability token in place of this placeholder,
# so the token never travels in a URL (referrers, shell history, server logs).
_TOKEN_PLACEHOLDER = "__SKILLOPT_DASHBOARD_TOKEN__"
# A custom header cannot be set by a cross-origin HTML form, so requiring it
# forces a preflight that this server deliberately does not answer.
_TOKEN_HEADER = "X-SkillOpt-Dashboard-Token"
_MAX_BODY_BYTES = 1 << 20  # 1 MiB: these payloads are prompts and config, not uploads

# Hostnames that really mean "this machine". A rebound name like evil.com
# resolving to 127.0.0.1 arrives with its own Host and is rejected.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "[::1]"})

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


def _coerce_config_value(key: str, value: Any) -> Any:
    """Coerce a submitted value to the type of the built-in default.

    ``load_config`` does not type-coerce, so a string where a number belongs
    reaches the arithmetic downstream (``edit_budget: "4"``). The default's
    type is the schema — this is the only place that knows it, and it is
    enforced server-side because the client is not the only possible caller.
    """
    default = DEFAULTS.get(key)
    if isinstance(default, bool):  # before int: bool is a subclass of int
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"true", "1", "yes", "on"}:
            return True
        if text in {"false", "0", "no", "off"}:
            return False
        raise ValueError(f"{key} must be a boolean")
    if isinstance(default, int):
        try:
            return int(str(value).strip())
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be an integer") from None
    if isinstance(default, float):
        try:
            return float(str(value).strip())
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be a number") from None
    if isinstance(default, str):
        if isinstance(value, (dict, list)):
            raise ValueError(f"{key} must be a string")
        return str(value)
    return value


def _write_config(updates: Dict[str, Any]) -> Dict[str, Any]:
    """Merge validated updates into the user config file.

    Raises ValueError if any submitted value cannot be coerced; nothing is
    written in that case, so a bad field cannot half-apply a form.
    """
    path = _user_config_file()
    current = _read_json(path) or {}
    accepted: Dict[str, Any] = {}
    removed = []
    for k, v in updates.items():
        if k not in _EDITABLE_KEYS:
            continue
        if v is None or v == "":
            removed.append(k)  # empty resets the key to the built-in default
        else:
            accepted[k] = _coerce_config_value(k, v)
    for k in removed:
        current.pop(k, None)
    current.update(accepted)
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
            mode = "dry-run" if dry_run else "run"
            cmd = [sys.executable, "-m", "skillopt_sleep", mode,
                   "--project", project, "--progress"]
            no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
            try:
                # The child inherits a dup of this descriptor, so the parent
                # closes its own copy immediately rather than leaving the log
                # held open (and, on Windows, locked) for the server's life.
                with open(self.log_path, "w", encoding="utf-8") as log:
                    self.proc = subprocess.Popen(
                        cmd, stdout=log, stderr=subprocess.STDOUT,
                        creationflags=no_window, cwd=project or None,
                    )
            except OSError as exc:
                # A failed spawn must not raise out of the request thread —
                # that would drop the connection and leave the UI hanging.
                self.proc = None
                return {"ok": False, "error": f"could not start {mode}: {exc}"}
            self.mode = mode
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


def _night_dir(project: str, ts: str) -> Optional[str]:
    """Resolve a night id to its staging directory, or None if it is not one.

    ``os.path.basename`` alone is not containment: it leaves ``".."`` intact,
    so ``/api/night/..`` resolved to the staging parent and ``/api/adopt``
    would have copied whatever it found there over the live SKILL.md and
    CLAUDE.md. Reject anything that is not a plain single path component, then
    confirm the *resolved* path is still under the staging root so a symlink
    planted in staging cannot redirect the read either.
    """
    name = str(ts or "")
    if not name or name in {".", ".."}:
        return None
    if name != os.path.basename(name):  # separators, drive letters, absolutes
        return None
    if os.sep in name or (os.altsep and os.altsep in name):
        return None

    root = staging_root(project)
    candidate = os.path.join(root, name)
    try:
        real_root = os.path.realpath(root)
        real_candidate = os.path.realpath(candidate)
        if os.path.commonpath([real_root, real_candidate]) != real_root:
            return None
        if real_candidate == real_root:
            return None
    except (OSError, ValueError):  # unrelated roots / different drives
        return None
    if not os.path.isdir(real_candidate):
        return None
    return candidate


def _split_host(value: str) -> Tuple[str, str]:
    """Split a Host header into (hostname, port), tolerating IPv6 brackets."""
    host = (value or "").strip()
    if host.startswith("["):  # [::1]:8321
        close = host.find("]")
        if close < 0:
            return "", ""
        return host[:close + 1].lower(), host[close + 2:] if host[close + 1:close + 2] == ":" else ""
    if host.count(":") > 1:  # bare IPv6 with no brackets
        return host.lower(), ""
    name, _, port = host.partition(":")
    return name.lower(), port


def _is_loopback_host(value: str, port: int) -> bool:
    name, host_port = _split_host(value)
    if name not in _LOOPBACK_HOSTS:
        return False
    # A Host with no port means the default port for the scheme, which is
    # never the port this server was given.
    return host_port == str(port)


def _allowed_origins(port: int) -> frozenset:
    return frozenset(
        f"http://{host}:{port}"
        for host in ("127.0.0.1", "localhost", "[::1]")
    )


class DashboardHandler(BaseHTTPRequestHandler):
    project: str = ""
    run_state: _RunState
    token: str = ""
    port: int = 0

    # ── plumbing ──────────────────────────────────────────────────────────
    def log_message(self, fmt: str, *args: Any) -> None:  # quiet server
        pass

    # ── request authorization ─────────────────────────────────────────────
    def _authorize(self, *, mutating: bool) -> bool:
        """Gate one request. Returns False after writing the error response.

        Loopback ``Host`` is required on every request, which is what defeats
        DNS rebinding: the browser sends the attacker's name, not ours. State
        changing requests additionally need a trusted ``Origin``, a JSON
        content type, and the per-process capability token — each of which a
        cross-origin page is independently unable to produce.
        """
        if not _is_loopback_host(self.headers.get("Host", ""), self.port):
            self._json({"error": "forbidden: bad host"}, 403)
            return False
        if not mutating:
            return True

        origin = self.headers.get("Origin")
        if not origin or origin not in _allowed_origins(self.port):
            # Missing Origin is rejected too: it is not evidence of a
            # same-origin caller, and the real page always sends one.
            self._json({"error": "forbidden: bad origin"}, 403)
            return False

        ctype = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
        if ctype != "application/json":
            # Form and text/plain bodies are exactly the shapes a cross-origin
            # <form> can send without a preflight.
            self._json({"error": "unsupported media type: expected application/json"}, 415)
            return False

        presented = self.headers.get(_TOKEN_HEADER) or ""
        if not self.token or not hmac.compare_digest(presented, self.token):
            self._json({"error": "forbidden: bad token"}, 403)
            return False
        return True

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

    def _drain(self, limit: int) -> None:
        """Discard up to ``limit`` bytes of an unwanted request body."""
        remaining = limit
        while remaining > 0:
            chunk = self.rfile.read(min(65536, remaining))
            if not chunk:
                return
            remaining -= len(chunk)

    def _body(self) -> Optional[Dict[str, Any]]:
        """Parse a required JSON object body, or write a 4xx and return None.

        An absent, empty, oversized or non-object body is an error rather than
        an empty dict: treating ``{}`` as "no arguments supplied" is what let a
        contentless POST to /api/run start a real run.
        """
        try:
            declared = int(self.headers.get("Content-Length", "") or "")
        except ValueError:
            self._json({"error": "invalid Content-Length"}, 400)
            return None
        if declared <= 0:
            self._json({"error": "request body required"}, 400)
            return None
        if declared > _MAX_BODY_BYTES:
            # Drain a bounded amount first so a well-behaved client can read
            # the 413 rather than seeing a connection reset mid-upload, then
            # hang up instead of consuming an unbounded stream.
            self._drain(min(declared, _MAX_BODY_BYTES * 2))
            self.close_connection = True
            self._json({"error": "request body too large"}, 413)
            return None

        raw = self.rfile.read(declared)
        if len(raw) != declared:
            self._json({"error": "truncated request body"}, 400)
            return None
        try:
            obj = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            self._json({"error": "malformed JSON body"}, 400)
            return None
        if not isinstance(obj, dict):
            self._json({"error": "request body must be a JSON object"}, 400)
            return None
        return obj

    # ── GET ───────────────────────────────────────────────────────────────
    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        if not self._authorize(mutating=False):
            return
        path = self.path.split("?", 1)[0]
        if path in {"/", "/index.html"}:
            html = _read_text(_HTML_PATH, limit=5_000_000)
            # Hand the page its capability token. Only a same-origin document
            # can read this body, so only it can make mutating calls.
            html = html.replace(_TOKEN_PLACEHOLDER, self.token)
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
            ts = unquote(path[len("/api/night/"):])
            d = _night_dir(self.project, ts)
            if d is None:
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
        if not self._authorize(mutating=True):
            return
        path = self.path.split("?", 1)[0]
        body = self._body()
        if body is None:  # _body already answered with the specific 4xx
            return
        if path == "/api/config":
            updates = body.get("updates") or {}
            if not isinstance(updates, dict):
                self._json({"error": "updates must be an object"}, 400)
                return
            try:
                saved = _write_config(updates)
            except ValueError as exc:
                self._json({"error": str(exc)}, 400)
                return
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
            d = _night_dir(self.project, body.get("ts", ""))
            if d is None:
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


def make_server(project: str = "", port: int = 8321) -> ThreadingHTTPServer:
    """Bind a dashboard server on loopback with a fresh capability token.

    The handler is a per-server subclass so its token and port cannot leak
    between servers (the tests run several), and so the token is scoped to one
    process lifetime: restarting the dashboard invalidates every old page.
    """
    project = os.path.abspath(project or os.getcwd())
    token = secrets.token_urlsafe(32)

    class _Handler(DashboardHandler):
        pass

    _Handler.project = project
    _Handler.run_state = _RunState()
    _Handler.token = token
    httpd = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    # Bound port, which may be ephemeral when port=0 was requested. Host and
    # Origin are validated against this exact value.
    _Handler.port = httpd.server_address[1]
    return httpd


def serve(project: str = "", port: int = 8321, open_browser: bool = True) -> int:
    project = os.path.abspath(project or os.getcwd())
    httpd = make_server(project, port)
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
