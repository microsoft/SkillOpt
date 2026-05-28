"""
SkillOpt WebUI — Configure, launch, and monitor training from your browser.

Usage:
    python -m skillopt_webui.app [--port PORT] [--share]
"""
import argparse
import glob
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import gradio as gr
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ─── Config helpers ──────────────────────────────────────────────────────────

def discover_configs() -> list[str]:
    """Find all YAML configs under configs/."""
    pattern = str(PROJECT_ROOT / "configs" / "**" / "*.yaml")
    paths = sorted(glob.glob(pattern, recursive=True))
    return [os.path.relpath(p, PROJECT_ROOT) for p in paths
            if "_base_" not in p]


def load_config(path: str) -> dict:
    """Load a YAML config file."""
    with open(PROJECT_ROOT / path) as f:
        return yaml.safe_load(f)


def config_to_display(cfg: dict) -> str:
    """Pretty-print config for display."""
    return yaml.dump(cfg, default_flow_style=False, sort_keys=False)


# ─── Training process management ────────────────────────────────────────────

class TrainingManager:
    """Manages a single training subprocess."""

    def __init__(self):
        self._lock = threading.Lock()
        self.process = None
        self.log_lines: list[str] = []
        self.stage = "Idle"
        self.step = 0
        self.total_steps = 0
        self.epoch = 0
        self.total_epochs = 0
        self.running = False

    def start(self, config_path: str, overrides: dict) -> str:
        with self._lock:
            if self.running:
                return "⚠️ Training already running. Stop it first."

        cmd = [
            sys.executable, "scripts/train.py",
            "--config", config_path,
        ]
        cfg_options = []
        for k, v in overrides.items():
            if v is not None and v != "":
                cfg_options.append(f"{k}={v}")
        if cfg_options:
            cmd.append("--cfg-options")
            cmd.extend(cfg_options)

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        # Auto-load API credentials from .secrets/*.env
        secrets_dir = PROJECT_ROOT / ".secrets"
        if secrets_dir.is_dir():
            for env_file in sorted(secrets_dir.glob("*.env")):
                for line in env_file.read_text().splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        env[k] = v
        # Propagate OPTIMIZER_* to base AZURE_OPENAI_* when base is missing,
        # so target/default endpoints inherit from optimizer config.
        _propagate = [
            ("ENDPOINT", ""), ("API_VERSION", ""), ("AUTH_MODE", ""),
            ("MANAGED_IDENTITY_CLIENT_ID", ""), ("AD_SCOPE", ""),
            ("API_KEY", ""),
        ]
        for suffix, _ in _propagate:
            base_key = f"AZURE_OPENAI_{suffix}"
            optimizer_key = f"OPTIMIZER_AZURE_OPENAI_{suffix}"
            if not env.get(base_key) and env.get(optimizer_key):
                env[base_key] = env[optimizer_key]

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(PROJECT_ROOT),
                bufsize=1,
                env=env,
                start_new_session=True,  # create process group for clean kill
            )
        except Exception as e:
            return f"❌ Failed to start training: {e}"

        with self._lock:
            self.process = proc
            self.log_lines = [f"$ {' '.join(cmd)}\n"]
            self.stage = "Starting"
            self.step = 0
            self.total_steps = 0
            self.epoch = 0
            self.total_epochs = 0
            self.running = True

        thread = threading.Thread(target=self._read_output, daemon=True)
        thread.start()

        return "✅ Training started!"

    def _read_output(self):
        for line in self.process.stdout:
            with self._lock:
                self.log_lines.append(line)
                self._parse_stage(line)
                if len(self.log_lines) > 5000:
                    self.log_lines = self.log_lines[-4000:]
        self.process.wait()
        with self._lock:
            self.running = False
            self.stage = f"Finished (exit={self.process.returncode})"

    def _parse_stage(self, line: str):
        line_lower = line.lower()
        if "1/6 rollout" in line_lower or ("rollout" in line_lower and "worker" in line_lower):
            self.stage = "🎯 Rollout"
        elif "2/6 reflect" in line_lower or ("reflect" in line_lower and "patch" in line_lower):
            self.stage = "🔍 Reflect"
        elif "3/6 aggregate" in line_lower or "merge" in line_lower:
            self.stage = "🔗 Aggregate"
        elif "4/6 select" in line_lower:
            self.stage = "✂️ Select"
        elif "5/6 update" in line_lower:
            self.stage = "📝 Update"
        elif "6/6" in line_lower or ("gate" in line_lower and "score" in line_lower):
            self.stage = "🚦 Gate"
        elif "slow update" in line_lower:
            self.stage = "🔄 Slow Update"
        elif "meta skill" in line_lower:
            self.stage = "🧠 Meta Skill"
        elif "baseline" in line_lower and "evaluate" in line_lower:
            self.stage = "📊 Baseline"
        if "[step" in line_lower:
            try:
                parts = line.split("[STEP")[1].split("]")[0].split("/")
                self.step = int(parts[0].strip())
                self.total_steps = int(parts[1].strip())
            except (IndexError, ValueError):
                pass
        if "[epoch" in line_lower:
            try:
                parts = line.split("[EPOCH")[1].split("]")[0].split("/")
                self.epoch = int(parts[0].strip())
                self.total_epochs = int(parts[1].strip())
            except (IndexError, ValueError):
                pass

    def stop(self) -> str:
        with self._lock:
            if self.process and self.running:
                try:
                    # Kill entire process group (children included)
                    os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                except (ProcessLookupError, OSError):
                    self.process.terminate()
                self.process.wait(timeout=5)
                self.running = False
                self.stage = "Stopped"
                return "🛑 Training stopped."
            return "No training running."

    def get_logs(self) -> str:
        with self._lock:
            return "".join(self.log_lines[-500:])

    def get_colored_logs_html(self) -> str:
        """Render last 300 log lines with color-coded stages."""
        import html as html_mod
        with self._lock:
            lines = list(self.log_lines[-300:])
        parts = []
        for line in lines:
            # Rebrand: display "skillopt" instead of "reflact" in logs
            line_display = line.replace("reflact", "skillopt").replace("ReflACT", "SkillOpt").replace("Reflact", "Skillopt").replace("REFLACT", "SKILLOPT")
            escaped = html_mod.escape(line_display.rstrip("\n"))
            low = line.lower()
            if "[epoch" in low:
                color = "#f59e0b"  # amber
                weight = "700"
            elif "[step" in low:
                color = "#8b5cf6"  # purple
                weight = "700"
            elif "rollout]" in low or "1/6" in low:
                color = "#3b82f6"  # blue
            elif "reflect" in low or "2/6" in low:
                color = "#f97316"  # orange
            elif "aggregate" in low or "3/6" in low or "merge" in low:
                color = "#06b6d4"  # cyan
            elif "select" in low or "4/6" in low:
                color = "#ec4899"  # pink
            elif "update" in low or "5/6" in low:
                color = "#10b981"  # green
            elif "gate" in low or "6/6" in low:
                color = "#ef4444"  # red
            elif "slow update" in low:
                color = "#f59e0b"  # amber
                weight = "700"
            elif "meta skill" in low:
                color = "#a855f7"  # violet
                weight = "700"
            elif "baseline" in low:
                color = "#6366f1"  # indigo
                weight = "700"
            elif "[rollout]" in low:
                # per-item rollout progress
                if "hard=1" in line:
                    color = "#22c55e"  # green for correct
                elif "hard=0" in line:
                    color = "#f87171"  # red for wrong
                elif "timeout" in low:
                    color = "#fbbf24"  # yellow for timeout
                else:
                    color = "#94a3b8"  # gray
                weight = "400"
            elif "error" in low or "fail" in low:
                color = "#ef4444"
                weight = "700"
            elif "========" in line:
                color = "#64748b"  # separator
                weight = "400"
            else:
                color = "#e2e8f0"  # default light gray
                weight = "400"
            if "weight" not in dir():
                weight = "400"
            parts.append(f'<span style="color:{color};font-weight:{weight}">{escaped}</span>')
            weight = "400"  # reset

        log_html = "<br>".join(parts) if parts else '<span style="color:#94a3b8">No logs yet. Click Refresh after launching training.</span>'
        return f'''<div id="log-container" style="
            height:500px;overflow-y:auto;background:#0f172a;padding:16px;
            border-radius:10px;font-family:'JetBrains Mono',Consolas,monospace;
            font-size:12.5px;line-height:1.6;border:1px solid #1e293b;
            box-shadow:inset 0 2px 4px rgba(0,0,0,0.3);">{log_html}</div>'''

    def get_progress_html(self) -> str:
        """Render a visual progress bar."""
        s = self.get_status()
        step = s["step"]
        total = s["total_steps"]
        epoch = self.epoch
        total_epochs = self.total_epochs
        pct = s["progress"] * 100

        if not self.running and step == 0:
            return '<div style="color:#94a3b8;text-align:center;padding:12px;">Waiting for training to start...</div>'

        # Color based on progress
        if pct < 25:
            bar_color = "linear-gradient(90deg, #3b82f6, #6366f1)"
        elif pct < 50:
            bar_color = "linear-gradient(90deg, #6366f1, #8b5cf6)"
        elif pct < 75:
            bar_color = "linear-gradient(90deg, #8b5cf6, #a855f7)"
        else:
            bar_color = "linear-gradient(90deg, #a855f7, #22c55e)"

        stage_icon = self.stage if self.stage != "Idle" else "⏳"
        status_dot = "🟢" if self.running else ("✅" if "Finished" in self.stage else "⚪")

        epoch_str = f"Epoch {epoch}/{total_epochs}" if total_epochs > 0 else ""
        step_str = f"Step {step}/{total}" if total > 0 else ""

        return f'''
        <div style="background:#1e293b;border-radius:12px;padding:16px;border:1px solid #334155;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
            <span style="color:#e2e8f0;font-weight:700;font-size:1rem;">{status_dot} {stage_icon}</span>
            <span style="color:#94a3b8;font-size:0.9rem;">{epoch_str} &nbsp; {step_str}</span>
            <span style="color:#e2e8f0;font-weight:700;font-size:1rem;">{pct:.1f}%</span>
          </div>
          <div style="background:#0f172a;border-radius:8px;height:20px;overflow:hidden;border:1px solid #334155;">
            <div style="height:100%;width:{pct}%;background:{bar_color};
                        border-radius:8px;transition:width 0.5s ease;
                        box-shadow:0 0 12px rgba(99,102,241,0.4);"></div>
          </div>
        </div>'''

    def get_status(self) -> dict:
        with self._lock:
            progress = 0
            if self.total_steps > 0:
                progress = self.step / self.total_steps
            return {
                "running": self.running,
                "stage": self.stage,
                "step": self.step,
                "total_steps": self.total_steps,
                "progress": progress,
            }


manager = TrainingManager()


# ─── Human Review helpers ───────────────────────────────────────────────────

def _outputs_root() -> Path:
    return PROJECT_ROOT / "outputs"


def discover_runs() -> list[str]:
    """Return run directories (relative to PROJECT_ROOT) that contain a
    ``human_review/`` folder OR look like an active SkillOpt run."""
    root = _outputs_root()
    if not root.exists():
        return []
    runs: list[str] = []
    for run_dir in sorted(root.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        # Either has a human_review subdir, or has the standard run layout
        if (run_dir / "human_review").exists() or (run_dir / "history.json").exists() or (run_dir / "skills").exists():
            runs.append(str(run_dir.relative_to(PROJECT_ROOT)).replace("\\", "/"))
    return runs


def _find_pending_review(run_rel: str) -> tuple[Path | None, dict | None]:
    """Find the oldest pending_review.json under ``{run}/human_review/`` that
    does not yet have a paired response file. Returns (path, parsed_json)."""
    if not run_rel:
        return None, None
    run_dir = PROJECT_ROOT / run_rel
    review_root = run_dir / "human_review"
    if not review_root.exists():
        return None, None
    candidates: list[tuple[float, Path]] = []
    for step_dir in review_root.iterdir():
        if not step_dir.is_dir():
            continue
        pending = step_dir / "pending_review.json"
        response = step_dir / "pending_review_response.json"
        if pending.exists() and not response.exists():
            candidates.append((pending.stat().st_mtime, pending))
    if not candidates:
        return None, None
    candidates.sort()
    path = candidates[0][1]
    try:
        with open(path, "r", encoding="utf-8") as f:
            return path, json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return path, {"_error": f"failed to read {path}: {exc}"}


def _format_edit_choices(ranked_edits: list[dict]) -> list[tuple[str, int]]:
    """Build (label, index) pairs for a CheckboxGroup over ranked edits."""
    choices = []
    for i, e in enumerate(ranked_edits):
        if not isinstance(e, dict):
            continue
        if "op" in e:  # patch-mode edit
            op = e.get("op", "?")
            target = (e.get("target") or "")[:60]
            content = (e.get("content") or "")[:80]
            label = f"[{i}] {op}"
            if target:
                label += f' target="{target}"'
            if content:
                label += f' content="{content}"'
        elif "title" in e and "instruction" in e:  # rewrite-mode suggestion
            label = f'[{i}] {e.get("type", "?")}: {e.get("title", "")[:60]}'
        else:  # full-rewrite candidate
            label = f'[{i}] {e.get("title", "")[:60]} ({e.get("source_type", "")})'
        choices.append((label, i))
    return choices


def _write_response(run_rel: str, step: int, payload: dict) -> str:
    """Write the response file next to the matching pending_review.json."""
    if not run_rel or step is None:
        return "⚠️ No pending review selected."
    resp_path = (
        PROJECT_ROOT / run_rel / "human_review" / f"step_{int(step):04d}"
        / "pending_review_response.json"
    )
    resp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(resp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return f"✅ Submitted {payload.get('action', '?')} for step {step}."


# ─── Pipeline Stage HTML ────────────────────────────────────────────────────

STAGES = ["Rollout", "Reflect", "Aggregate", "Select", "Update", "Gate"]
STAGE_ICONS = ["🎯", "🔍", "🔗", "✂️", "📝", "🚦"]


def render_pipeline_html(active_stage: str = "") -> str:
    """Render animated pipeline HTML."""
    html = '<div style="display:flex;align-items:center;justify-content:center;gap:4px;padding:20px;flex-wrap:wrap;">'
    for i, (name, icon) in enumerate(zip(STAGES, STAGE_ICONS)):
        is_active = name.lower() in active_stage.lower() if active_stage else False
        bg = "#6366f1" if is_active else "#f3f4f6"
        color = "white" if is_active else "#374151"
        border = "3px solid #4f46e5" if is_active else "2px solid #d1d5db"
        shadow = "0 0 20px rgba(99,102,241,0.4)" if is_active else "none"
        pulse = "animation: pulse 1.5s ease-in-out infinite;" if is_active else ""
        html += f'''
        <div style="display:flex;flex-direction:column;align-items:center;padding:12px 16px;
                    border-radius:12px;background:{bg};color:{color};border:{border};
                    min-width:80px;box-shadow:{shadow};transition:all 0.3s;{pulse}">
          <span style="font-size:1.5rem">{icon}</span>
          <span style="font-weight:700;font-size:0.85rem;margin-top:4px">{name}</span>
        </div>'''
        if i < len(STAGES) - 1:
            arrow_color = "#6366f1" if is_active else "#d1d5db"
            html += f'<div style="font-size:1.2rem;color:{arrow_color}">→</div>'
    html += '</div>'
    html += '<style>@keyframes pulse{0%,100%{transform:scale(1)}50%{transform:scale(1.05)}}</style>'
    return html


# ─── Human-review helpers ───────────────────────────────────────────────────

def _discover_runs() -> list[str]:
    """Find run directories under outputs/ that contain a human_review folder."""
    candidates = sorted(
        glob.glob(str(PROJECT_ROOT / "outputs" / "*")),
        key=os.path.getmtime,
        reverse=True,
    )
    runs = []
    for p in candidates:
        if os.path.isdir(p):
            runs.append(os.path.relpath(p, PROJECT_ROOT))
    return runs


def _find_pending_review(run_dir: str) -> tuple[str | None, dict | None]:
    """Return (request_path, parsed_request) for the latest pending review."""
    if not run_dir:
        return None, None
    abs_run = run_dir if os.path.isabs(run_dir) else str(PROJECT_ROOT / run_dir)
    pattern = os.path.join(abs_run, "human_review", "step_*", "pending_review.json")
    matches = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not matches:
        return None, None
    latest = matches[-1]
    try:
        with open(latest, encoding="utf-8") as f:
            return latest, json.load(f)
    except Exception:
        return latest, None


def _edit_label(idx: int, edit: dict) -> str:
    op = edit.get("op") or edit.get("type") or "?"
    target = (edit.get("target") or "")[:50]
    content = (
        edit.get("content") or edit.get("instruction") or edit.get("title") or ""
    )[:80]
    tgt_part = f' target="{target}"' if target else ""
    return f"[{idx}] {op}{tgt_part} -> {content!r}"


def _render_review_status(req_path: str | None, req: dict | None) -> str:
    if not req_path:
        return (
            '<div style="padding:12px;background:#1e293b;border-radius:8px;'
            'color:#94a3b8;border:1px solid #334155;">'
            "No pending review. The panel polls every 2 seconds — when the "
            "trainer reaches the gate at the next step, the request will "
            "appear here.</div>"
        )
    if not req:
        return (
            f'<div style="padding:12px;background:#7f1d1d;border-radius:8px;'
            f'color:#fecaca;border:1px solid #b91c1c;">'
            f"Found request at {req_path} but failed to parse JSON.</div>"
        )
    step = req.get("step", "?")
    cs = req.get("candidate_score", 0.0)
    curs = req.get("current_score", 0.0)
    bs = req.get("best_score", 0.0)
    bstep = req.get("best_step", 0)
    arrow = "↑" if cs > curs else ("=" if cs == curs else "↓")
    arrow_color = "#22c55e" if cs > curs else ("#94a3b8" if cs == curs else "#ef4444")
    retry = req.get("retry_attempt", 0)
    max_r = req.get("max_retries", 0)
    retry_str = (
        f' &nbsp;|&nbsp; <span style="color:#fbbf24;">retry {retry}/{max_r}</span>'
        if retry > 0
        else ""
    )
    return (
        f'<div style="padding:14px;background:#1e293b;border-radius:8px;'
        f'color:#e2e8f0;border:1px solid #6366f1;font-size:1rem;">'
        f"<b>Awaiting review — step {step}</b>{retry_str}<br>"
        f"<span style='color:#94a3b8;'>current</span> {curs:.4f} "
        f"<span style='color:{arrow_color};font-weight:700;'>{arrow}</span> "
        f"<span style='color:#e2e8f0;'>candidate</span> "
        f"<b style='color:{arrow_color};'>{cs:.4f}</b> &nbsp;|&nbsp; "
        f"<span style='color:#94a3b8;'>best</span> {bs:.4f} (step {bstep})"
        f"</div>"
    )


def _refresh_review(run_dir: str):
    """Read pending review for the run and return tuple of component values."""
    req_path, req = _find_pending_review(run_dir)
    status_html = _render_review_status(req_path, req)
    if not req:
        return (
            status_html, "", "", gr.update(choices=[], value=[]),
            "", req_path or "",
        )
    current_skill = req.get("current_skill", "")
    candidate_skill = req.get("candidate_skill", "")
    edits = req.get("ranked_edits") or []
    labels = [_edit_label(i, e) for i, e in enumerate(edits)]
    return (
        status_html,
        current_skill,
        candidate_skill,
        gr.update(choices=labels, value=labels),  # all selected by default
        "",  # clear critique
        req_path or "",
    )


def _submit_review(
    action: str,
    run_dir: str,
    req_path: str,
    edited_skill: str,
    selected_labels: list[str],
    critique: str,
) -> str:
    """Write pending_review_response.json. Returns status message for the UI."""
    if not req_path:
        return "❌ No pending review to respond to."
    if not os.path.exists(req_path):
        return f"❌ Request file vanished: {req_path}"
    try:
        with open(req_path, encoding="utf-8") as f:
            req = json.load(f)
    except Exception as exc:
        return f"❌ Could not re-read request: {exc}"

    response: dict = {"action": action, "critique": critique or ""}

    # Only attach edited_skill if user actually changed it
    if edited_skill and edited_skill != req.get("candidate_skill", ""):
        response["edited_skill"] = edited_skill

    if action == "apply_selected_edits":
        edits = req.get("ranked_edits") or []
        all_labels = [_edit_label(i, e) for i, e in enumerate(edits)]
        indices = [i for i, lbl in enumerate(all_labels) if lbl in (selected_labels or [])]
        response["selected_edit_indices"] = indices

    resp_path = req_path.replace("pending_review.json", "pending_review_response.json")
    try:
        with open(resp_path, "w", encoding="utf-8") as f:
            json.dump(response, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        return f"❌ Failed to write response: {exc}"

    return f"✅ Sent {action!r} → trainer will resume in ~1s. ({resp_path})"


# ─── Gradio UI ──────────────────────────────────────────────────────────────

def build_ui():
    configs = discover_configs()

    with gr.Blocks(
        title="SkillOpt WebUI",
    ) as app:
        gr.Markdown("# 🧠 SkillOpt Training Dashboard")
        gr.Markdown("*SKILLOPT: Executive Strategy for Self-Evolving Agent Skills — Configure, launch, and monitor training.*")

        with gr.Tabs():
            # ── Tab 1: Configure & Launch ────────────────────────────
            with gr.Tab("⚙️ Configure & Launch"):
                with gr.Row():
                    with gr.Column(scale=1):
                        config_dropdown = gr.Dropdown(
                            choices=configs,
                            label="Config File",
                            value=configs[0] if configs else None,
                        )
                        config_preview = gr.Code(
                            label="Config Preview",
                            language="yaml",
                            interactive=False,
                        )

                    with gr.Column(scale=1):
                        gr.Markdown("### Hyperparameters (DL Analogy)")
                        lr = gr.Slider(1, 32, value=4, step=1,
                                       label="Learning Rate (max edits/step)")
                        scheduler = gr.Dropdown(
                            ["cosine", "linear", "constant", "autonomous"],
                            value="cosine",
                            label="LR Scheduler",
                        )
                        num_epochs = gr.Slider(1, 8, value=4, step=1,
                                               label="Epochs")
                        batch_size = gr.Slider(10, 100, value=40, step=5,
                                               label="Batch Size (tasks per step)")
                        analyst_workers = gr.Slider(1, 32, value=16, step=1,
                                                    label="Analyst Workers (parallel reflection)")
                        use_slow_update = gr.Checkbox(value=True,
                                                       label="Slow Update (epoch-boundary momentum)")
                        use_meta_skill = gr.Checkbox(value=True,
                                                      label="Meta Skill (cross-epoch optimizer memory)")
                        use_gate = gr.Checkbox(value=True,
                                                label="Gate (validation-based accept/reject)")
                        human_feedback_enabled = gr.Checkbox(
                            value=False,
                            label="Human Feedback (pause at each gate for review in the Human Review tab)",
                        )

                        with gr.Row():
                            launch_btn = gr.Button("🚀 Launch Training",
                                                    variant="primary", size="lg")
                            stop_btn = gr.Button("🛑 Stop", variant="stop")

                        status_text = gr.Textbox(label="Status", interactive=False)

                def on_config_change(path):
                    if path:
                        try:
                            return config_to_display(load_config(path))
                        except Exception as e:
                            return f"Error: {e}"
                    return ""

                config_dropdown.change(on_config_change, config_dropdown, config_preview)

                def on_launch(cfg_path, lr_val, sched, epochs, batch, workers,
                              slow_update, meta_skill, gate, human_fb):
                    overrides = {
                        "optimizer.learning_rate": lr_val,
                        "optimizer.lr_scheduler": sched,
                        "train.num_epochs": epochs,
                        "train.batch_size": batch,
                        "gradient.analyst_workers": workers,
                        "optimizer.use_slow_update": slow_update,
                        "optimizer.use_meta_skill": meta_skill,
                        "evaluation.use_gate": gate,
                        "human_feedback.enabled": human_fb,
                    }
                    return manager.start(cfg_path, overrides)

                launch_btn.click(
                    on_launch,
                    [config_dropdown, lr, scheduler, num_epochs, batch_size,
                     analyst_workers, use_slow_update, use_meta_skill, use_gate,
                     human_feedback_enabled],
                    status_text,
                )
                stop_btn.click(lambda: manager.stop(), outputs=status_text)

            # ── Tab 2: Monitor ───────────────────────────────────────
            with gr.Tab("📊 Monitor"):
                pipeline_html = gr.HTML(
                    value=render_pipeline_html(),
                    label="Pipeline Stage",
                )

                progress_html = gr.HTML(
                    value=manager.get_progress_html(),
                    label="Progress",
                )

                log_html = gr.HTML(
                    value=manager.get_colored_logs_html(),
                    label="Training Logs",
                )

                refresh_btn = gr.Button("🔄 Refresh Logs", variant="primary", size="lg")

                def on_refresh():
                    s = manager.get_status()
                    pipeline = render_pipeline_html(s["stage"])
                    progress = manager.get_progress_html()
                    logs = manager.get_colored_logs_html()
                    return pipeline, progress, logs

                refresh_btn.click(
                    on_refresh,
                    outputs=[pipeline_html, progress_html, log_html],
                )

            # ── Tab 3: Results ───────────────────────────────────────
            with gr.Tab("📈 Results"):
                gr.Markdown("### Output Explorer")
                output_dir = gr.Textbox(
                    label="Output Directory",
                    value="outputs/",
                    interactive=True,
                )
                scan_btn = gr.Button("🔍 Scan Results")
                results_table = gr.Dataframe(
                    headers=["Experiment", "Benchmark", "Best Score", "Steps"],
                    label="Experiments",
                )

                def scan_outputs(out_dir):
                    rows = []
                    base = PROJECT_ROOT / out_dir
                    if not base.exists():
                        return rows
                    for bench_dir in sorted(base.iterdir()):
                        if not bench_dir.is_dir():
                            continue
                        for run_dir in sorted(bench_dir.iterdir()):
                            if not run_dir.is_dir():
                                continue
                            cfg_file = run_dir / "config.yaml"
                            score = "—"
                            steps = "—"
                            if cfg_file.exists():
                                try:
                                    c = yaml.safe_load(cfg_file.read_text())
                                    steps = str(c.get("train", {}).get("num_steps", "—"))
                                except Exception:
                                    pass
                            # Try to find best score from logs
                            for log_f in run_dir.glob("**/*.jsonl"):
                                try:
                                    with open(log_f) as f:
                                        for line in f:
                                            d = json.loads(line)
                                            if "score" in d:
                                                score = f"{d['score']:.4f}"
                                except Exception:
                                    pass
                            rows.append([
                                run_dir.name,
                                bench_dir.name,
                                score,
                                steps,
                            ])
                    return rows

                scan_btn.click(scan_outputs, output_dir, results_table)

            # ── Tab 4: Human Review ──────────────────────────────────
            with gr.Tab("🧑‍⚖️ Human Review"):
                gr.Markdown(
                    "### Review pending skill candidates from a paused training run\n"
                    "When training runs with `human_feedback.enabled=true`, the "
                    "trainer pauses at each gate and writes a request file here. "
                    "Pick the run below, then choose an action. The panel "
                    "auto-refreshes every 2 seconds."
                )

                with gr.Row():
                    review_run_dir = gr.Dropdown(
                        choices=_discover_runs(),
                        label="Run directory (most recent first)",
                        value=(_discover_runs() or [""])[0],
                        allow_custom_value=True,
                        scale=4,
                    )
                    review_rescan_btn = gr.Button("🔄 Rescan runs", scale=1)

                review_status = gr.HTML(value=_render_review_status(None, None))

                with gr.Row():
                    with gr.Column(scale=1):
                        review_current = gr.Code(
                            label="Current skill (read-only)",
                            language="markdown",
                            interactive=False,
                            lines=25,
                        )
                    with gr.Column(scale=1):
                        review_candidate = gr.Code(
                            label="Candidate skill (editable — your edits will be applied if you Accept)",
                            language="markdown",
                            interactive=True,
                            lines=25,
                        )

                review_edits = gr.CheckboxGroup(
                    choices=[],
                    label="Ranked edits — uncheck any you want to drop, then click 'Apply selected edits & re-evaluate'",
                )

                review_critique = gr.Textbox(
                    label="Critique (free-form; flows into the next step's optimizer prompt)",
                    placeholder="e.g. 'The replace target was too broad — narrow it to just the action-selection paragraph.'",
                    lines=4,
                )

                with gr.Row():
                    btn_accept = gr.Button("✅ Accept", variant="primary")
                    btn_accept_new_best = gr.Button("🏆 Accept as new best", variant="primary")
                    btn_reject = gr.Button("❌ Reject", variant="stop")
                    btn_apply_selected = gr.Button("✂️ Apply selected edits & re-evaluate")
                    btn_retry = gr.Button("🔁 Retry")

                review_result = gr.Textbox(label="Last action", interactive=False)

                # Hidden state holding the current pending request path
                req_path_state = gr.Textbox(value="", visible=False)

                review_outputs = [
                    review_status, review_current, review_candidate,
                    review_edits, review_critique, req_path_state,
                ]

                # Manual refresh on run change
                review_run_dir.change(
                    _refresh_review, review_run_dir, review_outputs,
                )
                review_rescan_btn.click(
                    lambda: gr.update(choices=_discover_runs()),
                    outputs=review_run_dir,
                )

                # Auto-poll every 2 seconds
                review_timer = gr.Timer(2.0)
                review_timer.tick(_refresh_review, review_run_dir, review_outputs)

                # Wire buttons — each captures the action name and reuses the same handler
                review_inputs = [
                    review_run_dir, req_path_state, review_candidate,
                    review_edits, review_critique,
                ]
                btn_accept.click(
                    lambda *a: _submit_review("accept", *a),
                    review_inputs, review_result,
                )
                btn_accept_new_best.click(
                    lambda *a: _submit_review("accept_new_best", *a),
                    review_inputs, review_result,
                )
                btn_reject.click(
                    lambda *a: _submit_review("reject", *a),
                    review_inputs, review_result,
                )
                btn_apply_selected.click(
                    lambda *a: _submit_review("apply_selected_edits", *a),
                    review_inputs, review_result,
                )
                btn_retry.click(
                    lambda *a: _submit_review("retry", *a),
                    review_inputs, review_result,
                )

    return app


def main():
    parser = argparse.ArgumentParser(description="SkillOpt WebUI")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Server host. Use 0.0.0.0 for public access.")
    args = parser.parse_args()

    app = build_ui()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(primary_hue="indigo"),
    )


if __name__ == "__main__":
    main()
