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
import socket
import subprocess
import sys
import threading
from pathlib import Path
from urllib.parse import urlparse

import gradio as gr
import yaml

from skillopt.config import flatten_config
from skillopt.config import load_config as load_merged_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _ensure_under_project(path: Path) -> Path:
    """Resolve *path* and fail closed unless it stays under PROJECT_ROOT."""
    resolved = path.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        raise ValueError(f"Path escapes project root: {path}")
    return resolved


def _resolve_project_path(user_path: str, *, subdir: str | None = None) -> Path:
    """Resolve a UI-supplied path, optionally constrained to a project subdir.

    UI callbacks must never trust Gradio component values (e.g. dropdown text):
    traversal, absolute paths, and symlinks are all rejected here, at the point
    the path is about to be consumed.
    """
    if not user_path:
        raise ValueError("Path is empty")
    candidate = Path(user_path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    resolved = _ensure_under_project(candidate)
    if subdir is not None:
        allowed = (PROJECT_ROOT / subdir).resolve()
        try:
            resolved.relative_to(allowed)
        except ValueError:
            raise ValueError(f"Path must be under {subdir!r}: {user_path}")
    return resolved

# Gradio moved where `theme` lives across versions: <=5 uses `Blocks(theme=...)`,
# >=6 moved it to `launch()`. Detect the installed major so the WebUI works on
# any supported version without an ignored-argument warning or a TypeError.
_GRADIO_MAJOR = int((getattr(gr, "__version__", "4.0").split(".")[0]) or 4)


# ─── Config helpers ──────────────────────────────────────────────────────────

def discover_configs() -> list[str]:
    """Find all YAML configs under configs/."""
    pattern = str(PROJECT_ROOT / "configs" / "**" / "*.yaml")
    paths = sorted(glob.glob(pattern, recursive=True))
    return [os.path.relpath(p, PROJECT_ROOT) for p in paths
            if "_base_" not in p]


def load_config(path: str) -> dict:
    """Load a YAML config file."""
    config_file = _resolve_project_path(path, subdir="configs")
    with open(config_file) as f:
        return yaml.safe_load(f)


def scan_outputs(out_dir: str) -> list:
    """Digest experiment results strictly under PROJECT_ROOT.

    The Output Explorer callback. Any path that escapes PROJECT_ROOT is
    rejected (empty result) at the point data is read, so a traversal arg
    can never read files outside the project.
    """
    rows = []
    if not out_dir:
        return rows
    try:
        base = _resolve_project_path(out_dir)
    except ValueError:
        return rows
    if not base.exists() or not base.is_dir():
        return rows
    for bench_dir in sorted(base.iterdir()):
        try:
            bench_dir = _ensure_under_project(bench_dir)
        except ValueError:
            continue
        if not bench_dir.is_dir():
            continue
        for run_dir in sorted(bench_dir.iterdir()):
            try:
                run_dir = _ensure_under_project(run_dir)
            except ValueError:
                continue
            if not run_dir.is_dir():
                continue
            cfg_file = run_dir / "config.yaml"
            score = "—"
            steps = "—"
            if cfg_file.exists():
                try:
                    cfg_file = _ensure_under_project(cfg_file)
                    c = yaml.safe_load(cfg_file.read_text())
                    steps = str(c.get("train", {}).get("num_steps", "—"))
                except Exception:
                    pass
            # Try to find best score from logs
            for log_f in run_dir.glob("**/*.jsonl"):
                try:
                    log_f = _ensure_under_project(log_f)
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


def config_to_display(cfg: dict) -> str:
    """Pretty-print config for display."""
    return yaml.dump(cfg, default_flow_style=False, sort_keys=False)


def config_preview(path: str) -> str:
    """Registered config-preview callback: YAML text for an in-tree config."""
    if not path:
        return ""
    try:
        return config_to_display(load_config(path))
    except Exception as exc:
        return f"Error: {exc}"


def _can_connect_to_url(url: str, timeout: float = 0.5) -> bool:
    parsed = urlparse(url)
    host = parsed.hostname
    if not host:
        return False
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _load_env_file(path: Path, env: dict[str, str]) -> None:
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip().strip("\"'")


def build_training_env() -> dict[str, str]:
    """Build the environment shared by preflight and the training subprocess."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    dot_env = PROJECT_ROOT / ".env"
    if dot_env.is_file():
        _load_env_file(dot_env, env)

    secrets_dir = PROJECT_ROOT / ".secrets"
    if secrets_dir.is_dir():
        for env_file in sorted(secrets_dir.glob("*.env")):
            _load_env_file(env_file, env)

    # Propagate OPTIMIZER_* to base AZURE_OPENAI_* when base is missing,
    # so target/default endpoints inherit from optimizer config.
    for suffix in (
        "ENDPOINT", "API_VERSION", "AUTH_MODE", "MANAGED_IDENTITY_CLIENT_ID",
        "AD_SCOPE", "API_KEY",
    ):
        base_key = f"AZURE_OPENAI_{suffix}"
        optimizer_key = f"OPTIMIZER_AZURE_OPENAI_{suffix}"
        if not env.get(base_key) and env.get(optimizer_key):
            env[base_key] = env[optimizer_key]
    return env


def validate_training_config(
    config_path: str,
    overrides: dict,
    env: dict[str, str] | None = None,
) -> str | None:
    """Return an actionable preflight error, or None when training can start."""
    env = env or os.environ
    cfg_options = [
        f"{key}={value}" for key, value in overrides.items()
        if value is not None and value != ""
    ]
    try:
        config_file = _resolve_project_path(config_path, subdir="configs")
        cfg = flatten_config(load_merged_config(str(config_file), cfg_options))
    except Exception as exc:
        return f"❌ Invalid config: {exc}"

    shared_endpoint = (
        cfg.get("azure_openai_endpoint")
        or cfg.get("azure_endpoint")
        or env.get("AZURE_OPENAI_ENDPOINT")
    )
    missing_openai_roles = []
    for role in ("optimizer", "target"):
        if cfg.get(f"{role}_backend") != "openai_chat":
            continue
        role_endpoint = (
            cfg.get(f"{role}_azure_openai_endpoint")
            or env.get(f"{role.upper()}_AZURE_OPENAI_ENDPOINT")
            or shared_endpoint
        )
        if not role_endpoint:
            missing_openai_roles.append(role)
    if missing_openai_roles:
        configured_backend = cfg.get("model_backend")
        detail = ""
        if configured_backend in {"qwen", "qwen_chat"}:
            detail = (
                "\nNote: model.backend is qwen, but explicit optimizer_backend/"
                "target_backend values are still openai_chat."
            )
        return (
            "❌ Model backend is not ready: missing Azure/OpenAI-compatible endpoint "
            f"for {', '.join(missing_openai_roles)}.\n"
            "Set model.azure_openai_endpoint (or AZURE_OPENAI_ENDPOINT), or change "
            "the role backends to the backend you intend to use."
            f"{detail}"
        )

    qwen_failures = []
    qwen_shared = (
        cfg.get("qwen_chat_base_url")
        or env.get("QWEN_CHAT_BASE_URL")
        or "http://localhost:8000/v1"
    )
    for role in ("optimizer", "target"):
        if cfg.get(f"{role}_backend") != "qwen_chat":
            continue
        base_url = (
            cfg.get(f"{role}_qwen_chat_base_url")
            or env.get(f"{role.upper()}_QWEN_CHAT_BASE_URL")
            or qwen_shared
        )
        if not _can_connect_to_url(str(base_url)):
            qwen_failures.append(f"{role}={base_url}")
    if qwen_failures:
        return (
            "❌ Model backend is not ready: cannot connect to qwen_chat endpoint "
            f"for {', '.join(qwen_failures)}.\n"
            "Start your OpenAI-compatible Qwen/vLLM server, or set "
            "model.qwen_chat_base_url / OPTIMIZER_QWEN_CHAT_BASE_URL / "
            "TARGET_QWEN_CHAT_BASE_URL to the correct URL."
        )
    return None


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

        env = build_training_env()
        preflight_error = validate_training_config(config_path, overrides, env)
        if preflight_error:
            return preflight_error

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


# ─── Gradio UI ──────────────────────────────────────────────────────────────

def build_ui():
    configs = discover_configs()

    _blocks_kwargs = {"title": "SkillOpt WebUI"}
    if _GRADIO_MAJOR < 6:
        _blocks_kwargs["theme"] = gr.themes.Soft(primary_hue="indigo")
    with gr.Blocks(**_blocks_kwargs) as app:
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
                        config_preview_box = gr.Code(
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

                        with gr.Row():
                            launch_btn = gr.Button("🚀 Launch Training",
                                                    variant="primary", size="lg")
                            stop_btn = gr.Button("🛑 Stop", variant="stop")

                        status_text = gr.Textbox(label="Status", interactive=False)

                config_dropdown.change(config_preview, config_dropdown, config_preview_box)

                def on_launch(cfg_path, lr_val, sched, epochs, batch, workers,
                              slow_update, meta_skill, gate):
                    overrides = {
                        "optimizer.learning_rate": lr_val,
                        "optimizer.lr_scheduler": sched,
                        "train.num_epochs": epochs,
                        "train.batch_size": batch,
                        "gradient.analyst_workers": workers,
                        "optimizer.use_slow_update": slow_update,
                        "optimizer.use_meta_skill": meta_skill,
                        "evaluation.use_gate": gate,
                    }
                    return manager.start(cfg_path, overrides)

                launch_btn.click(
                    on_launch,
                    [config_dropdown, lr, scheduler, num_epochs, batch_size,
                     analyst_workers, use_slow_update, use_meta_skill, use_gate],
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

                scan_btn.click(scan_outputs, output_dir, results_table)

    return app


def build_launch_kwargs(server_name: str, server_port: int, share: bool) -> dict:
    """Build the launch kwargs exactly as production ``main()`` applies them.

    Gradio 6 moved the theme to ``launch()``; applying it here (rather than in
    ``build_ui``) keeps the theme on the right object for the installed major
    without an ignored-argument warning. Tests call this so they exercise the
    same path a real ``skillopt-webui`` run does instead of injecting their own.
    """
    kwargs = dict(server_name=server_name, server_port=server_port, share=share)
    if _GRADIO_MAJOR >= 6:
        kwargs["theme"] = gr.themes.Soft(primary_hue="indigo")
    return kwargs


def resolve_auth(auth_user, auth_pass, env=None):
    """Resolve basic-auth credentials from the CLI flags and the environment.

    Returns ``(user, password)`` when authentication is configured, or ``None``
    for a deliberately unconfigured local run. Raises ``ValueError`` when a
    configuration is present but invalid.

    Precedence is per field: a flag supplied on the command line wins over its
    environment variable. "Not supplied" and "supplied but blank" are different
    states, because treating them as the same one fails open - ``--auth-user
    ""`` must not quietly fall back to ``SKILLOPT_WEBUI_USER``, and a pair of
    blank variables (an empty secret, a template that was never filled in) must
    not read as "no authentication requested" when the operator meant to
    require login.
    """
    env = os.environ if env is None else env

    def pick(cli_value, env_name, flag):
        if cli_value is not None:
            return cli_value, flag
        if env_name in env:
            return env[env_name], env_name
        return None, None

    user, user_src = pick(auth_user, "SKILLOPT_WEBUI_USER", "--auth-user")
    password, password_src = pick(auth_pass, "SKILLOPT_WEBUI_PASS", "--auth-pass")

    if user_src is None and password_src is None:
        return None
    if user_src is None or password_src is None:
        missing = "--auth-pass" if user_src is not None else "--auth-user"
        raise ValueError(f"{user_src or password_src} is set but {missing} is not")
    for value, src in ((user, user_src), (password, password_src)):
        if not value.strip():
            raise ValueError(f"{src} is blank")
    return (user, password)


def main():
    parser = argparse.ArgumentParser(description="SkillOpt WebUI")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="Server host. Default is localhost; use 0.0.0.0 "
                             "to expose publicly (no auth, use with care).")
    parser.add_argument("--auth-user", type=str, default=None,
                        help="Username for basic auth (or set SKILLOPT_WEBUI_USER).")
    parser.add_argument("--auth-pass", type=str, default=None,
                        help="Password for basic auth (or set SKILLOPT_WEBUI_PASS).")
    args = parser.parse_args()

    if args.host and args.host not in ("127.0.0.1", "localhost", "::1"):
        print(
            f"⚠ warning: binding SkillOpt WebUI on {args.host} with no auth "
            "exposes the Output Explorer (reads any path you type) and the "
            "training controls to reachable clients. Prefer --host 127.0.0.1; "
            "use 0.0.0.0 only if you understand the risk.",
            file=sys.stderr,
        )

    if args.share:
        print(
            "⚠ warning: --share creates a public tunnel (gradio.live) with no "
            "authentication by default. Anyone with the URL can start/stop "
            "training and browse the filesystem via Output Explorer. "
            "Use --auth-user / --auth-pass (or SKILLOPT_WEBUI_USER / "
            "SKILLOPT_WEBUI_PASS) to require login.",
            file=sys.stderr,
        )

    # Fail-closed: authentication requires BOTH credentials, from any source.
    # Supplying only one of them - or supplying blank ones - must not silently
    # launch the UI unauthenticated (a deployment could expose the training
    # controls without login).
    try:
        auth = resolve_auth(args.auth_user, args.auth_pass)
    except ValueError as exc:
        print(
            f"SKILLOPT_WEBUI authentication is misconfigured: {exc}. Provide both "
            "--auth-user and --auth-pass (or both SKILLOPT_WEBUI_USER and "
            "SKILLOPT_WEBUI_PASS) to require login, or neither to run without "
            "authentication. Refusing to start.",
            file=sys.stderr,
        )
        sys.exit(1)

    app = build_ui()
    launch_kwargs = build_launch_kwargs(args.host, args.port, args.share)
    if auth:
        launch_kwargs["auth"] = auth
    app.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
