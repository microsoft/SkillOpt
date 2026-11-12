"""SkillOpt-Sleep — built-in nightly scheduler.

Installs/removes a crontab entry (on Unix) or a Scheduled Task (on Windows) that
runs the sleep cycle automatically, so the user doesn't have to wire it manually.
"""
from __future__ import annotations

import hashlib
import os
import shlex
import shutil
import subprocess
import sys
from typing import List, Optional, Tuple

_BEGIN = "# >>> skillopt-sleep (managed) >>>"
_END = "# <<< skillopt-sleep (managed) <<<"


def _have_crontab() -> bool:
    return shutil.which("crontab") is not None


def _read_crontab() -> str:
    try:
        proc = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
        return proc.stdout if proc.returncode == 0 else ""
    except Exception:
        return ""


def _write_crontab(content: str) -> bool:
    try:
        proc = subprocess.run(["crontab", "-"], input=content, text=True,
                              capture_output=True)
        return proc.returncode == 0
    except Exception:
        return False


def _split_managed(crontab: str) -> Tuple[str, List[str]]:
    """Return (text_outside_block, managed_lines_inside_block)."""
    lines = crontab.splitlines()
    outside: List[str] = []
    managed: List[str] = []
    in_block = False
    for ln in lines:
        if ln.strip() == _BEGIN:
            in_block = True
            continue
        if ln.strip() == _END:
            in_block = False
            continue
        (managed if in_block else outside).append(ln)
    return "\n".join(outside).rstrip(), managed


def _have_schtasks() -> bool:
    return shutil.which("schtasks") is not None


def _win_task_name(project: str) -> str:
    project = os.path.abspath(project)
    digest = hashlib.sha256(project.encode("utf-8")).hexdigest()[:20]
    return f"SkillOpt-Sleep-{digest}"


def _create_win_task(task_name: str, command: str, hour: int, minute: int) -> bool:
    try:
        st = f"{hour:02d}:{minute:02d}"
        cmd = ["schtasks", "/create", "/tn", task_name, "/tr", command, "/sc", "daily", "/st", st, "/f"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        return proc.returncode == 0
    except Exception:
        return False


def _delete_win_task(task_name: str) -> bool:
    try:
        cmd = ["schtasks", "/delete", "/tn", task_name, "/f"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        return proc.returncode == 0
    except Exception:
        return False


def _list_win_tasks() -> List[str]:
    try:
        proc = subprocess.run(["schtasks", "/query", "/fo", "csv"], capture_output=True, text=True)
        if proc.returncode != 0:
            return []
        tasks = []
        for line in proc.stdout.splitlines():
            if not line.startswith('"'):
                continue
            parts = line.split(",")
            if len(parts) > 0:
                name = parts[0].strip('"')
                if name.startswith("SkillOpt-Sleep-"):
                    tasks.append(name)
        return tasks
    except Exception:
        return []


def _runner_cmd(project: str, backend: str, extra: str, python: str,
                scope: str = "invoked") -> str:
    if scope not in ("invoked", "all"):
        raise ValueError(f"scope must be 'invoked' or 'all', got {scope!r}")
    _validate_scheduler_text("project", project)
    _validate_scheduler_text("python", python)
    _validate_scheduler_text("repository", _repo_root())
    _validate_scheduler_text("backend", backend)
    if extra:
        _validate_scheduler_text("extra scheduler arguments", extra)
    logdir = os.path.join(project, ".skillopt-sleep")
    log = os.path.join(logdir, "cron.log")
    args = [
        python,
        "-m",
        "skillopt_sleep",
        "run",
        "--project",
        project,
        "--scope",
        scope,
        "--backend",
        backend,
        *shlex.split(extra),
    ]
    if sys.platform == "win32":
        helper_script = os.path.join(logdir, "run.ps1")
        try:
            os.makedirs(logdir, exist_ok=True)
            quoted = " ".join(_powershell_literal(value) for value in args[1:])
            content = (
                "$ErrorActionPreference = 'Stop'\n"
                f"Set-Location -LiteralPath {_powershell_literal(_repo_root())}\n"
                f"& {_powershell_literal(python)} {quoted} "
                f"*>> {_powershell_literal(log)}\n"
            )
            with open(helper_script, "w", encoding="utf-8") as f:
                f.write(content)
        except OSError as exc:
            raise ValueError(f"could not create the Windows scheduler helper: {exc}") from exc
        return (
            "powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass "
            f"-File {_windows_command_quote(helper_script)}"
        )
    cmd = shlex.join(args)
    return (
        f"mkdir -p {shlex.quote(logdir)}; "
        f"cd {shlex.quote(_repo_root())} && {cmd} >> {shlex.quote(log)} 2>&1"
    )


def _repo_root() -> str:
    # the package lives at <repo>/skillopt_sleep/; repo root is its parent
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _validate_scheduler_text(label: str, value: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        raise ValueError(f"{label} cannot contain control characters")


def _powershell_literal(value: str) -> str:
    """Single-quoted PowerShell data literal."""
    _validate_scheduler_text("scheduler argument", value)
    return "'" + value.replace("'", "''") + "'"


def _windows_command_quote(value: str) -> str:
    """Quote a Windows path whose allowed characters cannot include quotes."""
    _validate_scheduler_text("scheduler path", value)
    if '"' in value:
        raise ValueError("scheduler path cannot contain a double quote")
    return f'"{value}"'


def _project_marker(project: str) -> str:
    canonical = os.path.abspath(project)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"# project-sha256={digest}"


def schedule(project: str, *, backend: str = "mock", hour: int = 3, minute: int = 17,
             extra: str = "", python: Optional[str] = None,
             scope: str = "invoked") -> Tuple[bool, str]:
    """Install (or replace) the nightly entry for ``project``.

    Returns (installed, message). If the scheduler backend is unavailable, installed=False and
    the message contains instructions to add manually.
    """
    if type(hour) is not int or not 0 <= hour <= 23:
        return False, "hour must be an integer from 0 through 23"
    if type(minute) is not int or not 0 <= minute <= 59:
        return False, "minute must be an integer from 0 through 59"
    project = os.path.abspath(project)
    python = python or sys.executable or "python3"
    try:
        runner_cmd = _runner_cmd(project, backend, extra, python, scope)
    except (ValueError, OSError) as exc:
        return False, f"Refusing unsafe scheduler configuration: {exc}"

    if sys.platform == "win32":
        if not _have_schtasks():
            return False, "schtasks.exe not found on this system. Add this command to your scheduler manually:\n" + runner_cmd
        tn = _win_task_name(project)
        ok = _create_win_task(tn, runner_cmd, hour, minute)
        if ok:
            return True, (f"Scheduled nightly at {hour:02d}:{minute:02d} for {project} "
                          f"(backend={backend}) via Windows Task Scheduler. Task Name: {tn}\n"
                          f"Logs -> {project}/.skillopt-sleep/cron.log\n"
                          f"Runs `skillopt_sleep run`; it only STAGES a proposal — adopt is still manual.")
        return False, f"Failed to write scheduled task. Command to run manually:\nschtasks /create /tn \"{tn}\" /tr \"{runner_cmd}\" /sc daily /st {hour:02d}:{minute:02d} /f"

    cron_line = f"{minute} {hour} * * *  {runner_cmd}  {_project_marker(project)}"

    if not _have_crontab():
        return False, ("crontab not found on this system. Add this line to your "
                       "scheduler manually:\n" + cron_line)

    outside, managed = _split_managed(_read_crontab())
    # drop any existing line for this project, then add the new one
    marker = _project_marker(project)
    managed = [ln for ln in managed if marker not in ln and ln.strip()]
    managed.append(cron_line)

    block = _BEGIN + "\n" + "\n".join(managed) + "\n" + _END
    new_crontab = (outside + "\n\n" + block + "\n").lstrip("\n")
    ok = _write_crontab(new_crontab)
    if ok:
        return True, (f"Scheduled nightly at {hour:02d}:{minute:02d} for {project} "
                      f"(backend={backend}). Logs -> {project}/.skillopt-sleep/cron.log\n"
                      f"Runs `skillopt_sleep run`; it only STAGES a proposal — adopt is still manual.")
    return False, "Failed to write crontab. Line to add manually:\n" + cron_line


def unschedule(project: Optional[str] = None, *, all_projects: bool = False) -> Tuple[bool, str]:
    """Remove the entry for ``project`` (or the whole managed block with all_projects)."""
    if sys.platform == "win32":
        if not _have_schtasks():
            return False, "schtasks.exe not found on this system."
        if all_projects:
            tasks = _list_win_tasks()
            ok = True
            for t in tasks:
                if not _delete_win_task(t):
                    ok = False
            return ok, ("Removed all scheduled tasks." if ok else "Failed to remove some tasks.")
        elif project:
            tn = _win_task_name(project)
            ok = _delete_win_task(tn)
            try:
                logdir = os.path.join(project, ".skillopt-sleep")
                helper = os.path.join(logdir, "run.ps1")
                if os.path.exists(helper):
                    os.remove(helper)
            except Exception:
                pass
            return ok, ("Removed." if ok else "Failed to remove scheduled task (does it exist?).")
        return False, "No project specified to unschedule."

    if not _have_crontab():
        return False, "crontab not found; nothing to remove."
    outside, managed = _split_managed(_read_crontab())
    if all_projects:
        managed = []
    elif project:
        marker = _project_marker(project)
        managed = [ln for ln in managed if marker not in ln and ln.strip()]
    if managed:
        block = _BEGIN + "\n" + "\n".join(managed) + "\n" + _END
        new_crontab = (outside + "\n\n" + block + "\n").lstrip("\n")
    else:
        new_crontab = outside.rstrip() + "\n"
    ok = _write_crontab(new_crontab)
    return ok, ("Removed." if ok else "Failed to update crontab.")


def list_scheduled() -> List[str]:
    if sys.platform == "win32":
        if not _have_schtasks():
            return []
        return _list_win_tasks()
    _outside, managed = _split_managed(_read_crontab())
    return [ln for ln in managed if ln.strip()]
