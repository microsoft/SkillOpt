#!/usr/bin/env python3
"""dsh-skillopt — engine bootstrap helper (mirrors the official run-sleep.sh).

Resolves the skillopt_sleep engine the same way the official SkillOpt plugin
runner does, so the dsh tools work in every install shape:

  1. Source checkout: a `skillopt_sleep/` package next to this script (or under
     SKILLOPT_SLEEP_REPO) is importable — run from that root.
  2. A Python >= 3.10 interpreter is picked (python3.12 -> 3.11 -> 3.10 ->
     python3), skipping Python 2 / too-old versions.
  3. Fallbacks: `skillopt-sleep` CLI on PATH (uv tool / pipx / pip installs),
     then `python -m skillopt_sleep` against an installed package.

Usage:
    python scripts/sleep.py status
    python scripts/sleep.py run --backend mock --project .
    python scripts/sleep.py adopt --project .
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT_CANDIDATES = [
    HERE.parent,                       # plugin repo root (dsh-skillopt/)
    HERE / ".." / "..",                # SkillOpt checkout: plugins/.../scripts -> repo root
]

PYTHON_CANDIDATES = ["python3.12", "python3.11", "python3.10", "python3"]


def find_repo_root() -> Path | None:
    """A directory containing an importable `skillopt_sleep` package."""
    env = os.environ.get("SKILLOPT_SLEEP_REPO")
    candidates = list(REPO_ROOT_CANDIDATES)
    if env:
        candidates.insert(0, Path(env))
    for cand in candidates:
        root = cand.resolve()
        if (root / "skillopt_sleep").is_dir():
            return root
    # search upward from CWD (same last-resort as the official runner)
    d = Path.cwd()
    while d != d.parent:
        if (d / "skillopt_sleep").is_dir():
            return d
        d = d.parent
    return None


def pick_python() -> str | None:
    """First candidate with version >= 3.10, or None."""
    for cand in PYTHON_CANDIDATES:
        path = shutil.which(cand)
        if not path:
            continue
        try:
            ver = subprocess.run(
                [path, "-c", "import sys; print('%d%d' % sys.version_info[:2])"],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip()
        except Exception:
            continue
        if ver and int(ver) >= 310:
            return path
    # explicit python on PATH (may be < 3.10; let the engine fail loudly)
    return shutil.which("python") or shutil.which("python3")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", nargs="?", default="status",
                    choices=["status", "dry-run", "run", "adopt", "harvest",
                             "schedule", "unschedule"])
    ap.add_argument("args", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    # 1. source checkout: run from repo root so skillopt_sleep/ is importable
    repo_root = find_repo_root()
    cwd = str(repo_root) if repo_root else None

    # 2. python >= 3.10
    python = pick_python()
    if not python:
        print("[sleep] ERROR: need Python >= 3.10 (found none).", file=sys.stderr)
        return 1

    # 3a. installed package via python -m
    probe = subprocess.run(
        [python, "-c", "import skillopt_sleep"],
        capture_output=True, cwd=cwd,
    )
    if probe.returncode == 0:
        cmd = [python, "-m", "skillopt_sleep", args.action, *args.args]
        print("+", " ".join(cmd), file=sys.stderr)
        return subprocess.call(cmd, cwd=cwd)

    # 3b. skillopt-sleep CLI on PATH (uv tool / pipx / pip)
    cli = shutil.which("skillopt-sleep")
    if cli:
        cmd = [cli, args.action, *args.args]
        print("+", " ".join(cmd), file=sys.stderr)
        return subprocess.call(cmd, cwd=cwd)

    print(
        "skillopt_sleep not importable and no skillopt-sleep CLI on PATH.\n"
        "Install it with:  pip install skillopt\n"
        "or use a source checkout of https://github.com/microsoft/SkillOpt\n"
        "(set SKILLOPT_SLEEP_REPO to its path).",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
