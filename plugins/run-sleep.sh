#!/usr/bin/env bash
# SkillOpt-Sleep shared runner — used by all platform plugins (Claude Code,
# Codex, Copilot). Resolves the repo root (which contains the skillopt_sleep
# package), picks a Python >= 3.10, and execs the engine CLI.
#
# Usage: run-sleep.sh <run|dry-run|status|adopt|harvest|...> [args...]
set -euo pipefail

# This script lives at <repo>/plugins/run-sleep.sh, so the repo root (which
# holds skillopt_sleep/) is one level up. CLAUDE_PLUGIN_ROOT (if set by Claude
# Code) points at the plugin dir; the engine is then two levels above it.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$SCRIPT_DIR/../skillopt_sleep" ]; then
  REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
elif [ -n "${CLAUDE_PLUGIN_ROOT:-}" ] && [ -d "$CLAUDE_PLUGIN_ROOT/../../skillopt_sleep" ]; then
  REPO_ROOT="$(cd "$CLAUDE_PLUGIN_ROOT/../.." && pwd)"
elif [ -n "${SKILLOPT_SLEEP_REPO:-}" ] && [ -d "$SKILLOPT_SLEEP_REPO/skillopt_sleep" ]; then
  REPO_ROOT="$SKILLOPT_SLEEP_REPO"
fi
# Deliberately no search upward from CWD: the CWD is the (possibly untrusted)
# project being worked on, and a skillopt_sleep/ planted there would be run.
if [ "$#" -eq 0 ]; then set -- status; fi

if [ -n "${REPO_ROOT:-}" ]; then
  # Source checkout: run from repo root so skillopt_sleep/ is importable.
  PY=""
  # Allow explicit Python override (useful on macOS with old system Python).
  if [ -n "${SKILLOPT_SLEEP_PYTHON:-}" ]; then
    PY="$SKILLOPT_SLEEP_PYTHON"
  else
    for cand in python3.12 python3.11 python3.10 python3; do
      if command -v "$cand" >/dev/null 2>&1; then
        ver="$("$cand" -c 'import sys; print("%d%d" % sys.version_info[:2])' 2>/dev/null || echo 0)"
        if [ "${ver:-0}" -ge 310 ]; then PY="$cand"; break; fi
      fi
    done
  fi
  if [ -z "$PY" ]; then
    echo "[sleep] ERROR: need Python >= 3.10 (found none)." >&2
    exit 1
  fi
  cd "$REPO_ROOT"
  exec "$PY" -m skillopt_sleep "$@"
fi

# No source checkout found — fall back to an installed engine.

# Fallback 1: skillopt-sleep CLI on PATH (uv tool install / pipx / pip install).
# Checked before the import fallback because uv tool install / pipx isolate the
# package from the system Python's import path, so `python -c "import
# skillopt_sleep"` would fail even though the CLI is available.
if command -v skillopt-sleep >/dev/null 2>&1; then
  exec skillopt-sleep "$@"
fi

# Fallback 2: importable as a module (pip install into the active Python).
# Pick a Python >= 3.10 and check importability. `python -c`/`-m` put the CWD
# (the user's project) first on sys.path, so a planted skillopt_sleep.py there
# would run; strip it before importing. The CWD itself is kept because the
# engine uses it as the default project.
NO_CWD_IMPORT="import sys; sys.path[:] = [p for p in sys.path if p]; import skillopt_sleep"
NO_CWD_RUN="import sys; sys.path[:] = [p for p in sys.path if p]; import runpy; runpy.run_module('skillopt_sleep', run_name='__main__', alter_sys=True)"
PY=""
for cand in python3.12 python3.11 python3.10 python3; do
  if command -v "$cand" >/dev/null 2>&1; then
    ver="$("$cand" -c 'import sys; print("%d%d" % sys.version_info[:2])' 2>/dev/null || echo 0)"
    if [ "${ver:-0}" -ge 310 ] && "$cand" -c "$NO_CWD_IMPORT" >/dev/null 2>&1; then
      PY="$cand"; break
    fi
  fi
done
if [ -n "$PY" ]; then
  exec "$PY" -c "$NO_CWD_RUN" "$@"
fi

echo "[sleep] ERROR: could not locate the skillopt_sleep package." >&2
echo "[sleep] Install it with 'uv tool install skillopt' or 'pip install skillopt'," >&2
echo "[sleep] or set SKILLOPT_SLEEP_REPO to a clone of the SkillOpt repo." >&2
exit 1
