#!/usr/bin/env python3
"""dsh-skillopt — bootstrap helper.

Installs/verifies the skillopt_sleep engine and runs a sleep-cycle action the
same way the dsh tools do. Useful for testing the plumbing outside the agent.

Usage:
    python scripts/sleep.py status
    python scripts/sleep.py run --backend mock --project .
    python scripts/sleep.py adopt --project .
"""
import argparse
import shutil
import subprocess
import sys

try:
    import skillopt_sleep  # noqa: F401
    ENGINE_OK = True
except ImportError:
    ENGINE_OK = False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", nargs="?", default="status",
                    choices=["status", "dry-run", "run", "adopt", "harvest",
                             "schedule", "unschedule"])
    ap.add_argument("args", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    if not ENGINE_OK:
        print("skillopt_sleep not importable. Install it with:", file=sys.stderr)
        print("  pip install skillopt", file=sys.stderr)
        print("or use a source checkout of https://github.com/microsoft/SkillOpt",
              file=sys.stderr)
        return 2

    python = shutil.which("python") or "python"
    cmd = [python, "-m", "skillopt_sleep", args.action, *args.args]
    print("+", " ".join(cmd), file=sys.stderr)
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
