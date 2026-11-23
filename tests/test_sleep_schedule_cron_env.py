"""The scheduled nightly job must find the CLIs it was scheduled with.

cron starts jobs with a minimal PATH (/usr/bin:/bin), so a codex or claude
installed with npm/nvm, Homebrew or ~/.local/bin was not found and every model
call of the night failed. A "%" in the cron command must also be escaped: cron
ends the command at the first unescaped "%" and sends the rest to stdin.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
import unittest
from unittest import mock

from skillopt_sleep import scheduler

# The environment cron gives a job: no login shell, no profile, minimal PATH.
CRON_ENV = {"PATH": "/usr/bin:/bin", "HOME": "/", "SHELL": "/bin/sh"}


def _as_cron_runs_it(cron_line: str) -> str:
    """What cron hands to /bin/sh for a crontab line, per crontab(5)."""
    command = cron_line.split(None, 5)[5]
    out = []
    escaped = False
    for ch in command:
        if escaped:
            if ch != "%":
                out.append("\\")
            out.append(ch)
            escaped = False
        elif ch == "\\":
            escaped = True
        elif ch == "%":
            break  # the rest of the line would become the job's stdin
        else:
            out.append(ch)
    return "".join(out)


def _fake_python(project: str) -> tuple[str, str]:
    """An executable that records the PATH it was started with."""
    seen = os.path.join(project, "seen_path")
    path = os.path.join(project, "python")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f'#!/bin/sh\nprintf "%s" "$PATH" > {shlex.quote(seen)}\n')
    os.chmod(path, 0o755)
    return path, seen


def _schedule(project: str, python: str, user_path: str) -> tuple[bool, str, str]:
    """Run schedule() with crontab I/O stubbed; return (ok, message, cron line)."""
    written = []

    def write(content):
        written.append(content)
        return True

    with mock.patch.dict(os.environ, {"PATH": user_path}), \
         mock.patch.object(scheduler, "_have_crontab", return_value=True), \
         mock.patch.object(scheduler, "_read_crontab", return_value=""), \
         mock.patch.object(scheduler, "_write_crontab", side_effect=write):
        ok, msg = scheduler.schedule(project, backend="mock", python=python)
    line = next(ln for ln in written[0].splitlines() if "project-sha256=" in ln) if written else ""
    return ok, msg, line


@unittest.skipIf(os.name == "nt", "cron entries are POSIX only")
class TestScheduledJobEnvironment(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        # Real directories, since only existing PATH entries are kept.
        self.tools = os.path.join(tmp.name, "tools", "bin")
        os.makedirs(self.tools)

    def _path_seen_by_job(self, user_path: str) -> str:
        with tempfile.TemporaryDirectory() as project:
            python, seen = _fake_python(project)
            ok, msg, line = _schedule(project, python, user_path)
            self.assertTrue(ok, msg)
            # The listing still starts with the project's own log directory.
            self.assertTrue(line.split(None, 5)[5].startswith("mkdir -p "), line)
            subprocess.run(["/bin/sh", "-c", _as_cron_runs_it(line)], env=CRON_ENV, check=True)
            with open(seen, encoding="utf-8") as f:
                return f.read()

    def test_job_runs_with_the_path_it_was_scheduled_from(self):
        user_path = f"{self.tools}:/usr/bin:/bin"
        self.assertEqual(self._path_seen_by_job(user_path), user_path)

    def test_relative_missing_and_repeated_entries_are_dropped(self):
        missing = os.path.join(self.tools, "missing")
        user_path = f"bin:{self.tools}::.:{missing}:{self.tools}:/usr/bin:/bin"
        self.assertEqual(self._path_seen_by_job(user_path), f"{self.tools}:/usr/bin:/bin")

    def test_percent_in_the_project_path_survives_cron(self):
        with tempfile.TemporaryDirectory(suffix=" 100%-done") as project:
            python, seen = _fake_python(project)
            ok, msg, line = _schedule(project, python, f"{self.tools}:/usr/bin:/bin")
            self.assertTrue(ok, msg)
            subprocess.run(["/bin/sh", "-c", _as_cron_runs_it(line)], env=CRON_ENV, check=True)
            with open(seen, encoding="utf-8") as f:
                self.assertEqual(f.read(), f"{self.tools}:/usr/bin:/bin")

    def test_path_too_long_for_cron_is_left_out(self):
        long_dirs = []
        for i in range(40):
            d = os.path.join(self.tools, f"toolchain-{i:02d}-" + "x" * 20)
            os.makedirs(d)
            long_dirs.append(d)
        with tempfile.TemporaryDirectory() as project:
            python, seen = _fake_python(project)
            ok, msg, line = _schedule(project, python, ":".join(long_dirs + ["/usr/bin", "/bin"]))
            self.assertTrue(ok, msg)
            self.assertIn("too long to include PATH", msg)
            self.assertLessEqual(len(line.split(None, 5)[5].encode()), 998)
            subprocess.run(["/bin/sh", "-c", _as_cron_runs_it(line)], env=CRON_ENV, check=True)
            with open(seen, encoding="utf-8") as f:
                self.assertEqual(f.read(), "/usr/bin:/bin")


if __name__ == "__main__":
    unittest.main()
