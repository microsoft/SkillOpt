"""Tests for plugins/run-sleep.sh engine resolution and fallback logic.

Verifies that the runner:
  1. Uses the source checkout when skillopt_sleep/ is present (existing path).
  2. Falls back to the skillopt-sleep CLI on PATH when no source dir is found.
  3. Falls back to `python -m skillopt_sleep` when the package is importable
     but no CLI is on PATH.
  4. Errors with a helpful message when no engine can be found.

Pure-stdlib (unittest). No API key, no third-party deps.
Run:  python3 -m pytest tests/test_run_sleep_fallback.py -v
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RUNNER = os.path.join(REPO, "plugins", "run-sleep.sh")
BUNDLED = os.path.join(REPO, "plugins", "claude-code", "scripts", "run-sleep.sh")


def _run(script, args, env, cwd):
    """Run a sleep runner script and return (returncode, stdout, stderr)."""
    cmd = ["bash", script] + args
    proc = subprocess.run(
        cmd, capture_output=True, text=True, env=env, cwd=cwd, timeout=30,
    )
    return proc.returncode, proc.stdout, proc.stderr


@unittest.skipIf(sys.platform == "win32", "Unix bash runner tests require POSIX/bash environment")
class TestRunnerSourceCheckout(unittest.TestCase):
    """When skillopt_sleep/ is found in the repo, the source-checkout path is used."""

    def test_source_checkout_runs_engine(self):
        # The repo root contains skillopt_sleep/, so running from the repo
        # should find it via the SCRIPT_DIR/../skillopt_sleep check.
        env = {k: v for k, v in os.environ.items()
               if k not in ("SKILLOPT_SLEEP_REPO", "CLAUDE_PLUGIN_ROOT")}
        rc, out, err = _run(RUNNER, ["--help"], env, REPO)
        self.assertEqual(rc, 0, f"stderr:\n{err}")
        self.assertIn("skillopt_sleep", out)
        self.assertIn("run", out)
        self.assertIn("status", out)


@unittest.skipIf(sys.platform == "win32", "Unix bash runner tests require POSIX/bash environment")
class TestRunnerCliFallback(unittest.TestCase):
    """When no source dir is found, fall back to the skillopt-sleep CLI on PATH."""

    def setUp(self):
        # Copy the runner to a temp dir so SCRIPT_DIR/../skillopt_sleep doesn't
        # resolve to the repo's source checkout (SCRIPT_DIR is the script's
        # location, not CWD, so running the repo copy always finds the source).
        self._tmp = tempfile.TemporaryDirectory()
        self._script = os.path.join(self._tmp.name, "run-sleep.sh")
        shutil.copy2(RUNNER, self._script)
        # Build a fake skillopt-sleep CLI on PATH that records its args.
        self._bindir = os.path.join(self._tmp.name, "bin")
        os.makedirs(self._bindir)
        self._fake_cli = os.path.join(self._bindir, "skillopt-sleep")
        with open(self._fake_cli, "w") as f:
            f.write("#!/usr/bin/env bash\n")
            f.write('echo "fake-cli invoked: $@"\n')
        os.chmod(self._fake_cli, 0o755)

    def tearDown(self):
        self._tmp.cleanup()

    def _env_without_source(self):
        """Env with no source dir resolvable and fake CLI on PATH."""
        env = {k: v for k, v in os.environ.items()
               if k not in ("SKILLOPT_SLEEP_REPO", "CLAUDE_PLUGIN_ROOT")}
        # Prepend fake bin dir so our skillopt-sleep is found first.
        env["PATH"] = self._bindir + os.pathsep + env.get("PATH", "")
        return env

    def test_cli_fallback_used_when_no_source_dir(self):
        env = self._env_without_source()
        rc, out, err = _run(self._script, ["status", "--project", "/tmp"], env, "/tmp")
        self.assertEqual(rc, 0, f"stderr:\n{err}")
        self.assertIn("fake-cli invoked: status --project /tmp", out)

    def test_cli_fallback_passes_through_all_args(self):
        env = self._env_without_source()
        rc, out, err = _run(
            self._script, ["run", "--project", "/tmp", "--backend", "mock"], env, "/tmp",
        )
        self.assertEqual(rc, 0, f"stderr:\n{err}")
        self.assertIn("run --project /tmp --backend mock", out)

    def test_bundled_copy_also_uses_cli_fallback(self):
        """The byte-identical bundled copy in plugins/claude-code/scripts/ must
        also fall back to the CLI — it's what the marketplace install ships."""
        bundled_copy = os.path.join(self._tmp.name, "bundled-run-sleep.sh")
        shutil.copy2(BUNDLED, bundled_copy)
        env = self._env_without_source()
        rc, out, err = _run(bundled_copy, ["status"], env, "/tmp")
        self.assertEqual(rc, 0, f"stderr:\n{err}")
        self.assertIn("fake-cli invoked: status", out)


@unittest.skipIf(sys.platform == "win32", "Unix bash runner tests require POSIX/bash environment")
class TestRunnerNoEngine(unittest.TestCase):
    """When no source dir, no CLI on PATH, and no importable package, error cleanly."""

    def test_error_message_mentions_install_options(self):
        # Copy the runner to a temp dir so SCRIPT_DIR/../skillopt_sleep doesn't
        # resolve to the repo's source checkout.
        with tempfile.TemporaryDirectory() as tmp:
            script = os.path.join(tmp, "run-sleep.sh")
            shutil.copy2(RUNNER, script)
            # Build an env with a minimal PATH so skillopt-sleep isn't found,
            # and no source dir resolvable.
            env = {
                "PATH": "/usr/bin:/bin",
                "HOME": os.environ.get("HOME", "/tmp"),
            }
            # Use a Python that almost certainly doesn't have skillopt_sleep
            # importable (system python3). If it does, skip — we can't easily
            # simulate "not installed" in that case.
            try:
                subprocess.run(
                    [sys.executable, "-c", "import skillopt_sleep"],
                    capture_output=True, check=True, timeout=10,
                )
                has_package = True
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                has_package = False

            if has_package:
                self.skipTest("skillopt_sleep is importable in this env; "
                              "cannot test the no-engine error path")

            rc, out, err = _run(script, ["status"], env, "/tmp")
            self.assertNotEqual(rc, 0)
            self.assertIn("could not locate the skillopt_sleep package", err)
            self.assertIn("uv tool install skillopt", err)
            self.assertIn("pip install skillopt", err)
            self.assertIn("SKILLOPT_SLEEP_REPO", err)


@unittest.skipIf(sys.platform == "win32", "Unix bash runner tests require POSIX/bash environment")
class TestRunnerIgnoresCwdPlants(unittest.TestCase):
    """Files planted in the (untrusted) project directory must never be run."""

    PLANT = (
        "import os\n"
        "open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'PWNED'), 'w').close()\n"
    )

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        # Runner copied outside the repo so no source checkout is resolvable.
        self._script = os.path.join(self._tmp.name, "runner", "run-sleep.sh")
        os.makedirs(os.path.dirname(self._script))
        shutil.copy2(RUNNER, self._script)
        self._project = os.path.join(self._tmp.name, "project")
        os.makedirs(self._project)
        # PATH with a Python >= 3.10 but no skillopt-sleep CLI.
        self._bindir = os.path.join(self._tmp.name, "bin")
        os.makedirs(self._bindir)
        os.symlink(sys.executable, os.path.join(self._bindir, "python3"))
        self._env = {
            "PATH": self._bindir + os.pathsep + "/usr/bin:/bin",
            "HOME": os.environ.get("HOME", "/tmp"),
        }

    def tearDown(self):
        self._tmp.cleanup()

    def test_planted_module_file_in_cwd_not_imported(self):
        with open(os.path.join(self._project, "skillopt_sleep.py"), "w") as f:
            f.write(self.PLANT)
        _run(self._script, ["status"], self._env, self._project)
        self.assertFalse(os.path.exists(os.path.join(self._project, "PWNED")))

    def test_planted_package_dir_in_cwd_not_used_as_repo(self):
        pkg = os.path.join(self._project, "skillopt_sleep")
        os.makedirs(pkg)
        for name in ("__init__.py", "__main__.py"):
            with open(os.path.join(pkg, name), "w") as f:
                f.write(self.PLANT)
        _run(self._script, ["status"], self._env, self._project)
        self.assertFalse(os.path.exists(os.path.join(pkg, "PWNED")))


class TestRunnerBundledMatchesShared(unittest.TestCase):
    """The bundled copy must stay in sync with the shared runner."""

    def test_bundled_equals_shared(self):
        with open(RUNNER) as f:
            shared = f.read()
        with open(BUNDLED) as f:
            bundled = f.read()
        self.assertEqual(shared, bundled,
                         "plugins/claude-code/scripts/run-sleep.sh has drifted "
                         "from plugins/run-sleep.sh — they must stay in sync")


if __name__ == "__main__":
    unittest.main()
