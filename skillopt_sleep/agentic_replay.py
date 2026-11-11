"""Tiered Agentic Replay helper and disposable scratch workspace (Issue #155).

This module implements the first workspace-lifecycle slice for agentic replay:
- A disposable scratch copy (NOT a linked git worktree and NOT an OS/container
  sandbox) with an independent bare harness git directory outside the diffed tree.
- Strict allowlist environment scrubbing for child subprocesses, redirecting
  ``HOME`` / ``USERPROFILE`` / XDG / CLI config directories to empty locations
  outside the diffed working directory.
  Note: there is NO OS, user, or network isolation boundary in this slice. A
  child running under the same UID can still read files by absolute path
  (such as ``~/.ssh`` or ``~/.claude``) or open network sockets directly.
- Deterministic task verification against workspace state/tests, with no
  verifier configured treated as ``inconclusive`` (``insufficient_evidence``).
- Explicit opt-in execution (``enabled=False`` by default) for injectable
  test/runner callables; live CLI binary execution is not enabled here.
"""
from __future__ import annotations

import inspect
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from skillopt_sleep.types import EditRecord, ReplayResult, TaskRecord

# Outcomes
OUTCOME_PASS = "pass"
OUTCOME_FAIL = "fail"
OUTCOME_INCONCLUSIVE = "inconclusive"

# Typed inconclusive reasons
REASON_TIMEOUT = "timeout"
REASON_INFRA = "infra"
REASON_FIXTURE_FAILURE = "fixture_failure"
REASON_BUDGET_EXHAUSTED = "budget_exhausted"
REASON_INSUFFICIENT_EVIDENCE = "insufficient_evidence"

# Strict OS-only allowlist for child subprocesses (matches Backend isolation ethos)
_ENV_ALLOWLIST_COMMON = (
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "TERM",
    "COLORTERM",
)
_ENV_ALLOWLIST_WINDOWS = (
    "SYSTEMROOT",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
    "OS",
    "NUMBER_OF_PROCESSORS",
    "PROCESSOR_ARCHITECTURE",
    "SYSTEMDRIVE",
)


class FixtureSetupError(RuntimeError):
    """Raised when fixture validation, git materialization, or baseline setup fails."""


@dataclass
class AgenticReplayConfig:
    """Configuration, opt-in gate, and budgets for agentic replay."""

    enabled: bool = False               # Strict boolean opt-in; disabled by default
    fixture_root: str = ""              # Explicit root required for directory fixtures
    max_turns: int = 10                 # Turn/action cap per agentic replay
    max_tokens: int = 0                 # Cumulative token cap across run (0 = unlimited)
    timeout_sec: float = 60.0           # Wall-clock timeout in seconds
    max_replays_per_run: int = 5        # Per-night/run cap on agentic replays
    isolate_env: bool = True            # Use allowlisted env + isolated empty config dirs
    min_decidable_actions: int = 1      # Minimum actions required to score vs inconclusive
    min_coverage: float = 0.5           # Minimum pairwise decidable val fraction for gate
    pinned_commit: str = ""             # Optional commit materialized via git archive
    max_output_bytes: int = 64 * 1024   # Cap buffered stdout/stderr per subprocess
    max_diff_bytes: int = 256 * 1024    # Cap buffered unified diff per replay

    def __post_init__(self) -> None:
        self.enabled = self.enabled is True
        self.isolate_env = bool(self.isolate_env)


def agentic_config_from_sleep_config(cfg: Mapping[str, Any]) -> AgenticReplayConfig:
    """Build an :class:`AgenticReplayConfig` from a SleepConfig/dict mapping."""
    return AgenticReplayConfig(
        enabled=cfg.get("agentic_replay", False) is True,
        fixture_root=str(cfg.get("agentic_fixture_root", "") or ""),
        max_turns=int(cfg.get("agentic_max_turns", 10) or 0),
        max_tokens=int(cfg.get("agentic_max_tokens", 0) or 0),
        timeout_sec=float(cfg.get("agentic_timeout_sec", 60.0) or 60.0),
        max_replays_per_run=int(cfg.get("agentic_max_replays", 5) or 0),
        isolate_env=bool(cfg.get("agentic_isolate_env", True)),
        min_decidable_actions=int(cfg.get("agentic_min_actions", 1) or 0),
        min_coverage=float(cfg.get("agentic_min_coverage", 0.5) or 0.5),
    )


def _ensure_within_root(target: str, root: str) -> str:
    """Resolve ``target`` and verify it stays strictly within ``root``."""
    if not root or not str(root).strip():
        raise FixtureSetupError("fixture_root must be configured when loading fixture directories")
    real_root = os.path.realpath(root)
    if not os.path.isdir(real_root):
        raise FixtureSetupError(f"fixture_root is not an existing directory: {root!r}")
    candidate = target if os.path.isabs(target) else os.path.join(real_root, target)
    real_target = os.path.realpath(candidate)
    try:
        common = os.path.commonpath([real_root, real_target])
    except ValueError as exc:
        raise FixtureSetupError(f"Fixture path {target!r} is outside fixture_root {root!r}") from exc
    if common != real_root:
        raise FixtureSetupError(f"Fixture path {target!r} escapes fixture_root {root!r}")
    if not os.path.isdir(real_target):
        raise FixtureSetupError(f"Fixture path is not a directory: {target!r}")
    return real_target


def _read_stream_capped(stream: Any, cap_bytes: int, dest: List[Tuple[bytes, bool]]) -> None:
    """Drain ``stream`` in chunks, retaining at most ``cap_bytes`` bytes in memory."""
    buf = bytearray()
    truncated = False
    limit = max(0, int(cap_bytes))
    try:
        while True:
            chunk = stream.read(65536)
            if not chunk:
                break
            if len(buf) < limit:
                take = limit - len(buf)
                buf.extend(chunk[:take])
                if len(chunk) > take:
                    truncated = True
            else:
                truncated = True
    except OSError:
        pass
    finally:
        try:
            stream.close()
        except OSError:
            pass
    dest.append((bytes(buf), truncated))


def _kill_process_tree(proc: subprocess.Popen[bytes]) -> None:
    """Terminate ``proc`` and any descendants started in its process group."""
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                timeout=5.0,
            )
        except (subprocess.SubprocessError, OSError):
            pass
        try:
            proc.kill()
        except OSError:
            pass
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            try:
                proc.kill()
            except OSError:
                pass


def _run_bounded_subprocess(
    cmd: Sequence[str],
    *,
    cwd: str,
    env: Optional[Dict[str, str]] = None,
    stdin_bytes: Optional[bytes] = None,
    timeout_sec: float = 60.0,
    max_output_bytes: int = 64 * 1024,
) -> Tuple[bool, int, bytes, bytes, bool, bool]:
    """Run ``cmd`` in its own process group with bounded output capture.

    Returns ``(timed_out, returncode, stdout_bytes, stderr_bytes, stdout_trunc, stderr_trunc)``.
    """
    if isinstance(cmd, (str, bytes)) or not cmd:
        raise ValueError("cmd must be a non-empty sequence of string arguments")
    cmd_list = [str(part) for part in cmd]

    popen_kwargs: Dict[str, Any] = {
        "cwd": cwd,
        "env": env,
        "stdin": subprocess.PIPE if stdin_bytes is not None else subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True

    proc = subprocess.Popen(cmd_list, **popen_kwargs)

    out_holder: List[Tuple[bytes, bool]] = []
    err_holder: List[Tuple[bytes, bool]] = []
    out_thread = threading.Thread(
        target=_read_stream_capped,
        args=(proc.stdout, max_output_bytes, out_holder),
        daemon=True,
    )
    err_thread = threading.Thread(
        target=_read_stream_capped,
        args=(proc.stderr, max_output_bytes, err_holder),
        daemon=True,
    )
    out_thread.start()
    err_thread.start()

    in_thread: Optional[threading.Thread] = None
    if stdin_bytes is not None and proc.stdin is not None:
        def _feed_stdin() -> None:
            try:
                proc.stdin.write(stdin_bytes)  # type: ignore[union-attr]
                proc.stdin.flush()  # type: ignore[union-attr]
            except OSError:
                pass
            finally:
                try:
                    proc.stdin.close()  # type: ignore[union-attr]
                except OSError:
                    pass

        in_thread = threading.Thread(target=_feed_stdin, daemon=True)
        in_thread.start()

    timed_out = False
    try:
        proc.wait(timeout=max(0.05, float(timeout_sec)))
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_process_tree(proc)
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            pass

    if in_thread is not None:
        in_thread.join(timeout=1.0)
    out_thread.join(timeout=1.5)
    err_thread.join(timeout=1.5)

    out_bytes, out_trunc = out_holder[0] if out_holder else (b"", False)
    err_bytes, err_trunc = err_holder[0] if err_holder else (b"", False)
    rc = proc.returncode if proc.returncode is not None else -1
    return timed_out, rc, out_bytes, err_bytes, out_trunc, err_trunc


class AgenticWorkspace:
    """Disposable scratch copy for running one agentic replay.

    Directory layout places ``home``, ``tmp``, XDG/CLI config directories,
    prompt payload files, and ``harness.git`` alongside—NOT inside—``work``,
    so agent tool writes to ``$HOME`` or git metadata cannot pollute ``get_diff()``.
    """

    def __init__(
        self,
        base_dir: Optional[str] = None,
        pinned_commit: Optional[str] = None,
        isolate_env: bool = True,
        *,
        fixture_root: str = "",
        max_diff_bytes: int = 256 * 1024,
        max_output_bytes: int = 64 * 1024,
    ) -> None:
        self.base_dir = base_dir
        self.pinned_commit = pinned_commit or ""
        self.isolate_env = bool(isolate_env)
        self.fixture_root = fixture_root
        self.max_diff_bytes = max(1024, int(max_diff_bytes))
        self.max_output_bytes = max(1024, int(max_output_bytes))

        self._root_dir = tempfile.mkdtemp(prefix="skillopt_agentic_")
        self.workspace_path = os.path.join(self._root_dir, "work")
        self.home_dir = os.path.join(self._root_dir, "home")
        self._tmp_dir = os.path.join(self._root_dir, "tmp")
        self._xdg_config_home = os.path.join(self._root_dir, "xdg_config")
        self._xdg_data_home = os.path.join(self._root_dir, "xdg_data")
        self._xdg_cache_home = os.path.join(self._root_dir, "xdg_cache")
        self._xdg_state_home = os.path.join(self._root_dir, "xdg_state")
        self._claude_config_dir = os.path.join(self._root_dir, "claude_config")
        self._codex_home = os.path.join(self._root_dir, "codex_home")
        self._prompt_dir = os.path.join(self._root_dir, "prompt")
        self._harness_git_dir = os.path.join(self._root_dir, "harness.git")
        self._empty_hooks_dir = os.path.join(self._root_dir, "empty_hooks")
        self._empty_gitconfig = os.path.join(self._root_dir, "empty_gitconfig")
        self._is_setup = False

    def __enter__(self) -> "AgenticWorkspace":
        self.setup()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.cleanup()

    def _ensure_layout_dirs(self) -> None:
        for d in (
            self.workspace_path,
            self.home_dir,
            self._tmp_dir,
            self._xdg_config_home,
            self._xdg_data_home,
            self._xdg_cache_home,
            self._xdg_state_home,
            self._claude_config_dir,
            self._codex_home,
            self._prompt_dir,
            self._harness_git_dir,
            self._empty_hooks_dir,
        ):
            os.makedirs(d, exist_ok=True)
        with open(self._empty_gitconfig, "wb"):
            pass

    def _allowlisted_env(self) -> Dict[str, str]:
        """Minimal allowlisted child environment with isolated empty home/config dirs."""
        self._ensure_layout_dirs()
        allowlist = list(_ENV_ALLOWLIST_COMMON)
        if os.name == "nt":
            allowlist.extend(_ENV_ALLOWLIST_WINDOWS)
        env: Dict[str, str] = {}
        for key in allowlist:
            val = os.environ.get(key)
            if val is not None:
                env[key] = val

        env.update({
            "HOME": self.home_dir,
            "USERPROFILE": self.home_dir,
            "TMPDIR": self._tmp_dir,
            "TMP": self._tmp_dir,
            "TEMP": self._tmp_dir,
            "XDG_CONFIG_HOME": self._xdg_config_home,
            "XDG_DATA_HOME": self._xdg_data_home,
            "XDG_CACHE_HOME": self._xdg_cache_home,
            "XDG_STATE_HOME": self._xdg_state_home,
            "CLAUDE_CONFIG_DIR": self._claude_config_dir,
            "CODEX_HOME": self._codex_home,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": self._empty_gitconfig,
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_ASKPASS": "",
            "SSH_ASKPASS": "",
            "NO_COLOR": "1",
        })
        if os.name == "nt":
            drive, tail = os.path.splitdrive(self.home_dir)
            if drive:
                env["HOMEDRIVE"] = drive
                env["HOMEPATH"] = tail or "\\"
        return env

    def build_isolated_env(self) -> Dict[str, str]:
        """Return the child subprocess environment according to ``isolate_env``."""
        self._ensure_layout_dirs()
        if self.isolate_env:
            return self._allowlisted_env()
        return dict(os.environ)

    def _safe_git_prefix(self, git_dir: str, work_tree: Optional[str] = None) -> List[str]:
        cmd = [
            "git",
            f"--git-dir={git_dir}",
        ]
        if work_tree is not None:
            cmd.append(f"--work-tree={work_tree}")
        cmd.extend([
            "-c", f"core.hooksPath={self._empty_hooks_dir}",
            "-c", "core.fsmonitor=",
            "-c", "diff.external=",
            "-c", "commit.gpgsign=false",
            "-c", "tag.gpgsign=false",
        ])
        return cmd

    def _run_harness_git(
        self,
        args: Sequence[str],
        *,
        git_dir: Optional[str] = None,
        work_tree: Optional[str] = None,
        cwd: Optional[str] = None,
        timeout_sec: float = 30.0,
        max_output_bytes: Optional[int] = None,
    ) -> Tuple[bytes, bool]:
        """Run harness-side git strictly under allowlisted env without repo config hooks."""
        env = self._allowlisted_env()
        cmd = (
            self._safe_git_prefix(git_dir, work_tree)
            if git_dir is not None
            else [
                "git",
                "-c", f"core.hooksPath={self._empty_hooks_dir}",
                "-c", "core.fsmonitor=",
                "-c", "diff.external=",
                "-c", "commit.gpgsign=false",
                "-c", "tag.gpgsign=false",
            ]
        )
        cmd.extend(args)
        timed_out, rc, out_b, err_b, out_trunc, _ = _run_bounded_subprocess(
            cmd,
            cwd=cwd or self._root_dir,
            env=env,
            timeout_sec=timeout_sec,
            max_output_bytes=max_output_bytes or self.max_output_bytes,
        )
        if timed_out:
            raise FixtureSetupError(f"Harness git command timed out: {' '.join(args[:3])}")
        if rc != 0:
            err_msg = err_b.decode("utf-8", errors="replace").strip()
            raise FixtureSetupError(
                f"Harness git command failed (exit {rc}): {err_msg or ' '.join(args[:3])}"
            )
        return out_b, out_trunc

    def _check_no_dotgit_pointers_or_alternates(
        self,
        directory: str,
        *,
        allow_top_git_dir: bool,
    ) -> None:
        """Reject linked ``.git`` files/symlinks and ``objects/info/alternates``."""
        top_dotgit = os.path.join(directory, ".git")
        if os.path.islink(top_dotgit):
            raise FixtureSetupError(f"Symlinked .git is not allowed in {directory!r}")
        if os.path.isfile(top_dotgit):
            raise FixtureSetupError(
                f"Linked .git file is not allowed without archive materialization: {directory!r}"
            )
        if os.path.isdir(top_dotgit):
            if not allow_top_git_dir:
                raise FixtureSetupError(f"Unexpected .git directory in {directory!r}")
            alt = os.path.join(top_dotgit, "objects", "info", "alternates")
            if os.path.exists(alt) or os.path.islink(alt):
                raise FixtureSetupError(f"Git alternates file is not allowed: {alt!r}")

        for dirpath, dirnames, filenames in os.walk(directory, followlinks=False):
            # Top-level .git dir was checked above; skip descending into it for nested checks
            if dirpath == directory and ".git" in dirnames:
                dirnames.remove(".git")
            for name in filenames:
                if name == ".git":
                    raise FixtureSetupError(
                        f"Nested .git file/submodule pointer is not allowed: {os.path.join(dirpath, name)!r}"
                    )
            for name in list(dirnames):
                full = os.path.join(dirpath, name)
                if name == ".git":
                    if os.path.islink(full) or not allow_top_git_dir:
                        raise FixtureSetupError(f"Nested .git entry is not allowed: {full!r}")
                if os.path.islink(full):
                    # Check symlink does not point outside directory
                    real_dir = os.path.realpath(directory)
                    real_target = os.path.realpath(full)
                    try:
                        common = os.path.commonpath([real_dir, real_target])
                    except ValueError as exc:
                        raise FixtureSetupError(f"Symlink escapes directory: {full!r}") from exc
                    if common != real_dir:
                        raise FixtureSetupError(f"Symlink escapes directory: {full!r}")

    def _check_source_repo_alternates(self, source_dir: str) -> None:
        """Verify git object store for ``source_dir`` has no ``objects/info/alternates``."""
        for flag in ("--git-dir", "--git-common-dir"):
            out_b, _ = self._run_harness_git(["-C", source_dir, "rev-parse", flag])
            raw_dir = out_b.decode("utf-8", errors="replace").strip()
            if not raw_dir:
                continue
            resolved = raw_dir if os.path.isabs(raw_dir) else os.path.join(source_dir, raw_dir)
            alt = os.path.join(os.path.realpath(resolved), "objects", "info", "alternates")
            if os.path.exists(alt) or os.path.islink(alt):
                raise FixtureSetupError(f"Git alternates file is not allowed in fixture: {alt!r}")

    def _materialize_via_git_archive(self, source_dir: str, pinned_commit: str) -> None:
        """Materialize ``pinned_commit`` from ``source_dir`` without copying any ``.git``."""
        if not re.fullmatch(r"[A-Za-z0-9._/@^{}~-]+", pinned_commit) or pinned_commit.startswith("-"):
            raise FixtureSetupError(f"Invalid pinned_commit reference: {pinned_commit!r}")
        self._check_source_repo_alternates(source_dir)

        sha_bytes, _ = self._run_harness_git([
            "-C", source_dir, "rev-parse", "--verify", "--end-of-options", f"{pinned_commit}^{{commit}}",
        ])
        resolved_sha = sha_bytes.decode("utf-8", errors="replace").strip()
        if not re.fullmatch(r"[0-9a-fA-F]{7,64}", resolved_sha):
            raise FixtureSetupError(f"Unexpected commit resolution for {pinned_commit!r}")

        archive_path = os.path.join(self._root_dir, "fixture_archive.tar")
        env = self._allowlisted_env()
        cmd = [
            "git",
            "-C", source_dir,
            "-c", f"core.hooksPath={self._empty_hooks_dir}",
            "-c", "core.fsmonitor=",
            "-c", "commit.gpgsign=false",
            "archive",
            "--format=tar",
            resolved_sha,
        ]
        try:
            with open(archive_path, "wb") as out_f:
                popen_kw: Dict[str, Any] = {
                    "cwd": self._root_dir,
                    "env": env,
                    "stdin": subprocess.DEVNULL,
                    "stdout": out_f,
                    "stderr": subprocess.PIPE,
                }
                if os.name == "nt":
                    popen_kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
                else:
                    popen_kw["start_new_session"] = True
                proc = subprocess.Popen(cmd, **popen_kw)
                err_holder: List[Tuple[bytes, bool]] = []
                err_t = threading.Thread(
                    target=_read_stream_capped,
                    args=(proc.stderr, self.max_output_bytes, err_holder),
                    daemon=True,
                )
                err_t.start()
                try:
                    proc.wait(timeout=30.0)
                except subprocess.TimeoutExpired as exc:
                    _kill_process_tree(proc)
                    err_t.join(timeout=1.0)
                    raise FixtureSetupError("git archive timed out") from exc
                err_t.join(timeout=1.0)
                if proc.returncode != 0:
                    err_b = err_holder[0][0] if err_holder else b""
                    raise FixtureSetupError(
                        f"git archive failed for {pinned_commit!r}: "
                        f"{err_b.decode('utf-8', errors='replace').strip()}"
                    )

            real_work = os.path.realpath(self.workspace_path)
            with tarfile.open(archive_path, mode="r:") as tf:
                safe_members: List[tarfile.TarInfo] = []
                for member in tf.getmembers():
                    if member.name in ("", ".") or member.name.startswith("./"):
                        rel = member.name[2:] if member.name.startswith("./") else ""
                    else:
                        rel = member.name
                    norm_parts = [p for p in rel.replace("\\", "/").split("/") if p]
                    if not norm_parts:
                        if member.isdir():
                            continue
                        raise FixtureSetupError("Unsafe empty archive member")
                    if os.path.isabs(rel) or os.path.splitdrive(rel)[0] or ".." in norm_parts:
                        raise FixtureSetupError(f"Archive path escape rejected: {member.name!r}")
                    if ".git" in norm_parts:
                        raise FixtureSetupError(f"Archive member inside .git rejected: {member.name!r}")
                    if member.isdev() or member.isfifo() or member.ischr() or member.isblk():
                        raise FixtureSetupError(f"Special device member rejected: {member.name!r}")
                    dest_path = os.path.realpath(os.path.join(real_work, *norm_parts))
                    if os.path.commonpath([real_work, dest_path]) != real_work:
                        raise FixtureSetupError(f"Archive target escapes workspace: {member.name!r}")
                    if member.issym() or member.islnk():
                        link_base = real_work if member.islnk() else os.path.dirname(dest_path)
                        link_target = os.path.realpath(os.path.join(link_base, member.linkname))
                        if os.path.commonpath([real_work, link_target]) != real_work:
                            raise FixtureSetupError(
                                f"Archive link target escapes workspace: {member.linkname!r}"
                            )
                    safe_members.append(member)
                tf.extractall(path=self.workspace_path, members=safe_members)
        except (tarfile.TarError, OSError) as exc:
            raise FixtureSetupError(f"Failed extracting archived fixture: {exc}") from exc
        finally:
            if os.path.exists(archive_path):
                try:
                    os.remove(archive_path)
                except OSError:
                    pass

    def _copy_plain_fixture(self, source_dir: str) -> None:
        """Copy plain fixture files into ``workspace_path`` without copying ``.git``."""
        self._check_no_dotgit_pointers_or_alternates(source_dir, allow_top_git_dir=True)
        real_source = os.path.realpath(source_dir)
        for dirpath, dirnames, filenames in os.walk(real_source, followlinks=False):
            dirnames[:] = [d for d in dirnames if d != ".git"]
            rel_dir = os.path.relpath(dirpath, real_source)
            dst_dir = (
                self.workspace_path
                if rel_dir == "."
                else os.path.join(self.workspace_path, rel_dir)
            )
            os.makedirs(dst_dir, exist_ok=True)
            for fname in filenames:
                if fname == ".git":
                    continue
                src_file = os.path.join(dirpath, fname)
                if os.path.islink(src_file):
                    real_target = os.path.realpath(src_file)
                    try:
                        common = os.path.commonpath([real_source, real_target])
                    except ValueError as exc:
                        raise FixtureSetupError(f"Fixture file symlink escapes root: {src_file!r}") from exc
                    if common != real_source:
                        raise FixtureSetupError(f"Fixture file symlink escapes root: {src_file!r}")
                    continue
                shutil.copy2(src_file, os.path.join(dst_dir, fname))

    def setup(self) -> None:
        """Materialize workspace copy and initialize separate harness + local git baselines."""
        if self._is_setup:
            return
        self._ensure_layout_dirs()

        try:
            self._run_harness_git(["init", "--bare", self._harness_git_dir])

            if self.base_dir:
                source_dir = (
                    _ensure_within_root(self.base_dir, self.fixture_root)
                    if self.fixture_root
                    else os.path.realpath(self.base_dir)
                )
                if not os.path.isdir(source_dir):
                    raise FixtureSetupError(f"Fixture directory does not exist: {self.base_dir!r}")

                if self.pinned_commit:
                    self._materialize_via_git_archive(source_dir, self.pinned_commit)
                else:
                    self._copy_plain_fixture(source_dir)
            elif self.pinned_commit:
                raise FixtureSetupError("pinned_commit requires a base_dir git fixture")

            # Seed harness bare git dir with baseline commit (never touches workspace .git)
            self._run_harness_git(
                ["add", "-A"],
                git_dir=self._harness_git_dir,
                work_tree=self.workspace_path,
                cwd=self.workspace_path,
            )
            self._run_harness_git(
                [
                    "-c", "user.name=SkillOpt Agentic Scratch",
                    "-c", "user.email=agentic@skillopt.local",
                    "commit", "--allow-empty", "-m", "baseline",
                ],
                git_dir=self._harness_git_dir,
                work_tree=self.workspace_path,
                cwd=self.workspace_path,
            )

            # Give the scratch copy its own self-contained .git so agent commands like
            # `git rev-parse HEAD` work without ever seeing the host repository.
            self._run_harness_git(["-C", self.workspace_path, "init"])
            self._run_harness_git(["-C", self.workspace_path, "add", "-A"])
            self._run_harness_git([
                "-C", self.workspace_path,
                "-c", "user.name=SkillOpt Agentic Scratch",
                "-c", "user.email=agentic@skillopt.local",
                "commit", "--allow-empty", "-m", "baseline",
            ])
            self._is_setup = True
        except FixtureSetupError:
            raise
        except (subprocess.SubprocessError, OSError) as exc:
            raise FixtureSetupError(f"Workspace setup failed: {exc}") from exc

    def build_prompt_text(self, task: TaskRecord, skill: str = "", memory: str = "") -> str:
        """Render a structured text prompt for the task, skill, and memory."""
        parts = [f"Task: {task.intent}"]
        if task.context_excerpt:
            parts.append(f"Context:\n{task.context_excerpt}")
        if skill:
            parts.append(f"Skill:\n{skill}")
        if memory:
            parts.append(f"Memory:\n{memory}")
        return "\n\n".join(parts)

    def write_prompt_payload(self, task: TaskRecord, skill: str = "", memory: str = "") -> str:
        """Write prompt JSON outside ``workspace_path`` so it stays out of ``get_diff()``."""
        self._ensure_layout_dirs()
        prompt_file = os.path.join(self._prompt_dir, "task_payload.json")
        payload = {
            "task_id": task.id,
            "intent": task.intent,
            "context_excerpt": task.context_excerpt,
            "skill": skill,
            "memory": memory,
            "prompt": self.build_prompt_text(task, skill=skill, memory=memory),
        }
        with open(prompt_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return prompt_file

    def run_subprocess(
        self,
        cmd: Sequence[str],
        *,
        stdin_text: Optional[str] = None,
        timeout_sec: float = 60.0,
        env: Optional[Dict[str, str]] = None,
        max_output_bytes: Optional[int] = None,
    ) -> Tuple[bool, int, str, str]:
        """Execute a list-argv command inside ``workspace_path`` with bounded pipes.

        Returns ``(timed_out, returncode, stdout_text, stderr_text)``.
        """
        if not self._is_setup:
            self.setup()
        effective_env = env if env is not None else self.build_isolated_env()
        cap = self.max_output_bytes if max_output_bytes is None else max_output_bytes
        stdin_b = stdin_text.encode("utf-8", errors="replace") if stdin_text is not None else None
        timed_out, rc, out_b, err_b, out_trunc, err_trunc = _run_bounded_subprocess(
            cmd,
            cwd=self.workspace_path,
            env=effective_env,
            stdin_bytes=stdin_b,
            timeout_sec=timeout_sec,
            max_output_bytes=cap,
        )
        out_txt = out_b.decode("utf-8", errors="replace")
        err_txt = err_b.decode("utf-8", errors="replace")
        if out_trunc:
            out_txt += f"\n... [stdout truncated at {cap} bytes]"
        if err_trunc:
            err_txt += f"\n... [stderr truncated at {cap} bytes]"
        return timed_out, rc, out_txt, err_txt

    def get_diff(self) -> str:
        """Compute unified diff via harness ``harness.git`` without reading ``work/.git/config``."""
        if not self._is_setup or not os.path.isdir(self.workspace_path):
            return ""
        self._check_no_dotgit_pointers_or_alternates(self.workspace_path, allow_top_git_dir=True)
        self._run_harness_git(
            ["add", "-N", "."],
            git_dir=self._harness_git_dir,
            work_tree=self.workspace_path,
            cwd=self.workspace_path,
        )
        diff_b, truncated = self._run_harness_git(
            ["diff", "--no-ext-diff", "--no-textconv", "HEAD"],
            git_dir=self._harness_git_dir,
            work_tree=self.workspace_path,
            cwd=self.workspace_path,
            max_output_bytes=self.max_diff_bytes,
        )
        text = diff_b.decode("utf-8", errors="replace")
        if truncated:
            text += f"\n... [diff truncated at {self.max_diff_bytes} bytes]\n"
        return text

    def cleanup(self) -> None:
        """Remove temporary scratch directories (clearing read-only git object attrs)."""
        if not os.path.exists(self._root_dir):
            self._is_setup = False
            return

        def _handle_readonly(func: Any, path: str, _exc: Any) -> None:
            try:
                os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
                func(path)
            except OSError:
                pass

        if sys.version_info >= (3, 12):
            shutil.rmtree(self._root_dir, onexc=_handle_readonly)
        else:
            shutil.rmtree(self._root_dir, onerror=_handle_readonly)
        # Ensure clean removal even if a transient handle delayed deletion
        if os.path.exists(self._root_dir):
            shutil.rmtree(self._root_dir, ignore_errors=True)
        self._is_setup = False


class TaskVerifier:
    """Deterministic task verifier acting as the primary acceptance gate."""

    def __init__(
        self,
        test_cmd: Optional[Sequence[str]] = None,
        expected_files: Optional[Dict[str, Optional[str]]] = None,
        verifier_fn: Optional[Callable[[str], Tuple[bool, str]]] = None,
        timeout_sec: float = 30.0,
        max_output_bytes: int = 64 * 1024,
    ) -> None:
        if test_cmd is not None and isinstance(test_cmd, (str, bytes)):
            raise ValueError("test_cmd must be a list/sequence of string arguments")
        self.test_cmd = [str(part) for part in test_cmd] if test_cmd else None
        self.expected_files = dict(expected_files) if expected_files else {}
        self.verifier_fn = verifier_fn
        self.timeout_sec = float(timeout_sec)
        self.max_output_bytes = max(1024, int(max_output_bytes))

    def verify(
        self,
        workspace_path: str,
        *,
        env: Optional[Dict[str, str]] = None,
    ) -> Tuple[str, float, float, str, str]:
        """Run checks against ``workspace_path``.

        Returns ``(outcome, hard, soft, rationale, inconclusive_reason)``.
        """
        if not self.verifier_fn and not self.expected_files and not self.test_cmd:
            return (
                OUTCOME_INCONCLUSIVE,
                0.0,
                0.0,
                "No task verifier configured",
                REASON_INSUFFICIENT_EVIDENCE,
            )

        if self.verifier_fn is not None:
            try:
                passed, msg = self.verifier_fn(workspace_path)
                if passed:
                    return (OUTCOME_PASS, 1.0, 1.0, msg or "Verifier passed", "")
                return (OUTCOME_FAIL, 0.0, 0.0, msg or "Verifier failed", "")
            except Exception as exc:
                return (
                    OUTCOME_INCONCLUSIVE,
                    0.0,
                    0.0,
                    f"Verifier error: {exc}",
                    REASON_INFRA,
                )

        real_ws = os.path.realpath(workspace_path)
        for rel_path, expected_substr in self.expected_files.items():
            full_path = os.path.realpath(os.path.join(real_ws, rel_path))
            try:
                common = os.path.commonpath([real_ws, full_path])
            except ValueError:
                common = ""
            if common != real_ws:
                return (
                    OUTCOME_INCONCLUSIVE,
                    0.0,
                    0.0,
                    f"Expected file path escapes workspace: {rel_path}",
                    REASON_FIXTURE_FAILURE,
                )
            if not os.path.exists(full_path):
                return (OUTCOME_FAIL, 0.0, 0.0, f"Expected file missing: {rel_path}", "")
            if expected_substr is not None:
                try:
                    with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()
                except OSError as exc:
                    return (OUTCOME_FAIL, 0.0, 0.0, f"Cannot read {rel_path}: {exc}", "")
                if expected_substr not in content:
                    return (
                        OUTCOME_FAIL,
                        0.0,
                        0.0,
                        f"Expected content missing in {rel_path}",
                        "",
                    )

        if self.test_cmd:
            try:
                timed_out, rc, out_b, err_b, out_trunc, err_trunc = _run_bounded_subprocess(
                    self.test_cmd,
                    cwd=workspace_path,
                    env=env,
                    timeout_sec=self.timeout_sec,
                    max_output_bytes=self.max_output_bytes,
                )
                if timed_out:
                    return (
                        OUTCOME_INCONCLUSIVE,
                        0.0,
                        0.0,
                        f"Verifier command timed out after {self.timeout_sec}s",
                        REASON_TIMEOUT,
                    )
                if rc == 0:
                    return (OUTCOME_PASS, 1.0, 1.0, "Verifier test command passed", "")
                out_s = (out_b + err_b).decode("utf-8", errors="replace").strip()
                suffix = " [truncated]" if (out_trunc or err_trunc) else ""
                return (
                    OUTCOME_FAIL,
                    0.0,
                    0.0,
                    f"Verifier exited with {rc}: {out_s[:300]}{suffix}",
                    "",
                )
            except (subprocess.SubprocessError, OSError, ValueError) as exc:
                return (
                    OUTCOME_INCONCLUSIVE,
                    0.0,
                    0.0,
                    f"Verifier infrastructure error: {exc}",
                    REASON_INFRA,
                )

        return (OUTCOME_PASS, 1.0, 1.0, "All configured verification checks passed", "")


def _invoke_runner(
    runner: Callable[..., Any],
    workspace_path: str,
    env: Dict[str, str],
    context: Dict[str, Any],
) -> Any:
    """Call ``runner`` with ``(workspace_path, env, context)`` or ``(workspace_path, env)``."""
    try:
        sig = inspect.signature(runner)
    except (ValueError, TypeError):
        return runner(workspace_path, env, context)
    params = list(sig.parameters.values())
    accepts_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
    positional = [
        p for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if accepts_varargs or len(positional) >= 3:
        return runner(workspace_path, env, context)
    if any(p.kind == inspect.Parameter.KEYWORD_ONLY and p.name == "context" for p in params):
        return runner(workspace_path, env, context=context)
    return runner(workspace_path, env)


class AgenticReplayBackend:
    """Tiered opt-in agentic replay backend running in disposable scratch copies."""

    name = "agentic"

    def __init__(
        self,
        config: Optional[AgenticReplayConfig] = None,
        *,
        base_backend: Optional[Any] = None,
        agent_runner: Optional[Callable[..., Any]] = None,
        default_verifier: Optional[TaskVerifier] = None,
    ) -> None:
        self.config = config or AgenticReplayConfig()
        self.base_backend = base_backend
        self.agent_runner = agent_runner
        self.default_verifier = default_verifier
        self._lock = threading.Lock()
        self.replays_run = 0
        self.tokens_spent = 0
        self.preferences = ""

    def reset_run_budget(self) -> None:
        """Reset per-night/run replay and token counters."""
        with self._lock:
            self.replays_run = 0
            self.tokens_spent = 0

    def tokens_used(self) -> int:
        with self._lock:
            local = self.tokens_spent
        base = self.base_backend.tokens_used() if self.base_backend is not None else 0
        return local + base

    def attempt(
        self, task: TaskRecord, skill: str, memory: str, sample_id: int = 0
    ) -> str:
        if self.base_backend is not None:
            return self.base_backend.attempt(task, skill, memory, sample_id=sample_id)
        return ""

    def reflect(
        self,
        failures: List[Tuple[TaskRecord, ReplayResult]],
        successes: List[Tuple[TaskRecord, ReplayResult]],
        skill: str,
        memory: str,
        *,
        edit_budget: int = 4,
        evolve_skill: bool = True,
        evolve_memory: bool = True,
        **kwargs: Any,
    ) -> List[EditRecord]:
        """Delegate reflection to ``base_backend``, excluding inconclusive runs."""
        filtered_failures = [
            (t, r) for (t, r) in failures
            if getattr(r, "outcome", "") != OUTCOME_INCONCLUSIVE
        ]
        filtered_successes = [
            (t, r) for (t, r) in successes
            if getattr(r, "outcome", "") != OUTCOME_INCONCLUSIVE
        ]
        if self.base_backend is not None:
            return self.base_backend.reflect(
                filtered_failures,
                filtered_successes,
                skill,
                memory,
                edit_budget=edit_budget,
                evolve_skill=evolve_skill,
                evolve_memory=evolve_memory,
                **kwargs,
            )
        return []

    def judge(self, task: TaskRecord, response: str) -> Tuple[float, float, str]:
        if self.base_backend is not None:
            return self.base_backend.judge(task, response)
        return (0.0, 0.0, "No base backend judge available")

    def _reserve_budget(self) -> Optional[str]:
        """Thread-safely reserve one replay slot, returning fail reason if exhausted."""
        with self._lock:
            if (
                self.config.max_replays_per_run > 0
                and self.replays_run >= self.config.max_replays_per_run
            ):
                return "Max agentic replays budget reached for this run"
            if (
                self.config.max_tokens > 0
                and self.tokens_spent >= self.config.max_tokens
            ):
                return "Max agentic token budget reached for this run"
            self.replays_run += 1
        return None

    def _resolve_fixture_dir(
        self, task: TaskRecord, fixture_dir: Optional[str]
    ) -> Optional[str]:
        """Resolve an optional fixture directory exclusively from ``fixture_root``."""
        requested = fixture_dir
        if not requested and isinstance(task.judge, Mapping):
            raw = task.judge.get("fixture_dir")
            if isinstance(raw, str) and raw.strip():
                requested = raw.strip()
        if requested:
            return _ensure_within_root(requested, self.config.fixture_root)
        if self.config.fixture_root:
            return _ensure_within_root(self.config.fixture_root, self.config.fixture_root)
        return None

    def replay_agentic_task(
        self,
        task: TaskRecord,
        skill: str = "",
        memory: str = "",
        *,
        verifier: Optional[TaskVerifier] = None,
        fixture_dir: Optional[str] = None,
        sample_id: int = 0,
    ) -> ReplayResult:
        """Run one opt-in agentic replay inside a disposable scratch copy."""
        del sample_id
        t0 = time.monotonic()

        if self.config.enabled is not True:
            return ReplayResult(
                id=task.id,
                hard=0.0,
                soft=0.0,
                outcome=OUTCOME_INCONCLUSIVE,
                inconclusive_reason=REASON_INFRA,
                fail_reason="Agentic replay is disabled (set agentic_replay=True to opt in)",
                task_type="agentic",
            )

        if self.agent_runner is None:
            return ReplayResult(
                id=task.id,
                hard=0.0,
                soft=0.0,
                outcome=OUTCOME_INCONCLUSIVE,
                inconclusive_reason=REASON_INFRA,
                fail_reason="No agent_runner configured (live CLI execution is disabled in this slice)",
                task_type="agentic",
            )

        budget_err = self._reserve_budget()
        if budget_err is not None:
            return ReplayResult(
                id=task.id,
                hard=0.0,
                soft=0.0,
                outcome=OUTCOME_INCONCLUSIVE,
                inconclusive_reason=REASON_BUDGET_EXHAUSTED,
                fail_reason=budget_err,
                task_type="agentic",
            )

        try:
            resolved_fixture = self._resolve_fixture_dir(task, fixture_dir)
        except FixtureSetupError as exc:
            return ReplayResult(
                id=task.id,
                hard=0.0,
                soft=0.0,
                outcome=OUTCOME_INCONCLUSIVE,
                inconclusive_reason=REASON_FIXTURE_FAILURE,
                fail_reason=str(exc),
                latency_ms=(time.monotonic() - t0) * 1000.0,
                task_type="agentic",
            )

        ws = AgenticWorkspace(
            base_dir=resolved_fixture,
            pinned_commit=self.config.pinned_commit or None,
            isolate_env=self.config.isolate_env,
            fixture_root=self.config.fixture_root,
            max_diff_bytes=self.config.max_diff_bytes,
            max_output_bytes=self.config.max_output_bytes,
        )
        try:
            try:
                ws.setup()
            except FixtureSetupError as exc:
                return ReplayResult(
                    id=task.id,
                    hard=0.0,
                    soft=0.0,
                    outcome=OUTCOME_INCONCLUSIVE,
                    inconclusive_reason=REASON_FIXTURE_FAILURE,
                    fail_reason=str(exc),
                    latency_ms=(time.monotonic() - t0) * 1000.0,
                    task_type="agentic",
                )

            env = ws.build_isolated_env()
            prompt_file = ws.write_prompt_payload(task, skill=skill, memory=memory)
            prompt_text = ws.build_prompt_text(task, skill=skill, memory=memory)
            prompt_ctx = {
                "task": task,
                "task_id": task.id,
                "intent": task.intent,
                "context_excerpt": task.context_excerpt,
                "skill": skill,
                "memory": memory,
                "prompt": prompt_text,
                "prompt_file": prompt_file,
                "workspace": ws,
                "timeout_sec": self.config.timeout_sec,
                "max_turns": self.config.max_turns,
                "max_output_bytes": self.config.max_output_bytes,
            }

            response_text = ""
            actions_taken = 0
            turns_taken = 0
            tokens_used_now = 0
            runner_rc = 0
            runner_outcome = ""
            runner_reason = ""
            runner_fail_msg = ""

            try:
                res = _invoke_runner(self.agent_runner, ws.workspace_path, env, prompt_ctx)
                if isinstance(res, dict):
                    response_text = str(res.get("response", ""))
                    actions_taken = int(res.get("actions", 1))
                    turns_taken = int(res.get("turns", actions_taken))
                    tokens_used_now = max(0, int(res.get("tokens", 0) or 0))
                    raw_rc = res.get("returncode", res.get("exit_code", 0))
                    runner_rc = 0 if raw_rc is None else int(raw_rc)
                    runner_outcome = str(res.get("outcome", "") or "")
                    runner_reason = str(res.get("inconclusive_reason", "") or "")
                    runner_fail_msg = str(res.get("fail_reason", "") or "")
                elif isinstance(res, tuple) and len(res) >= 2:
                    response_text = str(res[0])
                    actions_taken = int(res[1])
                    turns_taken = actions_taken
                else:
                    response_text = str(res)
                    actions_taken = 1
                    turns_taken = 1
            except subprocess.TimeoutExpired:
                return ReplayResult(
                    id=task.id,
                    hard=0.0,
                    soft=0.0,
                    outcome=OUTCOME_INCONCLUSIVE,
                    inconclusive_reason=REASON_TIMEOUT,
                    fail_reason=f"Agent execution timed out after {self.config.timeout_sec}s",
                    latency_ms=(time.monotonic() - t0) * 1000.0,
                    task_type="agentic",
                )
            except FixtureSetupError as exc:
                return ReplayResult(
                    id=task.id,
                    hard=0.0,
                    soft=0.0,
                    outcome=OUTCOME_INCONCLUSIVE,
                    inconclusive_reason=REASON_FIXTURE_FAILURE,
                    fail_reason=str(exc),
                    latency_ms=(time.monotonic() - t0) * 1000.0,
                    task_type="agentic",
                )
            except Exception as exc:
                return ReplayResult(
                    id=task.id,
                    hard=0.0,
                    soft=0.0,
                    outcome=OUTCOME_INCONCLUSIVE,
                    inconclusive_reason=REASON_INFRA,
                    fail_reason=f"Agent runner infrastructure error: {exc}",
                    latency_ms=(time.monotonic() - t0) * 1000.0,
                    task_type="agentic",
                )

            with self._lock:
                self.tokens_spent += tokens_used_now
                over_token_budget = (
                    self.config.max_tokens > 0 and self.tokens_spent > self.config.max_tokens
                )

            if runner_outcome == OUTCOME_INCONCLUSIVE:
                return ReplayResult(
                    id=task.id,
                    hard=0.0,
                    soft=0.0,
                    response=response_text[:4000],
                    tokens=tokens_used_now,
                    outcome=OUTCOME_INCONCLUSIVE,
                    inconclusive_reason=runner_reason or REASON_INFRA,
                    fail_reason=runner_fail_msg or "Runner reported inconclusive outcome",
                    latency_ms=(time.monotonic() - t0) * 1000.0,
                    task_type="agentic",
                )

            if runner_rc != 0:
                return ReplayResult(
                    id=task.id,
                    hard=0.0,
                    soft=0.0,
                    response=response_text[:4000],
                    tokens=tokens_used_now,
                    outcome=OUTCOME_INCONCLUSIVE,
                    inconclusive_reason=REASON_INFRA,
                    fail_reason=runner_fail_msg or f"Agent process exited with code {runner_rc}",
                    latency_ms=(time.monotonic() - t0) * 1000.0,
                    task_type="agentic",
                )

            if over_token_budget or (
                self.config.max_turns > 0 and turns_taken > self.config.max_turns
            ):
                return ReplayResult(
                    id=task.id,
                    hard=0.0,
                    soft=0.0,
                    response=response_text[:4000],
                    tokens=tokens_used_now,
                    outcome=OUTCOME_INCONCLUSIVE,
                    inconclusive_reason=REASON_BUDGET_EXHAUSTED,
                    fail_reason=(
                        "Max agentic token budget exceeded"
                        if over_token_budget
                        else f"Turn budget exceeded ({turns_taken} > {self.config.max_turns})"
                    ),
                    latency_ms=(time.monotonic() - t0) * 1000.0,
                    task_type="agentic",
                )

            if actions_taken < self.config.min_decidable_actions:
                return ReplayResult(
                    id=task.id,
                    hard=0.0,
                    soft=0.0,
                    response=response_text[:4000],
                    tokens=tokens_used_now,
                    outcome=OUTCOME_INCONCLUSIVE,
                    inconclusive_reason=REASON_INSUFFICIENT_EVIDENCE,
                    fail_reason=(
                        f"Actions taken ({actions_taken}) below decidable floor "
                        f"({self.config.min_decidable_actions})"
                    ),
                    latency_ms=(time.monotonic() - t0) * 1000.0,
                    task_type="agentic",
                )

            try:
                diff = ws.get_diff()
            except FixtureSetupError as exc:
                return ReplayResult(
                    id=task.id,
                    hard=0.0,
                    soft=0.0,
                    response=response_text[:4000],
                    tokens=tokens_used_now,
                    outcome=OUTCOME_INCONCLUSIVE,
                    inconclusive_reason=REASON_FIXTURE_FAILURE,
                    fail_reason=str(exc),
                    latency_ms=(time.monotonic() - t0) * 1000.0,
                    task_type="agentic",
                )

            active_verifier = (
                verifier
                or self.default_verifier
                or TaskVerifier(
                    timeout_sec=self.config.timeout_sec,
                    max_output_bytes=self.config.max_output_bytes,
                )
            )
            outcome, hard, soft, rationale, inc_reason = active_verifier.verify(
                ws.workspace_path, env=env
            )
            latency_ms = (time.monotonic() - t0) * 1000.0
            fail_reason = "" if outcome == OUTCOME_PASS else rationale

            return ReplayResult(
                id=task.id,
                hard=hard,
                soft=soft,
                response=response_text[:4000],
                fail_reason=fail_reason,
                task_type="agentic",
                judge_rationale=rationale,
                tokens=tokens_used_now,
                latency_ms=latency_ms,
                outcome=outcome,
                inconclusive_reason=inc_reason,
                diff=diff,
            )
        finally:
            ws.cleanup()
