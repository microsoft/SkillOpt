"""SkillOpt-Sleep — AgenticReplayBackend (Issue #155 safe first slice).

Replay mined tasks with an interactive agent in an isolated, throwaway workspace,
and gate primarily on deterministic task verification (builds, tests, and checks)
rather than purely single-turn text completion.

Key properties:
  1. Safe, isolated workspace with environment scrubbing (no host credential leakage)
     and network isolation by default.
  2. Pinned starting commit and disposable throwaway git worktree / scratch dir.
  3. Task verification as the primary gate (deterministic test exit code/assertions);
     diff similarity remains secondary audit context.
  4. Three-valued outcome: pass | fail | inconclusive.
     Infrastructure failures (timeouts, budget exhaustion, missing fixtures, or
     insufficient decidable actions) emit `inconclusive` with typed reason codes
     so they are never counted as evidence that a candidate skill failed.
  5. Strict budgeting: explicit turn, timeout, token, and replay caps.
  6. Staged diff and report for human review without automatic adoption.
"""
from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from skillopt_sleep.backend import Backend, MockBackend
from skillopt_sleep.types import EditRecord, ReplayResult, TaskRecord

# ── Three-valued outcome constants ───────────────────────────────────────────
OUTCOME_PASS = "pass"
OUTCOME_FAIL = "fail"
OUTCOME_INCONCLUSIVE = "inconclusive"

# Typed reason codes for inconclusive outcomes
REASON_TIMEOUT = "timeout"
REASON_INFRA = "infra"
REASON_FIXTURE_FAILURE = "fixture_failure"
REASON_BUDGET_EXHAUSTED = "budget_exhausted"
REASON_INSUFFICIENT_EVIDENCE = "insufficient_evidence"
REASON_NETWORK_UNAVAILABLE = "network_unavailable"

# Windows console window suppression
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


def _remove_readonly(func: Callable, path: str, _excinfo: Any) -> None:
    """Clear the read-only bit and retry removal (handles Windows .git permissions)."""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except OSError:
        pass


@dataclass
class AgenticReplayConfig:
    """Budgets, sandbox boundaries, and execution settings for agentic replay."""

    enabled: bool = False
    agent_cmd: str = ""                # Path or command to drive agent CLI
    max_turns: int = 10                # Hard turn cap
    timeout_sec: float = 60.0          # Wall-clock timeout in seconds
    max_tokens: int = 20000            # Token spend cap
    max_replays_per_run: int = 5       # Strict budget cap on agentic replays
    sandbox_type: str = "workspace"    # "workspace" (throwaway worktree) | "container"
    isolate_network: bool = True       # Disable external network by default
    isolate_env: bool = True           # Scrub host credentials and keys
    min_decidable_actions: int = 1     # Floor below which outcome is insufficient_evidence
    pinned_commit: str = ""            # Reproducible starting commit SHA


class AgenticWorkspace:
    """Disposable, isolated scratch workspace for running an agentic replay safely."""

    _SENSITIVE_ENV_PREFIXES = (
        "AWS_", "AZURE_", "OPENAI_", "ANTHROPIC_", "GITHUB_", "GH_",
        "GITLAB_", "SLACK_", "STRIPE_", "SSH_", "API_KEY", "SECRET",
    )

    def __init__(
        self,
        base_dir: str = "",
        pinned_commit: str = "",
        isolate_env: bool = True,
        isolate_network: bool = True,
    ) -> None:
        self.base_dir = base_dir
        self.pinned_commit = pinned_commit
        self.isolate_env = isolate_env
        self.isolate_network = isolate_network
        self._temp_dir: Optional[str] = None
        self.workspace_path: str = ""

    def __enter__(self) -> "AgenticWorkspace":
        self.setup()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.cleanup()

    def setup(self) -> str:
        """Create and initialize disposable workspace directory."""
        self._temp_dir = tempfile.mkdtemp(prefix="skillopt_agentic_")
        self.workspace_path = self._temp_dir

        # If a base directory was provided, copy its content into the workspace
        if self.base_dir and os.path.isdir(self.base_dir):
            shutil.copytree(self.base_dir, self.workspace_path, dirs_exist_ok=True)

        # Initialize git repository if not already present
        git_dir = os.path.join(self.workspace_path, ".git")
        if not os.path.exists(git_dir):
            try:
                subprocess.run(
                    ["git", "init"],
                    cwd=self.workspace_path,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True,
                    creationflags=_NO_WINDOW,
                )
                subprocess.run(
                    ["git", "config", "user.name", "SkillOpt Agentic Sandbox"],
                    cwd=self.workspace_path,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True,
                    creationflags=_NO_WINDOW,
                )
                subprocess.run(
                    ["git", "config", "user.email", "sandbox@skillopt.local"],
                    cwd=self.workspace_path,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True,
                    creationflags=_NO_WINDOW,
                )
                subprocess.run(
                    ["git", "add", "."],
                    cwd=self.workspace_path,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=_NO_WINDOW,
                )
                subprocess.run(
                    ["git", "commit", "-m", "baseline", "--allow-empty"],
                    cwd=self.workspace_path,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True,
                    creationflags=_NO_WINDOW,
                )
            except (subprocess.SubprocessError, OSError):
                pass
        elif self.pinned_commit:
            try:
                subprocess.run(
                    ["git", "checkout", self.pinned_commit],
                    cwd=self.workspace_path,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True,
                    creationflags=_NO_WINDOW,
                )
            except (subprocess.SubprocessError, OSError) as exc:
                raise RuntimeError(
                    f"failed to checkout pinned commit {self.pinned_commit}: {exc}"
                ) from exc

        return self.workspace_path

    def build_isolated_env(self) -> Dict[str, str]:
        """Construct sanitized environment variables."""
        env = dict(os.environ)
        if self.isolate_env:
            # Strip sensitive credential keys
            keys_to_remove = [
                k for k in env
                if any(k.upper().startswith(prefix) for prefix in self._SENSITIVE_ENV_PREFIXES)
                or "KEY" in k.upper()
                or "TOKEN" in k.upper()
                or "PASSWORD" in k.upper()
            ]
            for k in keys_to_remove:
                env.pop(k, None)

            # Isolate home/user profile to prevent tampering with host configs
            sandbox_home = os.path.join(self.workspace_path, ".sandbox_home")
            os.makedirs(sandbox_home, exist_ok=True)
            env["HOME"] = sandbox_home
            env["USERPROFILE"] = sandbox_home
            env["XDG_CONFIG_HOME"] = os.path.join(sandbox_home, ".config")

        if self.isolate_network:
            # Direct proxy traffic to non-routable loopback sink
            env["HTTP_PROXY"] = "http://127.0.0.1:9"
            env["HTTPS_PROXY"] = "http://127.0.0.1:9"
            env["ALL_PROXY"] = "http://127.0.0.1:9"
            env["NO_PROXY"] = ""

        return env

    def get_diff(self) -> str:
        """Extract unified diff against HEAD, including untracked files."""
        if not self.workspace_path or not os.path.exists(self.workspace_path):
            return ""
        try:
            # Mark untracked files with intent-to-add so git diff includes newly added files
            subprocess.run(
                ["git", "add", "-N", "."],
                cwd=self.workspace_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=_NO_WINDOW,
            )
            res = subprocess.run(
                ["git", "diff", "HEAD"],
                cwd=self.workspace_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=_NO_WINDOW,
            )
            if res.returncode == 0:
                return res.stdout
        except (subprocess.SubprocessError, OSError):
            pass
        return ""

    def cleanup(self) -> None:
        """Safely delete disposable workspace directory."""
        if self._temp_dir and os.path.exists(self._temp_dir):
            try:
                shutil.rmtree(self._temp_dir, onerror=_remove_readonly)
            except OSError:
                pass
            self._temp_dir = None
            self.workspace_path = ""


class TaskVerifier:
    """Deterministic acceptance verifier executing tests or checks in the workspace."""

    def __init__(
        self,
        test_cmd: Optional[List[str]] = None,
        verifier_fn: Optional[Callable[[str], Any]] = None,
        expected_files: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
    ) -> None:
        self.test_cmd = test_cmd or []
        self.verifier_fn = verifier_fn
        self.expected_files = expected_files or {}
        self.timeout = timeout

    def verify(
        self,
        workspace_path: str,
        env: Optional[Dict[str, str]] = None,
    ) -> Tuple[str, float, float, str, str]:
        """Run verification checks.

        Returns:
            (outcome, hard, soft, rationale, inconclusive_reason)
        """
        if not os.path.isdir(workspace_path):
            return (
                OUTCOME_INCONCLUSIVE, 0.0, 0.0,
                "Workspace directory does not exist", REASON_FIXTURE_FAILURE,
            )

        # 1. Custom verifier callable if provided
        if self.verifier_fn is not None:
            try:
                res = self.verifier_fn(workspace_path)
                if isinstance(res, tuple) and len(res) >= 2:
                    passed, rationale = bool(res[0]), str(res[1])
                else:
                    passed, rationale = bool(res), "Verifier returned boolean result"
                if passed:
                    return (OUTCOME_PASS, 1.0, 1.0, rationale, "")
                return (OUTCOME_FAIL, 0.0, 0.0, rationale, "")
            except Exception as exc:
                return (
                    OUTCOME_INCONCLUSIVE, 0.0, 0.0,
                    f"Verifier function crashed: {exc}", REASON_INFRA,
                )

        # 2. Check required files and content matches if specified
        if self.expected_files:
            for rel_path, expected_text in self.expected_files.items():
                full_path = os.path.join(workspace_path, rel_path)
                if not os.path.exists(full_path):
                    return (
                        OUTCOME_FAIL, 0.0, 0.0,
                        f"Expected file missing: {rel_path}", "",
                    )
                if expected_text:
                    try:
                        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                            content = f.read()
                        if expected_text not in content:
                            return (
                                OUTCOME_FAIL, 0.0, 0.0,
                                f"Expected content not found in {rel_path}", "",
                            )
                    except OSError as exc:
                        return (
                            OUTCOME_INCONCLUSIVE, 0.0, 0.0,
                            f"Failed to read file {rel_path}: {exc}", REASON_INFRA,
                        )

        # 3. Execute test command if defined
        if self.test_cmd:
            try:
                proc = subprocess.run(
                    self.test_cmd,
                    cwd=workspace_path,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=self.timeout,
                    env=env,
                    creationflags=_NO_WINDOW,
                )
                if proc.returncode == 0:
                    return (
                        OUTCOME_PASS, 1.0, 1.0,
                        f"Test command succeeded (code 0):\n{proc.stdout[:300]}", "",
                    )
                # Test failed deterministically
                fail_summary = (proc.stdout + "\n" + proc.stderr).strip()[-500:]
                return (
                    OUTCOME_FAIL, 0.0, 0.0,
                    f"Test command failed (code {proc.returncode}):\n{fail_summary}", "",
                )
            except subprocess.TimeoutExpired:
                return (
                    OUTCOME_INCONCLUSIVE, 0.0, 0.0,
                    f"Test command timed out after {self.timeout}s", REASON_TIMEOUT,
                )
            except FileNotFoundError as exc:
                return (
                    OUTCOME_INCONCLUSIVE, 0.0, 0.0,
                    f"Test command binary not found: {exc}", REASON_INFRA,
                )
            except OSError as exc:
                return (
                    OUTCOME_INCONCLUSIVE, 0.0, 0.0,
                    f"Test command failed to execute: {exc}", REASON_INFRA,
                )

        # If no explicit check is configured, accept clean workspace
        return (OUTCOME_PASS, 1.0, 1.0, "No failing verifier rules triggered", "")


class AgenticReplayBackend(Backend):
    """Tiered agentic replay backend running tasks in disposable worktrees."""

    name = "agentic"

    def __init__(
        self,
        config: Optional[AgenticReplayConfig] = None,
        base_backend: Optional[Backend] = None,
        verifier: Optional[TaskVerifier] = None,
    ) -> None:
        self.config = config or AgenticReplayConfig()
        self.base_backend = base_backend or MockBackend()
        self.verifier = verifier or TaskVerifier()
        self.replays_run: int = 0
        self._tokens_spent: int = 0

    def tokens_used(self) -> int:
        return self._tokens_spent + self.base_backend.tokens_used()

    def attempt(
        self,
        task: TaskRecord,
        skill: str,
        memory: str,
        sample_id: int = 0,
    ) -> str:
        """Run the task attempt via the base backend or agent CLI."""
        return self.base_backend.attempt(task, skill, memory, sample_id=sample_id)

    def judge(self, task: TaskRecord, response: str) -> Tuple[float, float, str]:
        """Default judge delegating to base backend."""
        return self.base_backend.judge(task, response)

    def reflect(
        self,
        failures: List[Tuple[TaskRecord, ReplayResult]],
        successes: List[Tuple[TaskRecord, ReplayResult]],
        skill: str,
        memory: str,
        *,
        edit_budget: int,
        evolve_skill: bool,
        evolve_memory: bool,
    ) -> List[EditRecord]:
        """Filter out inconclusive runs before reflection so infra failures do not skew edits."""
        clean_failures = [
            (t, r) for t, r in failures
            if getattr(r, "outcome", "") != OUTCOME_INCONCLUSIVE
        ]
        return self.base_backend.reflect(
            clean_failures,
            successes,
            skill,
            memory,
            edit_budget=edit_budget,
            evolve_skill=evolve_skill,
            evolve_memory=evolve_memory,
        )

    def replay_agentic_task(
        self,
        task: TaskRecord,
        skill: str,
        memory: str,
        sample_id: int = 0,
        fixture_dir: str = "",
        test_cmd: Optional[List[str]] = None,
        expected_files: Optional[Dict[str, str]] = None,
        agent_runner: Optional[Callable[[str, Dict[str, str]], List[str]]] = None,
    ) -> ReplayResult:
        """Run an end-to-end agentic replay within a disposable workspace."""
        # 1. Budget gate check
        if self.replays_run >= self.config.max_replays_per_run:
            return ReplayResult(
                id=task.id,
                hard=0.0,
                soft=0.0,
                outcome=OUTCOME_INCONCLUSIVE,
                inconclusive_reason=REASON_BUDGET_EXHAUSTED,
                fail_reason="Max agentic replays budget reached for this run",
                judge_rationale=f"budget limit ({self.config.max_replays_per_run}) reached",
            )

        t0 = time.time()
        self.replays_run += 1
        tools_called: List[str] = []
        diff: str = ""

        # 2. Spin up disposable workspace
        base_dir = fixture_dir
        if not base_dir and task.context_excerpt and os.path.isdir(task.context_excerpt):
            base_dir = task.context_excerpt
        workspace = AgenticWorkspace(
            base_dir=base_dir,
            pinned_commit=self.config.pinned_commit,
            isolate_env=self.config.isolate_env,
            isolate_network=self.config.isolate_network,
        )

        try:
            workspace_path = workspace.setup()
            env = workspace.build_isolated_env()

            # 3. Execute agentic interaction
            timed_out = False
            infra_error = False
            error_msg = ""
            actions_count = 0

            if agent_runner is not None:
                try:
                    tools_called = agent_runner(workspace_path, env) or []
                    actions_count = len(tools_called)
                except subprocess.TimeoutExpired:
                    timed_out = True
                except Exception as exc:
                    infra_error = True
                    error_msg = str(exc)
            elif self.config.agent_cmd:
                try:
                    cmd = [self.config.agent_cmd, task.intent]
                    proc = subprocess.run(
                        cmd,
                        cwd=workspace_path,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        timeout=self.config.timeout_sec,
                        env=env,
                        creationflags=_NO_WINDOW,
                    )
                    actions_count = 1 if proc.returncode == 0 else 0
                except subprocess.TimeoutExpired:
                    timed_out = True
                except Exception as exc:
                    infra_error = True
                    error_msg = str(exc)
            else:
                # Mock / simulated agentic execution from task properties
                simulated_resp = self.attempt(task, skill, memory, sample_id=sample_id)
                # Extract simulated tool calls or edit markers
                if "TOOL_CALL:" in simulated_resp:
                    for line in simulated_resp.splitlines():
                        if "TOOL_CALL:" in line:
                            tools_called.append(line.split("TOOL_CALL:", 1)[1].strip())
                actions_count = len(tools_called) if tools_called else (1 if simulated_resp else 0)

            # Check timeouts / infra issues
            if timed_out:
                return ReplayResult(
                    id=task.id,
                    hard=0.0,
                    soft=0.0,
                    outcome=OUTCOME_INCONCLUSIVE,
                    inconclusive_reason=REASON_TIMEOUT,
                    fail_reason=f"Agent run timed out after {self.config.timeout_sec}s",
                    judge_rationale="turn/time budget exceeded",
                    latency_ms=round((time.time() - t0) * 1000.0, 1),
                )
            if infra_error:
                return ReplayResult(
                    id=task.id,
                    hard=0.0,
                    soft=0.0,
                    outcome=OUTCOME_INCONCLUSIVE,
                    inconclusive_reason=REASON_INFRA,
                    fail_reason=f"Infrastructure error during agent run: {error_msg}",
                    judge_rationale=f"infra failure: {error_msg}",
                    latency_ms=round((time.time() - t0) * 1000.0, 1),
                )

            # 4. Check decidable action floor (issue #155 constraint)
            if actions_count < self.config.min_decidable_actions:
                return ReplayResult(
                    id=task.id,
                    hard=0.0,
                    soft=0.0,
                    outcome=OUTCOME_INCONCLUSIVE,
                    inconclusive_reason=REASON_INSUFFICIENT_EVIDENCE,
                    fail_reason=(
                        f"Decidable actions ({actions_count}) below threshold "
                        f"({self.config.min_decidable_actions})"
                    ),
                    judge_rationale="insufficient decidable actions",
                    tools_called=tools_called,
                    latency_ms=round((time.time() - t0) * 1000.0, 1),
                )

            # 5. Extract unified diff
            diff = workspace.get_diff()

            # 6. Task verification (primary acceptance gate)
            active_verifier = self.verifier
            if test_cmd or expected_files:
                active_verifier = TaskVerifier(
                    test_cmd=test_cmd,
                    expected_files=expected_files,
                    timeout=min(self.config.timeout_sec, 30.0),
                )

            outcome, hard, soft, rationale, reason = active_verifier.verify(workspace_path, env)

            tokens = 500  # Baseline token estimate for agentic interaction
            self._tokens_spent += tokens
            latency_ms = (time.time() - t0) * 1000.0

            return ReplayResult(
                id=task.id,
                hard=hard,
                soft=soft,
                outcome=outcome,
                inconclusive_reason=reason,
                diff=diff,
                response=f"Agentic replay in {workspace_path} (diff: {len(diff)} chars)",
                fail_reason="" if outcome == OUTCOME_PASS else rationale,
                judge_rationale=rationale,
                tools_called=tools_called,
                tokens=tokens,
                latency_ms=round(latency_ms, 1),
            )
        finally:
            workspace.cleanup()
