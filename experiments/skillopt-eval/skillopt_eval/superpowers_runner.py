"""Live runner: Claude Code + pinned Superpowers bootstrap.

Delegates each scenario to ``SuperpowersEvaluator`` - the same adapter used by
``scripts/smoke_superpowers.sh`` - so candidate injection goes through the
isolated-overlay path (temp clone at a pinned SHA; the installed Superpowers
tree and the source skill file are never touched).

Requirements (fail-closed when absent):
  * POSIX host (bash shims; ``_run_scenario`` refuses non-POSIX)
  * ``claude`` CLI on PATH (or SKILLOPT_CLAUDE_BIN)
  * ANTHROPIC_API_KEY, or SKILLOPT_HOST_AUTH=1 for trusted candidates
  * network access to fetch the pinned Superpowers SHA

The adapter reports estimated tokens and wall latency but not turn/tool-call
counts; those fields stay None in the CSV rather than being fabricated.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from skillopt_sleep.adapters.superpowers import (
    DEFAULT_SHA,
    SuperpowersEvaluator,
    _get_scenarios,
)

from .harness import RunRecord, skill_hash, write_trajectory


class SuperpowersRunner:
    """Wraps the live adapter as an experiment Runner."""

    name = "superpowers"

    def __init__(
        self,
        skill: str,
        pinned_sha: str = DEFAULT_SHA,
        version: str = "v6.1.1",
        model: str = "claude",
        timeout: int = 120,
        token_cap: int = 0,
    ):
        self.skill = skill
        self.pinned_sha = pinned_sha
        self.model = model
        self.evaluator = SuperpowersEvaluator(
            skill=skill, superpowers_version=version,
            timeout=timeout, token_cap=token_cap,
        )

    def available_scenarios(self) -> List[str]:
        return sorted(s["id"] for s in _get_scenarios(self.skill))

    def run(
        self,
        scenario_id: str,
        skill_text: Optional[str],
        seed: int,
        workdir: Path,
    ) -> RunRecord:
        candidate_path: Optional[str] = None
        if skill_text is not None:
            # stage the candidate in a throwaway file; never edits the source
            staged = workdir / f"candidate-{scenario_id}-{seed}.md"
            staged.write_text(skill_text, encoding="utf-8")
            candidate_path = str(staged)

        results = self.evaluator.evaluate(
            candidate_path,
            scenario_filter=scenario_id,
            pinned_sha=self.pinned_sha,
        )
        s = results.scenarios[0]
        record = RunRecord(
            run_id=f"sp-{scenario_id}-{seed}-{results.pinned_sha[:8]}",
            condition="",
            scenario_id=scenario_id,
            split="",
            seed_index=seed,
            passed=s.passed if not s.error else False,
            score=(sum(1 for c in s.checks if c["passed"]) / len(s.checks)
                   if s.checks and not s.error else None),
            tokens=s.tokens,
            tool_calls=None,   # adapter does not expose tool-call counts
            turns=None,        # adapter does not expose turn counts
            latency_ms=s.latency_ms,
            failure=_classify_failure(s),
            error=s.error,
            checks=s.checks,
            model=self.model,
            skill_hash=skill_hash(skill_text or ""),
            pinned_sha=results.pinned_sha,
        )
        record.trajectory_file = write_trajectory(
            workdir, record, s.output, extra={"evidence": s.evidence}
        )
        return record


def _classify_failure(s) -> str:
    if not s.error:
        return "" if s.passed else "check_failed"
    if s.error == "TIMEOUT":
        return "timeout"
    if s.error.startswith("EXIT_"):
        return "nonzero_exit"
    if s.error == "BOOTSTRAP_SKILL_MISSING":
        return "incomplete_trajectory"
    return "harness_error"
