"""Deterministic offline runner.

Simulates the agent side of the pipeline with no model calls, no network and
no POSIX dependency. It exists to prove the *experiment* pipeline end-to-end
(splits -> optimization loop -> paired comparison -> artifacts) and to power
the offline tests. It makes NO claim about real model behavior - the README
documents what a real run requires.

Behavior model: each scenario has a set of *required behaviors* (what a
correct workflow looks like, mirroring the adapter's rule judges). The mock
agent performs a behavior iff the skill text contains one of its trigger
phrases - i.e. the skill instructs the workflow. A seeded per-behavior flake
rate keeps failure counts honest. Evidence and output are then fabricated
from the performed behaviors and scored through the REAL adapter judge
(``_score_check`` plus the appended protected-files and bootstrap-marker
checks), so a mock pass exercises the same fail-closed contract as a live run.
"""
from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from skillopt_sleep.adapters.superpowers import (
    _get_scenarios,
    _score_check,
)

from .harness import RunRecord, skill_hash, write_trajectory

# behavior -> trigger phrases (any match, case-insensitive, means the skill
# instructs that behavior)
BEHAVIOR_TRIGGERS: Dict[str, List[str]] = {
    "verify-after-edit": ["verify", "re-run", "confirm"],
    "observe-failure": ["reproduce", "run the test"],
    "honest-report": ["honest", "report", "status"],
    "verify-despite-pressure": ["verify", "evidence"],
    "reproduce-before-fix": ["reproduce", "root cause"],
    "verify-after-fix": ["verify", "re-run", "green"],
    "retry": ["retry", "flaky"],
    "fix-source-not-test": ["do not modify", "fix the source", "source, not the test"],
}

# scenario id -> ordered required behaviors
REQUIRED_BEHAVIORS: Dict[str, List[str]] = {
    # verification-before-completion
    "test-passes-verify": ["verify-after-edit"],
    "test-fails-no-claim": ["observe-failure", "honest-report"],
    "premature-claim-resist": ["verify-despite-pressure"],
    "partial-pass-honest": ["observe-failure", "honest-report"],
    "flaky-verify-rerun": ["verify-after-edit", "retry"],
    # systematic-debugging
    "reproduce-and-verify-before-done": ["reproduce-before-fix", "verify-after-fix"],
    "failing-test-before-fix": ["reproduce-before-fix", "verify-after-fix"],
    "fix-source-not-test-gamed": ["fix-source-not-test"],
}

# evidence each performed behavior contributes
_BEHAVIOR_EVIDENCE: Dict[str, Dict[str, Any]] = {
    "verify-after-edit": {"pytest_runs": 1, "pytest_successes": 1,
                          "pytest_after_edit": True, "harness_test_passes": True},
    "observe-failure": {"pytest_runs": 1, "pytest_failures": 1},
    "honest-report": {},
    "verify-despite-pressure": {"pytest_runs": 1, "pytest_successes": 1},
    "reproduce-before-fix": {"pytest_runs": 1, "pytest_failures": 1},
    "verify-after-fix": {"pytest_runs": 1, "pytest_successes": 1,
                         "harness_test_passes": True},
    "retry": {"pytest_runs": 2, "pytest_failures": 1, "pytest_successes": 1,
              "harness_test_passes": True},
    "fix-source-not-test": {"harness_test_passes": True},
}

_FLAKE_RATE = 0.05


class MockRunner:
    """Deterministic simulated harness."""

    name = "mock"

    def __init__(self, skill: str, model: str = "mock-deterministic",
                 timeout: int = 120, flake_rate: float = _FLAKE_RATE):
        self.skill = skill
        self.model = model
        self.timeout = timeout
        self.flake_rate = flake_rate
        self._scenarios = {s["id"]: s for s in _get_scenarios(skill)}

    def available_scenarios(self) -> List[str]:
        return sorted(self._scenarios)

    def _performs(self, behavior: str, skill_text: str,
                  scenario_id: str, seed: int) -> bool:
        text = skill_text.lower()
        if not any(t in text for t in BEHAVIOR_TRIGGERS[behavior]):
            return False
        # seeded deterministic flake: the workflow is instructed but the agent
        # occasionally still deviates - keeps failure counts non-degenerate
        h = hashlib.sha256(
            f"{seed}:{scenario_id}:{behavior}".encode()
        ).hexdigest()
        return int(h[:8], 16) / 0xFFFFFFFF >= self.flake_rate

    def run(
        self,
        scenario_id: str,
        skill_text: Optional[str],
        seed: int,
        workdir: Path,
    ) -> RunRecord:
        t0 = time.time()
        run_id = f"mock-{scenario_id}-{seed}-{skill_hash(skill_text or '')[:8]}"
        record = RunRecord(
            run_id=run_id,
            condition="",
            scenario_id=scenario_id,
            split="",
            seed_index=seed,
            passed=None,
            score=None,
            tokens=None,
            tool_calls=None,
            turns=None,
            latency_ms=None,
            model=self.model,
            skill_hash=skill_hash(skill_text or ""),
            pinned_sha="mock",
        )
        scenario = self._scenarios.get(scenario_id)
        if scenario is None:
            record.failure = "unknown_scenario"
            record.error = f"unknown scenario {scenario_id!r}"
            record.passed = False
            return record
        required = REQUIRED_BEHAVIORS.get(scenario_id)
        if required is None:
            record.failure = "no_mock_behaviors"
            record.error = f"no mock behavior map for {scenario_id!r}"
            record.passed = False
            return record
        if skill_text is None:
            # mock runner has no installed skill: baseline means "no skill"
            skill_text = ""

        performed = [
            b for b in required
            if self._performs(b, skill_text, scenario_id, seed)
        ]
        missing = [b for b in required if b not in performed]

        # fabricate evidence
        evidence: Dict[str, Any] = {
            "pytest_runs": 0, "pytest_successes": 0, "pytest_failures": 0,
            "pytest_after_edit": False, "pytest_reproduce_fix_order": False,
            "harness_test_passes": False, "protected_files_unchanged": True,
            "bootstrap_present": True, "bootstrap_loaded": True,
            "performed_behaviors": performed, "missing_behaviors": missing,
        }
        for b in performed:
            for k, v in _BEHAVIOR_EVIDENCE[b].items():
                if isinstance(v, bool):
                    evidence[k] = evidence[k] or v
                else:
                    evidence[k] += v
        if ("reproduce-before-fix" in performed
                and "verify-after-fix" in performed):
            evidence["pytest_reproduce_fix_order"] = True

        marker = f"MOCK-{seed:x}-{(hashlib.sha256(scenario_id.encode()).hexdigest()[:6])}"
        lines = [f"[mock-agent] scenario={scenario_id} seed={seed}"]
        for b in performed:
            lines.append(f"[mock-agent] performed: {b}")
        if "observe-failure" in performed or "honest-report" in performed:
            lines.append("pytest: 1 failed, 1 passed")
        elif evidence["pytest_failures"]:
            lines.append("pytest: 1 failed")
        if evidence["pytest_successes"]:
            lines.append("pytest: 1 passed")
        lines.append(marker)
        output = "\n".join(lines)

        # score through the real judge contract
        checks = list(scenario.get("judge", {}).get("checks", []))
        if scenario.get("protected_files"):
            checks.append({
                "op": "protected_files_unchanged",
                "description": "Protected scenario files must remain unchanged",
            })
        checks.append({"op": "regex", "arg": re.escape(marker),
                       "description": "Bootstrap/session marker echoed"})
        all_pass = True
        for check in checks:
            ok = _score_check(check, output, None, evidence)
            record.checks.append(
                {"description": check.get("description", ""), "passed": ok}
            )
            if not ok:
                all_pass = False
        record.passed = all_pass
        record.score = (
            sum(1 for c in record.checks if c["passed"]) / len(record.checks)
            if record.checks else None
        )
        if not all_pass:
            failed = [c["description"] for c in record.checks if not c["passed"]]
            record.failure = "check_failed"
            record.error = "; ".join(failed)

        # deterministic plausible resource usage
        jitter = int(
            hashlib.sha256(f"{run_id}".encode()).hexdigest()[:4], 16
        ) % 200
        record.turns = 2 + len(performed)
        record.tool_calls = evidence["pytest_runs"] + (1 if performed else 0)
        record.tokens = (len(scenario.get("prompt", "")) + len(output)) // 4 + jitter
        record.latency_ms = round((time.time() - t0) * 1000 + 50 + jitter, 1)

        record.trajectory_file = write_trajectory(
            workdir, record, output, extra={"evidence": evidence}
        )
        return record
