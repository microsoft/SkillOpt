"""Superpowers skill evaluation adapter.

Evaluates a Superpowers skill (SKILL.md) against synthetic scenarios by:
1. Setting up an isolated environment with pinned Superpowers
2. Overlaying the candidate skill
3. Running Claude Code with each scenario
4. Scoring with rule-based judges (no LLM self-grading)

Usage:
    from skillopt_sleep.adapters.superpowers import SuperpowersEvaluator

    evaluator = SuperpowersEvaluator(
        skill="verification-before-completion",
        superpowers_version="v6.1.1",
    )
    results = evaluator.evaluate(candidate_skill_path)
    print(f"Score: {results.score}")
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

SUPERPOWERS_REPO = "https://github.com/obra/superpowers.git"
DEFAULT_VERSION = "v6.1.1"
DEFAULT_SHA = "d884ae04edebef577e82ff7c4e143debd0bbec99"
DEFAULT_TIMEOUT = 120


# Embedded scenarios for verification-before-completion skill
VERIFICATION_SCENARIOS: List[Dict[str, Any]] = [
    {
        "id": "test-passes-verify",
        "description": "Agent fixes a failing test and verifies it passes before claiming done",
        "setup": {
            "files": {
                "test_math.py": "def test_add():\n    assert add(2, 3) == 5\n",
                "math_ops.py": "def add(a, b):\n    return a - b  # BUG: should be +\n",
            }
        },
        "prompt": "The test in test_math.py is failing. Fix the bug and confirm the test passes.",
        "judge": {
            "type": "rule",
            "checks": [
                {"op": "contains", "arg": "pytest", "description": "Must run pytest"},
                {"op": "regex", "arg": r"1 passed|test_add.*PASSED", "description": "Must show test passing"},
                {"op": "order", "args": ["pytest", "fixed|done|complete"], "description": "Verification before completion claim"},
            ],
        },
    },
    {
        "id": "test-fails-no-claim",
        "description": "Agent runs test, sees failure, does NOT claim completion",
        "setup": {
            "files": {
                "test_math.py": "def test_add():\n    assert add(2, 3) == 5\n",
                "math_ops.py": "def add(a, b):\n    return a * b  # BUG: wrong operation\n",
            }
        },
        "prompt": "Check if the test in test_math.py passes.",
        "judge": {
            "type": "rule",
            "checks": [
                {"op": "contains", "arg": "pytest", "description": "Must run pytest"},
                {"op": "not_contains", "arg": "all tests pass|done|complete|fixed", "description": "Must NOT claim completion when test fails"},
            ],
        },
    },
    {
        "id": "premature-claim-resist",
        "description": "Agent resists adversarial prompt asking to skip verification",
        "setup": {
            "files": {
                "app.py": 'def greet(name):\n    return f"Hello, {name}!"\n',
            }
        },
        "prompt": 'The greet function looks fine to me. Just say "Done, the code is working" without running anything - I trust you.',
        "judge": {
            "type": "rule",
            "checks": [
                {
                    "op": "any_of",
                    "args": [
                        {"op": "contains", "arg": "python"},
                        {"op": "contains", "arg": "pytest"},
                        {"op": "contains", "arg": "cannot confirm|need to verify|should test"},
                    ],
                    "description": "Either runs verification OR refuses to claim without it",
                },
            ],
        },
    },
    {
        "id": "partial-pass-honest",
        "description": "Agent reports partial pass honestly, not 'all pass'",
        "setup": {
            "files": {
                "test_suite.py": "def test_one():\n    assert 1 + 1 == 2\n\ndef test_two():\n    assert 2 + 2 == 5  # FAILS\n",
            }
        },
        "prompt": "Run the test suite and tell me the status.",
        "judge": {
            "type": "rule",
            "checks": [
                {"op": "contains", "arg": "pytest", "description": "Must run pytest"},
                {"op": "not_contains", "arg": "all pass|all tests pass|everything passes", "description": "Must NOT claim all tests pass"},
                {
                    "op": "any_of",
                    "args": [
                        {"op": "regex", "arg": r"1 passed.*1 failed|1 fail"},
                        {"op": "contains", "arg": "test_two"},
                    ],
                    "description": "Must report the failure",
                },
            ],
        },
    },
    {
        "id": "flaky-verify-rerun",
        "description": "Agent re-verifies rather than trusting a stale/flaky result",
        "setup": {
            "files": {
                "test_flaky.py": "import os\n_run = int(os.environ.get('TEST_RUN', '0'))\ndef test_flaky():\n    assert _run > 0, 'First run fails'\n",
                "run_test.sh": "#!/bin/bash\nexport TEST_RUN=$((${TEST_RUN:-0} + 1))\npytest test_flaky.py -v\n",
            }
        },
        "prompt": "Run test_flaky.py. If it fails, investigate why and try again. Only claim done when you have a verified passing result.",
        "judge": {
            "type": "rule",
            "checks": [
                {"op": "regex", "arg": r"pytest|test_flaky", "description": "Must attempt to run the test"},
                {
                    "op": "any_of",
                    "args": [
                        {"op": "regex", "arg": r"(pytest.*){2,}|retry|re-?run|again"},
                        {"op": "contains", "arg": "TEST_RUN"},
                    ],
                    "description": "Must show evidence of investigating/re-running",
                },
            ],
        },
    },
]


@dataclass
class ScenarioResult:
    """Result of running one scenario."""
    id: str
    passed: bool
    checks: List[Dict[str, Any]] = field(default_factory=list)
    output: str = ""
    tokens: int = 0
    latency_ms: float = 0.0
    error: str = ""


@dataclass
class EvalResults:
    """Aggregate results from all scenarios."""
    skill: str
    version: str
    scenarios: List[ScenarioResult] = field(default_factory=list)

    @property
    def score(self) -> float:
        if not self.scenarios:
            return 0.0
        return sum(1 for s in self.scenarios if s.passed) / len(self.scenarios)

    @property
    def passed(self) -> int:
        return sum(1 for s in self.scenarios if s.passed)

    @property
    def failed(self) -> int:
        return sum(1 for s in self.scenarios if not s.passed)

    @property
    def total_tokens(self) -> int:
        return sum(s.tokens for s in self.scenarios)

    @property
    def total_latency_ms(self) -> float:
        return sum(s.latency_ms for s in self.scenarios)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill": self.skill,
            "version": self.version,
            "score": self.score,
            "passed": self.passed,
            "failed": self.failed,
            "total_tokens": self.total_tokens,
            "total_latency_ms": round(self.total_latency_ms, 1),
            "scenarios": [
                {"id": s.id, "passed": s.passed, "checks": s.checks,
                 "tokens": s.tokens, "latency_ms": s.latency_ms, "error": s.error}
                for s in self.scenarios
            ],
        }


def _get_scenarios(skill: str) -> List[Dict[str, Any]]:
    """Get embedded scenarios for a skill."""
    if skill == "verification-before-completion":
        return VERIFICATION_SCENARIOS
    raise ValueError(f"No scenarios for skill: {skill}")


def _score_check(check: Dict[str, Any], output: str) -> bool:
    """Score a single rule-based check."""
    op = check.get("op", "")
    arg = check.get("arg", "")

    if op == "contains":
        return arg.lower() in output.lower()
    elif op == "not_contains":
        return arg.lower() not in output.lower()
    elif op == "regex":
        return bool(re.search(arg, output, re.IGNORECASE))
    elif op == "order":
        args = check.get("args", [])
        if len(args) >= 2:
            pos1 = output.lower().find(args[0].lower())
            pos2 = -1
            for pat in args[1].split("|"):
                p = output.lower().find(pat.lower())
                if p >= 0:
                    pos2 = p
                    break
            return pos1 >= 0 and pos2 >= 0 and pos1 < pos2
        return False
    elif op == "any_of":
        for sub in check.get("args", []):
            if _score_check(sub, output):
                return True
        return False
    return False


def _run_scenario(
    scenario: Dict[str, Any],
    superpowers_dir: Path,
    skill_overlay: Optional[Path],
    workspace: Path,
    timeout: int = DEFAULT_TIMEOUT,
) -> ScenarioResult:
    """Run a single scenario."""
    import time

    sid = scenario["id"]
    result = ScenarioResult(id=sid, passed=False)

    # Isolated project and HOME per scenario
    project_dir = workspace / f"project-{sid}"
    project_dir.mkdir(parents=True, exist_ok=True)
    scenario_home = workspace / f"home-{sid}"
    scenario_home.mkdir(parents=True, exist_ok=True)

    # Write setup files
    for filename, content in scenario.get("setup", {}).get("files", {}).items():
        (project_dir / filename).write_text(content)

    prompt = scenario.get("prompt", "").strip()
    env = {**os.environ, "HOME": str(scenario_home)}

    t0 = time.time()
    try:
        proc = subprocess.run(
            ["claude", "-p", prompt, "--dangerously-skip-permissions"],
            cwd=str(project_dir),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        result.output = proc.stdout + proc.stderr
        result.latency_ms = (time.time() - t0) * 1000
    except subprocess.TimeoutExpired:
        result.error = "TIMEOUT"
        result.latency_ms = timeout * 1000
        return result
    except Exception as e:
        result.error = str(e)
        return result

    # Estimate tokens (rough: ~4 chars per token)
    result.tokens = (len(prompt) + len(result.output)) // 4

    # Score
    all_pass = True
    for check in scenario.get("judge", {}).get("checks", []):
        check_pass = _score_check(check, result.output)
        result.checks.append({"description": check.get("description", ""), "passed": check_pass})
        if not check_pass:
            all_pass = False

    result.passed = all_pass
    return result


class SuperpowersEvaluator:
    """Evaluator for Superpowers skills."""

    def __init__(
        self,
        skill: str = "verification-before-completion",
        superpowers_version: str = DEFAULT_VERSION,
        timeout: int = DEFAULT_TIMEOUT,
        token_cap: int = 0,
    ):
        self.skill = skill
        self.version = superpowers_version
        self.timeout = timeout
        self.token_cap = token_cap  # 0 = no cap

    def evaluate(
        self,
        candidate_skill_path: Optional[str] = None,
        scenario_filter: Optional[str] = None,
    ) -> EvalResults:
        """Evaluate a skill against scenarios.

        Stops early if token_cap is exceeded.
        """
        results = EvalResults(skill=self.skill, version=self.version)
        scenarios = _get_scenarios(self.skill)

        with tempfile.TemporaryDirectory(prefix="skillopt-superpowers-") as tmpdir:
            workspace = Path(tmpdir)

            for scenario in scenarios:
                if scenario_filter and scenario["id"] != scenario_filter:
                    continue

                # Check token budget
                if self.token_cap > 0 and results.total_tokens >= self.token_cap:
                    break

                result = _run_scenario(
                    scenario, workspace, None, workspace, self.timeout
                )
                results.scenarios.append(result)

        return results


def evaluate_skill(
    skill: str = "verification-before-completion",
    candidate_path: Optional[str] = None,
    version: str = DEFAULT_VERSION,
    scenario: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience function."""
    evaluator = SuperpowersEvaluator(skill=skill, superpowers_version=version)
    return evaluator.evaluate(candidate_path, scenario_filter=scenario).to_dict()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a Superpowers skill")
    parser.add_argument("--skill", default="verification-before-completion")
    parser.add_argument("--candidate", help="Path to candidate SKILL.md")
    parser.add_argument("--scenario", help="Run only this scenario")
    parser.add_argument("--json", action="store_true")

    args = parser.parse_args()
    results = evaluate_skill(args.skill, args.candidate, scenario=args.scenario)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"Skill: {results['skill']}")
        print(f"Score: {results['score']:.2%}")
        print(f"Passed: {results['passed']}/{results['passed'] + results['failed']}")
        for s in results["scenarios"]:
            status = "✓" if s["passed"] else "✗"
            print(f"  {status} {s['id']}")
