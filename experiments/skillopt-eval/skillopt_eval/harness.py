"""Runner protocol and per-run record for the controlled experiment.

A *runner* is the pluggable execution harness: given a scenario id, a skill
document (or None for the installed baseline) and a seed, it executes the
agent exactly once and returns a :class:`RunRecord`. The experiment layer
never invokes a model directly - everything flows through this interface so
baseline and candidate conditions share harness, model, settings, budgets
and permissions by construction.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol


def skill_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class RunRecord:
    """One agent run on one scenario under one condition.

    ``score``/``passed`` are None when the run produced no scorable result
    (timeout, non-zero exit, malformed output, missing score, incomplete
    trajectory). Callers treat None as a failure - the comparison fails
    closed, never silently drops the task.
    """

    run_id: str
    condition: str            # "baseline" | "candidate" | "opt:<step>"
    scenario_id: str
    split: str                # train | validation | test
    seed_index: int
    passed: Optional[bool]
    score: Optional[float]
    tokens: Optional[int]
    tool_calls: Optional[int]
    turns: Optional[int]
    latency_ms: Optional[float]
    failure: str = ""         # timeout|nonzero_exit|malformed_output|missing_score|incomplete_trajectory|check_failed
    error: str = ""
    trajectory_file: str = ""
    checks: List[Dict[str, Any]] = field(default_factory=list)
    model: str = ""
    skill_hash: str = ""
    pinned_sha: str = ""

    def to_row(self) -> Dict[str, Any]:
        d = asdict(self)
        d["checks"] = json.dumps(d["checks"], separators=(",", ":"))
        return d

    def binary(self) -> int:
        """Binary outcome for paired stats; unscored runs fail closed to 0."""
        return 1 if self.passed else 0


class Runner(Protocol):
    """Pluggable execution harness."""

    name: str

    def available_scenarios(self) -> List[str]:
        """Scenario ids this runner can execute."""
        ...

    def run(
        self,
        scenario_id: str,
        skill_text: Optional[str],
        seed: int,
        workdir: Path,
    ) -> RunRecord:
        """Execute one scenario once.

        ``skill_text`` is None for the installed baseline condition.
        ``workdir`` is a per-run throwaway directory for trajectories.
        """
        ...


def write_trajectory(
    workdir: Path,
    record: RunRecord,
    output: str,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Persist the raw trajectory for a run; returns the stored file name.

    Trajectories are written verbatim (they may embed local paths/output and
    are gitignored by convention - see the smoke script's warning). Each file
    is one JSON document, never silently truncated.
    """
    traj_dir = workdir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    name = f"{record.run_id}.json"
    payload = {
        "run_id": record.run_id,
        "condition": record.condition,
        "scenario_id": record.scenario_id,
        "split": record.split,
        "seed_index": record.seed_index,
        "skill_hash": record.skill_hash,
        "model": record.model,
        "output": output,
    }
    if extra:
        payload.update(extra)
    (traj_dir / name).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return f"trajectories/{name}"
