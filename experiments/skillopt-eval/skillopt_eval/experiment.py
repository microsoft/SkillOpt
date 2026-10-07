"""Experiment orchestrator.

Pipeline: deterministic split -> bounded optimization behind a held-out
validation gate -> paired baseline-vs-best evaluation on the test split ->
paired statistics (evalkit) -> artifacts.

Contracts enforced here (not in the runners):

* identical settings: every run records the same model/settings/pinned SHA
* held-out gate: a candidate is accepted only when its validation score
  STRICTLY improves over the current best; equal or lower is rejected
* fail closed: a run with no score (timeout, non-zero exit, malformed
  output, missing score, incomplete trajectory) counts as a failure
* quality is the primary gate: cost deltas are reported but can never
  rescue a quality regression
* no general efficiency claim from a single task/seed/model/harness -
  RESULTS.md carries the caveat verbatim
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from skillopt_sleep import evalkit

from .config import ExperimentConfig
from .harness import Runner, RunRecord
from .optimizer import CandidateEdit, build_proposer
from .report import write_artifacts
from .splits import split_scenarios


@dataclass
class Attempt:
    step: int
    candidate_hash: str
    source: str
    description: str
    validation_score: Optional[float]
    decision: str           # accepted | rejected
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def _run_batch(
    runner: Runner,
    scenario_ids: List[str],
    skill_text: Optional[str],
    condition: str,
    split: str,
    seeds_per_task: int,
    workdir: Path,
) -> List[RunRecord]:
    records: List[RunRecord] = []
    for sid in scenario_ids:
        for seed in range(seeds_per_task):
            rec = runner.run(sid, skill_text, seed, workdir)
            rec.condition = condition
            rec.split = split
            if rec.passed is None:
                rec.passed = False  # fail closed
            records.append(rec)
    return records


def _mean_pass(records: List[RunRecord]) -> float:
    if not records:
        return 0.0
    return sum(1 for r in records if r.passed) / len(records)


def _mean_score(records: List[RunRecord]) -> Optional[float]:
    scores = [r.score for r in records if r.score is not None]
    if not scores:
        return None
    return sum(scores) / len(scores)


def run_experiment(
    cfg: ExperimentConfig,
    runner: Runner,
    baseline_text: Optional[str],
) -> Dict[str, Any]:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    all_ids = runner.available_scenarios()
    split = split_scenarios(all_ids, cfg.train, cfg.validation, cfg.test, cfg.seed)

    records: List[RunRecord] = []
    attempts: List[Attempt] = []

    # ---- optimization loop: propose bounded edits, gate on held-out validation
    best_text = baseline_text
    best_val = 0.0
    if cfg.optimize_steps > 0 and cfg.validation:
        best_val = _mean_pass(_run_batch(
            runner, split.validation, best_text,
            "baseline", "validation", cfg.seeds_per_task, out_dir,
        ))
    proposer = build_proposer(cfg.candidates, cfg.learning_rate, cfg.runner)
    rejected_hashes: List[str] = []
    for step in range(cfg.optimize_steps):
        edit: Optional[CandidateEdit] = proposer.propose(
            best_text or "", step, rejected_hashes
        )
        if edit is None:
            break
        train_records = _run_batch(
            runner, split.train, edit.text, f"opt:{step}",
            "train", cfg.seeds_per_task, out_dir,
        ) if split.train else []
        val_records = _run_batch(
            runner, split.validation, edit.text, f"opt:{step}",
            "validation", cfg.seeds_per_task, out_dir,
        )
        records.extend(train_records + val_records)
        val_score = _mean_pass(val_records)
        errored = [r for r in val_records if r.error and not r.checks]
        if edit.content_hash in rejected_hashes:
            decision, reason = "rejected", "duplicate of a rejected candidate"
        elif val_score > best_val:
            decision = "accepted"
            reason = (f"validation score {val_score:.4f} > "
                      f"{best_val:.4f} (strict improvement)")
            best_text, best_val = edit.text, val_score
        elif errored:
            decision, reason = "rejected", "validation run error (fail closed)"
            rejected_hashes.append(edit.content_hash)
        else:
            decision = "rejected"
            reason = (f"validation score {val_score:.4f} not strictly above "
                      f"{best_val:.4f}")
            rejected_hashes.append(edit.content_hash)
        attempts.append(Attempt(
            step=step, candidate_hash=edit.content_hash, source=edit.source,
            description=edit.description, validation_score=val_score,
            decision=decision, reason=reason,
        ))

    optimized = best_text is not baseline_text

    # ---- paired final evaluation on the held-out TEST split, interleaved
    test_records: List[RunRecord] = []
    for sid in split.test:
        for seed in range(cfg.seeds_per_task):
            for condition, text in (("baseline", baseline_text),
                                    ("candidate", best_text)):
                rec = runner.run(sid, text, seed, out_dir)
                rec.condition = condition
                rec.split = "test"
                if rec.passed is None:
                    rec.passed = False
                test_records.append(rec)
    records.extend(test_records)

    baseline_runs = [r for r in test_records if r.condition == "baseline"]
    candidate_runs = [r for r in test_records if r.condition == "candidate"]

    outcomes_a = {r.scenario_id: [r.binary()] for r in baseline_runs} \
        if cfg.seeds_per_task == 1 else _seeded_outcomes(baseline_runs, split.test)
    outcomes_b = {r.scenario_id: [r.binary()] for r in candidate_runs} \
        if cfg.seeds_per_task == 1 else _seeded_outcomes(candidate_runs, split.test)
    comparison = evalkit.compare(
        split.test, outcomes_a, outcomes_b, allow_graded=False,
    )

    metrics = {
        "baseline": _aggregate(baseline_runs),
        "candidate": _aggregate(candidate_runs),
    }
    verdict = _verdict(comparison, metrics)

    return write_artifacts(
        out_dir=out_dir,
        cfg=cfg,
        split=split,
        records=records,
        test_records=test_records,
        attempts=attempts,
        metrics=metrics,
        comparison=comparison,
        optimized=optimized,
        verdict=verdict,
        elapsed_s=time.time() - started,
    )


def _seeded_outcomes(records: List[RunRecord], ids: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for sid in ids:
        rs = sorted(
            (r for r in records if r.scenario_id == sid),
            key=lambda r: r.seed_index,
        )
        out[sid] = {"seeds": [r.binary() for r in rs]}
    return out


def _aggregate(runs: List[RunRecord]) -> Dict[str, Any]:
    def _sum(key):
        vals = [getattr(r, key) for r in runs if getattr(r, key) is not None]
        return sum(vals) if vals else None

    def _mean(key):
        vals = [getattr(r, key) for r in runs if getattr(r, key) is not None]
        return round(sum(vals) / len(vals), 2) if vals else None

    failures: Dict[str, int] = {}
    for r in runs:
        if not r.passed:
            kind = r.failure or "check_failed"
            failures[kind] = failures.get(kind, 0) + 1
    return {
        "n_runs": len(runs),
        "pass_rate": round(_mean_pass(runs), 4),
        "mean_score": _mean_score(runs),
        "total_tokens": _sum("tokens"),
        "mean_tokens": _mean("tokens"),
        "total_tool_calls": _sum("tool_calls"),
        "mean_tool_calls": _mean("tool_calls"),
        "total_turns": _sum("turns"),
        "mean_turns": _mean("turns"),
        "total_latency_ms": _sum("latency_ms"),
        "mean_latency_ms": _mean("latency_ms"),
        "failures": failures,
    }


def _verdict(comparison: evalkit.EvalReport,
             metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Quality is the gate; cost is informational."""
    significant = bool(
        comparison.mcnemar and comparison.mcnemar.significant
    ) or (comparison.bootstrap.low > 0 or comparison.bootstrap.high < 0)
    quality_improved = comparison.delta > 0 and significant
    quality_regressed = comparison.delta < 0 and significant
    cost_notes = []
    for key in ("mean_tokens", "mean_latency_ms", "mean_tool_calls"):
        a = metrics["baseline"].get(key)
        b = metrics["candidate"].get(key)
        if a is not None and b is not None:
            pct = (b - a) / a * 100 if a else 0.0
            cost_notes.append(f"{key}: {pct:+.1f}%")
    if quality_improved:
        summary = ("Candidate strictly improved held-out test quality; "
                   "adoption remains an explicit human decision.")
    elif quality_regressed:
        summary = ("Candidate REGRESSED held-out test quality - rejected; "
                   "cost differences cannot compensate.")
    else:
        summary = ("No significant quality difference on the held-out test "
                   "split; a result where no candidate beats the current "
                   "skill is still useful.")
    return {
        "quality_improved": quality_improved,
        "quality_regressed": quality_regressed,
        "significant": significant,
        "cost_notes": cost_notes,
        "summary": summary,
    }
