"""Artifact writers: raw_results.csv + results.json + RESULTS.md."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from skillopt_sleep import evalkit

from .config import ExperimentConfig
from .harness import RunRecord
from .splits import ScenarioSplit

NO_GENERAL_CLAIM = (
    "This report compares ONE skill on ONE harness/model/settings on the "
    "scenarios listed. It does not support a general efficiency claim about "
    "SkillOpt, Superpowers, or any model. Adoption of any candidate remains "
    "an explicit human decision."
)

CSV_FIELDS = [
    "run_id", "condition", "scenario_id", "split", "seed_index",
    "passed", "score", "tokens", "tool_calls", "turns", "latency_ms",
    "failure", "error", "model", "skill_hash", "pinned_sha",
    "trajectory_file", "checks",
]


def write_artifacts(
    out_dir: Path,
    cfg: ExperimentConfig,
    split: ScenarioSplit,
    records: List[RunRecord],
    test_records: List[RunRecord],
    attempts: List[Any],
    metrics: Dict[str, Any],
    comparison: evalkit.EvalReport,
    optimized: bool,
    verdict: Dict[str, Any],
    elapsed_s: float,
) -> Dict[str, Any]:
    _write_csv(out_dir / "raw_results.csv", records)
    results = _results_json(
        cfg, split, records, attempts, metrics, comparison,
        optimized, verdict, elapsed_s,
    )
    (out_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "RESULTS.md").write_text(
        _results_md(results, metrics, comparison, attempts, verdict),
        encoding="utf-8",
    )
    return results


def _write_csv(path: Path, records: List[RunRecord]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in records:
            writer.writerow(r.to_row())


def _results_json(
    cfg: ExperimentConfig,
    split: ScenarioSplit,
    records: List[RunRecord],
    attempts: List[Any],
    metrics: Dict[str, Any],
    comparison: evalkit.EvalReport,
    optimized: bool,
    verdict: Dict[str, Any],
    elapsed_s: float,
) -> Dict[str, Any]:
    return {
        "experiment": cfg.name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "skill": cfg.skill,
            "runner": cfg.runner,
            "model": cfg.model,
            "model_settings": cfg.model_settings,
            "superpowers_sha": cfg.superpowers_sha,
            "seed": cfg.seed,
            "seeds_per_task": cfg.seeds_per_task,
            "timeout_seconds": cfg.timeout_seconds,
            "token_cap": cfg.token_cap,
            "learning_rate": cfg.learning_rate,
            "baseline_skill": cfg.baseline_skill,
            "candidate_files": cfg.candidates,
            "split": split.to_dict(),
        },
        "optimization": {
            "optimized": optimized,
            "steps": cfg.optimize_steps,
            "attempts": [a.to_dict() for a in attempts],
        },
        "test_split": {
            "scenario_ids": split.test,
            "comparison": comparison.to_dict(),
        },
        "metrics": metrics,
        "verdict": verdict,
        "caveat": NO_GENERAL_CLAIM,
        "n_records": len(records),
        "elapsed_seconds": round(elapsed_s, 1),
    }


def _fmt(v: Any, unit: str = "") -> str:
    return f"{v}{unit}" if v is not None else "n/a"


def _results_md(
    results: Dict[str, Any],
    metrics: Dict[str, Any],
    comparison: evalkit.EvalReport,
    attempts: List[Any],
    verdict: Dict[str, Any],
) -> str:
    p = results["provenance"]
    b = metrics["baseline"]
    c = metrics["candidate"]
    lines = [
        f"# SkillOpt controlled experiment: {results['experiment']}",
        "",
        f"_{results['generated_at']} - runner `{p['runner']}` - "
        f"model `{p['model']}` - seed `{p['seed']}`_",
        "",
        "## Verdict",
        "",
        verdict["summary"],
        "",
        f"> {NO_GENERAL_CLAIM}",
        "",
        "## Held-out test comparison",
        "",
        f"- scenarios: {', '.join(results['test_split']['scenario_ids'])}",
        f"- baseline pass rate: {comparison.rate_a:.4f}",
        f"- candidate pass rate: {comparison.rate_b:.4f}",
        f"- delta: {comparison.delta:+.4f} "
        f"(bootstrap {comparison.bootstrap.alpha:.0%} CI "
        f"[{comparison.bootstrap.low:+.4f}, {comparison.bootstrap.high:+.4f}], "
        f"n_boot={comparison.bootstrap.n_boot})",
    ]
    if comparison.mcnemar is not None:
        m = comparison.mcnemar
        lines.append(
            f"- McNemar: a_only={m.a_only} b_only={m.b_only} "
            f"p_exact={m.p_exact:.4f} "
            f"({'significant' if m.significant else 'not significant'})"
        )
    for note in comparison.notes:
        lines.append(f"- note: {note}")
    lines += [
        "",
        "## Cost metrics (informational only - quality is the gate)",
        "",
        "| metric | baseline | candidate |",
        "|---|---|---|",
        f"| runs | {b['n_runs']} | {c['n_runs']} |",
        f"| pass rate | {b['pass_rate']} | {c['pass_rate']} |",
        f"| total tokens | {_fmt(b['total_tokens'])} | {_fmt(c['total_tokens'])} |",
        f"| mean tokens | {_fmt(b['mean_tokens'])} | {_fmt(c['mean_tokens'])} |",
        f"| total tool calls | {_fmt(b['total_tool_calls'])} | {_fmt(c['total_tool_calls'])} |",
        f"| mean turns | {_fmt(b['mean_turns'])} | {_fmt(c['mean_turns'])} |",
        f"| mean latency (ms) | {_fmt(b['mean_latency_ms'])} | {_fmt(c['mean_latency_ms'])} |",
        f"| failures | {json.dumps(b['failures'])} | {json.dumps(c['failures'])} |",
        "",
        "## Optimization attempts",
        "",
    ]
    if attempts:
        lines += [
            "| step | hash | decision | val score | reason |",
            "|---|---|---|---|---|",
        ]
        for a in attempts:
            lines.append(
                f"| {a.step} | `{a.candidate_hash[:8]}` | {a.decision} | "
                f"{_fmt(a.validation_score)} | {a.reason} |"
            )
    else:
        lines.append("_No optimization attempts recorded._")
    lines += [
        "",
        "## Reproduction",
        "",
        "```",
        "python run.py --config <config used for this run>",
        "```",
        "",
        "Provenance (from results.json):",
        "",
        "```json",
        json.dumps(p, indent=2),
        "```",
        "",
        "Raw per-run data: `raw_results.csv`; raw trajectories: `trajectories/`.",
    ]
    return "\n".join(lines) + "\n"
