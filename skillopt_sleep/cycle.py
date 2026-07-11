"""SkillOpt-Sleep — the nightly cycle orchestrator.

run_sleep_cycle() wires the stages:
    harvest -> mine -> replay -> consolidate(gate) -> stage  (-> optional adopt)

It is pure-Python and import-light; with backend="mock" it runs with no API
key and no third-party deps, which is what the deterministic experiment and
CI use. With backend="anthropic" it spends the user's budget for real lift.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import List, Optional

from skillopt_sleep.backend import Backend, get_backend
from skillopt_sleep.config import SleepConfig, load_config
from skillopt_sleep.dream import dream_consolidate
from skillopt_sleep.harvest_sources import harvest_for_config
from skillopt_sleep.memory import ensure_skill_scaffold
from skillopt_sleep.mine import mine
from skillopt_sleep.staging import adopt as adopt_staging
from skillopt_sleep.staging import redact_secrets
from skillopt_sleep.staging import write_staging
from skillopt_sleep.state import SleepState, _now_iso
from skillopt_sleep.types import SessionDigest, SleepReport, TaskRecord


@dataclass
class CycleOutcome:
    report: SleepReport
    staging_dir: str
    adopted: bool
    adopted_paths: List[str]


def _project_paths(cfg: SleepConfig) -> str:
    """Where live CLAUDE.md lives + which project we are evolving."""
    if cfg.get("projects") == "invoked" and cfg.get("invoked_project"):
        return cfg.get("invoked_project")
    # default: the invoked cwd
    return cfg.get("invoked_project") or os.getcwd()


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def _progress(cfg: SleepConfig, message: str) -> None:
    if cfg.get("progress", False):
        print(f"[sleep] {message}", file=sys.stderr, flush=True)


def _render_report_md(report: SleepReport, cfg: SleepConfig) -> str:
    lines = [
        f"# SkillOpt-Sleep — night {report.night} report",
        "",
        f"- project: `{report.project}`",
        f"- backend: `{cfg.get('backend')}`  replay: `{cfg.get('replay_mode')}`",
        f"- sessions harvested: {report.n_sessions}",
        f"- tasks mined: {report.n_tasks}  (replayed: {report.n_replayed})",
        f"- held-out score: {report.baseline_score:.3f} -> {report.candidate_score:.3f}",
        f"- gate: **{report.gate_action}** (accepted={report.accepted})",
        f"- tokens used: {report.tokens_used}",
        "",
    ]
    if report.edits:
        lines.append("## Accepted edits")
        for e in report.edits:
            lines.append(f"- [{e.target}/{e.op}] {e.content}  \n  _why: {e.rationale}_")
        lines.append("")
    if report.rejected_edits:
        lines.append("## Rejected by gate (kept as negative feedback)")
        for e in report.rejected_edits:
            lines.append(f"- [{e.target}/{e.op}] {e.content}")
        lines.append("")
    if report.notes:
        lines.append("## Notes")
        for n in report.notes:
            lines.append(f"- {n}")
        lines.append("")
    lines.append("_Review, then run `/sleep adopt` to apply, or discard this folder._")
    return "\n".join(lines)


def run_sleep_cycle(
    cfg: Optional[SleepConfig] = None,
    *,
    seed_tasks: Optional[List[TaskRecord]] = None,
    dry_run: bool = False,
    clock: Optional[float] = None,
    backend: Optional[Backend] = None,
) -> CycleOutcome:
    """Run one full sleep cycle and return the outcome.

    Parameters
    ----------
    cfg : SleepConfig
    seed_tasks : optional pre-built TaskRecords (used by the experiment to
        inject a known persona instead of harvesting ~/.claude).
    dry_run : harvest+mine+replay but DO NOT stage/adopt (report only).
    clock : fixed epoch seconds for deterministic timestamps in tests.
    backend : optional pre-built Backend; the handoff driver passes one so
        it can inspect the backend's pending calls after the run.
    """
    cfg = cfg or load_config()
    state = SleepState.load(cfg.state_path)
    night = state.begin_night(clock)
    project = _project_paths(cfg)
    started = _now_iso(clock)

    backend = backend or get_backend(
        cfg.get("backend", "mock"),
        model=cfg.get("model", ""),
        codex_path=cfg.get("codex_path", ""),
        project_dir=project,
    )
    _progress(cfg, f"night {night}: project={project} backend={backend.name}")

    # ── live skill/memory docs ───────────────────────────────────────────
    live_memory_path = os.path.join(project, "CLAUDE.md")
    live_skill_path = cfg.managed_skill_path()
    _progress(cfg, f"live skill: {live_skill_path}")
    raw_skill = _read(live_skill_path)
    skill = raw_skill
    memory = _read(live_memory_path)
    if not skill:
        skill = ensure_skill_scaffold(
            "", name=cfg.get("managed_skill_name", "skillopt-sleep-learned"),
            description="Preferences and procedures learned from past local agent sessions.",
        )
    target_filter = bool(
        cfg.get("target_task_filter", True)
        and cfg.get("target_skill_path", "")
        and raw_skill
    )

    # ── 1+2. harvest + mine (unless seed_tasks injected) ─────────────────
    digests: List[SessionDigest] = []
    if seed_tasks is not None:
        tasks = seed_tasks
        n_sessions = 0
        _progress(cfg, f"using {len(tasks)} seeded tasks")
    else:
        since = state.last_harvest_for(project)
        # On first run (no prior harvest), apply lookback_hours so we don't
        # scan the entire transcript history and trigger massive LLM mining.
        if since is None:
            lookback_hours = cfg.get("lookback_hours", 72)
            if lookback_hours is not None and lookback_hours > 0:
                import time
                ref_time = clock if clock is not None else time.time()
                cutoff = ref_time - lookback_hours * 3600
                since = _now_iso(cutoff)
        max_tasks = cfg.get("max_tasks_per_night", 40)
        max_sessions = cfg.get("max_sessions_per_night", 0) or max_tasks * 3
        candidate_limit = max_tasks
        if target_filter:
            candidate_limit = max(max_tasks, max_tasks * 3)
        _progress(
            cfg,
            f"harvest start: source={cfg.get('transcript_source')} max_sessions={max_sessions}",
        )
        digests = harvest_for_config(
            cfg,
            since_iso=since,
            limit=max_sessions,
        )
        n_sessions = len(digests)
        _progress(cfg, f"harvest done: sessions={n_sessions}")
        # When a real backend is configured, use it to mine checkable tasks from
        # the transcripts (rubric/rule judges); otherwise fall back to the
        # heuristic miner (no API, no checkable reference).
        llm_miner = None
        if cfg.get("backend", "mock") != "mock" and cfg.get("llm_mine", True):
            try:
                from skillopt_sleep.llm_miner import make_llm_miner
                llm_miner = make_llm_miner(
                    backend,
                    max_sessions=max_sessions,
                    max_tasks=candidate_limit,
                )
            except Exception:
                llm_miner = None
        _progress(
            cfg,
            f"mine start: max_tasks={max_tasks} candidate_limit={candidate_limit} "
            f"llm_mine={llm_miner is not None} target_filter={target_filter}",
        )
        tasks = mine(
            digests,
            max_tasks=max_tasks,
            candidate_limit=candidate_limit,
            holdout_fraction=cfg.get("holdout_fraction", 0.34),
            seed=cfg.get("seed", 42),
            llm_miner=llm_miner,
            target_skill_text=raw_skill if target_filter else "",
            target_skill_path=live_skill_path if target_filter else "",
        )
        _progress(cfg, f"mine done: tasks={len(tasks)}")

    report = SleepReport(
        night=night, project=project, started_at=started,
        n_sessions=n_sessions, n_tasks=len(tasks),
    )

    if not tasks:
        report.ended_at = _now_iso(clock)
        report.notes.append("no tasks mined — nothing to consolidate")
        state.set_last_harvest(project, started)
        state.record_night({"night": night, "accepted": False, "n_tasks": 0})
        if not dry_run:
            state.save()
        staging_dir = ""
        return CycleOutcome(report, staging_dir, False, [])

    # ── 3+4. replay + consolidate (gate), with opt-in dream + recall ──────
    # recall pulls similar past tasks from the persisted archive; dream_rollouts
    # / dream_factor enrich the training signal. With the defaults (recall_k=0,
    # dream_rollouts=1, dream_factor=0) this is exactly the prior single-shot
    # consolidate — behavior is unchanged unless the user opts in.
    _progress(cfg, "consolidate start")
    recall_k = int(cfg.get("recall_k", 0) or 0)
    history_tasks = []
    if recall_k > 0:
        history_tasks = [TaskRecord.from_dict(d) for d in state.task_archive()]
    result = dream_consolidate(
        backend, tasks, skill, memory,
        history_tasks=history_tasks,
        recall_k=recall_k,
        dream_rollouts=int(cfg.get("dream_rollouts", 1) or 1),
        dream_factor=int(cfg.get("dream_factor", 0) or 0),
        edit_budget=cfg.get("edit_budget", 4),
        gate_metric=cfg.get("gate_metric", "mixed"),
        gate_mixed_weight=cfg.get("gate_mixed_weight", 0.5),
        gate_mode=cfg.get("gate_mode", "on"),
        evolve_skill=cfg.get("evolve_skill", True),
        evolve_memory=cfg.get("evolve_memory", True),
        night=night,
    )
    # archive tonight's real (non-dream) tasks so future nights can recall them
    state.add_to_archive([t.to_dict() for t in tasks if t.origin != "dream"])
    _progress(
        cfg,
        f"consolidate done: gate={result.gate_action} accepted={result.accepted} "
        f"edits={len(result.applied_edits)} rejected={len(result.rejected_edits)}",
    )

    report.n_replayed = len(tasks)
    report.baseline_score = result.baseline_score
    report.candidate_score = result.candidate_score
    report.accepted = result.accepted
    report.gate_action = result.gate_action
    report.no_edits_reason = getattr(result, "no_edits_reason", "")
    report.edits = result.applied_edits
    report.rejected_edits = result.rejected_edits
    report.tokens_used = backend.tokens_used()
    report.ended_at = _now_iso(clock)

    # ── 5. stage (unless dry-run) ────────────────────────────────────────
    staging_dir = ""
    adopted = False
    adopted_paths: List[str] = []
    if not dry_run:
        _progress(cfg, "staging start")
        report_md = _render_report_md(report, cfg)
        proposed_skill = result.new_skill if (cfg.get("evolve_skill") and result.accepted) else None
        proposed_memory = result.new_memory if (cfg.get("evolve_memory") and result.accepted) else None
        staging_dir = write_staging(
            project,
            report=report,
            proposed_skill=proposed_skill,
            proposed_memory=proposed_memory,
            live_skill_path=live_skill_path,
            live_memory_path=live_memory_path,
            report_md=report_md,
        )
        # Observability: persist per-task held-out evidence + optimizer/codex errors so a
        # 0.0->0.0 night self-explains (empty responses vs failing checks vs no edits) — the
        # cycle previously captured none of this, making the gate a black box (#learning-stall).
        try:
            import json as _json
            # Backend stderr / optimizer replies / task responses can carry
            # credentials (e.g. a codex 401 stderr dump), so scrub secret-looking
            # substrings before persisting them to the on-disk diagnostics.
            with open(os.path.join(staging_dir, "diagnostics.json"), "w", encoding="utf-8") as _fh:
                _json.dump({
                    "night": night,
                    "backend": cfg.get("backend"),
                    "gate_mode": cfg.get("gate_mode"),
                    "n_tasks": len(tasks),
                    "baseline_score": result.baseline_score,
                    "candidate_score": result.candidate_score,
                    "accepted": result.accepted,
                    "n_applied_edits": len(result.applied_edits),
                    "n_rejected_edits": len(result.rejected_edits),
                    "call_error": redact_secrets(getattr(result, "call_error", "")),
                    "reflect_raw_head": redact_secrets(
                        (getattr(result, "reflect_raw", "") or "")[:1200]
                    ),
                    "holdout_detail": redact_secrets(getattr(result, "holdout_detail", [])),
                }, _fh, indent=2)
        except Exception:
            pass
        state.set_last_harvest(project, started)
        state.record_night({
            "night": night, "accepted": result.accepted,
            "baseline": result.baseline_score, "candidate": result.candidate_score,
            "n_tasks": len(tasks), "staging": staging_dir,
        })
        # ── 6. adopt (opt-in) ────────────────────────────────────────────
        if cfg.get("auto_adopt") and result.accepted:
            adopted_paths = adopt_staging(staging_dir)
            adopted = bool(adopted_paths)
        state.save()

    return CycleOutcome(report, staging_dir, adopted, adopted_paths)
