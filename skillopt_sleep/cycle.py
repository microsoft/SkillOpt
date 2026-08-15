"""SkillOpt-Sleep — the nightly cycle orchestrator.

run_sleep_cycle() wires the stages:
    harvest -> mine -> replay -> consolidate(gate) -> stage  (-> optional adopt)

It is pure-Python and import-light; with backend="mock" it runs with no API
key and no third-party deps, which is what the deterministic experiment and
CI use. With backend="anthropic" it spends the user's budget for real lift.
"""
from __future__ import annotations

import math
import os
import shutil
import sys
from dataclasses import dataclass
from typing import List, Optional

from skillopt_sleep import evidence
from skillopt_sleep.backend import Backend, CursorBackendError, build_backend
from skillopt_sleep.config import SleepConfig, load_config
from skillopt_sleep.dream import dream_consolidate
from skillopt_sleep.evidence import EvidenceLog
from skillopt_sleep.harvest_sources import harvest_for_config
from skillopt_sleep.memory import ensure_skill_scaffold
from skillopt_sleep.mine import group_tasks_by_skill_hint, mine
from skillopt_sleep.multi_skill import (
    SkillGroup,
    consolidate_groups,
    skill_group_reports,
)
from skillopt_sleep.staging import adopt as adopt_staging
from skillopt_sleep.staging import json_safe, redact_secrets, write_staging
from skillopt_sleep.state import SleepState, _now_iso
from skillopt_sleep.types import SessionDigest, SleepReport, TaskRecord


# ── Model-swap detection (F16) ───────────────────────────────
def _make_model_key(cfg: SleepConfig) -> str:
    """Stable string identifying the backend object(s) actually used.

    Model-change detection is advisory, so resolving its diagnostic key must
    never become an earlier failure point than construction of the real
    backend. Fall back to a credential-free description of the configured
    roles if a backend constructor cannot be used in this diagnostic path.
    """
    try:
        effective = build_backend(
            backend=cfg.get("backend", "mock"),
            model=cfg.get("model", ""),
            optimizer_backend=cfg.get("optimizer_backend", ""),
            optimizer_model=cfg.get("optimizer_model", ""),
            target_backend=cfg.get("target_backend", ""),
            target_model=cfg.get("target_model", ""),
            codex_path=cfg.get("codex_path", ""),
            pi_path=cfg.get("pi_path", ""),
            cursor_path=cfg.get("cursor_path", ""),
            opencode_path=cfg.get("opencode_path", ""),
            opencode_tool_replay=cfg.get("opencode_tool_replay", False),
            azure_endpoint=cfg.get("azure_endpoint", ""),
            project_dir=cfg.get("invoked_project", "") or os.getcwd(),
        )
    except Exception:
        backend = str(cfg.get("backend", "mock") or "mock")
        model = str(cfg.get("model", "") or "")
        split_keys = (
            "optimizer_backend",
            "optimizer_model",
            "target_backend",
            "target_model",
        )
        if not any(cfg.get(key, "") for key in split_keys):
            return f"configured:{backend}::{model}"
        optimizer_backend = str(cfg.get("optimizer_backend", "") or backend)
        optimizer_model = str(cfg.get("optimizer_model", "") or model)
        target_backend = str(cfg.get("target_backend", "") or backend)
        target_model = str(cfg.get("target_model", "") or model)
        return (
            f"configured:optimizer={optimizer_backend}::{optimizer_model};"
            f"target={target_backend}::{target_model}"
        )
    return _make_backend_key(effective)


def _make_backend_key(backend: Backend) -> str:
    """Describe resolved aliases/defaults without exposing credentials."""
    target = getattr(backend, "target", None)
    optimizer = getattr(backend, "optimizer", None)
    if target is not None and optimizer is not None:
        return (
            f"optimizer={_make_backend_key(optimizer)};"
            f"target={_make_backend_key(target)}"
        )
    name = str(getattr(backend, "name", backend.__class__.__name__) or "")
    model = str(getattr(backend, "model", "") or "")
    return f"{name}::{model}"


def _check_model_change(
    cfg: SleepConfig, state: SleepState, backend: Backend | None = None
) -> None:
    """Warn when the backend/model has changed since the last night.

    Skill text is backend-specific; adopting edits from a different model's
    reflections into a new model's skill file can cause regressions.
    This is advisory only — the cycle continues either way.
    """
    current_key = (
        _make_backend_key(backend) if backend is not None else _make_model_key(cfg)
    )
    prior_key = state.last_model_key
    if prior_key and state.last_model_key_format < 2:
        # Version 1 stored raw configuration rather than the resolved backend
        # model. Defaults and aliases make that value impossible to compare
        # truthfully, so migrate silently on the next successful night.
        return
    if prior_key and prior_key != current_key:
        print(
            f"[sleep] WARNING: model changed since last night "
            f"(was {prior_key!r}, now {current_key!r}). "
            "Learned skill text may not transfer cleanly. "
            "Consider starting from a fresh skill document.",
            file=sys.stderr,
        )


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


def _discard_unstaged_evidence(path: str) -> None:
    """Remove a pre-created evidence folder after a fail-closed Cursor call."""
    if not path:
        return
    shutil.rmtree(path, ignore_errors=True)
    # Avoid leaving an otherwise empty project .skillopt-sleep tree. Stop at
    # the first non-empty directory so existing nights are never disturbed.
    for parent in (os.path.dirname(path), os.path.dirname(os.path.dirname(path))):
        try:
            os.rmdir(parent)
        except OSError:
            break


def _markdown_table_text(value: object) -> str:
    """Keep untrusted evidence text inside one readable Markdown table cell."""
    text = " ".join(str(value).splitlines())
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("|", "&#124;")
        .replace("`", "&#96;")
    )


def _report_score(value: object) -> str:
    """Render an optional numeric score without breaking the report."""
    if value is None:
        return "—"
    try:
        score = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{score:.3f}" if math.isfinite(score) else "—"


def _render_report_md(report: SleepReport, cfg: SleepConfig) -> str:
    lines = [
        f"# SkillOpt-Sleep — night {report.night} report",
        "",
        f"- project: `{report.project}`",
        f"- backend: `{cfg.get('backend')}`  replay: `{cfg.get('replay_mode')}`",
        f"- sessions harvested: {report.n_sessions}",
        f"- tasks mined: {report.n_tasks}  (replayed: {report.n_replayed})",
        f"- held-out score: {_report_score(report.baseline_score)} "
        f"-> {_report_score(report.candidate_score)}",
        f"- gate: **{report.gate_action}** (accepted={report.accepted})",
        f"- no-regression gate: "
        f"{'enabled' if cfg.get('gate_no_regression', False) else 'disabled'}",
        f"- tokens used: {report.tokens_used}",
        "",
    ]
    gate_on = str(cfg.get("gate_mode", "on")).strip().lower() not in {
        "off", "none", "false", "greedy",
    }
    leaked_banner = report.holdout_leaked and gate_on
    if leaked_banner:
        lines[-1:] = [
            "> **Not validated.** The gate's validation slice was not disjoint "
            "from the tasks the optimizer saw (an overlapping set), so the "
            "comparison above cannot detect overfitting. Mine more tasks so a "
            "disjoint validation slice exists. Any edits below are unverified "
            "suggestions, not rejections.",
            "",
        ]
    if report.gate_trials:
        lines.append("## Held-out task changes")
        lines.append(
            "_These are observed score changes under the configured gate metric; "
            "they do not establish that a particular edit caused the change._"
        )
        lines.append("")
        for trial in report.gate_trials:
            target = _markdown_table_text(trial.get("target", "candidate"))
            accepted = bool(trial.get("accepted", False))
            blocked = bool(trial.get("blocked_by_regression", False))
            decision = "accepted" if accepted else "rejected"
            if blocked:
                decision += " (task regression)"
            baseline = _report_score(trial.get("baseline_score"))
            candidate = _report_score(trial.get("candidate_score"))
            lines.append(
                f"### `{target}` candidate — {decision} "
                f"({baseline} → {candidate})"
            )
            lines.append("")
            lines.append("| Task | Tags | Baseline | Candidate | Change |")
            lines.append("|---|---|---:|---:|---|")
            for delta in trial.get("task_deltas", []):
                task_id = _markdown_table_text(delta.get("task_id", ""))
                tags = _markdown_table_text(
                    ", ".join(str(tag) for tag in delta.get("tags", [])) or "—"
                )
                baseline_score = _report_score(delta.get("baseline_score"))
                candidate_score = _report_score(delta.get("candidate_score"))
                status = _markdown_table_text(delta.get("status", "unchanged"))
                lines.append(
                    f"| `{task_id}` | {tags} | {baseline_score} "
                    f"| {candidate_score} | {status} |"
                )
            lines.append("")
    if report.edits:
        lines.append("## Accepted edits")
        for e in report.edits:
            lines.append(f"- [{e.target}/{e.op}] {e.content}  \n  _why: {e.rationale}_")
        lines.append("")
    if report.rejected_edits:
        # On a leaked-holdout night the gate abstained rather than rejecting, so
        # these edits are unverified suggestions, not negative feedback.
        if leaked_banner:
            lines.append("## Unverified suggestions (not validated — review before adopting)")
        else:
            lines.append("## Rejected by gate (kept as negative feedback)")
        for e in report.rejected_edits:
            lines.append(f"- [{e.target}/{e.op}] {e.content}")
        lines.append("")
    if report.unmatched_edits:
        lines.append("## Proposed but changed nothing (never reached the gate)")
        lines.append(
            "_Anchor not found, replacement already present, duplicate/empty "
            "add, or an unknown op. "
            "These were never scored — check the anchor text if a rule you expected is missing._")
        for e in report.unmatched_edits:
            anchor = f"  \n  _anchor: `{e.anchor}`_" if e.anchor else ""
            lines.append(f"- [{e.target}/{e.op}] {e.content}{anchor}")
        lines.append("")
    if report.skill_groups:
        # The reviewer decides per skill, so the per-skill verdicts belong on
        # the page they actually read. Without this the rows reach report.json
        # only, and a human reviewing the night sees a single aggregate verdict
        # that no individual skill necessarily earned.
        lines.append("## Per-skill groups")
        lines.append(
            "_Each row is one skill's own evidence and its own gate decision. "
            "A rejected group does not block its neighbours, and an accepted "
            "one does not vouch for them._")
        lines.append("")
        lines.append("| Skill | Decision | Gate | Tasks | Held-out | Edits |")
        lines.append("|---|---|---|---|---|---|")
        for g in report.skill_groups:
            name = _markdown_table_text(g.skill_name or "_(no skill name)_")
            if g.status == "consolidated":
                decision = "accepted" if g.accepted else "rejected"
                scores = f"{g.baseline_score:.3f} → {g.candidate_score:.3f}"
                if g.gate_action == "reject_unverified":
                    # This is the one score the gate explicitly refuses to
                    # trust: it was measured on the same tasks the edits came
                    # from, which is how a reward hack reaches 1.000. Printing
                    # it bare reads as an improvement that was rejected for no
                    # reason, so say why the number does not count.
                    scores += " (unvalidated)"
                edits = f"{g.n_applied_edits} applied / {g.n_rejected_edits} rejected"
            else:
                # skipped and failed groups never reached the gate; showing a
                # 0.000 score for them would read as a measured result.
                decision = g.status
                scores = "—"
                edits = "—"
            reason = f" — {_markdown_table_text(g.reason)}" if g.reason else ""
            lines.append(
                f"| `{name}` | **{decision}**{reason} | {g.gate_action or '—'} "
                f"| {g.n_tasks} | {scores} | {edits} |")
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
    project = _project_paths(cfg)

    backend = backend or build_backend(
        backend=cfg.get("backend", "mock"),
        model=cfg.get("model", ""),
        optimizer_backend=cfg.get("optimizer_backend", ""),
        optimizer_model=cfg.get("optimizer_model", ""),
        target_backend=cfg.get("target_backend", ""),
        target_model=cfg.get("target_model", ""),
        codex_path=cfg.get("codex_path", ""),
        pi_path=cfg.get("pi_path", ""),
        cursor_path=cfg.get("cursor_path", ""),
        opencode_path=cfg.get("opencode_path", ""),
        opencode_tool_replay=cfg.get("opencode_tool_replay", False),
        azure_endpoint=cfg.get("azure_endpoint", ""),
        preferences=cfg.get("preferences", ""),
        project_dir=project,
    )
    _check_model_change(cfg, state, backend)  # F16: warn if model changed between nights
    night = state.begin_night(clock)
    started = _now_iso(clock)
    backend.preferences = cfg.get("preferences", "")
    _progress(cfg, f"night {night}: project={project} backend={backend.name}")

    # ── evidence log (the night's full evidentiary chain) ────────────────
    # Pre-create the staging dir so evidence.jsonl accumulates exactly where
    # the report will land; dry-runs log into the state dir instead.
    ev = None
    staging_dir_pre = ""
    # Callers may reuse a backend object across nights. Detach any logger from
    # an earlier run before honoring this run's evidence_log setting.
    evidence.attach(backend, None)
    if cfg.get("evidence_log", True):
        from skillopt_sleep.staging import _ts_dir, new_staging_dir
        if dry_run:
            ev_path = os.path.join(
                cfg.state_dir, "evidence", f"dryrun-{_ts_dir()}.jsonl")
        else:
            staging_dir_pre = new_staging_dir(project)
            ev_path = os.path.join(staging_dir_pre, "evidence.jsonl")
        ev = EvidenceLog(
            ev_path,
            max_chars=int(cfg.get("evidence_max_chars", 4000) or 4000),
            redact=bool(cfg.get("redact_secrets", True)),
        )
        evidence.attach(backend, ev)
        cycle_config = {k: cfg.get(k) for k in (
            "backend", "model", "optimizer_backend", "optimizer_model",
            "target_backend", "target_model", "gate_mode", "gate_metric",
            "gate_mixed_weight", "gate_no_regression", "edit_budget",
            "holdout_fraction",
            "dream_rollouts", "dream_factor", "recall_k",
            "max_tasks_per_night", "lookback_hours", "llm_mine",
            "evolve_skill", "evolve_memory")}
        cycle_config["opencode_tool_replay"] = (
            cfg.get("opencode_tool_replay", False) is True
        )
        ev.log("cycle", "start", night=night, project=project,
               backend=backend.name, model=cfg.get("model", ""),
               config=cycle_config)

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
        if ev is not None:
            # The transcript end of the evidentiary chain: which sessions were
            # even considered, and what signals they carried into mining.
            for d in digests:
                ev.log("harvest", "session", session_id=d.session_id,
                       project=d.project,
                       n_user_prompts=len(d.user_prompts),
                       user_prompts_head=[p[:200] for p in d.user_prompts[:6]],
                       assistant_final_head=(d.assistant_finals[-1][:300]
                                             if d.assistant_finals else ""),
                       feedback_signals=list(d.feedback_signals or []))
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
        try:
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
        except CursorBackendError:
            _discard_unstaged_evidence(staging_dir_pre)
            raise
        _progress(cfg, f"mine done: tasks={len(tasks)}")

    if ev is not None:
        # Final task pool with split assignment: which tasks train the edits
        # vs. which held-out tasks gate them (works for seeded tasks too).
        for t in tasks:
            ev.log("mine", "task_ready", task_id=t.id, split=t.split,
                   origin=t.origin, intent=t.intent[:300],
                   reference_kind=t.reference_kind,
                   checks=(t.judge or {}).get("checks", []),
                   rubric=(t.reference if t.reference_kind == "rubric" else ""),
                   source_sessions=list(t.source_sessions or []))

    report = SleepReport(
        night=night, project=project, started_at=started,
        n_sessions=n_sessions, n_tasks=len(tasks),
        gate_no_regression=bool(cfg.get("gate_no_regression", False)),
    )

    if not tasks:
        report.ended_at = _now_iso(clock)
        report.notes.append("no tasks mined — nothing to consolidate")
        state.set_last_harvest(project, started)
        state.record_night({"night": night, "accepted": False, "n_tasks": 0})
        if not dry_run:
            state.save()
        if ev is not None:
            ev.log("cycle", "end", night=night, outcome="no_tasks",
                   tokens_used=backend.tokens_used())
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
    try:
        result = dream_consolidate(
            backend, tasks, skill, memory,
            history_tasks=history_tasks,
            recall_k=recall_k,
            dream_rollouts=int(cfg.get("dream_rollouts", 1) or 1),
            dream_factor=int(cfg.get("dream_factor", 0) or 0),
            edit_budget=cfg.get("edit_budget", 4),
            gate_metric=cfg.get("gate_metric", "mixed"),
            gate_mixed_weight=cfg.get("gate_mixed_weight", 0.5),
            gate_no_regression=cfg.get("gate_no_regression", False),
            gate_mode=cfg.get("gate_mode", "on"),
            evolve_skill=cfg.get("evolve_skill", True),
            evolve_memory=cfg.get("evolve_memory", True),
            night=night,
        )
    except CursorBackendError:
        _discard_unstaged_evidence(staging_dir_pre)
        raise
    # archive tonight's real (non-dream) tasks so future nights can recall them
    state.add_to_archive([t.to_dict() for t in tasks if t.origin != "dream"])
    _progress(
        cfg,
        f"consolidate done: gate={result.gate_action} accepted={result.accepted} "
        f"edits={len(result.applied_edits)} rejected={len(result.rejected_edits)}"
        + (f" unmatched={len(result.unmatched_edits)}" if result.unmatched_edits else ""),
    )

    report.n_replayed = len(tasks)
    report.baseline_score = result.baseline_score
    report.candidate_score = result.candidate_score
    report.accepted = result.accepted
    report.gate_action = result.gate_action
    report.holdout_leaked = getattr(result, "holdout_leaked", False)
    report.no_edits_reason = getattr(result, "no_edits_reason", "")
    report.edits = result.applied_edits
    report.rejected_edits = result.rejected_edits
    report.unmatched_edits = result.unmatched_edits
    report.gate_trials = redact_secrets(getattr(result, "gate_trials", []))

    # ── 4b. optional per-skill group reporting ───────────────────────────
    # Off by default. When enabled, tonight's tasks are grouped by their skill
    # hint and each group is consolidated independently so the report carries a
    # row per skill instead of one aggregate verdict. This costs one extra
    # consolidation per hinted group, which is why it is opt-in rather than
    # automatic; a night whose evidence produces only the catch-all group adds
    # no rows and no calls.
    #
    # Each group currently starts from the same managed document. Resolving a
    # hinted group to its own live SKILL.md is the resolver's job and is not
    # wired here yet, so a row describes what that group's evidence did to the
    # managed skill, not to a separate file.
    if cfg.get("multi_skill_report", False):
        managed_name = cfg.get("managed_skill_name", "skillopt-sleep-learned")
        grouped = group_tasks_by_skill_hint(tasks, managed_name)
        only_catch_all = len(grouped) == 1 and managed_name in grouped
        if grouped and not only_catch_all:
            _progress(cfg, f"multi-skill report: groups={len(grouped)}")
            group_outcomes = consolidate_groups(
                backend,
                [SkillGroup(name, skill, rows) for name, rows in grouped.items()],
                memory,
                edit_budget=cfg.get("edit_budget", 4),
                gate_metric=cfg.get("gate_metric", "mixed"),
                gate_mixed_weight=cfg.get("gate_mixed_weight", 0.5),
                gate_no_regression=cfg.get("gate_no_regression", False),
                gate_mode=cfg.get("gate_mode", "on"),
                night=night,
            )
            group_rows = skill_group_reports(group_outcomes)
            # Skill hints and isolated backend failures are both untrusted
            # free text. Scrub them before either report.json or report.md is
            # rendered so staging cannot turn a failed call into a credential
            # leak.
            for row in group_rows:
                row.skill_name = str(redact_secrets(row.skill_name))
                row.reason = str(redact_secrets(row.reason))
            report.skill_groups = group_rows

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
            out_dir=staging_dir_pre,
        )
        if ev is not None:
            ev.log("stage", "staged", staging_dir=staging_dir,
                   has_skill=proposed_skill is not None,
                   has_memory=proposed_memory is not None,
                   accepted=result.accepted)
        # Observability: persist per-task held-out evidence + optimizer/codex errors so a
        # 0.0->0.0 night self-explains (empty responses vs failing checks vs no edits) — the
        # cycle previously captured none of this, making the gate a black box (#learning-stall).
        try:
            import json as _json
            # Backend stderr / optimizer replies / task responses can carry
            # credentials (e.g. a codex 401 stderr dump), so scrub secret-looking
            # substrings before persisting them to the on-disk diagnostics.
            with open(os.path.join(staging_dir, "diagnostics.json"), "w", encoding="utf-8") as _fh:
                _json.dump(
                    json_safe({
                        "night": night,
                        "backend": cfg.get("backend"),
                        "opencode_tool_replay": (
                            cfg.get("opencode_tool_replay", False) is True
                        ),
                        "gate_mode": cfg.get("gate_mode"),
                        "gate_no_regression": cfg.get("gate_no_regression", False),
                        "n_tasks": len(tasks),
                        "baseline_score": result.baseline_score,
                        "candidate_score": result.candidate_score,
                        "accepted": result.accepted,
                        "gate_action": result.gate_action,
                        "holdout_leaked": getattr(result, "holdout_leaked", False),
                        "n_applied_edits": len(result.applied_edits),
                        "n_rejected_edits": len(result.rejected_edits),
                        "n_unmatched_edits": len(result.unmatched_edits),
                        "call_error": redact_secrets(getattr(result, "call_error", "")),
                        "reflect_raw_head": redact_secrets(
                            (getattr(result, "reflect_raw", "") or "")[:1200]
                        ),
                        "holdout_detail": redact_secrets(
                            getattr(result, "holdout_detail", [])
                        ),
                        "gate_trials": report.gate_trials,
                    }),
                    _fh,
                    indent=2,
                    allow_nan=False,
                )
        except Exception:
            pass
        state.set_last_harvest(project, started)
        state.record_night({
            "night": night, "accepted": result.accepted,
            "baseline": result.baseline_score, "candidate": result.candidate_score,
            "n_tasks": len(tasks), "staging": staging_dir,
        })
        state.set_last_model_key(_make_backend_key(backend))  # F16: track resolved model
        # ── 6. adopt (opt-in) ────────────────────────────────────────────
        if cfg.get("auto_adopt") and result.accepted:
            adopted_paths = adopt_staging(staging_dir)
            adopted = bool(adopted_paths)
        state.save()

    if ev is not None:
        ev.log("cycle", "end", night=night, outcome="completed",
               gate_action=report.gate_action, accepted=report.accepted,
               baseline_score=report.baseline_score,
               candidate_score=report.candidate_score,
               n_applied_edits=len(report.edits),
               n_rejected_edits=len(report.rejected_edits),
               n_unmatched_edits=len(report.unmatched_edits),
               tokens_used=report.tokens_used, adopted=adopted)

    return CycleOutcome(report, staging_dir, adopted, adopted_paths)
