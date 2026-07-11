"""SkillOpt-Sleep — command-line interface.

    python -m skillopt_sleep run        # full cycle: harvest->mine->replay->gate->stage
    python -m skillopt_sleep dry-run    # same but report only, no staging/adopt
    python -m skillopt_sleep status     # show state + latest staged proposal
    python -m skillopt_sleep adopt      # apply the latest staged proposal (with backup)
    python -m skillopt_sleep harvest    # just print what would be mined (debug)

Common flags:
    --project PATH      project to evolve (default: cwd)
    --scope all|invoked harvest scope (default: invoked)
    --max-sessions N    cap transcript sessions per run
    --max-tasks N       cap mined tasks per run
    --target-skill-path PATH explicit live SKILL.md to stage/adopt
    --tasks-file PATH   reviewed TaskRecord JSON file to replay instead of harvesting
    --backend mock|claude|codex|copilot|handoff
    --source claude|codex|auto
    --model NAME
    --lookback-hours N
    --auto-adopt
    --json              machine-readable output
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

from skillopt_sleep.config import load_config
from skillopt_sleep.cycle import run_sleep_cycle
from skillopt_sleep.harvest_sources import harvest_for_config
from skillopt_sleep.mine import mine
from skillopt_sleep.staging import adopt as adopt_staging
from skillopt_sleep.staging import latest_staging
from skillopt_sleep.state import SleepState
from skillopt_sleep.tasks_file import load_tasks_file, make_tasks_payload, write_tasks_file


def _read_text(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def _report_payload(rep, outcome) -> Dict[str, Any]:
    return {
        "night": rep.night,
        "accepted": rep.accepted,
        "gate_action": rep.gate_action,
        "no_edits_reason": getattr(rep, "no_edits_reason", ""),
        "baseline": rep.baseline_score,
        "candidate": rep.candidate_score,
        "n_tasks": rep.n_tasks,
        "n_sessions": rep.n_sessions,
        "n_accepted_edits": len(rep.edits),
        "n_rejected_edits": len(rep.rejected_edits),
        "edits": [e.__dict__ for e in rep.edits],
        "rejected_edits": [e.__dict__ for e in rep.rejected_edits],
        "notes": rep.notes,
        "staging_dir": outcome.staging_dir,
        "adopted": outcome.adopted,
    }


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--project", default="")
    p.add_argument("--scope", default="", choices=["", "all", "invoked"])
    p.add_argument("--backend", default="",
                   choices=["", "mock", "claude", "codex", "copilot", "handoff"])
    p.add_argument("--model", default="")
    p.add_argument("--codex-path", default="", help="path to the real @openai/codex binary")
    p.add_argument("--claude-home", default="", help="override ~/.claude (also isolates state)")
    p.add_argument("--codex-home", default="", help="override ~/.codex for archived session harvest")
    p.add_argument("--source", default="", choices=["", "claude", "codex", "auto"],
                   help="session transcript source")
    p.add_argument("--lookback-hours", type=int, default=None,
                   help="harvest window in hours; 0 = scan full history")
    p.add_argument("--edit-budget", type=int, default=0)
    p.add_argument("--max-sessions", type=int, default=0,
                   help="cap harvested sessions before mining; default derives from max tasks")
    p.add_argument("--max-tasks", type=int, default=0,
                   help="cap mined tasks for this run")
    p.add_argument("--target-skill-path", default="",
                   help="explicit live SKILL.md path to evolve/stage/adopt")
    p.add_argument("--tasks-file", default="",
                   help="reviewed TaskRecord JSON file to replay instead of harvesting")
    p.add_argument("--progress", action="store_true",
                   help="print phase progress to stderr")
    p.add_argument("--auto-adopt", action="store_true")
    p.add_argument("--json", action="store_true")


def _cfg_from_args(args, task_meta: Dict[str, Any] | None = None) -> Any:
    overrides: Dict[str, Any] = {}
    if args.project:
        overrides["invoked_project"] = os.path.abspath(args.project)
        overrides["projects"] = "invoked"
    if args.scope:
        overrides["projects"] = args.scope
    if args.backend:
        overrides["backend"] = args.backend
    if args.model:
        overrides["model"] = args.model
    if getattr(args, "codex_path", ""):
        overrides["codex_path"] = os.path.abspath(args.codex_path)
    if getattr(args, "claude_home", ""):
        overrides["claude_home"] = os.path.abspath(args.claude_home)
    if getattr(args, "codex_home", ""):
        overrides["codex_home"] = os.path.abspath(args.codex_home)
    if getattr(args, "source", ""):
        overrides["transcript_source"] = args.source
    lh = getattr(args, "lookback_hours", None)
    if lh is not None:  # --lookback-hours was explicitly passed (0 = full history)
        overrides["lookback_hours"] = lh
    if getattr(args, "edit_budget", 0):
        overrides["edit_budget"] = args.edit_budget
    if getattr(args, "max_sessions", 0):
        overrides["max_sessions_per_night"] = args.max_sessions
    if getattr(args, "max_tasks", 0):
        overrides["max_tasks_per_night"] = args.max_tasks
    target_skill_path = getattr(args, "target_skill_path", "")
    if not target_skill_path and task_meta:
        target_skill_path = str(task_meta.get("target_skill_path") or "")
    if target_skill_path:
        path = os.path.expanduser(target_skill_path)
        if args.project and not os.path.isabs(path):
            path = os.path.join(os.path.abspath(args.project), path)
        overrides["target_skill_path"] = os.path.abspath(path)
    if getattr(args, "progress", False):
        overrides["progress"] = True
    if getattr(args, "auto_adopt", False):
        overrides["auto_adopt"] = True
    return load_config(**overrides)


def cmd_run(args, dry: bool = False) -> int:
    task_meta: Dict[str, Any] = {}
    tasks = None
    if getattr(args, "tasks_file", ""):
        # Load once before config so target_skill_path can default from metadata.
        tasks, task_meta = load_tasks_file(args.tasks_file)
    cfg = _cfg_from_args(args, task_meta=task_meta)
    if getattr(args, "tasks_file", ""):
        tasks, task_meta = load_tasks_file(
            args.tasks_file,
            holdout_fraction=cfg.get("holdout_fraction", 0.34),
            seed=cfg.get("seed", 42),
        )
        if cfg.get("backend", "mock") != "mock" and task_meta.get("reviewed") is not True:
            print(
                "[sleep] refusing real-backend replay from an unreviewed tasks file; "
                "inspect/redact it and set \"reviewed\": true first",
                file=sys.stderr,
            )
            return 2
    if cfg.get("backend", "mock") == "handoff":
        return _run_handoff(cfg, args, seed_tasks=tasks, task_meta=task_meta, dry=dry)
    outcome = run_sleep_cycle(cfg, seed_tasks=tasks, dry_run=dry)
    _print_run_report(outcome, args, task_meta)
    return 0


def _print_run_report(outcome, args, task_meta: Dict[str, Any]) -> None:
    rep = outcome.report
    if args.json:
        payload = _report_payload(rep, outcome)
        if task_meta:
            payload["tasks_file"] = task_meta.get("tasks_file", "")
            payload["tasks_reviewed"] = task_meta.get("reviewed", False)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"[sleep] night {rep.night}: {rep.n_sessions} sessions -> {rep.n_tasks} tasks")
        print(f"[sleep] held-out {rep.baseline_score:.3f} -> {rep.candidate_score:.3f} "
              f"=> {rep.gate_action} (accepted={rep.accepted})")
        for e in rep.edits:
            print(f"   + [{e.target}/{e.op}] {e.content}")
        if rep.rejected_edits:
            print("[sleep] rejected by gate:")
            for e in rep.rejected_edits:
                print(f"   - [{e.target}/{e.op}] {e.content}")
        if outcome.staging_dir:
            print(f"[sleep] staged: {outcome.staging_dir}")
            if not outcome.adopted:
                print("[sleep] review it, then: python -m skillopt_sleep adopt")
        if outcome.adopted:
            print(f"[sleep] auto-adopted: {', '.join(outcome.adopted_paths)}")


def _handoff_dir_for(cfg) -> str:
    project = cfg.get("invoked_project") or os.getcwd()
    return os.environ.get("SKILLOPT_SLEEP_HANDOFF_DIR", "") or os.path.join(
        project, ".skillopt-sleep-handoff"
    )


def _redact_deep(obj):
    """Redact secret-looking substrings in every string of a JSON-like tree."""
    from skillopt_sleep.staging import redact_secrets
    if isinstance(obj, str):
        return redact_secrets(obj)
    if isinstance(obj, list):
        return [_redact_deep(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _redact_deep(v) for k, v in obj.items()}
    return obj


def _flush_handoff(backend, args) -> int:
    prompts_path = backend.flush_pending()
    if args.json:
        print(json.dumps({
            "handoff_pending": len(backend.pending),
            "prompts": prompts_path,
            "answers_dir": backend.answers_dir,
        }, ensure_ascii=False, indent=2))
    else:
        print(f"[sleep] handoff: {len(backend.pending)} model call(s) need answers")
        print(f"[sleep] prompts: {prompts_path}")
        print(f"[sleep] write each raw answer to {backend.answers_dir}/<id>.md, "
              "then re-run this exact command to resume")
    return 3


def _handoff_mine_and_pin(cfg, args, backend, snapshot: str, dry: bool):
    """Harvest + mine with the same knobs as run_sleep_cycle (harvest window,
    target-skill filter, candidate-limit bump, LLM mining — routed through the
    handoff files like every other model call), then pin the result to
    ``tasks.json``. Session digests are pinned too, so the sessions created
    while answering prompts cannot change what gets mined between rounds.

    Returns ``(exit_code, tasks)``; ``tasks is None`` means exit now.
    """
    import time

    from skillopt_sleep.handoff_backend import PendingCalls
    from skillopt_sleep.state import SleepState, _now_iso
    from skillopt_sleep.types import SessionDigest

    project = cfg.get("invoked_project") or os.getcwd()
    state = SleepState.load(cfg.state_path)
    started = _now_iso()

    digests_path = os.path.join(backend.handoff_dir, "digests.json")
    digests = None
    if os.path.exists(digests_path):
        try:
            with open(digests_path, encoding="utf-8") as f:
                raw = json.load(f)
            known = set(SessionDigest.__dataclass_fields__)
            digests = [SessionDigest(**{k: v for k, v in d.items() if k in known})
                       for d in raw]
        except Exception:
            # Corrupted/truncated pin (e.g. an interrupted earlier round):
            # fall back to a fresh harvest instead of crashing the run.
            print("[sleep] handoff: digests.json unreadable — re-harvesting",
                  file=sys.stderr)
            digests = None
    if digests is None:
        since = state.last_harvest_for(project)
        lookback_hours = cfg.get("lookback_hours", 72)
        if since is None and lookback_hours and lookback_hours > 0:
            since = _now_iso(time.time() - lookback_hours * 3600)
        max_tasks = cfg.get("max_tasks_per_night", 40)
        session_limit = cfg.get("max_sessions_per_night", 0) or max_tasks * 3
        digests = harvest_for_config(cfg, since_iso=since, limit=session_limit)
        os.makedirs(backend.handoff_dir, exist_ok=True)
        with open(digests_path, "w", encoding="utf-8") as f:
            json.dump(_redact_deep([d.to_dict() for d in digests]), f,
                      ensure_ascii=False, indent=2)

    max_tasks = cfg.get("max_tasks_per_night", 40)
    session_limit = cfg.get("max_sessions_per_night", 0) or max_tasks * 3
    target_skill_path = cfg.managed_skill_path() if cfg.get("target_skill_path", "") else ""
    target_skill_text = _read_text(target_skill_path) if target_skill_path else ""
    candidate_limit = max_tasks
    if cfg.get("target_task_filter", True) and target_skill_text:
        candidate_limit = max(max_tasks, max_tasks * 3)
    llm_miner = None
    if cfg.get("llm_mine", True):
        try:
            from skillopt_sleep.llm_miner import make_llm_miner
            llm_miner = make_llm_miner(
                backend, max_sessions=session_limit, max_tasks=candidate_limit,
            )
        except Exception:
            llm_miner = None
    try:
        tasks = mine(
            digests,
            max_tasks=max_tasks,
            candidate_limit=candidate_limit,
            holdout_fraction=cfg.get("holdout_fraction", 0.34),
            seed=cfg.get("seed", 42),
            llm_miner=llm_miner,
            target_skill_text=target_skill_text,
            target_skill_path=target_skill_path,
        )
    except PendingCalls:
        tasks = []
    if backend.pending:
        # LLM mining needs answers before the task set can be pinned.
        return _flush_handoff(backend, args), None
    if not tasks:
        print("[sleep] handoff: no tasks mined — nothing to consolidate")
        if not dry:
            # Advance the harvest window like run_sleep_cycle's no-tasks
            # branch, or every later run re-scans the same stale window.
            state.set_last_harvest(project, started)
            state.save()
        return 0, None
    payload = make_tasks_payload(
        tasks,
        project=project,
        transcript_source=cfg.get("transcript_source", ""),
        n_sessions=len(digests),
        target_skill_path=target_skill_path,
    )
    # NOT marked reviewed: feeding this snapshot back through --tasks-file
    # with a real backend must still hit the human-review gate above. The
    # driver itself loads it directly, with the same trust as in-cycle mining.
    write_tasks_file(snapshot, _redact_deep(payload))
    print(f"[sleep] handoff: pinned {len(tasks)} tasks -> {snapshot}")
    return 0, tasks


def _run_handoff(cfg, args, *, seed_tasks, task_meta: Dict[str, Any], dry: bool) -> int:
    """Drive the handoff backend: run until model calls are needed, then
    write the prompt batch and exit 3; on a fully-answered run, finish
    normally. Session digests and mined tasks are pinned under the handoff
    dir on the first rounds so wall-clock time between rounds (including
    the very sessions that answer the prompts) cannot change the task set
    and invalidate earlier answers.
    """
    from skillopt_sleep.handoff_backend import HandoffBackend, PendingCalls

    hdir = _handoff_dir_for(cfg)
    backend = HandoffBackend(model=cfg.get("model", ""), handoff_dir=hdir)
    tasks = seed_tasks
    if tasks is None:
        snapshot = os.path.join(hdir, "tasks.json")
        if os.path.exists(snapshot):
            tasks, _meta = load_tasks_file(
                snapshot,
                holdout_fraction=cfg.get("holdout_fraction", 0.34),
                seed=cfg.get("seed", 42),
            )
        else:
            rc, tasks = _handoff_mine_and_pin(cfg, args, backend, snapshot, dry)
            if tasks is None:
                return rc
    outcome = None
    try:
        outcome = run_sleep_cycle(cfg, seed_tasks=tasks, dry_run=dry, backend=backend)
    except PendingCalls:
        pass
    if backend.pending:
        return _flush_handoff(backend, args)
    _print_run_report(outcome, args, task_meta)
    # A completed real run ends the night: archive the handoff dir so the
    # next night re-harvests instead of replaying the pinned snapshot.
    if not dry and outcome.staging_dir and os.path.isdir(hdir):
        import time
        done = f"{hdir}.night{outcome.report.night}.done"
        if os.path.exists(done):
            done = f"{done}.{int(time.time())}"
        os.rename(hdir, done)
        print(f"[sleep] handoff: archived round data -> {done}")
    return 0


def cmd_status(args) -> int:
    cfg = _cfg_from_args(args)
    state = SleepState.load(cfg.state_path)
    project = cfg.get("invoked_project") or os.getcwd()
    latest = latest_staging(project)
    info = {
        "night": state.night,
        "state_path": cfg.state_path,
        "project": project,
        "history_tail": state.data.get("history", [])[-5:],
        "latest_staging": latest,
        "slow_memory_chars": len(state.slow_memory),
    }
    if args.json:
        print(json.dumps(info, ensure_ascii=False, indent=2))
    else:
        print(f"[sleep] nights so far: {state.night}")
        print(f"[sleep] project: {project}")
        if latest:
            print(f"[sleep] latest staged proposal: {latest}")
            rp = os.path.join(latest, "report.md")
            if os.path.exists(rp):
                with open(rp) as f:
                    print("\n" + f.read())
        else:
            print("[sleep] no staged proposals yet.")
    return 0


def cmd_adopt(args) -> int:
    cfg = _cfg_from_args(args)
    project = cfg.get("invoked_project") or os.getcwd()
    target = args.staging or latest_staging(project)
    if not target or not os.path.isdir(target):
        print("[sleep] nothing to adopt (no staging dir).")
        return 1
    updated = adopt_staging(target)
    print(f"[sleep] adopted from {target}")
    for p in updated:
        print(f"   -> {p}")
    if not updated:
        print("[sleep] (proposal contained no accepted changes)")
    return 0


def cmd_harvest(args) -> int:
    cfg = _cfg_from_args(args)
    session_limit = cfg.get("max_sessions_per_night", 0) or cfg.get("max_tasks_per_night", 40) * 3
    target_skill_path = cfg.managed_skill_path() if cfg.get("target_skill_path", "") else ""
    target_skill_text = _read_text(target_skill_path) if target_skill_path else ""
    max_tasks = cfg.get("max_tasks_per_night", 40)
    candidate_limit = max_tasks
    if cfg.get("target_task_filter", True) and target_skill_text:
        candidate_limit = max(max_tasks, max_tasks * 3)
    digests = harvest_for_config(cfg, limit=session_limit)
    tasks = mine(
        digests,
        max_tasks=max_tasks,
        candidate_limit=candidate_limit,
        holdout_fraction=cfg.get("holdout_fraction", 0.34),
        seed=cfg.get("seed", 42),
        target_skill_text=target_skill_text,
        target_skill_path=target_skill_path,
    )
    payload = make_tasks_payload(
        tasks,
        project=cfg.get("invoked_project") or os.getcwd(),
        transcript_source=cfg.get("transcript_source", ""),
        n_sessions=len(digests),
        target_skill_path=target_skill_path,
    )
    output_path = ""
    if getattr(args, "output", ""):
        output_path = write_tasks_file(args.output, payload)
    if args.json:
        json_payload = dict(payload)
        if output_path:
            json_payload["output"] = output_path
        print(json.dumps(json_payload, ensure_ascii=False, indent=2))
    else:
        print(f"[sleep] {len(digests)} sessions -> {len(tasks)} tasks")
        if output_path:
            print(f"[sleep] wrote reviewed-task draft: {output_path}")
        for t in tasks:
            print(f"  [{t.split}/{t.outcome}] {t.intent[:90]}")
    return 0


def cmd_schedule(args) -> int:
    from skillopt_sleep.scheduler import schedule, list_scheduled
    cfg = _cfg_from_args(args)
    project = cfg.get("invoked_project") or os.getcwd()
    ok, msg = schedule(project, backend=cfg.get("backend", "mock"),
                       hour=args.hour, minute=args.minute,
                       extra=("--auto-adopt" if getattr(args, "auto_adopt", False) else ""))
    print("[sleep] " + msg)
    cur = list_scheduled()
    if cur:
        print("[sleep] currently scheduled:")
        for ln in cur:
            print("   " + ln[:140])
    return 0 if ok else 1


def cmd_unschedule(args) -> int:
    from skillopt_sleep.scheduler import unschedule
    cfg = _cfg_from_args(args)
    project = cfg.get("invoked_project") or os.getcwd()
    ok, msg = unschedule(project, all_projects=getattr(args, "all", False))
    print("[sleep] " + msg)
    return 0 if ok else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="skillopt_sleep", description="SkillOpt-Sleep nightly self-evolution")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run a full sleep cycle")
    _add_common(p_run)
    p_dry = sub.add_parser("dry-run", help="harvest+mine+replay, report only")
    _add_common(p_dry)
    p_status = sub.add_parser("status", help="show state + latest proposal")
    _add_common(p_status)
    p_adopt = sub.add_parser("adopt", help="apply latest staged proposal")
    _add_common(p_adopt)
    p_adopt.add_argument("--staging", default="", help="specific staging dir")
    p_harvest = sub.add_parser("harvest", help="debug: show mined tasks")
    _add_common(p_harvest)
    p_harvest.add_argument("--output", default="", help="write mined tasks JSON for review")
    p_sched = sub.add_parser("schedule", help="install a nightly cron entry for this project")
    _add_common(p_sched)
    p_sched.add_argument("--hour", type=int, default=3)
    p_sched.add_argument("--minute", type=int, default=17)
    p_unsched = sub.add_parser("unschedule", help="remove the nightly cron entry")
    _add_common(p_unsched)
    p_unsched.add_argument("--all", action="store_true", help="remove all managed entries")

    args = parser.parse_args(argv)
    if args.cmd == "run":
        return cmd_run(args, dry=False)
    if args.cmd == "dry-run":
        return cmd_run(args, dry=True)
    if args.cmd == "status":
        return cmd_status(args)
    if args.cmd == "adopt":
        return cmd_adopt(args)
    if args.cmd == "harvest":
        return cmd_harvest(args)
    if args.cmd == "schedule":
        return cmd_schedule(args)
    if args.cmd == "unschedule":
        return cmd_unschedule(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
