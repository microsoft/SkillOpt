"""SkillOpt-Sleep — command-line interface.

    python -m skillopt_sleep run        # full cycle: harvest->mine->replay->gate->stage
    python -m skillopt_sleep dry-run    # same but report only, no staging/adopt
    python -m skillopt_sleep status     # show state + latest staged proposal
    python -m skillopt_sleep adopt      # apply the latest staged proposal (with backup)
    python -m skillopt_sleep harvest    # just print what would be mined (debug)

Common flags:
    --project PATH      project to evolve (default: cwd)
    --scope all|invoked harvest scope (default: invoked)
    --backend mock|claude|codex|copilot
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
from skillopt_sleep.harvest import harvest
from skillopt_sleep.mine import mine
from skillopt_sleep.state import SleepState
from skillopt_sleep.staging import latest_staging, adopt as adopt_staging


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--project", default="")
    p.add_argument("--scope", default="", choices=["", "all", "invoked"])
    p.add_argument("--backend", default="", choices=["", "mock", "claude", "codex", "copilot"])
    p.add_argument("--model", default="")
    p.add_argument("--codex-path", default="", help="path to the real @openai/codex binary")
    p.add_argument("--claude-home", default="", help="override ~/.claude (also isolates state)")
    p.add_argument("--lookback-hours", type=int, default=0)
    p.add_argument("--edit-budget", type=int, default=0)
    p.add_argument("--auto-adopt", action="store_true")
    p.add_argument("--json", action="store_true")


def _cfg_from_args(args) -> Any:
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
    if getattr(args, "lookback_hours", 0):
        overrides["lookback_hours"] = args.lookback_hours
    if getattr(args, "edit_budget", 0):
        overrides["edit_budget"] = args.edit_budget
    if getattr(args, "auto_adopt", False):
        overrides["auto_adopt"] = True
    return load_config(**overrides)


def cmd_run(args, dry: bool = False) -> int:
    cfg = _cfg_from_args(args)
    outcome = run_sleep_cycle(cfg, dry_run=dry)
    rep = outcome.report
    if args.json:
        print(json.dumps({
            "night": rep.night, "accepted": rep.accepted,
            "gate_action": rep.gate_action,
            "baseline": rep.baseline_score, "candidate": rep.candidate_score,
            "n_tasks": rep.n_tasks, "n_sessions": rep.n_sessions,
            "edits": [e.__dict__ for e in rep.edits],
            "staging_dir": outcome.staging_dir, "adopted": outcome.adopted,
        }, ensure_ascii=False, indent=2))
    else:
        print(f"[sleep] night {rep.night}: {rep.n_sessions} sessions -> {rep.n_tasks} tasks")
        print(f"[sleep] held-out {rep.baseline_score:.3f} -> {rep.candidate_score:.3f} "
              f"=> {rep.gate_action} (accepted={rep.accepted})")
        for e in rep.edits:
            print(f"   + [{e.target}/{e.op}] {e.content}")
        if outcome.staging_dir:
            print(f"[sleep] staged: {outcome.staging_dir}")
            if not outcome.adopted:
                print("[sleep] review it, then: python -m skillopt_sleep adopt")
        if outcome.adopted:
            print(f"[sleep] auto-adopted: {', '.join(outcome.adopted_paths)}")
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
    digests = harvest(
        cfg.transcripts_dir,
        scope=cfg.get("projects", "invoked"),
        invoked_project=cfg.get("invoked_project", ""),
        limit=cfg.get("max_tasks_per_night", 40) * 3,
    )
    tasks = mine(digests, max_tasks=cfg.get("max_tasks_per_night", 40),
                 holdout_fraction=cfg.get("holdout_fraction", 0.34), seed=cfg.get("seed", 42))
    if args.json:
        print(json.dumps({
            "n_sessions": len(digests),
            "tasks": [t.to_dict() for t in tasks],
        }, ensure_ascii=False, indent=2))
    else:
        print(f"[sleep] {len(digests)} sessions -> {len(tasks)} tasks")
        for t in tasks:
            print(f"  [{t.split}/{t.outcome}] {t.intent[:90]}")
    return 0


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
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
