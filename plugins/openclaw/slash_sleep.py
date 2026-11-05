#!/usr/bin/env python3
"""slash_sleep.py — OpenClaw slash command equivalent of SkillOpt's /sleep.

Use from the main session as a /sleep command:
  /sleep status    — show current state + last 5 nights
  /sleep run       — trigger one cycle (all categories) right now
  /sleep run research-cron  — one cycle, single category
  /sleep adopt [night]      — adopt the most recent (or specified) staged proposal
  /sleep reject [night]     — discard the most recent (or specified) staging dir
  /sleep dry-run   — report-only cycle
  /sleep cost      — estimate per-night cost for current config

This script is a thin shell over run_sleep.py. It can be invoked either
manually from the main session or by an OpenClaw command handler.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime

SKILL_DIR = Path("/home/ethanclaw/.openclaw/workspace/skills/skillopt-sleep")
STATE_DIR = Path(os.path.expanduser("~/.skillopt-sleep"))  # default
STAGING_ROOT = STATE_DIR

def _resolve_state_dir():
    """Find the actual state dir.

    Priority: scan in order:
      1. ~/.skillopt-sleep/                 (default)
      2. /home/ethanclaw/.openclaw/workspace/.skillopt-sleep/  (when staging is there)
      3. /home/ethanclaw/.openclaw/.skillopt-sleep/            (parent of overridden claude_home)
    Pick the first one that has a state.json OR staging dir.
    """
    candidates = [
        Path(os.path.expanduser("~/.skillopt-sleep")),
        Path("/home/ethanclaw/.openclaw/workspace/.skillopt-sleep"),
        Path("/home/ethanclaw/.openclaw/.skillopt-sleep"),
    ]
    # Prefer the one with state.json
    for c in candidates:
        if (c / "state.json").exists():
            return c
    # Then the one with staging
    for c in candidates:
        if (c / "staging").exists():
            return c
    return candidates[0]

TESTS_DIR = SKILL_DIR / "tests"


def status() -> int:
    state_dir = _resolve_state_dir()
    state_file = state_dir / "state.json"
    staging_dir = state_dir / "staging"
    print(f"=== SkillOpt-Sleep status ===")
    print(f"  state dir: {state_dir}")
    print(f"  staging dir: {staging_dir}")
    if staging_dir.exists():
        stages = sorted(staging_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        print(f"  staging entries: {len(stages)}")
        for s in stages[:3]:
            print(f"    {s.name}")
    if not state_file.exists():
        print("  no state.json — run a cycle first (state is written at end of each non-dry-run)")
        return 0

    with open(state_file) as f:
        state = json.load(f)

    nights = state.get("history") or state.get("nights", [])
    print(f"  total nights: {len(nights)}")
    print(f"  accepted: {sum(1 for n in nights if n.get('accepted'))}")
    print(f"  rejected: {sum(1 for n in nights if not n.get('accepted'))}")
    if nights:
        last = nights[-1]
        print(f"  last night: {last.get('night')}")
        print(f"    accepted: {last.get('accepted')}")
        print(f"    baseline: {last.get('baseline'):.3f}  ->  candidate: {last.get('candidate'):.3f}")
        print(f"    staging: {last.get('staging') or '(none)'}")
    return 0


def run_category(category: str, *, dry_run: bool = False) -> int:
    cat_to_file = {
        "research-cron": "research-cron-tasks.json",
        "devops": "devops-tasks.json",
        "wiki": "wiki-tasks.json",
    }
    tasks_file = TESTS_DIR / cat_to_file.get(category, f"{category}-tasks.json")
    if not tasks_file.exists():
        print(f"ERROR: no tasks file for category '{category}': {tasks_file}")
        return 1

    cmd = [sys.executable, str(SKILL_DIR / "run_sleep.py")]
    if dry_run:
        cmd.append("--dry-run")
    cmd.extend(["--tasks", str(tasks_file)])

    print(f"=== /sleep run {category}{' (dry-run)' if dry_run else ''} ===")
    print(f"  cmd: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def run_all(*, dry_run: bool = False) -> int:
    rc = 0
    for cat in ("research-cron", "devops", "wiki"):
        r = run_category(cat, dry_run=dry_run)
        if r != 0:
            rc = r
    return rc


def adopt(night: str = None) -> int:
    state_dir = _resolve_state_dir()
    state_file = state_dir / "state.json"
    if not state_file.exists():
        print("ERROR: no state to adopt from")
        return 1
    with open(state_file) as f:
        state = json.load(f)
    nights = state.get("history") or state.get("nights", [])
    if not nights:
        print("ERROR: no nights recorded")
        return 1

    target = None
    if night:
        target = next((n for n in nights if str(n.get("night")) == night), None)
        if not target:
            print(f"ERROR: night '{night}' not found")
            return 1
    else:
        # most recent accepted
        candidates = [n for n in nights if n.get("accepted") and n.get("staging")]
        if not candidates:
            print("ERROR: no accepted nights with staging to adopt")
            return 1
        target = candidates[-1]

    staging = target["staging"]
    if not os.path.isdir(staging):
        print(f"ERROR: staging dir missing: {staging}")
        return 1

    print(f"=== /sleep adopt night {target['night']} ===")
    print(f"  staging: {staging}")
    print(f"  baseline: {target.get('baseline'):.3f}  candidate: {target.get('candidate'):.3f}")

    # Read proposed skill from staging
    manifest = Path(staging) / "manifest.json"
    if manifest.exists():
        with open(manifest) as f:
            m = json.load(f)
        proposed = m.get("proposed_skill")
        if proposed and Path(proposed).exists():
            live = STATE_DIR / "live_skill.md"
            backup = STATE_DIR / f"live_skill.md.bak-{target['night']}"
            if live.exists():
                shutil.copy2(live, backup)
                print(f"  backed up current live skill → {backup}")
            shutil.copy2(proposed, live)
            print(f"  adopted proposed skill → {live}")
            print()
            print("✅ Adoption complete. Next cycle will use the new skill.")
            return 0

    print("ERROR: no proposed_skill in manifest")
    return 1


def reject(night: str = None) -> int:
    state_dir = _resolve_state_dir()
    state_file = state_dir / "state.json"
    if not state_file.exists():
        print("ERROR: no state")
        return 1
    with open(state_file) as f:
        state = json.load(f)
    nights = state.get("history") or state.get("nights", [])
    target = None
    if night:
        target = next((n for n in nights if str(n.get("night")) == night), None)
    else:
        candidates = [n for n in reversed(nights) if n.get("staging")]
        target = candidates[0] if candidates else None

    if not target or not target.get("staging"):
        print("ERROR: nothing to reject")
        return 1

    staging = target["staging"]
    if os.path.isdir(staging):
        shutil.rmtree(staging)
        print(f"🗑️  Removed staging: {staging}")
    # remove from state
    state["history"] = [n for n in nights if n.get("night") != target["night"]]
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)
    print("✅ Rejected. State updated.")
    return 0


def schedule_cmd(hour: int, minute: int) -> int:
    """Install a nightly cron entry via the shared SkillOpt-Sleep scheduler.

    Note: this schedules the shared engine (``python -m skillopt_sleep run``),
    not the OpenClaw-specific ``run_sleep.py``.  Use ``run_sleep_cron.sh`` if
    you need the OpenClaw-native backend and category task files instead.
    """
    try:
        from skillopt_sleep.scheduler import schedule
    except ImportError:
        print("ERROR: skillopt_sleep.scheduler not available — is SkillOpt-Sleep installed?")
        return 1
    project = str(SKILL_DIR)
    ok, msg = schedule(project, hour=hour, minute=minute)
    print(msg)
    return 0 if ok else 1


def unschedule_cmd(all_projects: bool) -> int:
    """Remove cron entry via the shared SkillOpt-Sleep scheduler."""
    try:
        from skillopt_sleep.scheduler import unschedule
    except ImportError:
        print("ERROR: skillopt_sleep.scheduler not available — is SkillOpt-Sleep installed?")
        return 1
    project = str(SKILL_DIR)
    ok, msg = unschedule(project, all_projects=all_projects)
    print(msg)
    return 0 if ok else 1


def cost() -> int:
    """Estimate per-night cost based on the actual measurement from Phase 2.

    From the real dry-run: 5 devops tasks used 14,427 tokens total.
    That is ~2,885 tokens per task (all 3 phases combined).
    """
    cfg_path = SKILL_DIR / "config.json"
    cfg = {}
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
    cfg.pop("_comment", None)

    max_tasks = cfg.get("max_tasks_per_night", 12)
    model = cfg.get("model", "deepseek-v4-pro")
    # DeepSeek V4 pricing
    if "pro" in model:
        cost_in = 0.435  # per 1M
        cost_out = 0.87
    elif "flash" in model:
        cost_in = 0.14
        cost_out = 0.28
    else:
        cost_in, cost_out = 0.5, 1.0

    # Measured: ~2,900 tokens per task, 30% output / 70% input
    toks_per_task = 2900
    input_toks = int(toks_per_task * 0.7)
    output_toks = int(toks_per_task * 0.3)

    cost_in_total = (input_toks * max_tasks / 1_000_000) * cost_in
    cost_out_total = (output_toks * max_tasks / 1_000_000) * cost_out
    cost = cost_in_total + cost_out_total

    print(f"=== Cost estimate (per actual measurement) ===")
    print(f"  model: {model}")
    print(f"  max tasks/night: {max_tasks}")
    print(f"  ~tokens/night: {toks_per_task * max_tasks:,}")
    print(f"  cost/night: ${cost:.3f}")
    print(f"  cost/month (30 nights): ${cost*30:.2f}")
    print(f"  cost/year (365 nights): ${cost*365:.2f}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="OpenClaw /sleep command")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="show state + last 5 nights")
    p_run = sub.add_parser("run", help="trigger one cycle")
    p_run.add_argument("category", nargs="?", default=None,
                        choices=["research-cron", "devops", "wiki", None])
    p_run.add_argument("--dry-run", action="store_true")
    sub.add_parser("dry-run", help="report-only cycle (all categories)")
    p_adopt = sub.add_parser("adopt", help="adopt most recent accepted staging")
    p_adopt.add_argument("night", nargs="?", default=None)
    p_reject = sub.add_parser("reject", help="discard most recent staging")
    p_reject.add_argument("night", nargs="?", default=None)
    sub.add_parser("cost", help="estimate cost")
    p_schedule = sub.add_parser("schedule", help="install nightly cron entry")
    p_schedule.add_argument("--hour", type=int, default=3, help="hour (0-23)")
    p_schedule.add_argument("--minute", type=int, default=17, help="minute (0-59)")
    p_unschedule = sub.add_parser("unschedule", help="remove cron entry")
    p_unschedule.add_argument("--all", dest="all_projects", action="store_true",
                               help="remove entries for all projects")

    args = ap.parse_args()

    if args.cmd == "status":
        return status()
    if args.cmd == "run":
        if args.category:
            return run_category(args.category, dry_run=args.dry_run)
        return run_all(dry_run=args.dry_run)
    if args.cmd == "dry-run":
        return run_all(dry_run=True)
    if args.cmd == "adopt":
        return adopt(args.night)
    if args.cmd == "reject":
        return reject(args.night)
    if args.cmd == "cost":
        return cost()
    if args.cmd == "schedule":
        return schedule_cmd(args.hour, args.minute)
    if args.cmd == "unschedule":
        return unschedule_cmd(args.all_projects)
    return 1


if __name__ == "__main__":
    sys.exit(main())
