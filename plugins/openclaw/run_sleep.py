#!/usr/bin/env python3
"""run_sleep.py — OpenClaw entry point for SkillOpt-Sleep.

Runs one nightly sleep cycle:
  1. harvest recent session transcripts
  2. mine recurring task patterns
  3. replay tasks with current skill (baseline) + candidate skill (with proposed edit)
  4. gate candidate vs baseline on held-out accuracy
  5. stage the proposal in ~/.skillopt-sleep/staging/<night>/
  6. leave adoption to Ethan (auto_adopt=false)

Usage:
  python3 run_sleep.py                  # one cycle, default config
  python3 run_sleep.py --dry-run        # compute report only, no staging
  python3 run_sleep.py --tasks path.json  # use a pre-built task file
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure the skillopt_sleep package is importable (it lives in the cloned repo)
REPO = Path("/home/ethanclaw/.openclaw/workspace/SkillOpt")
sys.path.insert(0, str(REPO))

# Register our backend before importing cycle
from skillopt_sleep_openclaw import OpenClawDeepSeekBackend
import skillopt_sleep.backend as _b
_b._BACKENDS = getattr(_b, "_BACKENDS", {})
_b._BACKENDS["openclaw-deepseek"] = OpenClawDeepSeekBackend

# Patch get_backend to know about our backend
_orig_get_backend = _b.get_backend

def get_backend(name, model="", codex_path="", cursor_path="", project_dir=""):
    if name == "openclaw-deepseek":
        return OpenClawDeepSeekBackend(model=model or "deepseek-v4-pro")
    return _orig_get_backend(
        name,
        model=model,
        codex_path=codex_path,
        cursor_path=cursor_path,
        project_dir=project_dir,
    )

_b.get_backend = get_backend

from skillopt_sleep.cycle import run_sleep_cycle
from skillopt_sleep.config import load_config


def main() -> int:
    ap = argparse.ArgumentParser(description="OpenClaw SkillOpt-Sleep nightly cycle")
    ap.add_argument("--dry-run", action="store_true", help="Compute but don't stage")
    ap.add_argument("--config", default="/home/ethanclaw/.openclaw/workspace/skills/skillopt-sleep/config.json")
    ap.add_argument("--tasks", default=None, help="Path to pre-built tasks JSON")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    # Load config from file then override with our defaults
    overrides = {}
    if os.path.exists(args.config):
        with open(args.config) as f:
            overrides.update(json.load(f))
    overrides.pop("_comment", None)

    cfg = load_config(**overrides)

    seed_tasks = None
    if args.tasks:
        from skillopt_sleep.types import TaskRecord
        with open(args.tasks) as f:
            raw = json.load(f)
        # Translate our test-set fields → TaskRecord fields
        seed_tasks = []
        for t in raw:
            seed_tasks.append(TaskRecord(
                id=t['id'],
                project=t.get('project', 'openclaw'),
                intent=t.get('intent') or t.get('prompt', ''),
                context_excerpt=t.get('context_excerpt', ''),
                attempted_solution=t.get('attempted_solution', ''),
                outcome=t.get('outcome', 'unknown'),
                reference_kind=t.get('reference_kind', 'rubric'),
                reference=t.get('reference', ''),
                judge=t.get('judge', {}),
                tags=t.get('tags', []),
                source_sessions=t.get('source_sessions', []),
                split=t.get('split', 'train'),
            ))

    print(f"[skillopt-sleep] starting cycle...")
    print(f"  backend: {cfg.get('backend')}")
    print(f"  project: {cfg.get('invoked_project')}")
    print(f"  max tasks: {cfg.get('max_tasks_per_night')}")
    print(f"  edit budget: {cfg.get('edit_budget')}")
    print(f"  dry_run: {args.dry_run}")

    outcome = run_sleep_cycle(cfg, seed_tasks=seed_tasks, dry_run=args.dry_run)

    r = outcome.report
    print(f"\n=== Report — night {r.night} ===")
    print(f"  sessions harvested: {r.n_sessions}")
    print(f"  tasks mined: {r.n_tasks}  (replayed: {r.n_replayed})")
    print(f"  baseline: {r.baseline_score:.3f}  ->  candidate: {r.candidate_score:.3f}")
    print(f"  gate: {r.gate_action}  accepted={r.accepted}")
    print(f"  tokens: {r.tokens_used}")
    if r.edits:
        print(f"  applied edits ({len(r.edits)}):")
        for e in r.edits:
            print(f"    [{e.target}/{e.op}] {e.content[:80]}...")
    if r.rejected_edits:
        print(f"  rejected edits ({len(r.rejected_edits)}) — kept as negative feedback")
    if r.notes:
        for n in r.notes:
            print(f"  note: {n}")
    if outcome.staging_dir:
        print(f"\n  STAGED at: {outcome.staging_dir}")
        print(f"  Review with: ls {outcome.staging_dir}")

    return 0 if r.accepted or r.candidate_score >= r.baseline_score else 1


if __name__ == "__main__":
    sys.exit(main())
