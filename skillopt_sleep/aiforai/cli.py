"""Command-line entrypoint for AIForAI SkillOpt-Sleep."""

from __future__ import annotations

import argparse
import json
import os

from skillopt_sleep.aiforai.config import AiforaiConfig
from skillopt_sleep.aiforai.run import run_audit


def _parse_sources(raw: str) -> tuple[str, ...]:
    sources = [part.strip() for part in raw.split(",") if part.strip()]
    if not sources:
        return ("codex", "claude", "codewhale")
    return tuple(dict.fromkeys(sources))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="skillopt_sleep.aiforai",
        description="AIForAI SkillOpt-Sleep audit tooling",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    audit = sub.add_parser("audit", help="harvest trajectories and stage an audit report")
    audit.add_argument("--target-skill-repo", required=True)
    audit.add_argument("--sources", default="codex,claude,codewhale")
    audit.add_argument("--lookback-days", type=int, default=30)
    audit.add_argument("--max-tasks-per-source", type=int, default=40)
    audit.add_argument("--json", action="store_true")
    return parser


def _run_audit_command(args: argparse.Namespace) -> int:
    cfg = AiforaiConfig(
        target_skill_repo=os.path.abspath(args.target_skill_repo),
        sources=_parse_sources(args.sources),
        lookback_days=args.lookback_days,
        max_tasks_per_source=args.max_tasks_per_source,
    )
    result = run_audit(cfg)
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        total_sessions = sum(result.sessions_by_source.values())
        print(
            f"[aiforai] audit: {total_sessions} sessions -> "
            f"{result.checkable_tasks} checkable, {result.uncheckable_candidates} uncheckable"
        )
        print(f"[aiforai] staged: {result.staging_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "audit":
        return _run_audit_command(args)
    parser.print_help()
    return 2
