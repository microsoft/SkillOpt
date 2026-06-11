"""Command-line entrypoint for AIForAI SkillOpt-Sleep."""

from __future__ import annotations

import argparse
import json
import os

from skillopt_sleep.aiforai.config import AiforaiConfig
from skillopt_sleep.aiforai.run import SUPPORTED_SOURCES, adopt_latest, run_audit, run_mock_gate


def _parse_sources(raw: str) -> tuple[str, ...]:
    sources = [part.strip() for part in raw.split(",") if part.strip()]
    if not sources:
        raise argparse.ArgumentTypeError("at least one source must be provided")
    unsupported = [source for source in sources if source not in SUPPORTED_SOURCES]
    if unsupported:
        supported = ", ".join(SUPPORTED_SOURCES)
        invalid = ", ".join(dict.fromkeys(unsupported))
        raise argparse.ArgumentTypeError(
            f"unsupported source(s): {invalid}. Supported sources: {supported}"
        )
    return tuple(dict.fromkeys(sources))


def _positive_int_arg(name: str):
    def parse(raw: str) -> int:
        try:
            value = int(raw)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{name} must be an integer") from exc
        if value <= 0:
            raise argparse.ArgumentTypeError(f"{name} must be > 0")
        return value

    return parse


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="skillopt_sleep.aiforai",
        description="AIForAI SkillOpt-Sleep tooling",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    audit = sub.add_parser("audit", help="harvest trajectories and stage an audit report")
    audit.add_argument("--target-skill-repo", required=True)
    audit.add_argument(
        "--sources",
        type=_parse_sources,
        default=_parse_sources("codex,claude,codewhale"),
    )
    audit.add_argument("--lookback-days", type=_positive_int_arg("lookback-days"), default=30)
    audit.add_argument(
        "--max-tasks-per-source",
        type=_positive_int_arg("max-tasks-per-source"),
        default=40,
    )
    audit.add_argument("--json", action="store_true")

    run = sub.add_parser("run", help="harvest trajectories and stage a mock-gated proposal")
    run.add_argument("--target-skill-repo", required=True)
    run.add_argument(
        "--sources",
        type=_parse_sources,
        default=_parse_sources("codex,claude,codewhale"),
    )
    run.add_argument("--lookback-days", type=_positive_int_arg("lookback-days"), default=7)
    run.add_argument(
        "--max-tasks-per-source",
        type=_positive_int_arg("max-tasks-per-source"),
        default=40,
    )
    run.add_argument("--json", action="store_true")

    adopt = sub.add_parser("adopt", help="adopt the latest accepted staged proposal")
    adopt.add_argument("--target-skill-repo", required=True)
    return parser


def _run_audit_command(args: argparse.Namespace) -> int:
    cfg = AiforaiConfig(
        target_skill_repo=os.path.abspath(args.target_skill_repo),
        sources=args.sources,
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


def _run_mock_gate_command(args: argparse.Namespace) -> int:
    cfg = AiforaiConfig(
        target_skill_repo=os.path.abspath(args.target_skill_repo),
        sources=args.sources,
        lookback_days=args.lookback_days,
        max_tasks_per_source=args.max_tasks_per_source,
    )
    result = run_mock_gate(cfg)
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(f"[aiforai] run staged: {result.staging_dir}")
        print(f"[aiforai] accepted: {result.accepted}")
    return 0


def _adopt_command(args: argparse.Namespace) -> int:
    cfg = AiforaiConfig(
        target_skill_repo=os.path.abspath(args.target_skill_repo),
    )
    updated = adopt_latest(cfg)
    if not updated:
        print("[aiforai] no accepted staging proposal to adopt")
        return 1
    for path in updated:
        print(f"[aiforai] adopted: {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.cmd == "audit":
            return _run_audit_command(args)
        if args.cmd == "run":
            return _run_mock_gate_command(args)
        if args.cmd == "adopt":
            return _adopt_command(args)
    except ValueError as exc:
        parser.error(str(exc))
    parser.print_help()
    return 2
