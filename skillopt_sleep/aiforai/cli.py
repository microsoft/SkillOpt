"""Command-line entrypoint for AIForAI SkillOpt-Sleep."""

from __future__ import annotations

import argparse
import json
import os

from skillopt_sleep.aiforai.config import AiforaiConfig
from skillopt_sleep.aiforai.run import SUPPORTED_SOURCES, run_audit


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
        description="AIForAI SkillOpt-Sleep audit tooling",
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


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.cmd == "audit":
            return _run_audit_command(args)
    except ValueError as exc:
        parser.error(str(exc))
    parser.print_help()
    return 2
