"""Command line interface for Gitmoot-SkillOpt."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from . import package_version


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gitmoot-skillopt",
        description=(
            "Optimize Gitmoot agent templates from Gitmoot SkillOpt training "
            "packages and emit pending candidate packages for Gitmoot review."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {package_version()}",
    )

    subcommands = parser.add_subparsers(dest="command", metavar="<command>")

    optimize = subcommands.add_parser(
        "optimize",
        help="optimize a Gitmoot training package into a candidate package",
        description=(
            "Optimize a Gitmoot training package with the Gitmoot SkillOpt "
            "adapter and write a Gitmoot-compatible candidate package plus "
            "verified output artifacts."
        ),
    )
    optimize.add_argument(
        "--training-package",
        required=True,
        help="path to a Gitmoot skillopt training package JSON file",
    )
    optimize.add_argument(
        "--artifact-root",
        required=True,
        help="path to Gitmoot's content-addressed artifact blob root",
    )
    optimize.add_argument(
        "--out-root",
        required=True,
        help="directory for optimizer run output",
    )
    optimize.add_argument(
        "--candidate-output",
        required=True,
        help="path where the candidate package JSON should be written",
    )
    optimize.add_argument(
        "--dry-run",
        action="store_true",
        help="produce a candidate package with zero training epochs; useful for fixture smoke tests",
    )
    optimize.add_argument("--num-epochs", type=int, default=1, help="number of optimization epochs")
    optimize.add_argument("--batch-size", type=int, default=4, help="training batch size")
    optimize.add_argument("--seed", type=int, default=42, help="random seed")
    optimize.add_argument("--optimizer-model", default="gpt-5.5", help="optimizer model name")
    optimize.add_argument("--target-model", default="gpt-5.5", help="target model name")
    optimize.add_argument("--optimizer-backend", default="openai_chat", help="optimizer backend")
    optimize.add_argument("--target-backend", default="openai_chat", help="target backend")
    optimize.add_argument(
        "--skill-update-mode",
        default="patch",
        choices=["patch", "rewrite_from_suggestions", "full_rewrite_minibatch"],
        help="SkillOpt update mode",
    )
    optimize.set_defaults(func=_run_optimize)

    return parser


def _run_optimize(args: argparse.Namespace) -> int:
    from .optimize import run_optimize

    run_optimize(
        training_package=args.training_package,
        artifact_root=args.artifact_root,
        out_root=args.out_root,
        candidate_output=args.candidate_output,
        dry_run=args.dry_run,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        optimizer_model=args.optimizer_model,
        target_model=args.target_model,
        optimizer_backend=args.optimizer_backend,
        target_backend=args.target_backend,
        skill_update_mode=args.skill_update_mode,
    )
    print(f"wrote candidate package: {args.candidate_output}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = getattr(args, "func", None)
    if command is None:
        parser.print_help()
        return 0
    return int(command(args) or 0)
