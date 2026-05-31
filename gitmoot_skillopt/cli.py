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
            "Optimize a Gitmoot training package. This scaffold documents the "
            "stable command shape; package parsing and training execution are "
            "implemented in later goal tasks."
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
    optimize.set_defaults(func=_run_optimize)

    return parser


def _run_optimize(args: argparse.Namespace) -> int:
    del args
    raise SystemExit(
        "gitmoot-skillopt optimize is scaffolded; implementation follows in "
        "the Gitmoot adapter and package-contract tasks."
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = getattr(args, "func", None)
    if command is None:
        parser.print_help()
        return 0
    return int(command(args) or 0)
