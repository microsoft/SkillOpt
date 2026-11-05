"""Offline manifest and saved-score checker: python -m scripts.check_reproducibility."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from skillopt.utils.reproducibility import SPLITS, audit, format_text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--results", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--split", choices=tuple(SPLITS), default="test")
    parser.add_argument("--decimals", type=int, choices=range(13))
    parser.add_argument("--scale", type=int, choices=(1, 100), default=1)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)
    report = audit(
        args.manifest, source_root=args.source_root, results_path=args.results,
        summary_path=args.summary, evidence_path=args.evidence, split=args.split,
        decimals=args.decimals, scale=args.scale,
    )
    print(json.dumps(report, indent=2, allow_nan=False) if args.format == "json" else format_text(report))
    return {"PASS": 0, "FAIL": 1, "MISSING": 2}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
