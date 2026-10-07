#!/usr/bin/env python
"""Run one controlled baseline-vs-candidate experiment.

    python run.py --config config.mock.yaml

Emits raw_results.csv + results.json + RESULTS.md + trajectories/ under the
config's output_dir.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from skillopt_eval.config import ConfigError, ExperimentConfig
from skillopt_eval.experiment import run_experiment
from skillopt_eval.mock_runner import MockRunner
from skillopt_eval.superpowers_runner import SuperpowersRunner


def build_runner(cfg: ExperimentConfig):
    if cfg.runner == "mock":
        return MockRunner(
            skill=cfg.skill, model=cfg.model,
            timeout=cfg.timeout_seconds,
        )
    if cfg.runner == "superpowers":
        return SuperpowersRunner(
            skill=cfg.skill,
            pinned_sha=cfg.superpowers_sha,
            version=cfg.model_settings.get("superpowers_version", "v6.1.1"),
            model=cfg.model,
            timeout=cfg.timeout_seconds,
            token_cap=cfg.token_cap,
        )
    raise ConfigError(f"unknown runner {cfg.runner!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="path to experiment YAML")
    args = ap.parse_args()

    try:
        cfg = ExperimentConfig.load(args.config)
        runner = build_runner(cfg)
        baseline_text = None
        if cfg.baseline_skill:
            baseline_text = Path(cfg.baseline_skill).read_text(encoding="utf-8")
        results = run_experiment(cfg, runner, baseline_text)
    except (ConfigError, FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    out = Path(cfg.output_dir)
    v = results["verdict"]
    print(f"experiment: {results['experiment']} ({cfg.runner} runner)")
    print(f"verdict:    {v['summary']}")
    print(f"optimized:  {results['optimization']['optimized']}")
    print(f"artifacts:  {out / 'raw_results.csv'}")
    print(f"            {out / 'results.json'}")
    print(f"            {out / 'RESULTS.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
