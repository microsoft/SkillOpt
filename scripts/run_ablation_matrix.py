#!/usr/bin/env python3
"""Launch the SearchQA / SpreadsheetBench ablation matrix.

By default this script prints commands only. Pass --execute to actually start
runs. Every run writes to a unique out_root under the run root and logs stdout
/ stderr to logs/<run_id>.log.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = Path("/home/azureuser/workspace-gzy/miniconda3/envs/skillopt/bin/python")

T2_ENDPOINT = "https://t2vgoaigpt4o3.openai.azure.com/"
SEARCHAGENT5_ENDPOINT = "https://searchagent5.cognitiveservices.azure.com/"


@dataclass(frozen=True)
class Experiment:
    run_id: str
    benchmark: str
    config: str
    overrides: tuple[str, ...]


BENCH_CONFIG = {
    "searchqa": "configs/searchqa/default.yaml",
    "spreadsheetbench": "configs/spreadsheetbench/default.yaml",
    "livemathematicianbench": "configs/livemathematicianbench/default.yaml",
    "alfworld": "configs/alfworld/default.yaml",
    "docvqa": "configs/docvqa/default.yaml",
}

DEFAULT_SPLIT = {
    "searchqa": "data/ablation_splits/searchqa/2-1-7_seed42",
    "spreadsheetbench": "data/ablation_splits/spreadsheetbench/2-1-7_seed42",
    "livemathematicianbench": "data/ablation_splits/livemathematicianbench/2-1-7_seed42",
    "alfworld": "data/ablation_splits/alfworld/2-1-7_seed42",
    "docvqa": "/home/azureuser/zisu/SkillReflection/data/docvqa/splits",
}

DEFAULT_TRAIN_SIZE = {
    "searchqa": 400,
    "spreadsheetbench": 80,
    "livemathematicianbench": 35,
    "alfworld": 39,
    "docvqa": 1070,
}

BATCH_SIZE_VALUES: tuple[int | str, ...] = (8, 24, 40, 56, "full")

SPLITS = {
    "searchqa": {
        "1shot": ("data/ablation_splits/searchqa/1shot_seed42", ("optimizer.slow_update_samples=1",)),
        "1-1-8": ("data/ablation_splits/searchqa/1-1-8_seed42", ()),
        "2-1-7": ("data/ablation_splits/searchqa/2-1-7_seed42", ()),
        "4-1-5": ("data/ablation_splits/searchqa/4-1-5_seed42", ()),
    },
    "spreadsheetbench": {
        "1shot": ("data/ablation_splits/spreadsheetbench/1shot_seed42", ("optimizer.slow_update_samples=1",)),
        "1-1-8": ("data/ablation_splits/spreadsheetbench/1-1-8_seed42", ()),
        "2-1-7": ("data/ablation_splits/spreadsheetbench/2-1-7_seed42", ()),
        "4-1-5": ("data/ablation_splits/spreadsheetbench/4-1-5_seed42", ()),
    },
    "livemathematicianbench": {
        "1shot": ("data/ablation_splits/livemathematicianbench/1shot_seed42", ("optimizer.slow_update_samples=1",)),
        "1-1-8": ("data/ablation_splits/livemathematicianbench/1-1-8_seed42", ()),
        "2-1-7": ("data/ablation_splits/livemathematicianbench/2-1-7_seed42", ()),
        "4-1-5": ("data/ablation_splits/livemathematicianbench/4-1-5_seed42", ()),
    },
    "alfworld": {
        "1shot": ("data/ablation_splits/alfworld/1shot_seed42", ("optimizer.slow_update_samples=1",)),
        "1-1-8": ("data/ablation_splits/alfworld/1-1-8_seed42", ()),
        "2-1-7": ("data/ablation_splits/alfworld/2-1-7_seed42", ()),
        "4-1-5": ("data/ablation_splits/alfworld/4-1-5_seed42", ()),
    },
    "docvqa": {
        "1shot": ("data/ablation_splits/docvqa/1shot_seed42", ("optimizer.slow_update_samples=1",)),
        "1-1-8": ("data/ablation_splits/docvqa/1-1-8_seed42", ()),
        "2-1-7": ("/home/azureuser/zisu/SkillReflection/data/docvqa/splits", ()),
        "4-1-5": ("data/ablation_splits/docvqa/4-1-5_seed42", ()),
    },
}


def common_overrides(benchmark: str, out_root: Path) -> list[str]:
    return [
        "model.teacher_backend=openai_chat",
        "model.student_backend=openai_chat",
        "model.teacher=gpt-5.5",
        "model.student=gpt-5.5",
        f"model.teacher_azure_openai_endpoint={T2_ENDPOINT}",
        "model.teacher_azure_openai_api_version=2024-12-01-preview",
        "model.teacher_azure_openai_auth_mode=azure_cli",
        f"model.student_azure_openai_endpoint={T2_ENDPOINT}",
        "model.student_azure_openai_api_version=2024-12-01-preview",
        "model.student_azure_openai_auth_mode=azure_cli",
        "model.reasoning_effort=medium",
        "train.num_epochs=4",
        "train.train_size=0",
        "train.batch_size=40",
        "train.accumulation=1",
        "train.seed=42",
        "gradient.minibatch_size=8",
        "gradient.merge_batch_size=8",
        "gradient.analyst_workers=16",
        "gradient.use_deep_reflect=false",
        "optimizer.learning_rate=4",
        "optimizer.min_learning_rate=2",
        "optimizer.lr_scheduler=cosine",
        "optimizer.skill_update_mode=patch",
        "optimizer.use_slow_update=true",
        "optimizer.slow_update_samples=20",
        "optimizer.use_meta_skill=true",
        "optimizer.use_meta_reflect=false",
        "evaluation.use_gate=true",
        "evaluation.eval_test=true",
        "env.split_mode=split_dir",
        f"env.split_dir={DEFAULT_SPLIT[benchmark]}",
        f"env.out_root={out_root}",
    ]


def make_experiment(
    group: str,
    benchmark: str,
    suffix: str,
    run_root: Path,
    overrides: list[str],
) -> Experiment:
    run_id = f"{group}-{benchmark}-{suffix}"
    out_root = run_root / run_id
    all_overrides = common_overrides(benchmark, out_root)
    all_overrides.extend(overrides)
    return Experiment(
        run_id=run_id,
        benchmark=benchmark,
        config=BENCH_CONFIG[benchmark],
        overrides=tuple(all_overrides),
    )


def build_matrix(
    groups: set[str],
    benchmarks: list[str],
    run_root: Path,
    *,
    include_duplicate_defaults: bool = False,
) -> list[Experiment]:
    exps: list[Experiment] = []
    group_order = [
        "default",
        "split",
        "batch",
        "mbs",
        "lr",
        "sched",
        "slown",
        "mod",
        "smodel",
        "longpair",
        "lrctrl",
    ]

    for group in group_order:
        if group not in groups:
            continue
        for benchmark in benchmarks:
            if group == "default":
                exps.append(make_experiment("DEFAULT", benchmark, "5.5", run_root, []))
                continue

            if group == "split":
                for tag, (split_dir, extra) in SPLITS[benchmark].items():
                    if not include_duplicate_defaults and tag == "2-1-7":
                        continue
                    exps.append(make_experiment(
                        "SPLIT",
                        benchmark,
                        tag,
                        run_root,
                        [f"env.split_dir={split_dir}", *extra],
                    ))
                continue

            if group == "mbs":
                for value in (1, 2, 4, 8, 16, 32):
                    if not include_duplicate_defaults and value == 8:
                        continue
                    exps.append(make_experiment(
                        "MBS",
                        benchmark,
                        str(value),
                        run_root,
                        [f"gradient.minibatch_size={value}"],
                    ))
                continue

            if group == "batch":
                for value in BATCH_SIZE_VALUES:
                    if not include_duplicate_defaults and value == 40:
                        continue
                    batch_size = DEFAULT_TRAIN_SIZE[benchmark] if value == "full" else int(value)
                    exps.append(make_experiment(
                        "BATCH",
                        benchmark,
                        str(value),
                        run_root,
                        [
                            f"train.batch_size={batch_size}",
                            "gradient.minibatch_size=8",
                        ],
                    ))
                continue

            if group == "lr":
                for value in (1, 2, 4, 8, 16):
                    exps.append(make_experiment(
                        "LR",
                        benchmark,
                        str(value),
                        run_root,
                        [
                            "optimizer.lr_scheduler=constant",
                            "optimizer.min_learning_rate=1",
                            f"optimizer.learning_rate={value}",
                        ],
                    ))
                continue

            if group == "sched":
                for value in ("constant", "cosine", "linear"):
                    if not include_duplicate_defaults and value == "cosine":
                        continue
                    exps.append(make_experiment(
                        "SCHED",
                        benchmark,
                        value,
                        run_root,
                        [f"optimizer.lr_scheduler={value}"],
                    ))
                continue

            if group == "slown":
                for value in (5, 10, 20, 40):
                    if not include_duplicate_defaults and value == 20:
                        continue
                    exps.append(make_experiment(
                        "SLOWN",
                        benchmark,
                        str(value),
                        run_root,
                        [f"optimizer.slow_update_samples={value}"],
                    ))
                continue

            if group == "mod":
                settings = {
                    "slow-meta": ("true", "true"),
                    "slow-only": ("true", "false"),
                    "meta-only": ("false", "true"),
                    "none": ("false", "false"),
                }
                for tag, (slow, meta) in settings.items():
                    if not include_duplicate_defaults and tag == "slow-meta":
                        continue
                    exps.append(make_experiment(
                        "MOD",
                        benchmark,
                        tag,
                        run_root,
                        [
                            f"optimizer.use_slow_update={slow}",
                            f"optimizer.use_meta_skill={meta}",
                        ],
                    ))
                continue

            if group == "smodel":
                student_settings = {
                    "5.4": [
                        "model.student=gpt-5.4-pro",
                        f"model.student_azure_openai_endpoint={T2_ENDPOINT}",
                        "model.student_azure_openai_api_version=2025-03-01-preview",
                        "model.student_azure_openai_auth_mode=azure_cli",
                    ],
                    "5.4-mini": [
                        "model.student=gpt-5.4-mini",
                        f"model.student_azure_openai_endpoint={SEARCHAGENT5_ENDPOINT}",
                        "model.student_azure_openai_api_version=2024-12-01-preview",
                        "model.student_azure_openai_auth_mode=azure_cli",
                    ],
                    "5.5": [],
                }
                for tag, overrides in student_settings.items():
                    if not include_duplicate_defaults and tag == "5.5":
                        continue
                    exps.append(make_experiment("SMODEL", benchmark, tag, run_root, overrides))
                continue

            if group == "longpair":
                for value in ("changed", "unchanged"):
                    exps.append(make_experiment(
                        "LONGPAIR",
                        benchmark,
                        value,
                        run_root,
                        [f"optimizer.longitudinal_pair_policy={value}"],
                    ))
                continue

            if group == "lrctrl":
                settings = {
                    "autonomous": ["optimizer.lr_control_mode=autonomous"],
                    "full-rewrite": [
                        "optimizer.lr_control_mode=none",
                        "optimizer.skill_update_mode=full_rewrite_minibatch",
                    ],
                }
                for tag, overrides in settings.items():
                    exps.append(make_experiment("LRCTRL", benchmark, tag, run_root, overrides))
                continue

    return exps


def _build_matrix_legacy(
    groups: set[str],
    benchmarks: list[str],
    run_root: Path,
    *,
    include_duplicate_defaults: bool = False,
) -> list[Experiment]:
    exps: list[Experiment] = []
    for benchmark in benchmarks:
        if "default" in groups:
            exps.append(make_experiment("DEFAULT", benchmark, "5.5", run_root, []))

        if "split" in groups:
            for tag, (split_dir, extra) in SPLITS[benchmark].items():
                if not include_duplicate_defaults and tag == "2-1-7":
                    continue
                exps.append(make_experiment(
                    "SPLIT",
                    benchmark,
                    tag,
                    run_root,
                    [f"env.split_dir={split_dir}", *extra],
                ))

        if "mbs" in groups:
            for value in (1, 2, 4, 8, 16, 32):
                if not include_duplicate_defaults and value == 8:
                    continue
                exps.append(make_experiment(
                    "MBS",
                    benchmark,
                    str(value),
                    run_root,
                    [f"gradient.minibatch_size={value}"],
                ))

        if "batch" in groups:
            for value in BATCH_SIZE_VALUES:
                if not include_duplicate_defaults and value == 40:
                    continue
                batch_size = DEFAULT_TRAIN_SIZE[benchmark] if value == "full" else int(value)
                exps.append(make_experiment(
                    "BATCH",
                    benchmark,
                    str(value),
                    run_root,
                    [
                        f"train.batch_size={batch_size}",
                        "gradient.minibatch_size=8",
                    ],
                ))

        if "lr" in groups:
            for value in (1, 2, 4, 8, 16):
                exps.append(make_experiment(
                    "LR",
                    benchmark,
                    str(value),
                    run_root,
                    [
                        "optimizer.lr_scheduler=constant",
                        "optimizer.min_learning_rate=1",
                        f"optimizer.learning_rate={value}",
                    ],
                ))

        if "sched" in groups:
            for value in ("constant", "cosine", "linear"):
                if not include_duplicate_defaults and value == "cosine":
                    continue
                exps.append(make_experiment(
                    "SCHED",
                    benchmark,
                    value,
                    run_root,
                    [f"optimizer.lr_scheduler={value}"],
                ))

        if "slown" in groups:
            for value in (5, 10, 20, 40):
                if not include_duplicate_defaults and value == 20:
                    continue
                exps.append(make_experiment(
                    "SLOWN",
                    benchmark,
                    str(value),
                    run_root,
                    [f"optimizer.slow_update_samples={value}"],
                ))

        if "mod" in groups:
            settings = {
                "slow-meta": ("true", "true"),
                "slow-only": ("true", "false"),
                "meta-only": ("false", "true"),
                "none": ("false", "false"),
            }
            for tag, (slow, meta) in settings.items():
                if not include_duplicate_defaults and tag == "slow-meta":
                    continue
                exps.append(make_experiment(
                    "MOD",
                    benchmark,
                    tag,
                    run_root,
                    [
                        f"optimizer.use_slow_update={slow}",
                        f"optimizer.use_meta_skill={meta}",
                    ],
                ))

        if "smodel" in groups:
            student_settings = {
                "5.4": [
                    "model.student=gpt-5.4-pro",
                    f"model.student_azure_openai_endpoint={T2_ENDPOINT}",
                    "model.student_azure_openai_api_version=2025-03-01-preview",
                    "model.student_azure_openai_auth_mode=azure_cli",
                ],
                "5.4-mini": [
                    "model.student=gpt-5.4-mini",
                    f"model.student_azure_openai_endpoint={SEARCHAGENT5_ENDPOINT}",
                    "model.student_azure_openai_api_version=2024-12-01-preview",
                    "model.student_azure_openai_auth_mode=azure_cli",
                ],
                "5.5": [],
            }
            for tag, overrides in student_settings.items():
                if not include_duplicate_defaults and tag == "5.5":
                    continue
                exps.append(make_experiment("SMODEL", benchmark, tag, run_root, overrides))

        if "longpair" in groups:
            for value in ("changed", "unchanged"):
                exps.append(make_experiment(
                    "LONGPAIR",
                    benchmark,
                    value,
                    run_root,
                    [f"optimizer.longitudinal_pair_policy={value}"],
                ))

        if "lrctrl" in groups:
            settings = {
                "autonomous": ["optimizer.lr_control_mode=autonomous"],
                "full-rewrite": [
                    "optimizer.lr_control_mode=none",
                    "optimizer.skill_update_mode=full_rewrite_minibatch",
                ],
            }
            for tag, overrides in settings.items():
                exps.append(make_experiment("LRCTRL", benchmark, tag, run_root, overrides))

    return exps


def command_for(exp: Experiment) -> list[str]:
    return [
        str(PYTHON_BIN),
        "scripts/train.py",
        "--config",
        exp.config,
        "--cfg-options",
        *exp.overrides,
    ]


def active_run_ids(run_root: Path, valid_run_ids: set[str] | None = None) -> set[str]:
    try:
        raw = subprocess.check_output(["pgrep", "-af", "scripts/train.py"], text=True)
    except subprocess.CalledProcessError:
        return set()
    pattern = re.compile(re.escape(str(run_root)) + r"/([^\s]+)")
    active: set[str] = set()
    for line in raw.splitlines():
        for match in pattern.finditer(line):
            run_id = match.group(1).strip("'\"")
            if run_id.endswith(".log") or "/" in run_id:
                continue
            if valid_run_ids is not None and run_id not in valid_run_ids:
                continue
            active.add(run_id)
    return active


def completed_run_ids(run_root: Path) -> set[str]:
    return {
        path.parent.name
        for path in run_root.glob("*/summary.json")
        if path.is_file()
    }


def print_commands(exps: list[Experiment]) -> None:
    for exp in exps:
        cmd = command_for(exp)
        print(f"\n# {exp.run_id}")
        print(" ".join(subprocess.list2cmdline([part]) for part in cmd))


def run_commands(
    exps: list[Experiment],
    run_root: Path,
    max_parallel: int,
    run_retries: int,
) -> int:
    logs_dir = run_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    active: list[tuple[Experiment, subprocess.Popen, object]] = []
    valid_run_ids = {exp.run_id for exp in exps}
    skipped_completed = completed_run_ids(run_root)
    skipped_active = active_run_ids(run_root, valid_run_ids)
    pending: list[tuple[Experiment, int]] = [
        (exp, 0)
        for exp in exps
        if exp.run_id not in skipped_completed and exp.run_id not in skipped_active
    ]
    for run_id in sorted(skipped_completed):
        print(f"[SKIP_COMPLETED] {run_id}", flush=True)
    for run_id in sorted(skipped_active):
        print(f"[SKIP_ACTIVE] {run_id}", flush=True)
    failures = 0

    while pending or active:
        external_active = active_run_ids(run_root, valid_run_ids) - {exp.run_id for exp, _, _ in active}
        while pending and len(active) + len(external_active) < max_parallel:
            exp, attempt = pending.pop(0)
            log_path = logs_dir / f"{exp.run_id}.log"
            if attempt:
                log_path = logs_dir / f"{exp.run_id}.retry{attempt}.log"
            log_f = open(log_path, "w", encoding="utf-8")
            print(f"[START] {exp.run_id} attempt={attempt + 1} log={log_path}", flush=True)
            proc = subprocess.Popen(
                command_for(exp),
                cwd=PROJECT_ROOT,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
            )
            setattr(proc, "_attempt", attempt)
            active.append((exp, proc, log_f))

        time.sleep(5)
        still_active: list[tuple[Experiment, subprocess.Popen, object]] = []
        for exp, proc, log_f in active:
            rc = proc.poll()
            if rc is None:
                still_active.append((exp, proc, log_f))
                continue
            log_f.close()
            if rc == 0:
                print(f"[DONE]  {exp.run_id}", flush=True)
            else:
                if getattr(proc, "_attempt", 0) < run_retries:
                    next_attempt = getattr(proc, "_attempt", 0) + 1
                    pending.append((exp, next_attempt))
                    print(f"[RETRY] {exp.run_id} rc={rc} next_attempt={next_attempt + 1}", flush=True)
                else:
                    failures += 1
                    print(f"[FAIL]  {exp.run_id} rc={rc}", flush=True)
        active = still_active

    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["default"],
        choices=[
            "default",
            "split",
            "batch",
            "mbs",
            "lr",
            "sched",
            "slown",
            "mod",
            "smodel",
            "longpair",
            "lrctrl",
            "all",
        ],
        help="Experiment groups to include. Default: default.",
    )
    parser.add_argument(
        "--bench",
        nargs="+",
        default=["searchqa", "spreadsheetbench"],
        choices=["searchqa", "spreadsheetbench", "livemathematicianbench", "alfworld", "docvqa"],
    )
    parser.add_argument("--run-root", default="", help="Output root. Default: outputs/ablation_<UTC timestamp>.")
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--run-retries", type=int, default=1, help="Retry failed runs this many times. Default: 1.")
    parser.add_argument(
        "--include-duplicate-defaults",
        action="store_true",
        help="Also run ablation points that are exactly the default setting.",
    )
    parser.add_argument("--execute", action="store_true", help="Actually start runs. Without this, print commands only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    groups = set(args.groups)
    if "all" in groups:
        groups = {"default", "split", "batch", "mbs", "lr", "sched", "slown", "mod", "smodel"}

    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    run_root = Path(args.run_root) if args.run_root else PROJECT_ROOT / "outputs" / f"ablation_{ts}"
    if not run_root.is_absolute():
        run_root = PROJECT_ROOT / run_root
    run_root.mkdir(parents=True, exist_ok=True)

    exps = build_matrix(
        groups,
        args.bench,
        run_root,
        include_duplicate_defaults=args.include_duplicate_defaults,
    )
    print(f"run_root={run_root}")
    print(f"num_experiments={len(exps)}")
    print(f"groups={','.join(sorted(groups))}")
    print(f"bench={','.join(args.bench)}")

    if not args.execute:
        print_commands(exps)
        return

    max_parallel = max(1, int(args.max_parallel))
    failures = run_commands(
        exps,
        run_root,
        max_parallel=max_parallel,
        run_retries=max(0, int(args.run_retries)),
    )
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
