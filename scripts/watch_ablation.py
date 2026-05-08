#!/usr/bin/env python3
"""Watch an ablation run root and rerun final failures.

This watcher is intended to run in tmux next to scripts/run_ablation_matrix.py.
It writes STATUS.md on every poll and starts a direct rerun for any run that
the launcher marks as final [FAIL].
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

from run_ablation_matrix import PROJECT_ROOT, build_matrix, command_for


RUN_RE = re.compile(r"\[(START|DONE|FAIL|RETRY)\]\s+([^\s]+)")
ERROR_RE = re.compile(
    r"Traceback|RuntimeError|AuthenticationError|PermissionDenied|"
    r"DeploymentNotFound|LLM call failed|LLM message call failed|"
    r"BadRequestError|RateLimitError",
    re.IGNORECASE,
)


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def parse_launcher(path: Path) -> dict[str, list[str]]:
    events = {"START": [], "DONE": [], "FAIL": [], "RETRY": []}
    for line in read_text(path).splitlines():
        match = RUN_RE.search(line)
        if match:
            events[match.group(1)].append(match.group(2))
    return events


def active_run_ids(run_root: Path) -> list[str]:
    try:
        raw = subprocess.check_output(["pgrep", "-af", "scripts/train.py"], text=True)
    except subprocess.CalledProcessError:
        return []
    active: list[str] = []
    pattern = re.compile(re.escape(str(run_root)) + r"/([^\s]+)")
    for line in raw.splitlines():
        for match in pattern.finditer(line):
            active.append(match.group(1))
    return sorted(set(active))


def scan_errors(logs_dir: Path) -> dict[str, str]:
    errors: dict[str, str] = {}
    for log_path in sorted(logs_dir.glob("*.log")):
        text = read_text(log_path)
        match = ERROR_RE.search(text)
        if match:
            run_id = log_path.name.split(".watchrerun", 1)[0].removesuffix(".log")
            errors[run_id] = match.group(0)
    return errors


def load_state(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"reruns": {}}


def save_state(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def write_status(
    run_root: Path,
    total: int,
    events: dict[str, list[str]],
    active: list[str],
    completed: list[str],
    pending: list[str],
    errors: dict[str, str],
    reruns: dict[str, int],
) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    failed = sorted(set(events["FAIL"]))
    retrying = sorted(set(events["RETRY"]))
    lines = [
        "# Ablation Status",
        "",
        f"Updated: {now}",
        f"Run root: `{run_root}`",
        "",
        "| Metric | Count |",
        "| --- | ---: |",
        f"| Total planned | {total} |",
        f"| Completed summaries | {len(completed)} |",
        f"| Active train processes | {len(active)} |",
        f"| Pending/not summarized | {len(pending)} |",
        f"| Launcher final fails | {len(failed)} |",
        f"| Launcher retries | {len(retrying)} |",
        f"| Logs with error patterns | {len(errors)} |",
        "",
        "## Active",
        "",
        *(f"- `{run_id}`" for run_id in active),
        "",
        "## Final Fails",
        "",
        *(f"- `{run_id}` watcher_reruns={reruns.get(run_id, 0)}" for run_id in failed),
        "",
        "## Error Patterns",
        "",
        *(f"- `{run_id}`: `{err}`" for run_id, err in sorted(errors.items())),
        "",
        "## Recent Launcher",
        "",
        "```text",
        "\n".join(read_text(run_root / "launcher.log").splitlines()[-30:]),
        "```",
    ]
    (run_root / "STATUS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--watcher-retries", type=int, default=1)
    parser.add_argument("--groups", nargs="+", default=["all"])
    parser.add_argument("--bench", nargs="+", default=["searchqa", "spreadsheetbench"])
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    logs_dir = run_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_root / "watcher_state.json"

    groups = set(args.groups)
    if "all" in groups:
        groups = {"default", "split", "mbs", "lr", "sched", "slown", "mod", "smodel"}
    experiments = {
        exp.run_id: exp
        for exp in build_matrix(groups, args.bench, run_root, include_duplicate_defaults=False)
    }

    active_reruns: dict[str, subprocess.Popen] = {}
    while True:
        state = load_state(state_path)
        reruns = state.setdefault("reruns", {})
        events = parse_launcher(run_root / "launcher.log")
        active = active_run_ids(run_root)
        completed = sorted(
            run_id for run_id in experiments
            if (run_root / run_id / "summary.json").exists()
        )
        pending = sorted(set(experiments) - set(completed))
        errors = scan_errors(logs_dir)

        # Reap watcher-started reruns.
        for run_id, proc in list(active_reruns.items()):
            rc = proc.poll()
            if rc is None:
                continue
            active_reruns.pop(run_id, None)
            with open(logs_dir / f"{run_id}.watcher.log", "a", encoding="utf-8") as f:
                f.write(f"\n[WATCHER_DONE] rc={rc} time={time.time()}\n")

        for run_id in sorted(set(events["FAIL"])):
            if run_id not in experiments:
                continue
            if (run_root / run_id / "summary.json").exists():
                continue
            if run_id in active or run_id in active_reruns:
                continue
            count = int(reruns.get(run_id, 0))
            if count >= args.watcher_retries:
                continue
            reruns[run_id] = count + 1
            save_state(state_path, state)
            log_path = logs_dir / f"{run_id}.watchrerun{count + 1}.log"
            with open(log_path, "w", encoding="utf-8") as log_f:
                log_f.write(f"[WATCHER_START] run_id={run_id} attempt={count + 1}\n")
                log_f.flush()
                proc = subprocess.Popen(
                    command_for(experiments[run_id]),
                    cwd=PROJECT_ROOT,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    close_fds=True,
                )
            active_reruns[run_id] = proc

        save_state(state_path, state)
        write_status(
            run_root=run_root,
            total=len(experiments),
            events=events,
            active=active,
            completed=completed,
            pending=pending,
            errors=errors,
            reruns={k: int(v) for k, v in reruns.items()},
        )
        time.sleep(max(5, int(args.interval)))


if __name__ == "__main__":
    main()
