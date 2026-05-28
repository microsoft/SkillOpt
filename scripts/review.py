#!/usr/bin/env python3
"""Headless reviewer for human-in-the-loop SkillOpt runs.

Watches a run's ``human_review/`` directory for a ``pending_review.json``,
prints the candidate skill + score delta + ranked edits, and prompts the
reviewer for a decision. Writes ``pending_review_response.json`` next to it
so the trainer can unblock.

Usage
-----
    python scripts/review.py --run-dir outputs/<run_name>

Optionally point at a specific step file instead of the latest:

    python scripts/review.py --request outputs/<run>/human_review/step_0003/pending_review.json

The script loops by default so you can review every step in a single session.
Pass ``--once`` to handle one pending review and exit.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import Optional


VALID_ACTIONS = {
    "accept": "Accept the candidate as the new current skill",
    "accept_new_best": "Accept and mark as new best",
    "reject": "Reject — keep current skill unchanged",
    "retry": "Re-run the selection rollout (non-deterministic target model may differ)",
    "apply_selected_edits": "Cherry-pick edits by index (patch mode only)",
}


def _find_latest_request(run_dir: str) -> Optional[str]:
    pattern = os.path.join(run_dir, "human_review", "step_*", "pending_review.json")
    matches = sorted(glob.glob(pattern), key=os.path.getmtime)
    return matches[-1] if matches else None


def _print_request(req: dict) -> None:
    print("\n" + "=" * 70)
    print(f"  Pending review for step {req.get('step', '?')}")
    print("=" * 70)

    cs = req.get("candidate_score", 0.0)
    curs = req.get("current_score", 0.0)
    bs = req.get("best_score", 0.0)
    bstep = req.get("best_step", 0)
    arrow = "↑" if cs > curs else ("=" if cs == curs else "↓")
    print(f"  Score: current={curs:.4f}  candidate={cs:.4f} {arrow}"
          f"  best={bs:.4f} (step {bstep})")

    retry = req.get("retry_attempt", 0)
    max_retries = req.get("max_retries", 0)
    if retry > 0 or max_retries > 0:
        print(f"  Retry attempt: {retry}/{max_retries}")

    edits = req.get("ranked_edits") or []
    if edits:
        print(f"\n  Ranked edits ({len(edits)}):")
        for i, e in enumerate(edits):
            op = e.get("op") or e.get("type") or "?"
            target = (e.get("target") or "")[:60]
            content = (
                e.get("content") or e.get("instruction") or e.get("title") or ""
            )[:80]
            tgt_part = f' target="{target}"' if target else ""
            print(f"    [{i}] {op}{tgt_part}  →  {content!r}")

    update_mode = req.get("update_mode", "patch")
    print(f"\n  Update mode: {update_mode}")
    print("=" * 70)


def _show_skill(label: str, text: str, max_lines: int = 80) -> None:
    print(f"\n--- {label} ({len(text)} chars) ---")
    lines = text.splitlines()
    if len(lines) <= max_lines:
        for line in lines:
            print(line)
    else:
        for line in lines[: max_lines - 5]:
            print(line)
        print(f"... ({len(lines) - max_lines + 5} more lines omitted) ...")
        for line in lines[-5:]:
            print(line)


def _prompt_action() -> str:
    print("\nActions:")
    for k, v in VALID_ACTIONS.items():
        print(f"  {k:25s} {v}")
    while True:
        raw = input("\nAction: ").strip()
        if raw in VALID_ACTIONS:
            return raw
        print(f"  invalid; choose one of: {', '.join(VALID_ACTIONS)}")


def _prompt_selected_indices(n_edits: int) -> list[int]:
    while True:
        raw = input(
            f"  Edit indices to apply (comma-separated, 0..{n_edits - 1}): "
        ).strip()
        try:
            indices = [int(x.strip()) for x in raw.split(",") if x.strip()]
        except ValueError:
            print("  invalid; enter integers separated by commas")
            continue
        if all(0 <= i < n_edits for i in indices):
            return indices
        print(f"  out of range; values must be in 0..{n_edits - 1}")


def _prompt_critique() -> str:
    print(
        "\nCritique (free-form; flows into the next optimizer prompt). "
        "End with a blank line:"
    )
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "":
            break
        lines.append(line)
    return "\n".join(lines)


def _handle_request(req_path: str) -> None:
    with open(req_path, encoding="utf-8") as f:
        req = json.load(f)
    _print_request(req)

    show_skill = input(
        "\nShow candidate skill? [y/N/d=diff-style side-by-side preview] "
    ).strip().lower()
    if show_skill == "y":
        _show_skill("CANDIDATE SKILL", req.get("candidate_skill", ""))
    elif show_skill == "d":
        _show_skill("CURRENT SKILL", req.get("current_skill", ""), max_lines=40)
        _show_skill("CANDIDATE SKILL", req.get("candidate_skill", ""), max_lines=40)

    action = _prompt_action()

    response: dict = {"action": action}
    if action == "apply_selected_edits":
        edits = req.get("ranked_edits") or []
        if not edits:
            print("  no edits available — switching to reject")
            response["action"] = "reject"
        else:
            response["selected_edit_indices"] = _prompt_selected_indices(len(edits))

    response["critique"] = _prompt_critique()

    resp_path = req_path.replace("pending_review.json", "pending_review_response.json")
    with open(resp_path, "w", encoding="utf-8") as f:
        json.dump(response, f, ensure_ascii=False, indent=2)
    print(f"\n  → wrote {resp_path}")
    print("    Trainer will pick this up within ~1 second.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-dir", help="Path to a run's out_root directory")
    g.add_argument("--request", help="Path to a specific pending_review.json")
    p.add_argument("--once", action="store_true",
                   help="Handle one pending review and exit (default: loop)")
    p.add_argument("--poll", type=float, default=2.0,
                   help="Poll interval in seconds when waiting for a request")
    args = p.parse_args()

    if args.request:
        if not os.path.exists(args.request):
            print(f"  no request at {args.request}", file=sys.stderr)
            sys.exit(1)
        _handle_request(args.request)
        return

    print(f"  watching {args.run_dir}/human_review/ for pending reviews "
          f"(Ctrl+C to exit)")
    handled: set[str] = set()
    while True:
        req_path = _find_latest_request(args.run_dir)
        if req_path and req_path not in handled:
            try:
                _handle_request(req_path)
                handled.add(req_path)
                if args.once:
                    return
            except KeyboardInterrupt:
                print("\n  aborted")
                return
            except Exception as exc:  # noqa: BLE001
                print(f"  error handling {req_path}: {exc}", file=sys.stderr)
                handled.add(req_path)
        else:
            try:
                time.sleep(args.poll)
            except KeyboardInterrupt:
                print("\n  exiting")
                return


if __name__ == "__main__":
    main()
