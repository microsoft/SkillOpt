#!/usr/bin/env python3
"""Smoke test for the human-review file handoff — no LLM needed.

Exercises HumanReviewProvider in one thread while a fake reviewer writes a
response in another thread. Verifies that the request file is written, the
response is read, files are cleaned up, and each action shape round-trips.

Run::

    python scripts/smoke_human_gate.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skillopt.evaluation.human_gate import (
    HumanReviewProvider,
    HumanReviewRequest,
    HumanReviewResponse,
)


def _fake_reviewer(run_dir: str, step: int, response: dict, delay: float = 0.3) -> None:
    """After *delay* seconds, write a response file for the given step."""
    time.sleep(delay)
    resp_path = os.path.join(
        run_dir, "human_review", f"step_{step:04d}", "pending_review_response.json"
    )
    with open(resp_path, "w", encoding="utf-8") as f:
        json.dump(response, f)


def _run_case(label: str, response_payload: dict) -> HumanReviewResponse:
    with tempfile.TemporaryDirectory() as tmp:
        provider = HumanReviewProvider(run_dir=tmp, timeout_seconds=5)
        req = HumanReviewRequest(
            step=1,
            current_skill="# old skill\nbe helpful",
            candidate_skill="# new skill\nbe helpful and concise",
            current_score=0.42,
            candidate_score=0.55,
            best_score=0.50,
            best_step=0,
            ranked_edits=[
                {"op": "append", "content": "be concise", "target": ""},
                {"op": "replace", "content": "be helpful and concise", "target": "be helpful"},
            ],
            update_mode="patch",
        )

        t = threading.Thread(
            target=_fake_reviewer, args=(tmp, 1, response_payload), daemon=True
        )
        t.start()

        response = provider.request_review(req)
        t.join(timeout=2)

        # Files must have been cleaned up
        leftover = os.listdir(os.path.join(tmp, "human_review", "step_0001"))
        assert leftover == [], f"leftover files after handoff: {leftover}"
        assert response is not None, "provider returned None for a valid response"
        print(f"  [PASS] {label} -> action={response.action!r}"
              f" critique={response.critique[:40]!r}"
              f" edited_skill={bool(response.edited_skill)}"
              f" selected={response.selected_edit_indices}")
        return response


def _run_timeout_case() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        provider = HumanReviewProvider(run_dir=tmp, timeout_seconds=1)
        req = HumanReviewRequest(
            step=2, current_skill="x", candidate_skill="y",
            current_score=0, candidate_score=0,
            best_score=0, best_step=0,
        )
        t0 = time.time()
        response = provider.request_review(req)
        elapsed = time.time() - t0
        assert response is None, "expected None on timeout"
        assert 0.9 <= elapsed <= 2.0, f"timeout off: {elapsed}s"
        # Request file should still exist (caller decides what to do on timeout)
        req_path = os.path.join(
            tmp, "human_review", "step_0002", "pending_review.json"
        )
        assert os.path.exists(req_path), "request file removed after timeout"
        print(f"  [PASS] timeout returns None after {elapsed:.2f}s")


def main() -> int:
    print("Smoke-testing HumanReviewProvider file handoff...\n")

    _run_case("accept (no edit)", {
        "action": "accept",
        "critique": "score looks good; ship it",
    })

    r = _run_case("accept_new_best with direct skill edit", {
        "action": "accept_new_best",
        "edited_skill": "# hand-edited skill\nbe excellent",
        "critique": "tightened wording",
    })
    assert r.edited_skill == "# hand-edited skill\nbe excellent"

    _run_case("reject", {
        "action": "reject",
        "critique": "selection score went up but the change harms reasoning depth",
    })

    r = _run_case("apply_selected_edits subset", {
        "action": "apply_selected_edits",
        "selected_edit_indices": [0],
        "critique": "drop the replace; just keep the append",
    })
    assert r.selected_edit_indices == [0]

    _run_case("retry", {
        "action": "retry",
        "critique": "rollout looked flaky; re-run",
    })

    _run_timeout_case()

    print("\nAll smoke checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
