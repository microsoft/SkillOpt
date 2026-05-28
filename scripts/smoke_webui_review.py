#!/usr/bin/env python3
"""Smoke test for the WebUI human-review helpers — no Gradio launch needed.

Verifies the polling + parsing + response-writing functions work against
synthetic pending review files. Useful to catch regressions in the
review-panel logic without spinning up the full Gradio app.

Run::

    python scripts/smoke_webui_review.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skillopt_webui.app import (
    _edit_label,
    _find_pending_review,
    _render_review_status,
    _submit_review,
)


def _write_pending(run_dir: str, step: int, payload: dict) -> str:
    step_dir = os.path.join(run_dir, "human_review", f"step_{step:04d}")
    os.makedirs(step_dir, exist_ok=True)
    path = os.path.join(step_dir, "pending_review.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    return path


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        # Empty run dir -> no pending
        path, req = _find_pending_review(tmp)
        assert path is None and req is None, "expected no pending review"
        status = _render_review_status(None, None)
        assert "No pending review" in status, status
        print("  [PASS] empty run dir returns no pending review")

        # Write a pending request and confirm it's discovered
        payload = {
            "step": 5,
            "current_skill": "old",
            "candidate_skill": "new and improved",
            "current_score": 0.4,
            "candidate_score": 0.6,
            "best_score": 0.55,
            "best_step": 3,
            "ranked_edits": [
                {"op": "append", "content": "be careful", "target": ""},
                {"op": "replace", "content": "X", "target": "Y"},
            ],
            "update_mode": "patch",
            "retry_attempt": 0,
            "max_retries": 3,
        }
        req_path = _write_pending(tmp, 5, payload)
        found_path, found_req = _find_pending_review(tmp)
        assert found_path == req_path, f"path mismatch: {found_path} vs {req_path}"
        assert found_req is not None and found_req["step"] == 5
        print("  [PASS] pending review discovered and parsed")

        # Edit labels render
        labels = [_edit_label(i, e) for i, e in enumerate(payload["ranked_edits"])]
        assert labels[0].startswith("[0] append"), labels
        assert "target=" in labels[1] and "replace" in labels[1]
        print(f"  [PASS] edit labels: {labels!r}")

        # Status HTML contains the score arrow
        status = _render_review_status(req_path, payload)
        assert "Awaiting review" in status and "step 5" in status
        assert "0.6000" in status or "0.6" in status
        print("  [PASS] status HTML rendered with score delta")

        # Submit an accept with critique and direct skill edit
        msg = _submit_review(
            action="accept_new_best",
            run_dir=tmp,
            req_path=req_path,
            edited_skill="hand-edited skill",
            selected_labels=[],
            critique="looks great, tightened wording",
        )
        assert msg.startswith("[OK]") or "Sent" in msg or "OK" in msg or msg.startswith("✅"), msg
        resp_path = req_path.replace("pending_review.json", "pending_review_response.json")
        assert os.path.exists(resp_path), "response file not written"
        with open(resp_path, encoding="utf-8") as f:
            resp = json.load(f)
        assert resp["action"] == "accept_new_best"
        assert resp["edited_skill"] == "hand-edited skill"
        assert resp["critique"] == "looks great, tightened wording"
        print("  [PASS] accept_new_best response written with edited_skill + critique")

        # Submit apply_selected_edits — only the first label
        msg = _submit_review(
            action="apply_selected_edits",
            run_dir=tmp,
            req_path=req_path,
            edited_skill="",
            selected_labels=[labels[0]],
            critique="drop the replace",
        )
        with open(resp_path, encoding="utf-8") as f:
            resp = json.load(f)
        assert resp["selected_edit_indices"] == [0], resp
        print("  [PASS] apply_selected_edits maps labels back to indices")

        # Edge case: no req_path
        msg = _submit_review(
            action="accept", run_dir=tmp, req_path="",
            edited_skill="", selected_labels=[], critique="",
        )
        assert "No pending review" in msg
        print("  [PASS] empty req_path is rejected cleanly")

    print("\nAll WebUI review smoke checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
