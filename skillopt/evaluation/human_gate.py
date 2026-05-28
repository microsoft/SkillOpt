"""Human-in-the-loop review gate.

File-based handoff between the trainer and a reviewer. The trainer writes
``pending_review.json`` into the run's ``human_review/step_NNNN/`` directory
and polls for ``pending_review_response.json``. A reviewer (typically the
WebUI panel, but also ``scripts/review.py``) writes the response file.

Decoupling the trainer from any specific UI is intentional: the WebUI runs
as a separate subprocess (see ``skillopt_webui/app.py``), so the two cannot
share Python objects. Files in ``out_root`` are inspectable, restart-safe,
and trivially diagnosable when something goes wrong.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Literal, Optional


HumanAction = Literal[
    "accept_new_best",
    "accept",
    "reject",
    "retry",
    "apply_selected_edits",
]

TimeoutPolicy = Literal["fallback_to_gate", "accept", "reject"]


@dataclass(frozen=True)
class HumanReviewRequest:
    """What the trainer hands to the reviewer."""

    step: int
    current_skill: str
    candidate_skill: str
    current_score: float
    candidate_score: float
    best_score: float
    best_step: int
    ranked_edits: list[dict] = field(default_factory=list)
    update_mode: str = "patch"
    retry_attempt: int = 0
    max_retries: int = 0


@dataclass(frozen=True)
class HumanReviewResponse:
    """What the reviewer hands back."""

    action: HumanAction
    edited_skill: Optional[str] = None
    selected_edit_indices: Optional[list[int]] = None
    critique: str = ""


def _request_path(run_dir: str, step: int) -> str:
    return os.path.join(
        run_dir, "human_review", f"step_{step:04d}", "pending_review.json"
    )


def _response_path(run_dir: str, step: int) -> str:
    return os.path.join(
        run_dir, "human_review", f"step_{step:04d}", "pending_review_response.json"
    )


class HumanReviewProvider:
    """File-based pending-review queue scoped to a single training run.

    The trainer constructs one provider per run and calls
    :meth:`request_review` from the step loop. Each call blocks (polling)
    until a response file appears or until ``timeout_seconds`` elapses.
    """

    def __init__(
        self,
        run_dir: str,
        timeout_seconds: int = 0,
        on_timeout: TimeoutPolicy = "fallback_to_gate",
        poll_interval: float = 1.0,
    ) -> None:
        self.run_dir = run_dir
        self.timeout_seconds = max(0, int(timeout_seconds))
        self.on_timeout: TimeoutPolicy = on_timeout
        self.poll_interval = max(0.1, float(poll_interval))

    def request_review(
        self, req: HumanReviewRequest
    ) -> Optional[HumanReviewResponse]:
        """Write a pending-review file and block until response or timeout.

        Returns ``None`` if the timeout elapses with no response, in which
        case the caller applies its own :attr:`on_timeout` policy. Returns a
        :class:`HumanReviewResponse` otherwise.
        """
        req_path = _request_path(self.run_dir, req.step)
        resp_path = _response_path(self.run_dir, req.step)
        os.makedirs(os.path.dirname(req_path), exist_ok=True)

        # Stale-response cleanup so we don't pick up a previous attempt.
        if os.path.exists(resp_path):
            try:
                os.remove(resp_path)
            except OSError:
                pass

        payload = asdict(req)
        with open(req_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        t0 = time.time()
        while True:
            if os.path.exists(resp_path):
                # Tiny grace pause so the writer can finish flushing before
                # we read; cheaper than a lock file for a single-reviewer flow.
                time.sleep(0.05)
                with open(resp_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                response = self._parse_response(data)
                try:
                    os.remove(req_path)
                    os.remove(resp_path)
                except OSError:
                    pass
                return response

            if self.timeout_seconds > 0 and (time.time() - t0) > self.timeout_seconds:
                return None

            time.sleep(self.poll_interval)

    @staticmethod
    def _parse_response(data: dict) -> HumanReviewResponse:
        action = data.get("action")
        if action not in {
            "accept_new_best", "accept", "reject", "retry", "apply_selected_edits"
        }:
            raise ValueError(f"Invalid human review action: {action!r}")
        return HumanReviewResponse(
            action=action,
            edited_skill=data.get("edited_skill"),
            selected_edit_indices=data.get("selected_edit_indices"),
            critique=str(data.get("critique") or ""),
        )
