"""End-to-end regression for the optimizer-feedback trust boundary."""
from __future__ import annotations

import json
import os
import tempfile
from unittest import mock

from skillopt_sleep.backend import CliBackend
from skillopt_sleep.config import load_config
from skillopt_sleep.cycle import run_sleep_cycle
from skillopt_sleep.evidence import read_events
from skillopt_sleep.types import SessionDigest

PRIVATE_PATTERN = r"(?im)^\s*ROUTE:\s*consultation\s*$"
SAFE_DESCRIPTION = "Route recurring consultation requests through the consultation utility."
SAFE_RULE = "Route recurring consultation requests through the consultation utility."


class FeedbackFlowBackend(CliBackend):
    """Script every model response while exercising the real orchestration."""

    name = "feedback-flow"

    def __init__(self) -> None:
        super().__init__(model="scripted")
        self.reflect_prompts: list[str] = []

    def _call(self, prompt: str, *, max_tokens: int = 1024) -> str:
        if prompt.startswith("You are mining a user's past AI-assistant sessions"):
            return json.dumps(
                [
                    {
                        "intent": "Route a recurring consultation request",
                        "checks": [
                            {
                                "op": "regex",
                                "arg": PRIVATE_PATTERN,
                                "description": SAFE_DESCRIPTION,
                            }
                        ],
                        "rubric": "",
                        "satisfied": False,
                    },
                    {
                        "intent": "Handle a repeated consultation follow-up",
                        "checks": [
                            {
                                "op": "regex",
                                "arg": PRIVATE_PATTERN,
                                "description": SAFE_DESCRIPTION,
                            }
                        ],
                        "rubric": "",
                        "satisfied": False,
                    },
                ]
            )
        if prompt.startswith("Complete the following task for the user"):
            if SAFE_RULE in prompt:
                return "ROUTE: consultation"
            return "No route declaration here."
        if prompt.startswith("You are SkillOpt's optimizer"):
            self.reflect_prompts.append(prompt)
            return json.dumps(
                [
                    {
                        "op": "add",
                        "content": SAFE_RULE,
                        "anchor": "",
                        "rationale": "Preserve the mined user-visible routing behavior.",
                    }
                ]
            )
        raise AssertionError(f"unexpected scripted prompt: {prompt[:120]!r}")


def test_mined_description_reaches_proposal_without_verifier_syntax() -> None:
    with tempfile.TemporaryDirectory() as project, tempfile.TemporaryDirectory() as home:
        backend = FeedbackFlowBackend()
        digest = SessionDigest(
            session_id="synthetic-feedback-flow",
            project=project,
            user_prompts=["Please route this recurring consultation request correctly."],
            assistant_finals=["I handled it without the required route."],
            feedback_signals=["neg:wrong route"],
        )
        cfg = load_config(
            invoked_project=project,
            projects="invoked",
            backend="codex",
            claude_home=os.path.join(home, ".claude"),
            state_dir=os.path.join(home, "state"),
            max_tasks_per_night=2,
            val_fraction=0.5,
            test_fraction=0.0,
            target_task_filter=False,
            evolve_memory=False,
            auto_adopt=False,
        )

        with mock.patch(
            "skillopt_sleep.cycle.harvest_for_config", return_value=[digest]
        ):
            outcome = run_sleep_cycle(cfg, backend=backend)

        assert outcome.report.accepted is True
        assert len(backend.reflect_prompts) == 1
        optimizer_prompt = backend.reflect_prompts[0]
        assert SAFE_DESCRIPTION in optimizer_prompt
        assert PRIVATE_PATTERN not in optimizer_prompt
        assert "regex=" not in optimizer_prompt

        proposal_path = os.path.join(outcome.staging_dir, "proposed_SKILL.md")
        with open(proposal_path, encoding="utf-8") as handle:
            proposal = handle.read()
        assert SAFE_RULE in proposal
        assert PRIVATE_PATTERN not in proposal

        with open(
            os.path.join(outcome.staging_dir, "report.json"), encoding="utf-8"
        ) as handle:
            report = json.load(handle)
        assert report["edits"][0]["content"] == SAFE_RULE
        assert PRIVATE_PATTERN not in json.dumps(report)

        with open(
            os.path.join(outcome.staging_dir, "diagnostics.json"), encoding="utf-8"
        ) as handle:
            diagnostics = json.load(handle)
        assert any(
            PRIVATE_PATTERN in row["why"] for row in diagnostics["holdout_detail"]
        )
        assert SAFE_RULE in diagnostics["reflect_raw_head"]

        events = read_events(os.path.join(outcome.staging_dir, "evidence.jsonl"))
        miner = next(e for e in events if e["event"] == "miner_exchange")
        miner_reply = json.loads(miner["raw_reply"])
        assert miner_reply[0]["checks"][0]["arg"] == PRIVATE_PATTERN
        mined = [e for e in events if e["event"] == "task_mined"]
        assert len(mined) == 2
        assert all(e["checks"][0]["description"] == SAFE_DESCRIPTION for e in mined)
        assert all(e["checks"][0]["arg"] == PRIVATE_PATTERN for e in mined)

        replay = [e for e in events if e["stage"] == "replay" and e["event"] == "result"]
        assert any(PRIVATE_PATTERN in e["why"] for e in replay)
        reflection = next(e for e in events if e["event"] == "exchange")
        assert SAFE_DESCRIPTION in reflection["prompt"]
        assert PRIVATE_PATTERN not in reflection["prompt"]
        edits = next(e for e in events if e["event"] == "edits_returned")
        assert edits["edits"][0]["content"] == SAFE_RULE
        assert PRIVATE_PATTERN not in json.dumps(edits)
        assert any(e["stage"] == "stage" and e["event"] == "staged" for e in events)
