"""Synthetic evidence contracts for #154; no real preference inference is tested."""
import json
import pytest

from skillopt_sleep.types import SessionDigest
from skillopt_sleep.mine import heuristic_mine, mine
from skillopt_sleep.llm_miner import _digest_to_prompt

class ScopeEvidenceLoss(AssertionError):
    """Only this known retention gap is allowed to xfail."""


CASES = [
    ("durable", "For all future status reports, use bullet points.",
     "Keep that preference for future reports.", "eligible_for_review"),
    ("one_off", "For this report only, use bullet points.",
     "Do not apply that formatting to other reports.", "withhold_generalization"),
    ("reversal", "For all future status reports, use bullet points.",
     "I withdraw that preference; use prose in future reports.", "withhold_stale_rule"),
    ("ambiguous", "Maybe bullet points would be better.",
     "I am not sure whether I want that in future reports.", "withhold_pending_clarification"),
]


def sessions(case):
    name, first, later, _ = case
    return [SessionDigest(
        session_id=f"{name}-{i}", project="synthetic-project",
        started_at=f"2026-10-0{i+1}T00:00:00Z",
        user_prompts=["Prepare a status report for this project.", text],
        assistant_finals=["Synthetic status report."],
        feedback_signals=["Thanks"], n_user_turns=2,
    ) for i, text in enumerate((first, later))]


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_short_scope_evidence_reaches_real_miner_prompt(case):
    for digest in sessions(case):
        assert digest.user_prompts[-1] in _digest_to_prompt(digest)


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_scope_evidence_survives_individual_heuristic_mining(case):
    for digest in sessions(case):
        record = heuristic_mine([digest])[0]
        assert digest.user_prompts[-1] in record.context_excerpt
        assert record.source_sessions == [digest.session_id]


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
@pytest.mark.xfail(strict=True, raises=ScopeEvidenceLoss, reason="Proposed #154 contract: dedup currently retains first context only")
def test_paired_scope_evidence_survives_public_mine_and_dedup(case):
    digests = sessions(case)
    records = mine(digests, val_fraction=0, test_fraction=0)
    assert len(records) == 1
    assert records[0].source_sessions == [d.session_id for d in digests]
    payload = json.dumps(records[0].to_dict())
    missing = [d.session_id for d in digests if d.user_prompts[-1] not in payload]
    if missing:
        raise ScopeEvidenceLoss(f"Missing scope evidence from sessions: {missing}")
