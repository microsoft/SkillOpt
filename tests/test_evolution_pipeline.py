"""Tests for Hermes skill evolution governance prototype."""
from __future__ import annotations

import json

import pytest

from skillopt.evolution.pipeline import (
    evaluate_candidate,
    gate_candidate,
    promote_candidate,
    stage_candidate,
    write_review_request,
)

GOOD_SKILL = """---
name: demo-skill
description: Demo skill for safe candidate evaluation.
---

# Demo Skill

Use this skill to test staging.

```bash
qmd status
```
用途：確認 qmd index 狀態。

## Verification

Check cost and latency before promoting.
"""


def test_stage_eval_gate_review_promote_flow(tmp_path) -> None:
    registry = tmp_path / "registry"
    live = tmp_path / "live" / "SKILL.md"
    candidate = tmp_path / "candidate.md"
    live.parent.mkdir(parents=True)
    live.write_text(GOOD_SKILL.replace("Demo Skill", "Old Demo Skill"))
    candidate.write_text(GOOD_SKILL)

    manifest = stage_candidate(
        registry=registry,
        skill_name="demo-skill",
        candidate_path=candidate,
        base_skill_path=live,
    )
    candidate_id = manifest["candidate_id"]

    report = evaluate_candidate(registry=registry, skill_name="demo-skill", candidate_id=candidate_id)
    assert report["scores"]["safety"] == 1.0

    decision = gate_candidate(registry=registry, skill_name="demo-skill", candidate_id=candidate_id)
    assert decision.passed

    review_path = write_review_request(registry=registry, skill_name="demo-skill", candidate_id=candidate_id)
    assert review_path.exists()

    promotion = promote_candidate(
        registry=registry,
        skill_name="demo-skill",
        candidate_id=candidate_id,
        live_skill_path=live,
        approved_by="unit-test",
    )
    assert promotion["approved_by"] == "unit-test"
    assert live.read_text() == GOOD_SKILL


def test_gate_rejects_secret_candidate(tmp_path) -> None:
    registry = tmp_path / "registry"
    candidate = tmp_path / "candidate.md"
    candidate.write_text(GOOD_SKILL + "\napi_key=abcdefghijklmno\n")
    manifest = stage_candidate(registry=registry, skill_name="demo", candidate_path=candidate)
    candidate_id = manifest["candidate_id"]

    staged = registry / "staging" / "demo" / candidate_id / "candidate.md"
    assert "abcdefghijklmno" not in staged.read_text()
    assert manifest["candidate_redacted"] is True
    report = evaluate_candidate(registry=registry, skill_name="demo", candidate_id=candidate_id)
    decision = gate_candidate(registry=registry, skill_name="demo", candidate_id=candidate_id)

    assert report["scores"]["safety"] == 0.0
    assert not decision.passed
    gate = json.loads((registry / "staging" / "demo" / candidate_id / "gate.json").read_text())
    assert gate["passed"] is False


def test_eval_scores_regression_fixture_and_qmd_grounding(tmp_path) -> None:
    registry = tmp_path / "registry"
    candidate = tmp_path / "candidate.md"
    fixture = tmp_path / "fixture.json"
    candidate.write_text(
        GOOD_SKILL
        + "\n## Grounding\n"
        + "- qmd://hermes-agent-architecture/index.md records Hermes Agent runtime context.\n"
        + "- Use qmd search before editing skills.\n"
        + "\n## Regression coverage\n"
        + "This skill covers safety gate, human review, and promotion dry-run workflows.\n"
    )
    fixture.write_text(
        json.dumps(
            {
                "tasks": [
                    {"name": "safety", "required_terms": ["safety gate", "human review"]},
                    {"name": "dry-run", "required_terms": ["promotion dry-run"]},
                ]
            }
        )
    )
    manifest = stage_candidate(registry=registry, skill_name="demo", candidate_path=candidate)

    report = evaluate_candidate(
        registry=registry,
        skill_name="demo",
        candidate_id=manifest["candidate_id"],
        regression_fixture=fixture,
    )

    assert report["scores"]["qmd_grounding"] == 1.0
    assert report["scores"]["regression"] == 1.0
    assert report["findings"]["qmd_references"] == ["qmd://hermes-agent-architecture/index.md"]
    assert report["findings"]["regression"]["passed"] == 2


def test_review_request_contains_checklist_and_promote_dry_run_does_not_write_live(tmp_path) -> None:
    registry = tmp_path / "registry"
    live = tmp_path / "live" / "SKILL.md"
    candidate = tmp_path / "candidate.md"
    live.parent.mkdir(parents=True)
    live.write_text("old-live")
    candidate.write_text(GOOD_SKILL)
    manifest = stage_candidate(registry=registry, skill_name="demo", candidate_path=candidate)
    candidate_id = manifest["candidate_id"]
    evaluate_candidate(registry=registry, skill_name="demo", candidate_id=candidate_id)
    gate_candidate(registry=registry, skill_name="demo", candidate_id=candidate_id, min_score=0.6)

    with pytest.raises(ValueError, match="human_review.md is required"):
        promote_candidate(
            registry=registry,
            skill_name="demo",
            candidate_id=candidate_id,
            live_skill_path=live,
            approved_by="unit-test",
        )

    review_path = write_review_request(registry=registry, skill_name="demo", candidate_id=candidate_id)
    review_text = review_path.read_text()
    assert "## Human review checklist" in review_text
    assert "[ ] Read candidate.md" in review_text
    assert "[ ] Confirm promotion dry-run output" in review_text

    promotion = promote_candidate(
        registry=registry,
        skill_name="demo",
        candidate_id=candidate_id,
        live_skill_path=live,
        approved_by="unit-test",
        dry_run=True,
    )
    assert promotion["dry_run"] is True
    assert promotion["would_write"] == str(live)
    assert live.read_text() == "old-live"
