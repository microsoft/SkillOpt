from __future__ import annotations

import pytest

from gitmoot_skillopt.artifacts import content_hash
from gitmoot_skillopt.contracts import (
    CANDIDATE_PACKAGE_KIND,
    CONTRACT_VERSION,
    TRAINING_PACKAGE_KIND,
    CandidatePackage,
    ContractError,
    TrainingPackage,
)


def template_content(template_id: str = "planner", name: str = "Planner") -> str:
    return f"""---
id: {template_id}
name: {name}
description: Plans work.
kind: agent-template
version: 1
capabilities:
  - ask
runtime_compatibility:
  - codex
tags:
  - planning
inputs:
  - request
outputs:
  - plan
---
# {name}

Plan carefully.
"""


def metadata(template_id: str = "planner", name: str = "Planner") -> dict[str, object]:
    return {
        "id": template_id,
        "name": name,
        "description": "Plans work.",
        "kind": "agent-template",
        "version": 1,
        "capabilities": ["ask"],
        "runtime_compatibility": ["codex"],
        "tags": ["planning"],
        "inputs": ["request"],
        "outputs": ["plan"],
    }


def training_package_dict() -> dict[str, object]:
    content = template_content()
    return {
        "kind": TRAINING_PACKAGE_KIND,
        "contract_version": CONTRACT_VERSION,
        "template": {
            "id": "planner",
            "version_id": "planner@v1",
            "version_number": 1,
            "version_state": "current",
            "content_hash": content_hash(content.encode()),
            "source_repo": "jerryfane/gitmoot",
            "source_ref": "main",
            "source_path": "skills/gitmoot/agent-templates/planner.md",
            "resolved_commit": "abc123",
            "metadata": metadata(),
            "content": content,
        },
        "eval_run": {
            "id": "run-1",
            "template_id": "planner",
            "template_version_id": "planner@v1",
            "target_repo": "owner/repo",
            "state": "completed",
            "metadata": {"metric": "preference"},
        },
        "items": [
            {
                "id": "item-1",
                "title": "Item 1",
                "baseline_artifact_id": "baseline",
                "candidate_artifact_id": "candidate",
                "metadata": {"difficulty": "easy"},
            }
        ],
        "artifacts": [
            {
                "id": "baseline",
                "hash": "sha256:" + "2" * 64,
                "media_type": "text/markdown",
                "size_bytes": 8,
                "driver": "text",
            },
            {
                "id": "candidate",
                "hash": "3" * 64,
                "media_type": "text/markdown",
                "size_bytes": 9,
                "driver": "text",
            },
        ],
        "feedback_events": [
            {
                "run_id": "run-1",
                "item_id": "item-1",
                "choice": "b",
                "reasoning": "More concrete.",
                "reviewer": "alice",
                "source": "markdown",
                "source_url": "",
                "created_at": "2026-05-31T00:00:00Z",
            }
        ],
        "evaluator_config": {"mode": "pairwise"},
    }


def candidate_package_dict() -> dict[str, object]:
    return {
        "kind": CANDIDATE_PACKAGE_KIND,
        "contract_version": CONTRACT_VERSION,
        "template_id": "planner",
        "base_version_id": "planner@v1",
        "candidate": {
            "content": template_content(),
            "metadata": metadata(),
        },
        "eval_report": {"score": 0.8},
        "summary": {
            "diff_artifact_id": "diff",
            "score": 0.8,
            "preference_summary": "More actionable.",
            "metadata": {"items": 3},
        },
    }


def test_training_package_loads_and_round_trips(tmp_path):
    package_path = tmp_path / "training.json"
    package = TrainingPackage.from_dict(training_package_dict())
    package.dump(package_path)

    loaded = TrainingPackage.load(package_path)

    assert loaded.to_dict() == package.to_dict()
    assert loaded.artifacts[1].hash == "sha256:" + "3" * 64


def test_training_package_rejects_wrong_kind():
    data = training_package_dict()
    data["kind"] = "wrong"

    with pytest.raises(ContractError, match="kind must be"):
        TrainingPackage.from_dict(data)


def test_training_package_rejects_boolean_contract_version():
    data = training_package_dict()
    data["contract_version"] = True

    with pytest.raises(ContractError, match="contract_version must be an integer"):
        TrainingPackage.from_dict(data)


def test_training_package_rejects_boolean_size():
    data = training_package_dict()
    artifacts = list(data["artifacts"])  # type: ignore[arg-type]
    artifacts[0] = {**artifacts[0], "size_bytes": True}  # type: ignore[index]
    data["artifacts"] = artifacts

    with pytest.raises(ContractError, match="artifact.size_bytes must be an integer"):
        TrainingPackage.from_dict(data)


def test_training_package_rejects_missing_artifact_reference():
    data = training_package_dict()
    data["artifacts"] = []

    with pytest.raises(ContractError, match="references missing artifact"):
        TrainingPackage.from_dict(data)


def test_training_package_rejects_bad_hash():
    data = training_package_dict()
    artifacts = list(data["artifacts"])  # type: ignore[arg-type]
    artifacts[0] = {**artifacts[0], "hash": "sha256:not-hex"}  # type: ignore[index]
    data["artifacts"] = artifacts

    with pytest.raises(ValueError, match="artifact hash"):
        TrainingPackage.from_dict(data)


def test_training_package_rejects_template_content_hash_mismatch():
    data = training_package_dict()
    data["template"] = {**data["template"], "content_hash": "sha256:" + "1" * 64}  # type: ignore[arg-type]

    with pytest.raises(ContractError, match="content_hash mismatch"):
        TrainingPackage.from_dict(data)


def test_candidate_package_validates_content_metadata_consistency():
    package = CandidatePackage.from_dict(candidate_package_dict())

    assert package.template_id == "planner"
    assert package.to_dict()["candidate"]["metadata"] == metadata()


def test_candidate_package_rejects_mismatched_template_id():
    data = candidate_package_dict()
    data["template_id"] = "reviewer"

    with pytest.raises(ContractError, match="does not match package template_id"):
        CandidatePackage.from_dict(data)


def test_candidate_package_rejects_mismatched_metadata():
    data = candidate_package_dict()
    candidate = dict(data["candidate"])  # type: ignore[arg-type]
    candidate["metadata"] = {**metadata(), "name": "Different"}
    data["candidate"] = candidate

    with pytest.raises(ContractError, match="metadata does not match"):
        CandidatePackage.from_dict(data)


def test_candidate_package_rejects_incomplete_frontmatter():
    data = candidate_package_dict()
    data["candidate"] = {
        "content": "---\nid: planner\n---\n# Planner\n",
        "metadata": {"id": "planner"},
    }

    with pytest.raises(ContractError, match="template frontmatter missing name"):
        CandidatePackage.from_dict(data)


@pytest.mark.parametrize(
    ("content", "metadata_value"),
    [
        (template_content().replace("  - ask\n", "  - ask\n  - 123\n"), ["ask", 123]),
        (template_content().replace("  - ask\n", "  - ask\n  - ''\n"), ["ask", ""]),
    ],
)
def test_candidate_package_rejects_invalid_metadata_list_items(content, metadata_value):
    data = candidate_package_dict()
    candidate = dict(data["candidate"])  # type: ignore[arg-type]
    candidate["content"] = content
    candidate["metadata"] = {**metadata(), "capabilities": metadata_value}
    data["candidate"] = candidate

    with pytest.raises(ContractError, match="capabilities must contain strings"):
        CandidatePackage.from_dict(data)


def test_candidate_package_rejects_invalid_evaluation_metadata():
    data = candidate_package_dict()
    content = template_content().replace("outputs:\n  - plan\n", "outputs:\n  - plan\nevaluation:\n  metric: 123\n")
    candidate = dict(data["candidate"])  # type: ignore[arg-type]
    candidate["content"] = content
    candidate["metadata"] = {**metadata(), "evaluation": {"metric": 123}}
    data["candidate"] = candidate

    with pytest.raises(ContractError, match="metadata does not match|evaluation"):
        CandidatePackage.from_dict(data)


def test_candidate_package_rejects_boolean_summary_score():
    data = candidate_package_dict()
    summary = dict(data["summary"])  # type: ignore[arg-type]
    summary["score"] = True
    data["summary"] = summary

    with pytest.raises(ContractError, match="summary.score must be numeric"):
        CandidatePackage.from_dict(data)
