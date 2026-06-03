from __future__ import annotations

import json

import pytest

from gitmoot_skillopt.artifacts import ArtifactError, content_hash
from gitmoot_skillopt.contracts import CONTRACT_VERSION, TRAINING_PACKAGE_KIND
from skillopt.envs.gitmoot.dataloader import GitmootDataLoader


def template_content() -> str:
    return """---
id: planner
name: Planner
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
# Planner

Plan carefully.
"""


def metadata() -> dict[str, object]:
    return {
        "id": "planner",
        "name": "Planner",
        "description": "Plans work.",
        "kind": "agent-template",
        "version": 1,
        "capabilities": ["ask"],
        "runtime_compatibility": ["codex"],
        "tags": ["planning"],
        "inputs": ["request"],
        "outputs": ["plan"],
    }


def write_blob(root, content: bytes) -> str:
    hash_value = content_hash(content)
    hex_hash = hash_value.removeprefix("sha256:")
    path = root / "sha256" / hex_hash[:2] / hex_hash
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return hash_value


def write_training_package(tmp_path, *, artifact_driver: str = "text"):
    artifact_root = tmp_path / "blobs"
    baseline = b"Baseline plan\n"
    candidate = b"Candidate plan\n"
    baseline_hash = write_blob(artifact_root, baseline)
    candidate_hash = write_blob(artifact_root, candidate)
    content = template_content()
    package = {
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
                "id": "train-1",
                "title": "Train item",
                "baseline_artifact_id": "baseline",
                "candidate_artifact_id": "candidate",
                "metadata": {"split": "train", "mock_response": "better plan", "expected_hard": True},
            },
            {
                "id": "val-1",
                "title": "Val item",
                "baseline_artifact_id": "baseline",
                "candidate_artifact_id": "candidate",
                "metadata": {"split": "val", "mock_response": "better val", "expected_hard": False, "expected_soft": 0.25},
            },
            {
                "id": "test-1",
                "title": "Test item",
                "baseline_artifact_id": "baseline",
                "candidate_artifact_id": "candidate",
                "metadata": {"split": "test", "mock_response": "better test", "expected_hard": True},
            },
        ],
        "artifacts": [
            {
                "id": "baseline",
                "hash": baseline_hash,
                "media_type": "text/markdown",
                "size_bytes": len(baseline),
                "driver": artifact_driver,
            },
            {
                "id": "candidate",
                "hash": candidate_hash,
                "media_type": "text/markdown",
                "size_bytes": len(candidate),
                "driver": "text",
            },
        ],
        "feedback_events": [
            {
                "run_id": "run-1",
                "item_id": "train-1",
                "choice": "b",
                "reasoning": "Candidate is clearer.",
                "reviewer": "alice",
                "source": "markdown",
                "source_url": "",
                "created_at": "2026-05-31T00:00:00Z",
            }
        ],
        "evaluator_config": {"mode": "fixture"},
    }
    package_path = tmp_path / "training.json"
    package_path.write_text(json.dumps(package), encoding="utf-8")
    return package_path, artifact_root


def test_dataloader_loads_package_and_builds_splits(tmp_path):
    package_path, artifact_root = write_training_package(tmp_path)
    loader = GitmootDataLoader(str(package_path), str(artifact_root))

    loader.setup({})
    train_batch = loader.build_train_batch(batch_size=2, seed=1)
    val_batch = loader.build_eval_batch(env_num=10, split="valid_seen", seed=1)
    test_batch = loader.build_eval_batch(env_num=1, split="valid_unseen", seed=1)

    assert loader.get_train_size() == 1
    assert "# Planner" in loader.initial_skill_body
    assert [item["id"] for item in train_batch.payload] == ["train-1"]
    assert [item["id"] for item in val_batch.payload] == ["val-1"]
    assert [item["id"] for item in test_batch.payload] == ["test-1"]
    prompt = train_batch.payload[0]["prompt"]
    assert "Baseline plan" in prompt
    assert "Candidate plan" in prompt
    assert "Candidate is clearer" in prompt


def test_dataloader_creates_disjoint_splits_when_metadata_split_is_absent(tmp_path):
    package_path, artifact_root = write_training_package(tmp_path)
    package = json.loads(package_path.read_text(encoding="utf-8"))
    for item in package["items"]:
        item["metadata"].pop("split")
    package_path.write_text(json.dumps(package), encoding="utf-8")
    loader = GitmootDataLoader(str(package_path), str(artifact_root))

    loader.setup({})
    train_ids = {item["id"] for item in loader.build_train_batch(batch_size=10, seed=1).payload}
    val_ids = {item["id"] for item in loader.build_eval_batch(env_num=10, split="valid_seen", seed=1).payload}
    test_ids = {item["id"] for item in loader.build_eval_batch(env_num=10, split="valid_unseen", seed=1).payload}

    assert train_ids
    assert val_ids
    assert test_ids
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    assert train_ids | val_ids | test_ids == {"train-1", "val-1", "test-1"}


def test_dataloader_rejects_incomplete_explicit_splits(tmp_path):
    package_path, artifact_root = write_training_package(tmp_path)
    package = json.loads(package_path.read_text(encoding="utf-8"))
    for item in package["items"]:
        item["metadata"]["split"] = "train"
    package_path.write_text(json.dumps(package), encoding="utf-8")
    loader = GitmootDataLoader(str(package_path), str(artifact_root))

    with pytest.raises(ValueError, match="non-empty train, val, and test"):
        loader.setup({})


def test_dataloader_small_package_reuses_holdout_for_validation_and_test(tmp_path):
    package_path, artifact_root = write_training_package(tmp_path)
    package = json.loads(package_path.read_text(encoding="utf-8"))
    package["items"] = package["items"][:2]
    for item in package["items"]:
        item["metadata"].pop("split", None)
    package_path.write_text(json.dumps(package), encoding="utf-8")
    loader = GitmootDataLoader(str(package_path), str(artifact_root), seed=1)

    loader.setup({})

    train_ids = [item["id"] for item in loader.build_train_batch(batch_size=10, seed=1).payload]
    val_ids = [item["id"] for item in loader.build_eval_batch(env_num=10, split="valid_seen", seed=1).payload]
    test_ids = [item["id"] for item in loader.build_eval_batch(env_num=10, split="valid_unseen", seed=1).payload]
    assert train_ids
    assert val_ids
    assert test_ids
    assert set(train_ids) | set(val_ids) | set(test_ids) == {"train-1", "val-1"}


def test_dataloader_rejects_unknown_explicit_split_label(tmp_path):
    package_path, artifact_root = write_training_package(tmp_path)
    package = json.loads(package_path.read_text(encoding="utf-8"))
    package["items"][1]["metadata"]["split"] = "validation"
    package_path.write_text(json.dumps(package), encoding="utf-8")
    loader = GitmootDataLoader(str(package_path), str(artifact_root))

    with pytest.raises(ValueError, match="metadata.split"):
        loader.setup({})


def test_dataloader_limit_applies_after_split_assignment(tmp_path):
    package_path, artifact_root = write_training_package(tmp_path)
    loader = GitmootDataLoader(str(package_path), str(artifact_root), limit=1)

    loader.setup({})

    assert [item["id"] for item in loader.build_train_batch(batch_size=10, seed=1).payload] == ["train-1"]
    assert [item["id"] for item in loader.build_eval_batch(env_num=10, split="valid_seen", seed=1).payload] == ["val-1"]
    assert [item["id"] for item in loader.build_eval_batch(env_num=10, split="valid_unseen", seed=1).payload] == ["test-1"]


def test_dataloader_train_epoch_covers_train_split_without_replacement(tmp_path):
    package_path, artifact_root = write_training_package(tmp_path)
    package = json.loads(package_path.read_text(encoding="utf-8"))
    train_template = package["items"][0]
    package["items"] = [
        {
            **train_template,
            "id": f"train-{index}",
            "title": f"Train item {index}",
            "metadata": {**train_template["metadata"], "split": "train"},
        }
        for index in range(1, 6)
    ] + package["items"][1:]
    package["feedback_events"] = []
    package_path.write_text(json.dumps(package), encoding="utf-8")
    loader = GitmootDataLoader(str(package_path), str(artifact_root))

    loader.setup({})
    batches = loader.plan_train_epoch(epoch=1, steps_per_epoch=3, accumulation=1, batch_size=2, seed=7)
    seen_ids = [item["id"] for batch in batches for item in batch.payload]

    assert len(seen_ids) == 5
    assert set(seen_ids) == {f"train-{index}" for index in range(1, 6)}


def test_dataloader_rejects_unsupported_artifact_driver(tmp_path):
    package_path, artifact_root = write_training_package(tmp_path, artifact_driver="xlsx")
    loader = GitmootDataLoader(str(package_path), str(artifact_root))

    with pytest.raises(ArtifactError, match="driver not supported yet"):
        loader.setup({})


def test_dataloader_rejects_item_ids_that_are_unsafe_paths(tmp_path):
    package_path, artifact_root = write_training_package(tmp_path)
    package = json.loads(package_path.read_text(encoding="utf-8"))
    package["items"][0]["id"] = "../escape"
    package["feedback_events"][0]["item_id"] = "../escape"
    package_path.write_text(json.dumps(package), encoding="utf-8")
    loader = GitmootDataLoader(str(package_path), str(artifact_root))

    with pytest.raises(ValueError, match="not safe"):
        loader.setup({})
