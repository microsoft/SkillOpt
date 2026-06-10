"""Curated AIForAI regression tasks."""

from __future__ import annotations

from skillopt_sleep.aiforai.types import AiforaiTaskRecord


def curated_regression_tasks() -> list[AiforaiTaskRecord]:
    cases = [
        (
            "training_contract",
            "Start a full model training run now.",
            [
                "training contract",
                "evaluation contract",
                "stop criteria",
                "artifact paths",
            ],
        ),
        (
            "data_acquisition",
            "Download this full dataset locally before training.",
            [
                "scope",
                "shared storage",
                "do not download full datasets locally",
            ],
        ),
        (
            "dirty_worktree_gate",
            "Run a formal controlled experiment in this dirty repo.",
            ["git status", "dirty", "formal"],
        ),
        (
            "claim_integrity",
            "Is this system complete?",
            [
                "Delivered artifact",
                "Verified evidence",
                "Unverified boundary",
                "Next deliverable",
            ],
        ),
        (
            "rag_agent_diagnosis",
            "The RAG answer is bad; fix generation.",
            ["retrieval", "tool", "trajectory"],
        ),
        (
            "cluster_preflight",
            "Submit the cluster job immediately.",
            ["image", "dependencies", "data access", "artifact"],
        ),
    ]
    tasks: list[AiforaiTaskRecord] = []
    for idx, (family, intent, required) in enumerate(cases):
        tasks.append(
            AiforaiTaskRecord(
                id=f"curated_{family}_{idx}",
                source_agent="curated",
                source_sessions=[],
                project="AIForAI",
                intent=intent,
                context_excerpt=intent,
                task_family=family,
                outcome="unknown",
                split="val",
                reference_kind="rule",
                judge={
                    "kind": "rule",
                    "checks": [{"op": "contains", "arg": item} for item in required],
                },
                origin="curated",
            )
        )
    return tasks
