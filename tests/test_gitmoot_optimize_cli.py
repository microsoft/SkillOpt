from __future__ import annotations

import json

import pytest

from gitmoot_skillopt.cli import main
from gitmoot_skillopt.contracts import CANDIDATE_PACKAGE_KIND, CandidatePackage, TrainingPackage
from gitmoot_skillopt.optimize import write_candidate_package
from tests.test_gitmoot_dataloader import write_training_package


def test_optimize_requires_flags():
    with pytest.raises(SystemExit):
        main(["optimize"])


def test_optimize_rejects_invalid_package_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="training package"):
        main(
            [
                "optimize",
                "--training-package",
                str(tmp_path / "missing.json"),
                "--artifact-root",
                str(tmp_path),
                "--out-root",
                str(tmp_path / "out"),
                "--candidate-output",
                str(tmp_path / "out" / "candidate.json"),
                "--dry-run",
            ]
        )


def test_optimize_dry_run_writes_candidate_package_and_artifacts(tmp_path):
    package_path, artifact_root = write_training_package(tmp_path)
    out_root = tmp_path / "out"
    candidate_output = out_root / "candidate.json"

    result = main(
        [
            "optimize",
            "--training-package",
            str(package_path),
            "--artifact-root",
            str(artifact_root),
            "--out-root",
            str(out_root),
            "--candidate-output",
            str(candidate_output),
            "--dry-run",
        ]
    )

    assert result == 0
    data = json.loads(candidate_output.read_text(encoding="utf-8"))
    loaded = CandidatePackage.from_dict(data)
    assert loaded.kind == CANDIDATE_PACKAGE_KIND
    assert loaded.template_id == "planner"
    assert loaded.candidate.content.startswith("---\n")
    assert loaded.summary.diff_artifact_id == "run-1/candidate-diff"
    assert {artifact.id for artifact in loaded.artifacts} == {
        "run-1/candidate-diff",
        "run-1/eval-report",
        "run-1/preference-summary",
    }
    for artifact in loaded.artifacts:
        assert (out_root / "artifacts" / artifact.path).is_file()


def test_optimize_dry_run_accepts_explicit_artifact_dir(tmp_path):
    package_path, artifact_root = write_training_package(tmp_path)
    out_root = tmp_path / "out"
    artifact_dir = tmp_path / "candidate-artifacts"
    candidate_output = out_root / "candidate.json"

    result = main(
        [
            "optimize",
            "--training-package",
            str(package_path),
            "--artifact-root",
            str(artifact_root),
            "--out-root",
            str(out_root),
            "--candidate-output",
            str(candidate_output),
            "--artifact-dir",
            str(artifact_dir),
            "--dry-run",
        ]
    )

    assert result == 0
    data = json.loads(candidate_output.read_text(encoding="utf-8"))
    loaded = CandidatePackage.from_dict(data)
    for artifact in loaded.artifacts:
        assert (artifact_dir / artifact.path).is_file()
    assert not (out_root / "artifacts").exists()


def test_optimize_dry_run_does_not_start_trainer(tmp_path, monkeypatch):
    class FailingTrainer:
        def __init__(self, *args, **kwargs):
            raise AssertionError("dry-run must not initialize trainer")

    monkeypatch.setattr("gitmoot_skillopt.optimize.ReflACTTrainer", FailingTrainer)
    package_path, artifact_root = write_training_package(tmp_path)
    out_root = tmp_path / "out"

    result = main(
        [
            "optimize",
            "--training-package",
            str(package_path),
            "--artifact-root",
            str(artifact_root),
            "--out-root",
            str(out_root),
            "--candidate-output",
            str(out_root / "candidate.json"),
            "--dry-run",
        ]
    )

    assert result == 0


def test_optimize_threads_optional_reasoning_effort(tmp_path, monkeypatch):
    captured = {}

    def fake_run_optimize(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("gitmoot_skillopt.optimize.run_optimize", fake_run_optimize)

    result = main(
        [
            "optimize",
            "--training-package",
            "training.json",
            "--artifact-root",
            "blobs",
            "--out-root",
            "out",
            "--candidate-output",
            "out/candidate.json",
        ]
    )

    assert result == 0
    assert captured["reasoning_effort"] == ""

    result = main(
        [
            "optimize",
            "--training-package",
            "training.json",
            "--artifact-root",
            "blobs",
            "--out-root",
            "out",
            "--candidate-output",
            "out/candidate.json",
            "--reasoning-effort",
            "medium",
        ]
    )

    assert result == 0
    assert captured["reasoning_effort"] == "medium"


def test_optimize_threads_gate_metric(tmp_path, monkeypatch):
    captured = {}

    def fake_run_optimize(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("gitmoot_skillopt.optimize.run_optimize", fake_run_optimize)

    result = main(
        [
            "optimize",
            "--training-package",
            "training.json",
            "--artifact-root",
            "blobs",
            "--out-root",
            "out",
            "--candidate-output",
            "out/candidate.json",
            "--gate-metric",
            "soft",
        ]
    )

    assert result == 0
    assert captured["gate_metric"] == "soft"


def test_blocked_summary_writes_non_promotable_candidate_metadata(tmp_path):
    package_path, artifact_root = write_training_package(tmp_path)
    del artifact_root
    package = TrainingPackage.load(package_path)
    candidate_output = tmp_path / "out" / "candidate.json"
    block = {
        "blocker": "evaluator_not_run",
        "items": [
            {
                "id": "val-1",
                "blocker": "evaluator_not_run",
                "score_status": "unscored",
                "target_trace_path": "predictions/val-1/conversation.json",
                "evaluator_trace_path": "predictions/val-1/result.json",
            }
        ],
    }

    candidate = write_candidate_package(
        package=package,
        candidate_content=package.template.content,
        summary={
            "gate_status": "blocked",
            "gate_blocker": "evaluator_not_run",
            "gate_blockers": [block],
            "best_selection_hard": None,
            "baseline_selection_hard": None,
        },
        out_root=tmp_path / "out",
        artifact_dir=tmp_path / "out" / "artifacts",
        candidate_output=candidate_output,
        dry_run=False,
    )

    loaded = CandidatePackage.load(candidate_output)
    assert candidate.summary.score is None
    assert loaded.summary.score is None
    assert loaded.eval_report["gate_status"] == "blocked"
    assert loaded.eval_report["promotable"] is False
    assert loaded.summary.metadata["promotable"] is False
    assert loaded.summary.metadata["gate_blocker"] == "evaluator_not_run"
    assert loaded.summary.metadata["gate_blockers"][0]["items"][0]["id"] == "val-1"
