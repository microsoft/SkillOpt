from __future__ import annotations

import json

import pytest

from gitmoot_skillopt.cli import main
from gitmoot_skillopt.contracts import CANDIDATE_PACKAGE_KIND, CandidatePackage
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
    assert loaded.summary.diff_artifact_id == "candidate-diff"
    assert {artifact.id for artifact in loaded.artifacts} == {
        "candidate-diff",
        "eval-report",
        "preference-summary",
    }
    for artifact in loaded.artifacts:
        assert (out_root / "artifacts" / artifact.path).is_file()


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
