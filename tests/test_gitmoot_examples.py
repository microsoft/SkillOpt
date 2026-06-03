from __future__ import annotations

from pathlib import Path

from gitmoot_skillopt.artifacts import content_hash
from gitmoot_skillopt.cli import main
from gitmoot_skillopt.contracts import CandidatePackage, TrainingPackage

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "examples" / "gitmoot" / "mvp-fixture"


def test_gitmoot_mvp_fixture_dry_run_contract(tmp_path):
    package_path = FIXTURE_ROOT / "training.json"
    artifact_root = FIXTURE_ROOT / "blobs"
    out_root = tmp_path / "out"
    candidate_output = out_root / "candidate.json"

    training = TrainingPackage.load(package_path)
    assert training.template.id == "planner-fixture"
    assert training.eval_run.id == "fixture-run-1"

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
    candidate = CandidatePackage.load(candidate_output)
    assert candidate.template_id == "planner-fixture"
    assert candidate.summary.diff_artifact_id == "fixture-run-1/candidate-diff"
    assert candidate.summary.metadata["artifact_ids"] == [
        "fixture-run-1/candidate-diff",
        "fixture-run-1/eval-report",
        "fixture-run-1/preference-summary",
    ]

    for artifact in candidate.artifacts:
        artifact_path = out_root / "artifacts" / artifact.path
        assert artifact_path.is_file()
        assert content_hash(artifact_path.read_bytes()) == artifact.hash
