from __future__ import annotations

import json
import os

import pytest

from gitmoot_skillopt.cli import main
from gitmoot_skillopt.contracts import CANDIDATE_PACKAGE_KIND, CandidatePackage, TrainingPackage
from gitmoot_skillopt.optimize import write_candidate_package
from gitmoot_skillopt.preflight import PreflightResult, resolve_evaluator_config
from skillopt.envs.gitmoot.evaluator import evaluate_response
from tests.test_gitmoot_dataloader import write_training_package


def _landing_dimension_scores():
    return {
        "mobile_responsiveness": 0.9,
        "footer_presence_clarity": 0.8,
        "hero_quality": 0.9,
        "cta_clarity": 0.85,
        "visual_images_relevance": 0.75,
        "animation_motion_quality": 0.7,
        "text_overlap_readability": 0.95,
        "ranked_strength_preservation": 0.85,
    }


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


def test_optimize_dry_run_writes_no_candidate_package_and_artifacts(tmp_path, capsys):
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
    output = capsys.readouterr().out
    assert "wrote no-candidate package" in output
    assert "no_candidate_reason: candidate_content_unchanged" in output
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
    assert loaded.summary.score is None
    assert loaded.eval_report["promotable"] is False
    assert loaded.eval_report["no_candidate_reason"] == "candidate_content_unchanged"
    assert loaded.summary.metadata["promotable"] is False
    assert loaded.summary.metadata["no_candidate_reason"] == "candidate_content_unchanged"
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


def test_no_candidate_package_preserves_noop_retry_attempts(tmp_path):
    package_path, _artifact_root = write_training_package(tmp_path)
    package = TrainingPackage.load(package_path)
    out_root = tmp_path / "out"
    candidate_output = out_root / "candidate.json"
    retry_attempts = [
        {
            "attempt": 0,
            "reasons": ["no_meaningful_skill_change", "candidate_content_unchanged"],
            "retry_hints": {"improve": ["better mobile layout"]},
        }
    ]

    candidate = write_candidate_package(
        package=package,
        candidate_content=package.template.content,
        summary={
            "best_origin": "initial_skill",
            "total_accepts": 0,
            "no_candidate_triggers": ["no_meaningful_skill_change"],
            "noop_retry_attempts": retry_attempts,
        },
        out_root=out_root,
        artifact_dir=out_root / "artifacts",
        candidate_output=candidate_output,
        dry_run=False,
    )

    assert candidate.summary.score is None
    assert candidate.eval_report["no_candidate_reason"] == "no_meaningful_skill_change"
    assert candidate.eval_report["noop_retry_attempts"] == retry_attempts
    assert candidate.summary.metadata["noop_retry_attempts"] == retry_attempts
    assert "candidate_content_unchanged" in candidate.summary.metadata["no_candidate_triggers"]


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


def test_optimize_dry_run_skips_evaluator_resolution(tmp_path):
    package_path, artifact_root = write_training_package(tmp_path)
    package = json.loads(package_path.read_text(encoding="utf-8"))
    package["evaluator_config"] = {"mode": "legacy-manual-evaluator"}
    package_path.write_text(json.dumps(package), encoding="utf-8")
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
    assert (out_root / "candidate.json").is_file()


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


def test_optimize_threads_evaluator_options(monkeypatch):
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
            "--evaluator-id",
            "landing_page_v1",
            "--evaluator-model",
            "gpt-evaluator",
            "--evaluator-backend",
            "codex",
        ]
    )

    assert result == 0
    assert captured["evaluator_id"] == "landing_page_v1"
    assert captured["evaluator_model"] == "gpt-evaluator"
    assert captured["evaluator_backend"] == "codex"


def test_preflight_infers_landing_page_evaluator_for_vue_preview_manual_review(tmp_path):
    package_path, _artifact_root = write_training_package(tmp_path)
    package = json.loads(package_path.read_text(encoding="utf-8"))
    package["evaluator_config"] = {"driver": "manual-review"}
    package["items"][0]["metadata"] = {"split": "train", "artifact_type": "vue-preview"}
    package_path.write_text(json.dumps(package), encoding="utf-8")

    config = resolve_evaluator_config(TrainingPackage.load(package_path))

    assert config["mode"] == "landing_page_v1"
    assert config["evaluator_id"] == "landing_page_v1"
    assert config["driver"] == "manual-review"


def test_preflight_threads_evaluator_profile_contract(tmp_path):
    package_path, _artifact_root = write_training_package(tmp_path)
    package = json.loads(package_path.read_text(encoding="utf-8"))
    package.pop("evaluator_config", None)
    package["evaluator_profile"] = {
        "profile_id": "vue_landing_page_v1",
        "task_kind": "vue_landing_page",
        "artifact_contract": "vue_vite_bundle",
        "preview_adapter": "vue_vite",
        "checks": [{"id": "render_smoke", "type": "playwright", "required": True}],
        "judge": {"type": "screenshot_llm", "model": "gpt-profile-eval"},
    }
    package_path.write_text(json.dumps(package), encoding="utf-8")

    config = resolve_evaluator_config(TrainingPackage.load(package_path))

    assert config["mode"] == "landing_page_v1"
    assert config["evaluator_id"] == "landing_page_v1"
    assert config["profile_id"] == "vue_landing_page_v1"
    assert config["artifact_contract"] == "vue_vite_bundle"
    assert config["preview_adapter"] == "vue_vite"
    assert config["checks"][0]["id"] == "render_smoke"
    assert config["checks"][0]["required"] is True
    assert config["evaluator_model"] == "gpt-profile-eval"


def test_preflight_honors_evaluator_id_override_over_profile_mode(tmp_path):
    package_path, _artifact_root = write_training_package(tmp_path)
    package = json.loads(package_path.read_text(encoding="utf-8"))
    package["evaluator_profile"] = {
        "profile_id": "vue_landing_page_v1",
        "task_kind": "vue_landing_page",
        "artifact_contract": "vue_vite_bundle",
        "preview_adapter": "vue_vite",
    }
    package["evaluator_config"] = {"evaluator_id": "fixture"}
    package_path.write_text(json.dumps(package), encoding="utf-8")

    config = resolve_evaluator_config(TrainingPackage.load(package_path))

    assert config["mode"] == "fixture"
    assert config["evaluator_id"] == "fixture"
    assert config["profile_id"] == "vue_landing_page_v1"
    assert config["artifact_contract"] == "vue_vite_bundle"


def test_preflight_honors_driver_override_over_profile_mode(tmp_path):
    package_path, _artifact_root = write_training_package(tmp_path)
    package = json.loads(package_path.read_text(encoding="utf-8"))
    package["evaluator_profile"] = {
        "profile_id": "vue_landing_page_v1",
        "task_kind": "vue_landing_page",
        "artifact_contract": "vue_vite_bundle",
    }
    package["evaluator_config"] = {"driver": "fixture"}
    package_path.write_text(json.dumps(package), encoding="utf-8")

    config = resolve_evaluator_config(TrainingPackage.load(package_path))

    assert config["mode"] == "fixture"
    assert config["evaluator_id"] == "fixture"
    assert config["profile_id"] == "vue_landing_page_v1"


@pytest.mark.parametrize(
    ("raw_config", "expected_mode"),
    [
        ({}, "llm_judge"),
        ({"mode": "deterministic"}, "fixture"),
        ({"mode": "mock"}, "fixture"),
        ({"mode": "substring"}, "contains"),
        ({"mode": "llm-judge"}, "llm_judge"),
        ({"mode": "pairwise"}, "llm_judge"),
    ],
)
def test_preflight_preserves_adapter_evaluator_aliases(tmp_path, raw_config, expected_mode):
    package_path, _artifact_root = write_training_package(tmp_path)
    package = json.loads(package_path.read_text(encoding="utf-8"))
    if raw_config:
        package["evaluator_config"] = raw_config
    else:
        package.pop("evaluator_config", None)
    package_path.write_text(json.dumps(package), encoding="utf-8")

    config = resolve_evaluator_config(TrainingPackage.load(package_path))

    assert config["evaluator_id"] == expected_mode
    if raw_config:
        assert config["mode"] == expected_mode
    else:
        assert "mode" not in config


def test_preflight_default_does_not_override_item_evaluator_mode(tmp_path):
    package_path, _artifact_root = write_training_package(tmp_path)
    package = json.loads(package_path.read_text(encoding="utf-8"))
    package.pop("evaluator_config", None)
    package_path.write_text(json.dumps(package), encoding="utf-8")

    config = resolve_evaluator_config(TrainingPackage.load(package_path))
    score = evaluate_response(
        {"metadata": {"evaluator_mode": "fixture", "expected_hard": False, "expected_soft": 0.25}},
        "response",
        config,
    )

    assert config["evaluator_id"] == "llm_judge"
    assert "mode" not in config
    assert score["metadata"]["evaluator"] == "fixture"
    assert score["hard"] == 0
    assert score["soft"] == 0.25


def test_preflight_restores_optimizer_deployment_after_evaluator_canary(tmp_path, monkeypatch):
    from gitmoot_skillopt import preflight

    package_path, _artifact_root = write_training_package(tmp_path)
    package = json.loads(package_path.read_text(encoding="utf-8"))
    package["evaluator_config"] = {"mode": "llm_judge"}
    package_path.write_text(json.dumps(package), encoding="utf-8")
    captured = {}

    def fake_chat_target(**kwargs):
        captured["target_stage"] = kwargs["stage"]
        return "gitmoot-target-canary-ok", {}

    def fake_chat_optimizer(**kwargs):
        stage = kwargs["stage"]
        if stage == "gitmoot_preflight_optimizer":
            captured["optimizer_stage"] = stage
            captured["optimizer_deployment"] = os.environ.get("OPTIMIZER_DEPLOYMENT")
            return "gitmoot-optimizer-canary-ok", {}
        captured["evaluator_stage"] = stage
        captured["evaluator_deployment"] = os.environ.get("OPTIMIZER_DEPLOYMENT")
        return '{"hard": 1, "soft": 0.9, "fail_reason": "", "reasoning": "ok"}', {}

    monkeypatch.setattr(preflight, "chat_target", fake_chat_target)
    monkeypatch.setattr(preflight, "chat_optimizer", fake_chat_optimizer)

    result = preflight.run_optimizer_preflight(
        TrainingPackage.load(package_path),
        optimizer_backend="openai_chat",
        target_backend="openai_chat",
        optimizer_model="gpt-opt",
        target_model="gpt-target",
        evaluator_backend="codex",
        evaluator_model="gpt-eval",
    )

    assert result.optimizer_model == "gpt-opt"
    assert captured["optimizer_stage"] == "gitmoot_preflight_optimizer"
    assert captured["optimizer_deployment"] == "gpt-opt"
    assert captured["target_stage"] == "gitmoot_preflight_target"
    assert captured["evaluator_stage"] == "gitmoot_preflight_evaluator"
    assert captured["evaluator_deployment"] == "gpt-eval"
    assert os.environ["OPTIMIZER_DEPLOYMENT"] == "gpt-opt"


def test_preflight_blocks_required_render_smoke_when_npm_missing(tmp_path, monkeypatch):
    from gitmoot_skillopt import preflight

    package_path, _artifact_root = write_training_package(tmp_path)
    package = json.loads(package_path.read_text(encoding="utf-8"))
    package.pop("evaluator_config", None)
    package["evaluator_profile"] = {
        "profile_id": "vue_landing_page_v1",
        "task_kind": "vue_landing_page",
        "artifact_contract": "vue_vite_bundle",
        "checks": [{"id": "render_smoke", "type": "playwright", "required": True}],
    }
    package_path.write_text(json.dumps(package), encoding="utf-8")

    def fake_chat_target(**kwargs):
        return "gitmoot-target-canary-ok", {}

    def fake_chat_optimizer(**kwargs):
        if kwargs["stage"] == "gitmoot_preflight_optimizer":
            return "gitmoot-optimizer-canary-ok", {}
        return json.dumps(
            {
                "hard": 1,
                "soft": 0.9,
                "dimension_scores": _landing_dimension_scores(),
                "rationale": "ok",
                "fail_reason": "",
            }
        ), {}

    monkeypatch.setattr(preflight, "chat_target", fake_chat_target)
    monkeypatch.setattr(preflight, "chat_optimizer", fake_chat_optimizer)
    monkeypatch.setattr(preflight.shutil, "which", lambda command: None if command == "npm" else f"/bin/{command}")

    with pytest.raises(ValueError, match="requires npm"):
        preflight.run_optimizer_preflight(
            TrainingPackage.load(package_path),
            optimizer_backend="openai_chat",
            target_backend="openai_chat",
            optimizer_model="gpt-opt",
            target_model="gpt-target",
        )


def test_preflight_blocks_required_render_smoke_when_chromium_missing(tmp_path, monkeypatch):
    from gitmoot_skillopt import preflight

    package_path, _artifact_root = write_training_package(tmp_path)
    package = json.loads(package_path.read_text(encoding="utf-8"))
    package["evaluator_config"] = {
        "mode": "landing_page_v1",
        "artifact_contract": "vue_vite_bundle",
        "require_vue_render_smoke": True,
    }
    package_path.write_text(json.dumps(package), encoding="utf-8")

    class FakeChromium:
        def launch(self):
            raise RuntimeError("Executable doesn't exist")

    class FakePlaywright:
        chromium = FakeChromium()

    class FakeSyncPlaywright:
        def __enter__(self):
            return FakePlaywright()

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_chat_target(**kwargs):
        return "gitmoot-target-canary-ok", {}

    def fake_chat_optimizer(**kwargs):
        if kwargs["stage"] == "gitmoot_preflight_optimizer":
            return "gitmoot-optimizer-canary-ok", {}
        return json.dumps(
            {
                "hard": 1,
                "soft": 0.9,
                "dimension_scores": _landing_dimension_scores(),
                "rationale": "ok",
                "fail_reason": "",
            }
        ), {}

    monkeypatch.setattr(preflight, "chat_target", fake_chat_target)
    monkeypatch.setattr(preflight, "chat_optimizer", fake_chat_optimizer)
    monkeypatch.setattr(preflight.shutil, "which", lambda command: f"/bin/{command}")
    monkeypatch.setattr(preflight, "_import_sync_playwright", lambda: FakeSyncPlaywright)

    with pytest.raises(ValueError, match="requires Playwright Chromium"):
        preflight.run_optimizer_preflight(
            TrainingPackage.load(package_path),
            optimizer_backend="openai_chat",
            target_backend="openai_chat",
            optimizer_model="gpt-opt",
            target_model="gpt-target",
        )


def test_preflight_blocks_required_render_smoke_when_trusted_deps_missing(tmp_path, monkeypatch):
    from gitmoot_skillopt import preflight

    package_path, _artifact_root = write_training_package(tmp_path)
    package = json.loads(package_path.read_text(encoding="utf-8"))
    package["evaluator_config"] = {
        "mode": "landing_page_v1",
        "artifact_contract": "vue_vite_bundle",
        "require_vue_render_smoke": True,
    }
    package_path.write_text(json.dumps(package), encoding="utf-8")

    class FakeBrowser:
        def close(self):
            pass

    class FakeChromium:
        def launch(self):
            return FakeBrowser()

    class FakePlaywright:
        chromium = FakeChromium()

    class FakeSyncPlaywright:
        def __enter__(self):
            return FakePlaywright()

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_chat_target(**kwargs):
        return "gitmoot-target-canary-ok", {}

    def fake_chat_optimizer(**kwargs):
        if kwargs["stage"] == "gitmoot_preflight_optimizer":
            return "gitmoot-optimizer-canary-ok", {}
        return json.dumps(
            {
                "hard": 1,
                "soft": 0.9,
                "dimension_scores": _landing_dimension_scores(),
                "rationale": "ok",
                "fail_reason": "",
            }
        ), {}

    def fail_prepare_trusted_deps(work_path, timeout):
        del work_path, timeout
        raise RuntimeError("registry unavailable")

    monkeypatch.setattr(preflight, "chat_target", fake_chat_target)
    monkeypatch.setattr(preflight, "chat_optimizer", fake_chat_optimizer)
    monkeypatch.setattr(preflight.shutil, "which", lambda command: f"/bin/{command}")
    monkeypatch.setattr(preflight, "_import_sync_playwright", lambda: FakeSyncPlaywright)
    monkeypatch.setattr(preflight, "_prepare_trusted_vue_render_deps", fail_prepare_trusted_deps)

    with pytest.raises(ValueError, match="trusted Vue render dependencies"):
        preflight.run_optimizer_preflight(
            TrainingPackage.load(package_path),
            optimizer_backend="openai_chat",
            target_backend="openai_chat",
            optimizer_model="gpt-opt",
            target_model="gpt-target",
        )


def test_preflight_requires_exact_target_canary(tmp_path, monkeypatch):
    from gitmoot_skillopt import preflight

    package_path, _artifact_root = write_training_package(tmp_path)

    def fake_chat_optimizer(**kwargs):
        assert kwargs["stage"] == "gitmoot_preflight_optimizer"
        return "gitmoot-optimizer-canary-ok", {}

    def fake_chat_target(**kwargs):
        assert kwargs["stage"] == "gitmoot_preflight_target"
        return "target returned a refusal instead", {}

    monkeypatch.setattr(preflight, "chat_optimizer", fake_chat_optimizer)
    monkeypatch.setattr(preflight, "chat_target", fake_chat_target)

    with pytest.raises(ValueError, match="target canary returned unexpected response"):
        preflight.run_optimizer_preflight(
            TrainingPackage.load(package_path),
            optimizer_backend="openai_chat",
            target_backend="openai_chat",
            optimizer_model="gpt-opt",
            target_model="gpt-target",
        )


def test_preflight_preserves_package_evaluator_backend_and_model(tmp_path, monkeypatch):
    from gitmoot_skillopt import preflight
    from skillopt.model import get_optimizer_backend

    package_path, _artifact_root = write_training_package(tmp_path)
    package = json.loads(package_path.read_text(encoding="utf-8"))
    package["evaluator_config"] = {
        "mode": "llm_judge",
        "evaluator_backend": "codex",
        "evaluator_model": "gpt-package-eval",
    }
    package_path.write_text(json.dumps(package), encoding="utf-8")
    captured = {}

    def fake_chat_optimizer(**kwargs):
        stage = kwargs["stage"]
        if stage == "gitmoot_preflight_optimizer":
            captured["optimizer_backend"] = get_optimizer_backend()
            captured["optimizer_deployment"] = os.environ.get("OPTIMIZER_DEPLOYMENT")
            return "gitmoot-optimizer-canary-ok", {}
        captured["evaluator_backend"] = get_optimizer_backend()
        captured["evaluator_deployment"] = os.environ.get("OPTIMIZER_DEPLOYMENT")
        return '{"hard": 1, "soft": 0.9, "fail_reason": "", "reasoning": "ok"}', {}

    def fake_chat_target(**kwargs):
        return "gitmoot-target-canary-ok", {}

    monkeypatch.setattr(preflight, "chat_optimizer", fake_chat_optimizer)
    monkeypatch.setattr(preflight, "chat_target", fake_chat_target)

    result = preflight.run_optimizer_preflight(
        TrainingPackage.load(package_path),
        optimizer_backend="openai_chat",
        target_backend="openai_chat",
        optimizer_model="gpt-opt",
        target_model="gpt-target",
    )

    assert result.evaluator_backend == "codex"
    assert result.evaluator_model == "gpt-package-eval"
    assert result.evaluator_config["evaluator_backend"] == "codex"
    assert result.evaluator_config["evaluator_model"] == "gpt-package-eval"
    assert captured["optimizer_backend"] == "openai_chat"
    assert captured["optimizer_deployment"] == "gpt-opt"
    assert captured["evaluator_backend"] == "codex"
    assert captured["evaluator_deployment"] == "gpt-package-eval"


def test_preflight_accepts_azure_openai_backend_aliases(tmp_path, monkeypatch):
    from gitmoot_skillopt import preflight
    from skillopt.model import get_optimizer_backend

    package_path, _artifact_root = write_training_package(tmp_path)
    package = json.loads(package_path.read_text(encoding="utf-8"))
    package["evaluator_config"] = {"mode": "llm_judge"}
    package_path.write_text(json.dumps(package), encoding="utf-8")
    captured = {}

    def fake_chat_optimizer(**kwargs):
        stage = kwargs["stage"]
        if stage == "gitmoot_preflight_optimizer":
            captured["optimizer_backend"] = get_optimizer_backend()
            return "gitmoot-optimizer-canary-ok", {}
        captured["evaluator_backend"] = get_optimizer_backend()
        return '{"hard": 1, "soft": 0.9, "fail_reason": "", "reasoning": "ok"}', {}

    def fake_chat_target(**kwargs):
        captured["target_stage"] = kwargs["stage"]
        return "gitmoot-target-canary-ok", {}

    monkeypatch.setattr(preflight, "chat_optimizer", fake_chat_optimizer)
    monkeypatch.setattr(preflight, "chat_target", fake_chat_target)

    result = preflight.run_optimizer_preflight(
        TrainingPackage.load(package_path),
        optimizer_backend="azure_openai",
        target_backend="azure_openai",
        optimizer_model="gpt-opt",
        target_model="gpt-target",
        evaluator_backend="azure_openai",
        evaluator_model="gpt-eval",
    )

    assert result.optimizer_backend == "openai_chat"
    assert result.target_backend == "openai_chat"
    assert result.evaluator_backend == "openai_chat"
    assert result.evaluator_config["evaluator_backend"] == "openai_chat"
    assert captured["optimizer_backend"] == "openai_chat"
    assert captured["target_stage"] == "gitmoot_preflight_target"
    assert captured["evaluator_backend"] == "openai_chat"


def test_optimize_runs_preflight_before_real_trainer(tmp_path, monkeypatch):
    package_path, artifact_root = write_training_package(tmp_path)
    out_root = tmp_path / "out"
    captured = {}

    def fake_preflight(package, **kwargs):
        captured["preflight"] = kwargs
        return PreflightResult(
            optimizer_backend="codex",
            target_backend="codex_exec",
            evaluator_backend="openai_chat",
            optimizer_model="gpt-resolved-opt",
            target_model="gpt-resolved-target",
            evaluator_model="gpt-eval",
            evaluator_config={"mode": "landing_page_v1", "evaluator_id": "landing_page_v1"},
        )

    class FakeTrainer:
        def __init__(self, cfg, adapter):
            captured["cfg"] = cfg
            captured["adapter"] = adapter

        def train(self):
            return {
                "gate_status": "passed",
                "promotable": True,
                "best_selection_hard": 1.0,
                "baseline_selection_hard": 0.0,
                "best_step": 1,
                "total_steps": 1,
            }

    monkeypatch.setattr("gitmoot_skillopt.optimize.run_optimizer_preflight", fake_preflight)
    monkeypatch.setattr("gitmoot_skillopt.optimize.ReflACTTrainer", FakeTrainer)

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
            "--optimizer-backend",
            "openai_chat",
            "--target-backend",
            "codex_exec",
            "--optimizer-model",
            "gpt-opt",
            "--target-model",
            "gpt-target",
            "--evaluator-id",
            "landing_page_v1",
            "--evaluator-backend",
            "codex",
            "--evaluator-model",
            "gpt-eval",
        ]
    )

    assert result == 0
    assert captured["preflight"]["target_backend"] == "codex_exec"
    assert captured["preflight"]["evaluator_id"] == "landing_page_v1"
    assert captured["cfg"]["optimizer_backend"] == "codex"
    assert captured["cfg"]["target_backend"] == "codex_exec"
    assert captured["cfg"]["optimizer_model"] == "gpt-resolved-opt"
    assert captured["cfg"]["target_model"] == "gpt-resolved-target"
    assert captured["cfg"]["evaluator_config"]["mode"] == "landing_page_v1"
    assert (out_root / "candidate.json").is_file()


def test_optimize_preflight_failure_blocks_trainer(tmp_path, monkeypatch):
    package_path, artifact_root = write_training_package(tmp_path)
    out_root = tmp_path / "out"

    def fail_preflight(*args, **kwargs):
        raise ValueError("evaluator canary did not return JSON")

    class FailingTrainer:
        def __init__(self, *args, **kwargs):
            raise AssertionError("trainer must not start after preflight failure")

    monkeypatch.setattr("gitmoot_skillopt.optimize.run_optimizer_preflight", fail_preflight)
    monkeypatch.setattr("gitmoot_skillopt.optimize.ReflACTTrainer", FailingTrainer)

    with pytest.raises(ValueError, match="evaluator canary did not return JSON"):
        main(
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
            ]
        )

    assert not (out_root / "candidate.json").exists()


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


def test_initial_skill_best_origin_writes_no_candidate_metadata(tmp_path):
    package_path, artifact_root = write_training_package(tmp_path)
    del artifact_root
    package = TrainingPackage.load(package_path)
    candidate_output = tmp_path / "out" / "candidate.json"

    candidate = write_candidate_package(
        package=package,
        candidate_content=package.template.content.replace("Plan carefully.", "Plan carefully with a checklist."),
        summary={
            "gate_status": "passed",
            "promotable": True,
            "best_origin": "initial_skill",
            "total_accepts": 1,
            "best_selection_hard": 1,
            "baseline_selection_hard": 1,
        },
        out_root=tmp_path / "out",
        artifact_dir=tmp_path / "out" / "artifacts",
        candidate_output=candidate_output,
        dry_run=False,
    )

    loaded = CandidatePackage.load(candidate_output)
    assert candidate.summary.score is None
    assert loaded.eval_report["promotable"] is False
    assert loaded.eval_report["no_candidate_reason"] == "best_origin_initial_skill"
    assert loaded.summary.metadata["no_candidate_triggers"] == ["best_origin_initial_skill"]


def test_selection_reject_summary_writes_gate_rejection_package(tmp_path):
    package_path, artifact_root = write_training_package(tmp_path)
    del artifact_root
    package = TrainingPackage.load(package_path)
    candidate_output = tmp_path / "out" / "candidate.json"

    candidate = write_candidate_package(
        package=package,
        candidate_content=package.template.content,
        summary={
            "gate_status": "passed",
            "promotable": True,
            "best_origin": "initial_skill",
            "total_accepts": 0,
            "total_rejects": 1,
            "best_selection_hard": 1.0,
            "best_selection_soft": 0.89,
            "baseline_selection_hard": 1.0,
            "baseline_selection_soft": 0.89,
            "final_test_skipped_reason": "selection_gate_rejected_candidate",
            "gate_rejection": {
                "rejection_type": "candidate_score_regression",
                "retryable": True,
                "baseline": {"hard": 1.0, "soft": 0.89, "gate_score": 0.89},
                "candidate": {"hard": 1.0, "soft": 0.84, "gate_score": 0.84},
                "primary_reason": "candidate_quality_regressed",
                "human_reason": "The candidate lost selection evaluation against the baseline skill.",
                "optimizer_hint": "Use gate rejection evidence before spending final test budget.",
                "failed_dimensions": ["selection_gate", "human_feedback_alignment"],
                "evidence": ["Candidate gate score 0.8400 <= baseline gate score 0.8900."],
                "attempted_patch": "artifact delivery only",
                "retry_attempts": "0/0",
                "next_action": "Stop without final test eval.",
            },
        },
        out_root=tmp_path / "out",
        artifact_dir=tmp_path / "out" / "artifacts",
        candidate_output=candidate_output,
        dry_run=False,
    )

    loaded = CandidatePackage.load(candidate_output)
    assert candidate.summary.score is None
    assert loaded.eval_report["final_test_skipped_reason"] == "selection_gate_rejected_candidate"
    assert loaded.eval_report["gate_rejection"]["candidate"]["gate_score"] == 0.84
    assert loaded.summary.metadata["gate_rejection"]["baseline"]["gate_score"] == 0.89
    assert loaded.summary.gate_rejection is not None
    assert loaded.summary.gate_rejection.primary_reason == "candidate_quality_regressed"
    assert loaded.summary.gate_rejection.attempted_patch == "artifact delivery only"
