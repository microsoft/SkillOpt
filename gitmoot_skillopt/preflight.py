"""Preflight checks for Gitmoot SkillOpt optimizer runs."""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gitmoot_skillopt.contracts import TrainingPackage
from skillopt.envs.gitmoot.evaluator import (
    LANDING_PAGE_DIMENSIONS,
    LANDING_PAGE_EVALUATOR_ID,
    _prepare_trusted_vue_render_deps,
)
from skillopt.model import (
    chat_optimizer,
    chat_target,
    get_optimizer_backend,
    is_target_chat_backend,
    is_target_exec_backend,
    set_optimizer_backend,
    set_optimizer_deployment,
    set_target_backend,
    set_target_deployment,
)
from skillopt.model.backend_config import normalize_target_backend_name
from skillopt.model.codex_harness import prepare_workspace, run_target_exec
from skillopt.model.common import default_model_for_backend, normalize_backend_name
from skillopt.utils import extract_json

SUPPORTED_EVALUATORS = {LANDING_PAGE_EVALUATOR_ID, "llm_judge", "fixture", "contains"}
TRAINING_MODES = {"explore", "refine", "distill", "validate"}
OPTIMIZER_CANARY_TEXT = "gitmoot-optimizer-canary-ok"
TARGET_CANARY_TEXT = "gitmoot-target-canary-ok"


@dataclass(frozen=True)
class PreflightResult:
    optimizer_backend: str
    target_backend: str
    evaluator_backend: str
    optimizer_model: str
    target_model: str
    evaluator_model: str
    evaluator_config: dict[str, Any]


def resolve_evaluator_config(
    package: TrainingPackage,
    *,
    evaluator_id: str = "",
    evaluator_backend: str = "",
    evaluator_model: str = "",
) -> dict[str, Any]:
    profile_config = _evaluator_profile_config(package)
    package_config = _package_evaluator_config(package)
    config = {**profile_config, **package_config}
    _apply_evaluator_id_mode_override(config, package_config)
    explicit = _normal_evaluator_id(evaluator_id)
    configured = _configured_evaluator_id(config)
    inferred = _infer_evaluator_id(package)
    resolved = explicit or configured or inferred or "llm_judge"
    if not resolved:
        raise ValueError("evaluator id is required; pass --evaluator-id or use a package with a supported evaluator")
    if resolved not in SUPPORTED_EVALUATORS:
        raise ValueError(f"evaluator id {resolved!r} is not supported")
    if explicit or configured or inferred:
        config["mode"] = resolved
    config["evaluator_id"] = resolved
    if evaluator_backend:
        config["evaluator_backend"] = _runtime_backend_name(evaluator_backend)
    if evaluator_model:
        config["evaluator_model"] = str(evaluator_model).strip()
    return config


def run_optimizer_preflight(
    package: TrainingPackage,
    *,
    optimizer_backend: str,
    target_backend: str,
    optimizer_model: str,
    target_model: str,
    evaluator_id: str = "",
    evaluator_backend: str = "",
    evaluator_model: str = "",
) -> PreflightResult:
    resolved_optimizer_backend = _runtime_backend_name(optimizer_backend or "openai_chat")
    resolved_target_backend = _runtime_target_backend_name(target_backend or "openai_chat")
    resolved_optimizer_model = str(optimizer_model or default_model_for_backend(resolved_optimizer_backend)).strip()
    resolved_target_model = str(target_model or default_model_for_backend(resolved_target_backend)).strip()
    evaluator_config = resolve_evaluator_config(
        package,
        evaluator_id=evaluator_id,
        evaluator_backend=evaluator_backend,
        evaluator_model=evaluator_model,
    )
    resolved_evaluator_backend = _runtime_backend_name(evaluator_config.get("evaluator_backend") or resolved_optimizer_backend)
    resolved_evaluator_model = str(evaluator_config.get("evaluator_model") or resolved_optimizer_model).strip()
    if evaluator_config.get("evaluator_backend"):
        evaluator_config["evaluator_backend"] = resolved_evaluator_backend
    if evaluator_config.get("evaluator_model"):
        evaluator_config["evaluator_model"] = resolved_evaluator_model

    _require_model("optimizer", resolved_optimizer_model)
    _require_model("target", resolved_target_model)
    _require_model("evaluator", resolved_evaluator_model)
    set_optimizer_backend(resolved_optimizer_backend)
    set_target_backend(resolved_target_backend)
    set_optimizer_deployment(resolved_optimizer_model)
    set_target_deployment(resolved_target_model)
    _run_optimizer_canary()
    _run_target_canary(resolved_target_backend, resolved_target_model)
    _run_evaluator_canary(evaluator_config, resolved_evaluator_backend, resolved_evaluator_model)
    _run_required_render_smoke_canary(evaluator_config)
    return PreflightResult(
        optimizer_backend=resolved_optimizer_backend,
        target_backend=resolved_target_backend,
        evaluator_backend=resolved_evaluator_backend,
        optimizer_model=resolved_optimizer_model,
        target_model=resolved_target_model,
        evaluator_model=resolved_evaluator_model,
        evaluator_config=evaluator_config,
    )


def _normal_evaluator_id(value: str) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    if normalized in {"judge", "llm", "llmjudge", "manual_judge", "manual_review", "pairwise"}:
        return "llm_judge"
    if normalized == "landing_page":
        return LANDING_PAGE_EVALUATOR_ID
    if normalized in {"deterministic", "mock"}:
        return "fixture"
    if normalized == "substring":
        return "contains"
    return normalized


def _runtime_backend_name(value: str | None) -> str:
    normalized = normalize_backend_name(value)
    if normalized == "azure_openai":
        return "openai_chat"
    return normalized


def _runtime_target_backend_name(value: str | None) -> str:
    normalized = normalize_target_backend_name(value)
    if normalized == "azure_openai":
        return "openai_chat"
    return normalized


def _configured_evaluator_id(config: dict[str, Any]) -> str:
    for key in ("mode", "evaluator_id", "id"):
        value = str(config.get(key) or "")
        if key == "mode" and _is_training_mode(value):
            continue
        if resolved := _normal_evaluator_id(value):
            return resolved
    driver = str(config.get("driver") or "").strip().lower().replace("-", "_")
    if driver and driver != "manual_review":
        return _normal_evaluator_id(driver)
    return ""


def _apply_evaluator_id_mode_override(config: dict[str, Any], override_config: dict[str, Any]) -> None:
    if "mode" in override_config and not _is_training_mode(str(override_config.get("mode") or "")):
        return
    if _is_training_mode(str(config.get("mode") or "")):
        config.pop("mode", None)
    driver = str(override_config.get("driver") or "").strip().lower().replace("-", "_")
    evaluator_id = _normal_evaluator_id(str(override_config.get("evaluator_id") or override_config.get("id") or ""))
    if not evaluator_id and driver and driver != "manual_review":
        evaluator_id = _normal_evaluator_id(driver)
    if evaluator_id:
        config["mode"] = evaluator_id


def _package_evaluator_config(package: TrainingPackage) -> dict[str, Any]:
    config = dict(package.evaluator_config) if isinstance(package.evaluator_config, dict) else {}
    if _is_training_mode(str(config.get("mode") or "")):
        config.pop("mode", None)
    return config


def _is_training_mode(value: str) -> bool:
    return str(value or "").strip().lower().replace("-", "_") in TRAINING_MODES


def _infer_evaluator_id(package: TrainingPackage) -> str:
    driver = ""
    if isinstance(package.evaluator_config, dict):
        driver = str(package.evaluator_config.get("driver") or "").strip().lower().replace("-", "_")
    if driver == "manual_review" and _package_uses_vue_preview(package):
        return LANDING_PAGE_EVALUATOR_ID
    return ""


def _evaluator_profile_config(package: TrainingPackage) -> dict[str, Any]:
    profile = package.evaluator_profile
    if profile is None:
        return {}
    config: dict[str, Any] = {}
    if profile.artifact_contract:
        config["artifact_contract"] = profile.artifact_contract
    if profile.preview_adapter:
        config["preview_adapter"] = profile.preview_adapter
    if profile.task_kind:
        config["task_kind"] = profile.task_kind
    if profile.profile_id:
        config["profile_id"] = profile.profile_id
    if profile.checks:
        config["checks"] = [check.to_dict() for check in profile.checks]
    if profile.judge is not None and profile.judge.model:
        config["evaluator_model"] = profile.judge.model
    if _profile_requires_landing_page_mode(config):
        config["mode"] = LANDING_PAGE_EVALUATOR_ID
    return config


def _profile_requires_landing_page_mode(config: dict[str, Any]) -> bool:
    profile_id = str(config.get("profile_id") or "").strip().lower()
    task_kind = str(config.get("task_kind") or "").strip().lower()
    artifact_contract = str(config.get("artifact_contract") or "").strip().lower()
    return (
        profile_id in {LANDING_PAGE_EVALUATOR_ID, "vue_landing_page_v1"}
        or task_kind == "vue_landing_page"
        or artifact_contract in {"vue_vite_bundle", "vue-vite-bundle"}
    )


def _package_uses_vue_preview(package: TrainingPackage) -> bool:
    for item in package.items:
        metadata = item.metadata if isinstance(item.metadata, dict) else {}
        fields = [
            metadata.get("artifact_type"),
            metadata.get("output_type"),
            metadata.get("brief"),
            item.title,
        ]
        text = " ".join(str(field or "").lower() for field in fields)
        if "vue-preview" in text or ("landing" in text and "page" in text):
            return True
    return False


def _require_model(label: str, model: str) -> None:
    if not str(model or "").strip():
        raise ValueError(f"{label} model is required")


def _run_optimizer_canary() -> None:
    response, _usage = chat_optimizer(
        system="You are a Gitmoot SkillOpt preflight optimizer canary.",
        user=f"Return the exact text: {OPTIMIZER_CANARY_TEXT}",
        max_completion_tokens=64,
        retries=1,
        stage="gitmoot_preflight_optimizer",
    )
    if str(response or "").strip() != OPTIMIZER_CANARY_TEXT:
        raise ValueError("optimizer canary returned unexpected response")


def _run_target_canary(target_backend: str, target_model: str) -> None:
    del target_model
    if is_target_chat_backend():
        response, _usage = chat_target(
            system="You are a Gitmoot SkillOpt preflight target canary.",
            user=f"Return the exact text: {TARGET_CANARY_TEXT}",
            max_completion_tokens=64,
            retries=1,
            stage="gitmoot_preflight_target",
        )
    elif is_target_exec_backend():
        with tempfile.TemporaryDirectory(prefix="gitmoot-skillopt-target-canary-") as work_dir:
            prepare_workspace(
                work_dir=work_dir,
                skill_md="# SkillOpt Target Canary\n\nReturn the requested canary text.",
                task_text=f"Return the exact text: {TARGET_CANARY_TEXT}",
            )
            response, _raw = run_target_exec(
                work_dir=work_dir,
                prompt=f"Read task.md and return exactly: {TARGET_CANARY_TEXT}",
                model=os.environ.get("TARGET_DEPLOYMENT") or default_model_for_backend(target_backend),
                timeout=120,
                allow_file_edits=False,
            )
    else:
        raise ValueError(f"target backend {target_backend!r} is not supported")
    if str(response or "").strip() != TARGET_CANARY_TEXT:
        raise ValueError("target canary returned unexpected response")


def _run_evaluator_canary(evaluator_config: dict[str, Any], evaluator_backend: str, evaluator_model: str) -> None:
    evaluator_id = _configured_evaluator_id(evaluator_config)
    if evaluator_id in {"fixture", "contains"}:
        return
    previous_backend = get_optimizer_backend()
    previous_model = os.environ.get("OPTIMIZER_DEPLOYMENT", "")
    try:
        set_optimizer_backend(evaluator_backend)
        set_optimizer_deployment(evaluator_model)
        raw, _usage = chat_optimizer(
            system=_evaluator_canary_system_prompt(evaluator_id),
            user=_evaluator_canary_user_prompt(evaluator_id),
            max_completion_tokens=1024,
            retries=1,
            stage="gitmoot_preflight_evaluator",
        )
    finally:
        set_optimizer_backend(previous_backend)
        if previous_model:
            set_optimizer_deployment(previous_model)
    parsed = extract_json(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"evaluator canary for {evaluator_id} did not return JSON")
    _validate_evaluator_canary_json(evaluator_id, parsed)


def _evaluator_canary_system_prompt(evaluator_id: str) -> str:
    if evaluator_id == LANDING_PAGE_EVALUATOR_ID:
        dimensions = ", ".join(LANDING_PAGE_DIMENSIONS)
        return (
            "Return only JSON for a landing page evaluator canary. "
            "Use keys hard, soft, dimension_scores, rationale, and fail_reason. "
            f"dimension_scores must include: {dimensions}. "
            "Use hard 1 and soft 0.9."
        )
    return "Return only JSON with keys hard, soft, fail_reason, and reasoning. Use hard 1 and soft 0.9."


def _evaluator_canary_user_prompt(evaluator_id: str) -> str:
    if evaluator_id == LANDING_PAGE_EVALUATOR_ID:
        return "Evaluate this canary landing page: responsive hero, CTA, footer, no overlap, relevant visuals, light motion."
    return "Evaluate this canary response as passing."


def _validate_evaluator_canary_json(evaluator_id: str, parsed: dict[str, Any]) -> None:
    hard = parsed.get("hard")
    soft = parsed.get("soft")
    if hard is None or soft is None:
        raise ValueError(f"evaluator canary for {evaluator_id} missing hard/soft scores")
    try:
        float(soft)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"evaluator canary for {evaluator_id} returned invalid soft score") from exc
    if evaluator_id != LANDING_PAGE_EVALUATOR_ID:
        return
    dimensions = parsed.get("dimension_scores")
    if not isinstance(dimensions, dict):
        raise ValueError("landing_page_v1 evaluator canary missing dimension_scores")
    missing = [dimension for dimension in LANDING_PAGE_DIMENSIONS if dimension not in dimensions]
    if missing:
        raise ValueError(f"landing_page_v1 evaluator canary missing dimension scores: {', '.join(missing)}")
    if not str(parsed.get("rationale") or "").strip():
        raise ValueError("landing_page_v1 evaluator canary missing rationale")


def _run_required_render_smoke_canary(evaluator_config: dict[str, Any]) -> None:
    if not _requires_required_render_smoke(evaluator_config):
        return
    if shutil.which("npm") is None:
        raise ValueError("required render_smoke check requires npm to be installed")
    sync_playwright = _import_sync_playwright()
    if sync_playwright is None:
        raise ValueError("required render_smoke check requires the Python Playwright package")
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            browser.close()
    except Exception as exc:
        raise ValueError("required render_smoke check requires Playwright Chromium to launch") from exc
    try:
        with tempfile.TemporaryDirectory(prefix="gitmoot-render-preflight-") as work_dir:
            _prepare_trusted_vue_render_deps(Path(work_dir), timeout=120)
    except Exception as exc:
        raise ValueError("required render_smoke check requires trusted Vue render dependencies") from exc


def _requires_required_render_smoke(evaluator_config: dict[str, Any]) -> bool:
    if bool(evaluator_config.get("require_vue_render_smoke")):
        return True
    checks = evaluator_config.get("checks")
    if not isinstance(checks, list):
        return False
    for check in checks:
        if not isinstance(check, dict):
            continue
        check_id = str(check.get("id") or "").strip().lower().replace("-", "_")
        check_type = str(check.get("type") or "").strip().lower().replace("-", "_")
        if (check_id == "render_smoke" or check_type == "render_smoke") and bool(check.get("required", False)):
            return True
    return False


def _import_sync_playwright() -> Any | None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None
    return sync_playwright
