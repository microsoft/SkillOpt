"""Gitmoot rollout evaluators."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from skillopt.model import chat_optimizer, get_optimizer_backend, set_optimizer_backend, set_optimizer_deployment
from skillopt.model.common import default_model_for_backend
from skillopt.utils import extract_json

LANDING_PAGE_EVALUATOR_ID = "landing_page_v1"
LANDING_PAGE_EVALUATOR_VERSION = "v1"
LANDING_PAGE_DIMENSIONS = (
    "mobile_responsiveness",
    "footer_presence_clarity",
    "hero_quality",
    "cta_clarity",
    "visual_images_relevance",
    "animation_motion_quality",
    "text_overlap_readability",
    "ranked_strength_preservation",
)
LANDING_PAGE_TASK_METADATA_KEYS = (
    "source",
    "run_id",
    "review_id",
    "issue_number",
    "target_repo",
    "profile_id",
    "task_kind",
    "item_id",
    "option_id",
)
VUE_BUNDLE_REQUIRED_FILES = (
    "package.json",
    "index.html",
    "src/main.js",
    "src/App.vue",
)
TRUSTED_VUE_RENDER_PACKAGE_JSON = (
    '{"type":"module",'
    '"dependencies":{"@vitejs/plugin-vue":"5.2.1","vite":"5.4.11","vue":"3.5.13"}}'
)
TRUSTED_VITE_CONFIG = (
    "import { defineConfig } from 'vite';\n"
    "import vue from '@vitejs/plugin-vue';\n\n"
    "export default defineConfig({ base: './', plugins: [vue()] });\n"
)
_APP_VUE_FORBIDDEN_PATTERNS = (
    ("script_tag", re.compile(r"<\s*script\b", re.IGNORECASE)),
    (
        "import_statement",
        re.compile(r"(?:^|[;{}\n]\s*)import\s+(?:[\w*{}\s,]+?\s+from\s+['\"]|['\"][^'\"]+['\"])"),
    ),
    ("dynamic_import", re.compile(r"\bimport\s*\(")),
    ("require_call", re.compile(r"\brequire\s*\(", re.IGNORECASE)),
    ("import_meta", re.compile(r"\bimport\.meta\b", re.IGNORECASE)),
    ("css_import", re.compile(r"@import\b", re.IGNORECASE)),
    ("css_url", re.compile(r"\burl\s*\(", re.IGNORECASE)),
)
_VITE_BUILD_COMMAND_RE = re.compile(r"(?:^|[;&|]\s*)(?:npx\s+)?vite\s+build(?:\s|$)")
_A_TAG_RE = re.compile(r"<a\b[^>]*>", re.IGNORECASE)
_HREF_RE = re.compile(r"(?:^|[\s<])(?:v-bind:|:)?href\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^\s>]+))", re.IGNORECASE)
_JS_IMPORT_SPEC_RE = re.compile(
    r"(?:\b(?:import|export)\s*(?:[^'\"\n;]*?\s*from\s*)?['\"]([^'\"]+)['\"]|"
    r"\bimport\s*\(\s*['\"`]([^'\"`]+)['\"`]\s*\)|"
    r"\brequire\s*\(\s*['\"`]([^'\"`]+)['\"`]\s*\)|"
    r"\bimport\.meta\.glob\s*\(\s*['\"`]([^'\"`]+)['\"`])"
)
_CSS_IMPORT_SPEC_RE = re.compile(r"@import\s+(?:url\()?['\"]?([^'\"\s)]+)", re.IGNORECASE)
_CSS_URL_SPEC_RE = re.compile(r"\burl\s*\(\s*['\"]?([^'\"\s)]+)", re.IGNORECASE)
_HTML_SCRIPT_RE = re.compile(r"<script\b(?P<attrs>[^>]*)>(?P<body>.*?)</script\s*>", re.IGNORECASE | re.DOTALL)
_HTML_MODULE_TYPE_RE = re.compile(r"\btype\s*=\s*(?:['\"]module['\"]|module(?:\s|$))", re.IGNORECASE)
_ALLOWED_RENDER_BARE_IMPORTS = {"vue"}


class VueRenderEnvironmentError(RuntimeError):
    """Local render tooling is unavailable before the candidate can be scored."""


def evaluate_response(item: dict[str, Any], response: str, evaluator_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Evaluate one Gitmoot response.

    Supported deterministic modes are intended for fixtures and CI. When no
    supported explicit mode is present, an LLM judge compares the response
    against source, baseline, candidate, and feedback context.
    """
    config = evaluator_config if isinstance(evaluator_config, dict) else {}
    mode = str(config.get("mode") or item.get("metadata", {}).get("evaluator_mode") or "").strip().lower()
    if mode in {"fixture", "deterministic", "mock"}:
        return _fixture_score(item)
    if mode in {"contains", "substring"}:
        return _contains_score(item, response, config)
    if mode in {LANDING_PAGE_EVALUATOR_ID, "landing-page-v1", "landing_page"}:
        return _landing_page_score(item, response, config)
    return _judge_score(item, response, config)


def _fixture_score(item: dict[str, Any]) -> dict[str, Any]:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    hard = bool(metadata.get("expected_hard", True))
    soft = float(metadata.get("expected_soft", 1.0 if hard else 0.0))
    return {
        "hard": 1 if hard else 0,
        "soft": max(0.0, min(1.0, soft)),
        "fail_reason": "" if hard else str(metadata.get("fail_reason") or "fixture evaluator marked this item failed"),
        "metadata": {"evaluator": "fixture"},
    }


def _contains_score(item: dict[str, Any], response: str, config: dict[str, Any]) -> dict[str, Any]:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    required = str(config.get("required_text") or metadata.get("required_text") or "").strip()
    if not required:
        return {
            "hard": 0,
            "soft": 0.0,
            "fail_reason": "contains evaluator requires required_text",
            "metadata": {"evaluator": "contains"},
        }
    ok = required.lower() in response.lower()
    return {
        "hard": 1 if ok else 0,
        "soft": 1.0 if ok else 0.0,
        "fail_reason": "" if ok else f"response did not contain required text {required!r}",
        "metadata": {"evaluator": "contains", "required_text": required},
    }


def _judge_score(item: dict[str, Any], response: str, config: dict[str, Any]) -> dict[str, Any]:
    system = (
        "You are evaluating a Gitmoot SkillOpt candidate response. "
        "Return only JSON with keys hard, soft, fail_reason, and reasoning. "
        "hard must be 0 or 1. soft must be a number from 0 to 1."
    )
    user = "\n\n".join(
        [
            "## Evaluation Config",
            json.dumps(config, indent=2, sort_keys=True),
            "## Task Prompt",
            str(item.get("prompt") or ""),
            "## Candidate Response",
            response,
        ]
    )
    raw, _usage = _chat_evaluator(
        config,
        system=system,
        user=user,
        max_completion_tokens=2048,
        retries=2,
        stage="gitmoot_judge",
    )
    parsed = extract_json(raw)
    if not isinstance(parsed, dict):
        return {
            "hard": 0,
            "soft": 0.0,
            "fail_reason": "judge did not return JSON",
            "metadata": {"evaluator": "llm_judge", "raw": raw[:1000]},
        }
    hard = _parse_hard(parsed.get("hard"))
    try:
        soft = float(parsed.get("soft", hard))
    except (TypeError, ValueError):
        soft = float(hard)
    soft = max(0.0, min(1.0, soft))
    fail_reason = str(parsed.get("fail_reason") or "")
    return {
        "hard": hard,
        "soft": soft,
        "fail_reason": "" if hard else fail_reason or "judge marked this item failed",
        "metadata": {
            "evaluator": "llm_judge",
            "judge_derived": True,
            "reasoning": str(parsed.get("reasoning") or ""),
        },
    }


def _landing_page_score(item: dict[str, Any], response: str, config: dict[str, Any]) -> dict[str, Any]:
    render_result: dict[str, Any] | None = None
    requires_vue_bundle = _requires_vue_vite_bundle(item, config)
    if requires_vue_bundle:
        artifact_check = _check_vue_vite_bundle(response)
        if artifact_check is not None:
            return artifact_check
        render_enabled, render_required = _vue_render_smoke_policy(config)
        if render_enabled:
            render_result = _run_vue_render_smoke(response, item, {**config, "_render_smoke_required": render_required})
            if render_required and render_result.get("hard") == 0:
                return render_result

    check_context = _landing_page_check_context(requires_vue_bundle=requires_vue_bundle, render_result=render_result)
    raw, _usage = _chat_evaluator(
        config,
        system=_landing_page_system_prompt(),
        user=_landing_page_user_prompt(item, response, config, check_context),
        max_completion_tokens=4096,
        retries=2,
        stage="gitmoot_landing_page_judge",
    )
    parsed = extract_json(raw)
    if not isinstance(parsed, dict):
        raise ValueError("landing_page_v1 judge did not return JSON")
    score = _normalize_landing_page_score(parsed, raw=raw, check_context=check_context)
    if render_result is not None:
        _attach_render_metadata(score, render_result)
    return score


def _requires_vue_vite_bundle(item: dict[str, Any], config: dict[str, Any]) -> bool:
    if bool(config.get("require_vue_vite_bundle")):
        return True
    if _vue_render_smoke_policy(config)[0]:
        return True
    for source in (config, item.get("metadata") if isinstance(item.get("metadata"), dict) else {}):
        artifact_contract = str(source.get("artifact_contract") or source.get("output_contract") or "").strip().lower()
        if artifact_contract in {"vue_vite_bundle", "vue-vite-bundle"}:
            return True
        output_type = str(source.get("output_type") or "").strip().lower()
        if output_type in {"vue_vite_bundle", "vue-vite-bundle"}:
            return True
    return False


def _vue_render_smoke_policy(config: dict[str, Any]) -> tuple[bool, bool]:
    if bool(config.get("require_vue_render_smoke")):
        return True, True
    checks = config.get("checks")
    if not isinstance(checks, list):
        return False, False
    render_enabled = False
    render_required = False
    for check in checks:
        if not isinstance(check, dict):
            continue
        check_id = str(check.get("id") or "").strip().lower().replace("-", "_")
        check_type = str(check.get("type") or "").strip().lower().replace("-", "_")
        if check_id == "render_smoke" or check_type == "render_smoke":
            render_enabled = True
            render_required = render_required or bool(check.get("required", False))
    return render_enabled, render_required


def _check_vue_vite_bundle(response: str) -> dict[str, Any] | None:
    parsed = extract_json(response)
    failures: list[dict[str, Any]] = []
    evidence: list[str] = []
    if not isinstance(parsed, dict):
        return _vue_bundle_failure(
            [
                _failed_check(
                    "vue_vite_bundle.json",
                    "Generated response must be a JSON object containing a Vue/Vite preview bundle.",
                    ["response did not contain a parseable JSON object"],
                )
            ]
        )

    files = parsed.get("files")
    if not isinstance(files, list) or not files:
        return _vue_bundle_failure(
            [
                _failed_check(
                    "vue_vite_bundle.files",
                    "Vue/Vite preview bundle must include a non-empty files array.",
                    ["files missing or empty"],
                )
            ]
        )

    file_map: dict[str, str] = {}
    for index, file_entry in enumerate(files, start=1):
        if not isinstance(file_entry, dict):
            failures.append(
                _failed_check(
                    "vue_vite_bundle.files",
                    f"File entry {index} must be an object.",
                    [f"files[{index - 1}] is {type(file_entry).__name__}"],
                )
            )
            continue
        path = str(file_entry.get("path") or "").strip()
        if not path or "content" not in file_entry:
            failures.append(
                _failed_check(
                    "vue_vite_bundle.files",
                    f"File entry {index} must include path and content.",
                    [f"files[{index - 1}] missing path or content"],
                )
            )
            continue
        if not _is_safe_vue_bundle_path(path):
            failures.append(
                _failed_check(
                    "vue_vite_bundle.file_path",
                    "Vue/Vite preview bundle file paths must be relative safe paths.",
                    [f"files[{index - 1}].path is unsafe: {path!r}"],
                )
            )
            continue
        content = file_entry["content"]
        if not isinstance(content, str):
            failures.append(
                _failed_check(
                    "vue_vite_bundle.files",
                    f"File entry {index} content must be a string.",
                    [f"files[{index - 1}].content is {type(content).__name__}"],
                )
            )
            continue
        file_map[path] = content

    missing = [path for path in VUE_BUNDLE_REQUIRED_FILES if path not in file_map]
    if missing:
        failures.append(
            _failed_check(
                "vue_vite_bundle.required_files",
                "Vue/Vite preview bundle is missing required files.",
                [f"missing {path}" for path in missing],
            )
        )

    if "package.json" in file_map:
        package_failure = _check_package_json(file_map["package.json"])
        if package_failure is not None:
            failures.append(package_failure)

    if "src/App.vue" in file_map:
        failures.extend(_check_app_vue(file_map["src/App.vue"]))

    href_failures = _check_local_href_anchors(file_map)
    failures.extend(href_failures)

    if failures:
        for failure in failures:
            evidence.extend(str(item) for item in failure.get("evidence", []))
        return _vue_bundle_failure(failures, evidence=evidence)
    return None


def _run_vue_render_smoke(response: str, item: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    parsed = extract_json(response)
    file_map = _vue_bundle_file_map(parsed if isinstance(parsed, dict) else {})
    if "src/App.vue" not in file_map:
        return _vue_render_failure(
            "vue_render_smoke.bundle_unavailable",
            "Vue/Vite render smoke requires a validated bundle with src/App.vue.",
            ["src/App.vue missing after artifact validation"],
        )
    import_safety_failure = _check_vue_render_source_imports(file_map)
    if import_safety_failure is not None:
        return import_safety_failure
    sync_playwright = _import_sync_playwright()
    if sync_playwright is None:
        return _vue_render_environment_unavailable(
            "python package playwright is unavailable",
            required=bool(config.get("_render_smoke_required")),
        )

    render_config = {**config, "_render_smoke_response_hash": _response_fingerprint(response)}
    artifact_dir = _render_artifact_dir(item, render_config)
    timeout = int(config.get("render_smoke_timeout_seconds") or 120)
    with tempfile.TemporaryDirectory(prefix="gitmoot-vue-render-") as work_dir:
        work_path = Path(work_dir)
        try:
            _write_vue_render_workspace(work_path, file_map)
            vite_bin = _prepare_trusted_vue_render_deps(work_path, timeout)
            _run_render_command([str(vite_bin), "build", "--config", str(work_path / "vite.config.mjs")], work_path, timeout)
            dist_index = work_path / "dist" / "index.html"
            if not dist_index.is_file():
                return _vue_render_failure(
                    "vue_render_smoke.build_output",
                    "Vue/Vite build output is missing dist/index.html.",
                    ["dist/index.html missing after trusted Vite build"],
                )
            return _smoke_check_dist(sync_playwright, dist_index, artifact_dir)
        except VueRenderEnvironmentError as exc:
            return _vue_render_environment_unavailable(str(exc), required=bool(config.get("_render_smoke_required")))
        except Exception as exc:
            return _vue_render_failure(
                "vue_render_smoke.build_failed",
                "Vue/Vite render smoke build failed.",
                [str(exc)],
            )


def _vue_bundle_file_map(parsed: dict[str, Any]) -> dict[str, str]:
    files = parsed.get("files")
    if not isinstance(files, list):
        return {}
    file_map: dict[str, str] = {}
    for file_entry in files:
        if not isinstance(file_entry, dict):
            continue
        path = str(file_entry.get("path") or "").strip()
        content = file_entry.get("content")
        if path and isinstance(content, str):
            file_map[path] = content
    return file_map


def _check_vue_render_source_imports(file_map: dict[str, str]) -> dict[str, Any] | None:
    failed_checks: list[dict[str, Any]] = []
    for path, content in file_map.items():
        if not _is_safe_vue_render_workspace_path(path):
            continue
        if path.endswith((".js", ".mjs", ".ts", ".vue")):
            for specifier in _iter_js_import_specifiers(content):
                if not _is_safe_vue_render_import_specifier(path, specifier):
                    failed_checks.append(
                        _failed_check(
                            "vue_render_smoke.import_safety",
                            "Vue render smoke source imports must stay inside the submitted runtime bundle.",
                            [f"{path} imports unsafe specifier {specifier!r}"],
                        )
                    )
        if path.endswith(".html"):
            for specifier in _iter_html_module_import_specifiers(content):
                if not _is_safe_vue_render_import_specifier(path, specifier):
                    failed_checks.append(
                        _failed_check(
                            "vue_render_smoke.import_safety",
                            "Vue render smoke HTML module imports must stay inside the submitted runtime bundle.",
                            [f"{path} imports unsafe specifier {specifier!r}"],
                        )
                    )
        if path.endswith((".css", ".vue")):
            for specifier in _iter_css_resource_specifiers(content):
                if not _is_safe_vue_render_resource_specifier(path, specifier):
                    failed_checks.append(
                        _failed_check(
                            "vue_render_smoke.import_safety",
                            "Vue render smoke CSS resources must stay inside the submitted runtime bundle.",
                            [f"{path} references unsafe resource {specifier!r}"],
                        )
                    )
    if not failed_checks:
        return None
    evidence = [item for failure in failed_checks for item in failure.get("evidence", [])]
    return _vue_render_failure(
        failed_checks[0]["check"],
        failed_checks[0]["reason"],
        evidence,
        failed_checks=failed_checks,
        optimizer_hint="Keep imports and CSS resources relative to src/ or public/, and do not import host files, external URLs, or arbitrary packages.",
    )


def _iter_js_import_specifiers(content: str) -> list[str]:
    specifiers: list[str] = []
    for match in _JS_IMPORT_SPEC_RE.finditer(content):
        specifier = next((group for group in match.groups() if group), "")
        if specifier:
            specifiers.append(specifier)
    return specifiers


def _iter_html_module_import_specifiers(content: str) -> list[str]:
    specifiers: list[str] = []
    for match in _HTML_SCRIPT_RE.finditer(content):
        attrs = match.group("attrs") or ""
        if _HTML_MODULE_TYPE_RE.search(attrs):
            specifiers.extend(_iter_js_import_specifiers(match.group("body") or ""))
    return specifiers


def _iter_css_resource_specifiers(content: str) -> list[str]:
    specifiers: list[str] = []
    for pattern in (_CSS_IMPORT_SPEC_RE, _CSS_URL_SPEC_RE):
        for match in pattern.finditer(content):
            specifier = match.group(1).strip()
            if specifier:
                specifiers.append(specifier)
    return specifiers


def _is_safe_vue_render_import_specifier(source_path: str, specifier: str) -> bool:
    specifier = specifier.strip()
    if not specifier:
        return False
    lowered = specifier.lower()
    if lowered.startswith(("http:", "https:", "ws:", "wss:", "data:", "file:", "blob:")):
        return False
    clean_specifier = re.split(r"[?#]", specifier, maxsplit=1)[0]
    if not clean_specifier:
        return False
    if clean_specifier in _ALLOWED_RENDER_BARE_IMPORTS:
        return True
    if clean_specifier.startswith("/"):
        public_path = clean_specifier.lstrip("/")
        return _is_safe_vue_render_workspace_path(public_path) or _is_safe_vue_render_workspace_path(
            f"public/{public_path}"
        )
    if clean_specifier.startswith("."):
        source_parts = source_path.strip().replace("\\", "/").split("/")[:-1]
        target_parts: list[str] = []
        for part in [*source_parts, *clean_specifier.split("/")]:
            if part in {"", "."}:
                continue
            if part == "..":
                if not target_parts:
                    return False
                target_parts.pop()
                continue
            target_parts.append(part)
        return _is_safe_vue_render_workspace_path("/".join(target_parts))
    return False


def _is_safe_vue_render_resource_specifier(source_path: str, specifier: str) -> bool:
    specifier = specifier.strip()
    if not specifier:
        return False
    lowered = specifier.lower()
    if lowered.startswith(("http:", "https:", "ws:", "wss:", "data:", "file:", "blob:")):
        return False
    clean_specifier = re.split(r"[?#]", specifier, maxsplit=1)[0]
    if not clean_specifier:
        return False
    if clean_specifier.startswith(("/", ".")):
        return _is_safe_vue_render_import_specifier(source_path, clean_specifier)
    return _is_safe_vue_render_import_specifier(source_path, f"./{clean_specifier}")


def _write_vue_render_workspace(work_path: Path, file_map: dict[str, str]) -> None:
    for relative_path, content in file_map.items():
        if not _is_safe_vue_render_workspace_path(relative_path):
            continue
        target = work_path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    (work_path / "package.json").write_text(TRUSTED_VUE_RENDER_PACKAGE_JSON, encoding="utf-8")
    (work_path / "vite.config.mjs").write_text(TRUSTED_VITE_CONFIG, encoding="utf-8")


def _prepare_trusted_vue_render_deps(work_path: Path, timeout: int) -> Path:
    deps_path = _trusted_vue_render_deps_cache_dir()
    _ensure_private_render_cache_dir(deps_path.parent)
    _ensure_private_render_cache_dir(deps_path)
    ready_path = deps_path / ".gitmoot-ready"
    package_path = deps_path / "package.json"
    if not ready_path.is_file():
        package_path.write_text(TRUSTED_VUE_RENDER_PACKAGE_JSON, encoding="utf-8")
        try:
            _run_render_command(["npm", "install", "--ignore-scripts"], deps_path, timeout)
        except subprocess.SubprocessError as exc:
            raise VueRenderEnvironmentError(f"trusted Vue render dependencies could not be installed: {exc}") from exc
    node_modules = deps_path / "node_modules"
    vite_bin = node_modules / ".bin" / "vite"
    if not vite_bin.is_file():
        raise VueRenderEnvironmentError("trusted Vite binary was not installed")
    ready_path.write_text("ok\n", encoding="utf-8")
    (work_path / "node_modules").symlink_to(node_modules, target_is_directory=True)
    return vite_bin


def _trusted_vue_render_deps_cache_dir() -> Path:
    configured = str(os.environ.get("GITMOOT_RENDER_DEPS_CACHE") or "").strip()
    if configured:
        cache_root = Path(configured).expanduser()
    else:
        cache_home = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache").expanduser()
        cache_root = cache_home / "gitmoot-skillopt" / "vue-render-deps"
    package_hash = hashlib.sha256(TRUSTED_VUE_RENDER_PACKAGE_JSON.encode("utf-8")).hexdigest()[:16]
    return cache_root / package_hash


def _ensure_private_render_cache_dir(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not hasattr(os, "getuid"):
        return
    try:
        stat_result = path.stat()
    except OSError as exc:
        raise VueRenderEnvironmentError(f"render dependency cache is unavailable: {path}") from exc
    if stat_result.st_uid != os.getuid():
        raise VueRenderEnvironmentError(f"render dependency cache is not owned by the current user: {path}")
    if stat_result.st_mode & 0o077:
        path.chmod(stat_result.st_mode & ~0o077)


def _run_render_command(command: list[str], cwd: Path, timeout: int) -> None:
    executable = shutil.which(command[0])
    if executable is None:
        raise VueRenderEnvironmentError(f"{command[0]} is not available")
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if completed.returncode != 0:
        output = "\n".join(part for part in (completed.stdout.strip(), completed.stderr.strip()) if part)
        raise subprocess.SubprocessError(output or f"{' '.join(command)} failed with exit code {completed.returncode}")


def _smoke_check_dist(sync_playwright: Any, dist_index: Path, artifact_dir: Path) -> dict[str, Any]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    screenshots: list[dict[str, Any]] = []
    stage_status = [{"stage": "render_smoke", "status": "passed"}]
    viewports = (
        ("desktop", {"width": 1366, "height": 768}),
        ("mobile", {"width": 390, "height": 844}),
    )
    failed_checks: list[dict[str, Any]] = []
    evidence: list[str] = []
    with sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch()
        except Exception as exc:
            raise VueRenderEnvironmentError(f"Playwright Chromium is not available: {exc}") from exc
        try:
            for label, viewport in viewports:
                context = browser.new_context(viewport=viewport)
                _block_external_browser_requests(context, dist_index.parent)
                page = context.new_page()
                try:
                    page.goto(dist_index.as_uri(), wait_until="networkidle")
                    page.screenshot(path=str(artifact_dir / f"{label}.png"), full_page=True)
                    screenshots.append(
                        {
                            "label": label,
                            "path": str(artifact_dir / f"{label}.png"),
                            "viewport": viewport,
                        }
                    )
                    failed_checks.extend(_page_render_failures(page, label))
                finally:
                    context.close()
        finally:
            browser.close()
    if failed_checks:
        for failure in failed_checks:
            evidence.extend(str(item) for item in failure.get("evidence", []))
        return _vue_render_failure(
            failed_checks[0]["check"],
            failed_checks[0]["reason"],
            evidence,
            failed_checks=failed_checks,
            screenshots=screenshots,
        )
    return {
        "hard": 1,
        "soft": 1.0,
        "profile_id": "vue_landing_page_v1",
        "task_kind": "vue_landing_page",
        "dimension_scores": {"render_smoke": 1.0},
        "stage_status": stage_status,
        "metadata": {
            "render_smoke": {"screenshots": screenshots},
            "stage_status": stage_status,
            "dimension_scores": {"render_smoke": 1.0},
        },
    }


def _page_render_failures(page: Any, label: str) -> list[dict[str, Any]]:
    checks = page.evaluate(
        """() => {
            const doc = document.documentElement;
            const body = document.body;
            const visible = Array.from(document.querySelectorAll('body *')).some((el) => {
                const style = window.getComputedStyle(el);
                const rect = el.getBoundingClientRect();
                return style.display !== 'none' && style.visibility !== 'hidden' && rect.width > 0 && rect.height > 0;
            });
            const hero = document.querySelector('main, header, section, h1, h2');
            const heroRect = hero ? hero.getBoundingClientRect() : null;
            const footer = document.querySelector('footer');
            if (footer) {
                footer.scrollIntoView({block: 'end'});
            }
            const footerRect = footer ? footer.getBoundingClientRect() : null;
            return {
                nonblank: Boolean(body && body.children.length > 0 && visible),
                horizontalOverflow: doc.scrollWidth > window.innerWidth + 2,
                heroVisible: Boolean(heroRect && heroRect.width > 0 && heroRect.height > 0 && heroRect.bottom > 0 && heroRect.top < window.innerHeight),
                footerReachable: Boolean(footerRect && footerRect.width > 0 && footerRect.height > 0),
                scrollWidth: doc.scrollWidth,
                viewportWidth: window.innerWidth
            };
        }"""
    )
    failures: list[dict[str, Any]] = []
    if not checks.get("nonblank"):
        failures.append(_failed_check("vue_render_smoke.nonblank", f"{label} render is blank.", [f"{label} page has no visible body content"]))
    if checks.get("horizontalOverflow"):
        failures.append(
            _failed_check(
                "vue_render_smoke.horizontal_overflow",
                f"{label} render has horizontal overflow.",
                [f"{label} scrollWidth={checks.get('scrollWidth')} viewportWidth={checks.get('viewportWidth')}"],
            )
        )
    if not checks.get("heroVisible"):
        failures.append(_failed_check("vue_render_smoke.hero_visible", f"{label} render does not show a hero/main section.", [f"{label} hero/main section not visible"]))
    if not checks.get("footerReachable"):
        failures.append(_failed_check("vue_render_smoke.footer_reachable", f"{label} render has no reachable footer.", [f"{label} footer not found or not reachable"]))
    return failures


def _render_artifact_dir(item: dict[str, Any], config: dict[str, Any]) -> Path:
    configured = str(config.get("render_artifact_dir") or config.get("artifact_dir") or "").strip()
    if configured:
        return Path(configured).expanduser()
    item_id = _safe_render_artifact_segment(str(item.get("id") or "item"))
    response_hash = str(config.get("_render_smoke_response_hash") or "unknown").strip() or "unknown"
    return Path(tempfile.gettempdir()) / "gitmoot-render-smoke" / item_id / response_hash


def _response_fingerprint(response: str) -> str:
    return hashlib.sha256(response.encode("utf-8", errors="replace")).hexdigest()[:12]


def _block_external_browser_requests(context: Any, allowed_file_root: Path) -> None:
    allowed_root = allowed_file_root.resolve()
    if hasattr(context, "add_init_script"):
        context.add_init_script(
            """
            (() => {
              const BlockedWebSocket = function () {
                throw new Error('WebSocket connections are blocked during Gitmoot render smoke checks');
              };
              BlockedWebSocket.CLOSED = 3;
              BlockedWebSocket.CLOSING = 2;
              BlockedWebSocket.CONNECTING = 0;
              BlockedWebSocket.OPEN = 1;
              Object.defineProperty(window, 'WebSocket', { configurable: false, writable: false, value: BlockedWebSocket });
            })();
            """
        )

    def handle_route(route: Any) -> None:
        url = str(route.request.url)
        if url.startswith(("data:", "blob:", "about:")):
            route.continue_()
            return
        if url.startswith("file:") and _file_url_within_root(url, allowed_root):
            route.continue_()
            return
        route.abort()

    context.route("**/*", handle_route)


def _file_url_within_root(url: str, allowed_root: Path) -> bool:
    parsed = urlparse(url)
    if parsed.scheme != "file":
        return False
    try:
        path = Path(unquote(parsed.path)).resolve()
    except (OSError, RuntimeError, ValueError):
        return False
    return path == allowed_root or allowed_root in path.parents


def _is_safe_vue_bundle_path(path: str) -> bool:
    normalized = path.strip().replace("\\", "/")
    if not normalized or normalized.startswith("/"):
        return False
    parts = tuple(normalized.split("/"))
    if not parts:
        return False
    blocked_parts = {"", ".", "..", "node_modules", ".gitmoot-render-deps"}
    return not any(part in blocked_parts for part in parts)


def _is_safe_vue_render_workspace_path(path: str) -> bool:
    if not _is_safe_vue_bundle_path(path):
        return False
    normalized = path.strip().replace("\\", "/")
    if normalized == "index.html":
        return True
    if normalized == "package.json":
        return False
    parts = tuple(normalized.split("/"))
    return len(parts) > 1 and parts[0] in {"src", "public"}


def _safe_render_artifact_segment(value: str) -> str:
    segment = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip(".-")
    return segment[:80] or "item"


def _import_sync_playwright() -> Any | None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None
    return sync_playwright


def _vue_render_failure(
    check: str,
    reason: str,
    evidence: list[str],
    *,
    optimizer_hint: str = "Fix the Vue/Vite build or rendered layout before sending this landing page to the visual judge.",
    failed_checks: list[dict[str, Any]] | None = None,
    screenshots: list[dict[str, Any]] | None = None,
    primary_reason: str = "vue_render_smoke_failed",
) -> dict[str, Any]:
    failed_checks = failed_checks or [_failed_check(check, reason, evidence)]
    failure = {
        "primary_reason": primary_reason,
        "human_reason": reason,
        "optimizer_hint": optimizer_hint,
        "failed_checks": failed_checks,
        "evidence": evidence,
        "stage_status": [{"stage": "render_smoke", "status": "failed"}],
    }
    render_smoke_metadata: dict[str, Any] = {"failure": failure}
    if screenshots:
        render_smoke_metadata["screenshots"] = screenshots
    metadata: dict[str, Any] = {
        "evaluator": LANDING_PAGE_EVALUATOR_ID,
        "evaluator_version": LANDING_PAGE_EVALUATOR_VERSION,
        "dimension_scores": {"render_smoke": 0.0},
        "failure": failure,
        "render_smoke": render_smoke_metadata,
        "stage_status": failure["stage_status"],
    }
    return {
        "hard": 0,
        "soft": 0.0,
        "fail_reason": reason,
        "profile_id": "vue_landing_page_v1",
        "task_kind": "vue_landing_page",
        "dimension_scores": {"render_smoke": 0.0},
        "failure": failure,
        "stage_status": failure["stage_status"],
        "evaluator_id": LANDING_PAGE_EVALUATOR_ID,
        "evaluator_version": LANDING_PAGE_EVALUATOR_VERSION,
        "metadata": metadata,
    }


def _vue_render_environment_unavailable(reason: str, *, required: bool) -> dict[str, Any]:
    if not required:
        return _vue_render_skipped(f"Render smoke environment unavailable: {reason}")
    return _vue_render_failure(
        "vue_render_smoke.environment_unavailable",
        "Vue render smoke environment is unavailable.",
        [reason],
        optimizer_hint="Install npm, trusted Vue render dependencies, Playwright, and Playwright Chromium before running required render smoke checks.",
        primary_reason="vue_render_smoke_environment_unavailable",
    )


def _vue_render_skipped(reason: str) -> dict[str, Any]:
    stage_status = [{"stage": "render_smoke", "status": "skipped", "details": {"reason": reason}}]
    return {
        "hard": 1,
        "soft": 1.0,
        "profile_id": "vue_landing_page_v1",
        "task_kind": "vue_landing_page",
        "dimension_scores": {},
        "stage_status": stage_status,
        "metadata": {
            "render_smoke": {"skipped": True, "reason": reason},
            "stage_status": stage_status,
            "dimension_scores": {},
        },
    }


def _attach_render_metadata(score: dict[str, Any], render_result: dict[str, Any]) -> None:
    render_metadata = render_result.get("metadata") if isinstance(render_result.get("metadata"), dict) else {}
    metadata = score.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
        score["metadata"] = metadata
    if "render_smoke" in render_metadata:
        metadata["render_smoke"] = render_metadata["render_smoke"]
    stage_status = list(render_metadata.get("stage_status") or [])
    if stage_status:
        score["stage_status"] = [*stage_status, *list(score.get("stage_status") or [])]
        metadata["stage_status"] = score["stage_status"]
    dimensions = dict(metadata.get("dimension_scores") or {})
    dimensions.update(render_metadata.get("dimension_scores") or {})
    metadata["dimension_scores"] = dimensions
    score["dimension_scores"] = dimensions


def _check_package_json(content: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return _failed_check(
            "vue_vite_bundle.package_json",
            "package.json must be valid JSON.",
            ["package.json is not valid JSON"],
        )
    if not isinstance(parsed, dict):
        return _failed_check(
            "vue_vite_bundle.package_json",
            "package.json must be a JSON object.",
            ["package.json root is not an object"],
        )
    scripts = parsed.get("scripts")
    build_script = scripts.get("build") if isinstance(scripts, dict) else None
    if not isinstance(build_script, str) or _VITE_BUILD_COMMAND_RE.search(build_script) is None:
        return _failed_check(
            "vue_vite_bundle.package_json.build",
            "package.json must define a build script that runs vite build.",
            ["scripts.build missing vite build"],
        )
    return None


def _check_app_vue(content: str) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for pattern_id, pattern in _APP_VUE_FORBIDDEN_PATTERNS:
        match = pattern.search(content)
        if match is None:
            continue
        failures.append(
            _failed_check(
                f"vue_vite_bundle.app_vue.{pattern_id}",
                "src/App.vue contains forbidden code or external-loading patterns.",
                [f"src/App.vue matched {pattern.pattern!r}: {match.group(0)!r}"],
            )
        )
    return failures


def _check_local_href_anchors(files: dict[str, str]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for path, content in files.items():
        for anchor in _A_TAG_RE.finditer(content):
            href_match = _HREF_RE.search(anchor.group(0))
            if href_match is None:
                continue
            href = _normalize_href_value(next((group for group in href_match.groups() if group is not None), ""))
            if href.startswith("#"):
                continue
            failures.append(
                _failed_check(
                    "vue_vite_bundle.local_hrefs",
                    "All href attributes must be local # anchors.",
                    [f"{path} has href={href!r}"],
                )
            )
    return failures


def _normalize_href_value(value: str) -> str:
    href = value.strip()
    for _ in range(2):
        if len(href) >= 2 and href[0] in {"'", '"'} and href[-1] == href[0]:
            href = href[1:-1].strip()
    return href


def _failed_check(check: str, reason: str, evidence: list[str]) -> dict[str, Any]:
    return {
        "check": check,
        "severity": "hard_blocker",
        "reason": reason,
        "evidence": evidence,
    }


def _vue_bundle_failure(failed_checks: list[dict[str, Any]], *, evidence: list[str] | None = None) -> dict[str, Any]:
    first_reason = failed_checks[0]["reason"] if failed_checks else "Vue/Vite preview bundle failed validation."
    evidence = evidence or [item for check in failed_checks for item in check.get("evidence", [])]
    failure = {
        "primary_reason": "vue_vite_bundle_contract_failed",
        "human_reason": first_reason,
        "optimizer_hint": (
            "Return a JSON Vue/Vite preview bundle with package.json, index.html, src/main.js, "
            "and src/App.vue. Keep src/App.vue template/style-only, use vite build, and make links local # anchors."
        ),
        "failed_checks": failed_checks,
        "evidence": evidence,
        "stage_status": [{"stage": "artifact_contract", "status": "failed"}],
    }
    return {
        "hard": 0,
        "soft": 0.0,
        "fail_reason": first_reason,
        "profile_id": "vue_landing_page_v1",
        "task_kind": "vue_landing_page",
        "dimension_scores": {"artifact_contract": 0.0},
        "failure": failure,
        "stage_status": failure["stage_status"],
        "evaluator_id": LANDING_PAGE_EVALUATOR_ID,
        "evaluator_version": LANDING_PAGE_EVALUATOR_VERSION,
        "metadata": {
            "evaluator": LANDING_PAGE_EVALUATOR_ID,
            "evaluator_version": LANDING_PAGE_EVALUATOR_VERSION,
            "dimension_scores": {"artifact_contract": 0.0},
            "failure": failure,
            "stage_status": failure["stage_status"],
        },
    }


def _chat_evaluator(config: dict[str, Any], **kwargs) -> tuple[str, dict[str, Any]]:
    evaluator_backend = str(config.get("evaluator_backend") or "").strip()
    evaluator_model = str(config.get("evaluator_model") or "").strip()
    if not evaluator_backend and not evaluator_model:
        return chat_optimizer(**kwargs)

    previous_backend = get_optimizer_backend()
    previous_model = os.environ.get("OPTIMIZER_DEPLOYMENT", "")
    try:
        if evaluator_backend:
            set_optimizer_backend(evaluator_backend)
        if evaluator_model:
            set_optimizer_deployment(evaluator_model)
        return chat_optimizer(**kwargs)
    finally:
        set_optimizer_backend(previous_backend)
        set_optimizer_deployment(previous_model or default_model_for_backend(previous_backend))


def _landing_page_system_prompt() -> str:
    dimensions = ", ".join(LANDING_PAGE_DIMENSIONS)
    return (
        "You are evaluating a generated landing page for Gitmoot SkillOpt. "
        "Return only JSON with keys evaluator_id, evaluator_version, hard, soft, "
        "dimension_scores, rationale, fail_reason, and failure. "
        f"dimension_scores must contain exactly these 0-to-1 dimensions: {dimensions}. "
        "hard must be 1 only when the landing page is suitable to promote after review. "
        "soft must be a 0-to-1 overall quality score. Penalize missing mobile responsiveness, "
        "missing footer, weak hero/CTA, irrelevant or absent visuals, missing requested motion, "
        "text overlap/readability problems, and failure to preserve human-ranked strengths. "
        "When hard is 0, failure must include primary_reason, human_reason, optimizer_hint, "
        "failed_checks, and evidence so the optimizer can reuse the rejection."
    )


def _landing_page_user_prompt(
    item: dict[str, Any],
    response: str,
    config: dict[str, Any],
    check_context: dict[str, Any] | None = None,
) -> str:
    return "\n\n".join(
        [
            "## Landing Page Evaluation Config",
            json.dumps(config, indent=2, sort_keys=True),
            "## Deterministic And Render Check Context",
            _json_for_prompt(check_context or {}),
            "## Rubric",
            "\n".join(
                [
                    "- mobile_responsiveness: Works on mobile without overflow or unusable layout.",
                    "- footer_presence_clarity: Includes a clear footer with useful links or closing context.",
                    "- hero_quality: Hero is clear, polished, product-relevant, and visually strong.",
                    "- cta_clarity: Primary and final calls to action are obvious and well placed.",
                    "- visual_images_relevance: Graphics/images help explain the product and are not generic filler.",
                    "- animation_motion_quality: Motion exists when requested and supports comprehension.",
                    "- text_overlap_readability: Text does not overlap, occlude, or become unreadable.",
                    "- ranked_strength_preservation: Preserves strengths called out in human rankings/feedback.",
                ]
            ),
            "## Task Prompt, Artifacts, And Human Feedback",
            str(item.get("prompt") or ""),
            "## Structured Task Context",
            _json_for_prompt(_landing_page_task_context(item)),
            "## Generated Landing Page Response",
            response,
        ]
    )


def _landing_page_check_context(*, requires_vue_bundle: bool, render_result: dict[str, Any] | None) -> dict[str, Any]:
    context: dict[str, Any] = {
        "artifact_contract": {
            "stage": "artifact_contract",
            "status": "passed" if requires_vue_bundle else "not_required",
            "contract": "vue_vite_bundle",
            "required_files": list(VUE_BUNDLE_REQUIRED_FILES) if requires_vue_bundle else [],
        },
        "render_smoke": {"stage": "render_smoke", "status": "not_enabled"},
        "stage_status": [],
        "dimension_scores": {},
    }
    if render_result is None:
        return context
    render_metadata = render_result.get("metadata") if isinstance(render_result.get("metadata"), dict) else {}
    render_smoke = render_metadata.get("render_smoke") if isinstance(render_metadata.get("render_smoke"), dict) else {}
    stage_status = list(render_result.get("stage_status") or render_metadata.get("stage_status") or [])
    dimensions = dict(render_result.get("dimension_scores") or render_metadata.get("dimension_scores") or {})
    context["render_smoke"] = {
        "stage": "render_smoke",
        "status": _last_stage_status(stage_status, default="passed" if render_result.get("hard") else "failed"),
        "soft": render_result.get("soft"),
        "fail_reason": str(render_result.get("fail_reason") or ""),
        "failure": render_result.get("failure") or render_smoke.get("failure"),
        "screenshots": render_smoke.get("screenshots") or [],
        "skipped": bool(render_smoke.get("skipped", False)),
        "reason": render_smoke.get("reason"),
    }
    context["stage_status"] = stage_status
    context["dimension_scores"] = dimensions
    return context


def _landing_page_task_context(item: dict[str, Any]) -> dict[str, Any]:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    safe_metadata = {key: metadata[key] for key in LANDING_PAGE_TASK_METADATA_KEYS if key in metadata}
    return {
        "id": item.get("id"),
        "title": item.get("title"),
        "task_type": item.get("task_type"),
        "task_description": item.get("task_description"),
        "source": item.get("source") or metadata.get("source"),
        "metadata": safe_metadata,
        "artifacts": item.get("artifacts") or metadata.get("artifacts"),
        "feedback_events": item.get("feedback_events") or metadata.get("feedback_events"),
        "ranked_feedback_events": item.get("ranked_feedback_events") or metadata.get("ranked_feedback_events"),
        "ranked_artifacts": item.get("ranked_artifacts") or metadata.get("ranked_artifacts"),
    }


def _json_for_prompt(value: Any) -> str:
    return json.dumps(_clip_for_prompt(value), indent=2, sort_keys=True)


def _clip_for_prompt(value: Any, *, depth: int = 0) -> Any:
    if depth >= 5:
        return "[truncated nested value]"
    if isinstance(value, dict):
        clipped: dict[str, Any] = {}
        items = list(value.items())
        for key, nested in items[:24]:
            clipped[str(key)] = _clip_for_prompt(nested, depth=depth + 1)
        if len(items) > 24:
            clipped["__truncated_keys__"] = len(items) - 24
        return clipped
    if isinstance(value, list):
        clipped_items = [_clip_for_prompt(item, depth=depth + 1) for item in value[:24]]
        if len(value) > 24:
            clipped_items.append({"__truncated_items__": len(value) - 24})
        return clipped_items
    if isinstance(value, str):
        limit = 4000 if depth <= 2 else 1200
        if len(value) > limit:
            return f"{value[:limit]}... [truncated {len(value) - limit} chars]"
    return value


def _last_stage_status(stage_status: list[Any], *, default: str) -> str:
    for entry in reversed(stage_status):
        if isinstance(entry, dict) and entry.get("stage") == "render_smoke":
            status = str(entry.get("status") or "").strip()
            if status:
                return status
    return default


def _normalize_landing_page_score(
    parsed: dict[str, Any],
    *,
    raw: str,
    check_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    hard = _parse_landing_hard(parsed.get("hard"))
    soft = _parse_score(parsed.get("soft"), "soft")
    dimensions = _parse_dimension_scores(parsed.get("dimension_scores"))
    rationale = str(parsed.get("rationale") or parsed.get("reasoning") or "").strip()
    fail_reason = str(parsed.get("fail_reason") or "").strip()
    if hard not in {0, 1}:
        raise ValueError("landing_page_v1 hard must be 0 or 1")
    if not rationale:
        raise ValueError("landing_page_v1 rationale is required")
    if hard == 0 and not fail_reason:
        fail_reason = "landing_page_v1 judge marked this landing page below promotion quality"
    judge_stage = {"stage": "llm_judge", "status": "passed" if hard else "failed"}
    stage_status = [judge_stage]
    failure = _landing_page_judge_failure(parsed, fail_reason=fail_reason, rationale=rationale, dimensions=dimensions) if hard == 0 else None
    metadata: dict[str, Any] = {
        "evaluator": LANDING_PAGE_EVALUATOR_ID,
        "evaluator_version": LANDING_PAGE_EVALUATOR_VERSION,
        "dimension_scores": dimensions,
        "rationale": rationale,
        "raw": raw[:1000],
        "stage_status": stage_status,
        "check_context": check_context or {},
    }
    if failure is not None:
        metadata["failure"] = failure
    return {
        "hard": hard,
        "soft": soft,
        "fail_reason": "" if hard else fail_reason,
        "profile_id": "vue_landing_page_v1",
        "task_kind": "vue_landing_page",
        "dimension_scores": dimensions,
        "stage_status": stage_status,
        "evaluator_id": LANDING_PAGE_EVALUATOR_ID,
        "evaluator_version": LANDING_PAGE_EVALUATOR_VERSION,
        **({"failure": failure} if failure is not None else {}),
        "metadata": metadata,
    }


def _landing_page_judge_failure(
    parsed: dict[str, Any],
    *,
    fail_reason: str,
    rationale: str,
    dimensions: dict[str, float],
) -> dict[str, Any]:
    supplied = parsed.get("failure") if isinstance(parsed.get("failure"), dict) else {}
    low_dimensions = [
        {"check": f"landing_page_v1.{dimension}", "score": score}
        for dimension, score in dimensions.items()
        if score < 0.7
    ]
    primary_reason = str(
        supplied.get("primary_reason")
        or parsed.get("primary_reason")
        or (low_dimensions[0]["check"].replace("landing_page_v1.", "") if low_dimensions else "")
        or "landing_page_judge_rejected"
    )
    human_reason = str(supplied.get("human_reason") or fail_reason or rationale).strip()
    optimizer_hint = str(supplied.get("optimizer_hint") or parsed.get("optimizer_hint") or "").strip()
    if not optimizer_hint:
        if low_dimensions:
            weak = ", ".join(item["check"].replace("landing_page_v1.", "") for item in low_dimensions[:4])
            optimizer_hint = f"Improve the rejected landing page dimensions before trying again: {weak}."
        else:
            optimizer_hint = "Use the judge rationale and human feedback to revise the landing page before trying again."
    fallback_check = {
        "check": "landing_page_v1.llm_judge",
        "severity": "soft_quality_rejection",
        "reason": human_reason,
        "evidence": [rationale],
        "metadata": {"dimension_scores": low_dimensions},
    }
    failed_checks = _normalize_landing_page_failed_checks(supplied.get("failed_checks"), fallback_check)
    evidence = _normalize_string_list(supplied.get("evidence")) or [item for item in (fail_reason, rationale) if item]
    return {
        "primary_reason": primary_reason,
        "human_reason": human_reason,
        "optimizer_hint": optimizer_hint,
        "failed_checks": failed_checks,
        "evidence": evidence,
        "stage_status": [{"stage": "llm_judge", "status": "failed"}],
    }


def _normalize_landing_page_failed_checks(value: Any, fallback_check: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return [fallback_check]
    normalized: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        check = str(item.get("check") or "").strip()
        reason = str(item.get("reason") or "").strip()
        if not check and not reason:
            continue
        failed_check: dict[str, Any] = {
            "check": check or str(fallback_check.get("check")),
            "severity": str(item.get("severity") or fallback_check.get("severity") or "soft_quality_rejection"),
            "reason": reason or str(fallback_check.get("reason") or ""),
            "evidence": _normalize_string_list(item.get("evidence")),
        }
        metadata = item.get("metadata")
        if isinstance(metadata, dict):
            failed_check["metadata"] = metadata
        normalized.append(failed_check)
    return normalized or [fallback_check]


def _normalize_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            normalized.append(text)
    return normalized


def _parse_score(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"landing_page_v1 {label} must be numeric")
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"landing_page_v1 {label} must be numeric") from exc
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"landing_page_v1 {label} must be between 0 and 1")
    return score


def _parse_landing_hard(value: Any) -> int:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, int | float) and not isinstance(value, bool):
        if float(value) in {0.0, 1.0}:
            return int(float(value))
    if isinstance(value, str):
        normalized = value.strip().lower()
        try:
            numeric = float(normalized)
        except ValueError:
            numeric = None
        if numeric in {0.0, 1.0}:
            return int(numeric)
        if normalized in {"true", "yes", "pass", "passed", "success"}:
            return 1
        if normalized in {"false", "no", "fail", "failed", "failure"}:
            return 0
    raise ValueError("landing_page_v1 hard must be 0 or 1")


def _parse_dimension_scores(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ValueError("landing_page_v1 dimension_scores must be an object")
    missing = [dimension for dimension in LANDING_PAGE_DIMENSIONS if dimension not in value]
    if missing:
        raise ValueError(f"landing_page_v1 dimension_scores missing: {', '.join(missing)}")
    return {
        dimension: _parse_score(value[dimension], f"dimension_scores.{dimension}")
        for dimension in LANDING_PAGE_DIMENSIONS
    }


def _parse_hard(value: Any) -> int:
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, int | float):
        return 1 if float(value) > 0 else 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "pass", "passed", "success"}:
            return 1
        if normalized in {"0", "false", "no", "fail", "failed", "failure"}:
            return 0
        try:
            return 1 if float(normalized) > 0 else 0
        except ValueError:
            return 0
    return 0
