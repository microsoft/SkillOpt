"""Gitmoot rollout evaluators."""

from __future__ import annotations

import json
import os
import re
from typing import Any

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
VUE_BUNDLE_REQUIRED_FILES = (
    "package.json",
    "index.html",
    "src/main.js",
    "src/App.vue",
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
    if _requires_vue_vite_bundle(item, config):
        artifact_check = _check_vue_vite_bundle(response)
        if artifact_check is not None:
            return artifact_check

    raw, _usage = _chat_evaluator(
        config,
        system=_landing_page_system_prompt(),
        user=_landing_page_user_prompt(item, response, config),
        max_completion_tokens=4096,
        retries=2,
        stage="gitmoot_landing_page_judge",
    )
    parsed = extract_json(raw)
    if not isinstance(parsed, dict):
        raise ValueError("landing_page_v1 judge did not return JSON")
    return _normalize_landing_page_score(parsed, raw=raw)


def _requires_vue_vite_bundle(item: dict[str, Any], config: dict[str, Any]) -> bool:
    if bool(config.get("require_vue_vite_bundle")):
        return True
    for source in (config, item.get("metadata") if isinstance(item.get("metadata"), dict) else {}):
        artifact_contract = str(source.get("artifact_contract") or source.get("output_contract") or "").strip().lower()
        if artifact_contract in {"vue_vite_bundle", "vue-vite-bundle"}:
            return True
        output_type = str(source.get("output_type") or "").strip().lower()
        if output_type in {"vue_vite_bundle", "vue-vite-bundle"}:
            return True
    return False


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
        "dimension_scores, rationale, and fail_reason. "
        f"dimension_scores must contain exactly these 0-to-1 dimensions: {dimensions}. "
        "hard must be 1 only when the landing page is suitable to promote after review. "
        "soft must be a 0-to-1 overall quality score. Penalize missing mobile responsiveness, "
        "missing footer, weak hero/CTA, irrelevant or absent visuals, missing requested motion, "
        "text overlap/readability problems, and failure to preserve human-ranked strengths."
    )


def _landing_page_user_prompt(item: dict[str, Any], response: str, config: dict[str, Any]) -> str:
    return "\n\n".join(
        [
            "## Landing Page Evaluation Config",
            json.dumps(config, indent=2, sort_keys=True),
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
            "## Generated Landing Page Response",
            response,
        ]
    )


def _normalize_landing_page_score(parsed: dict[str, Any], *, raw: str) -> dict[str, Any]:
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
    return {
        "hard": hard,
        "soft": soft,
        "fail_reason": "" if hard else fail_reason,
        "evaluator_id": LANDING_PAGE_EVALUATOR_ID,
        "evaluator_version": LANDING_PAGE_EVALUATOR_VERSION,
        "metadata": {
            "evaluator": LANDING_PAGE_EVALUATOR_ID,
            "evaluator_version": LANDING_PAGE_EVALUATOR_VERSION,
            "dimension_scores": dimensions,
            "rationale": rationale,
            "raw": raw[:1000],
        },
    }


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
