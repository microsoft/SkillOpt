"""Staging → eval → gate → human review → promote for Hermes skills.

This module intentionally does not call external models and never writes to a
live skill unless promotion is explicitly approved. It is a local governance
prototype around SkillOpt-generated candidate artifacts such as ``best_skill.md``.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from skillopt.utils.redaction import REDACTED, redact_secrets, redact_string

_SECRET_RE = re.compile(
    r"(Bearer\s+[A-Za-z0-9._~+/=-]{12,}|sk-[A-Za-z0-9][A-Za-z0-9._-]{12,}|gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})",
    re.IGNORECASE,
)
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)(api[_-]?key|access[_-]?token|auth[_-]?token|password|secret)\s*[=:]\s*[^\s,;]{8,}",
)
_DANGEROUS_RE = re.compile(
    r"\b(rm\s+-rf|sudo\b|curl\b.*\|\s*(?:sh|bash)|chmod\s+777|launchctl\s+load|systemctl\s+enable)\b",
    re.IGNORECASE,
)
_QMD_REF_RE = re.compile(r"qmd://[^\s)\]>'\"]+")


@dataclass(frozen=True)
class GateDecision:
    passed: bool
    reasons: list[str]
    scores: dict[str, float]


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "skill"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(redact_secrets(payload), indent=2, ensure_ascii=False) + "\n")


def _secret_hits(text: str) -> list[str]:
    """Return secret-like substrings without exposing them in reports."""
    return _SECRET_RE.findall(text) + _SECRET_ASSIGNMENT_RE.findall(text)


def stage_candidate(
    *,
    registry: Path,
    skill_name: str,
    candidate_path: Path,
    base_skill_path: Path | None = None,
    source: str = "skillopt",
) -> dict[str, Any]:
    """Copy a generated candidate into the staging registry and record lineage."""
    candidate_path = candidate_path.expanduser().resolve()
    if not candidate_path.exists():
        raise FileNotFoundError(candidate_path)
    skill_slug = _safe_name(skill_name)
    candidate_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{_sha256(candidate_path)[:12]}"
    stage_dir = registry / "staging" / skill_slug / candidate_id
    stage_dir.mkdir(parents=True, exist_ok=False)
    staged_candidate = stage_dir / "candidate.md"
    raw_candidate = candidate_path.read_text()
    stage_secret_hits = _secret_hits(raw_candidate)
    staged_candidate.write_text(redact_string(raw_candidate))

    manifest: dict[str, Any] = {
        "candidate_id": candidate_id,
        "created_at": _utc_now(),
        "skill_name": skill_name,
        "source": source,
        "candidate_path": str(staged_candidate),
        "candidate_sha256": _sha256(staged_candidate),
        "source_candidate_sha256": _sha256(candidate_path),
        "candidate_redacted": bool(stage_secret_hits),
        "stage_sensitive_hits": [REDACTED for _ in stage_secret_hits],
        "status": "staged",
    }
    if base_skill_path is not None:
        base = base_skill_path.expanduser().resolve()
        if base.exists():
            manifest.update({"base_skill_path": str(base), "base_sha256": _sha256(base)})
    _write_json(stage_dir / "manifest.json", manifest)
    return manifest


def _score_tool_grounding(text: str) -> float:
    fences = re.findall(r"```.*?```", text, flags=re.DOTALL)
    if not fences:
        return 1.0
    explained = sum(1 for block in fences if "用途" in block or "purpose" in block.lower())
    nearby = sum(
        1
        for block in fences
        if re.search(r"用途|Purpose|purpose", text[max(0, text.find(block) - 200) : text.find(block) + len(block) + 300])
    )
    return min(1.0, max(explained, nearby) / len(fences))


def _score_qmd_grounding(text: str) -> tuple[float, list[str]]:
    """Score whether a candidate cites or instructs use of local qmd knowledge."""
    references = sorted(set(_QMD_REF_RE.findall(text)))
    mentions_qmd_lookup = bool(re.search(r"\bqmd\s+(?:search|query|get|multi-get|status)\b", text, flags=re.IGNORECASE))
    if references and mentions_qmd_lookup:
        return 1.0, references
    if references or mentions_qmd_lookup:
        return 0.8, references
    return 0.8, references


def _score_regression_fixture(text: str, fixture_path: Path | None) -> tuple[float, dict[str, Any]]:
    """Evaluate a deterministic regression fixture against candidate text.

    Fixture format: {"tasks": [{"name": "...", "required_terms": ["..."]}]}
    A task passes when every required term appears in the candidate text.
    """
    if fixture_path is None:
        return 0.8, {"path": None, "total": 0, "passed": 0, "failed": []}
    fixture_path = fixture_path.expanduser().resolve()
    fixture = _read_json(fixture_path)
    tasks = fixture.get("tasks", [])
    if not isinstance(tasks, list):
        raise ValueError("regression fixture must contain a list at 'tasks'")
    text_lower = text.lower()
    failed: list[dict[str, Any]] = []
    for idx, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise ValueError(f"regression task #{idx} must be an object")
        required_terms = task.get("required_terms", [])
        if not isinstance(required_terms, list):
            raise ValueError(f"regression task #{idx} required_terms must be a list")
        missing = [str(term) for term in required_terms if str(term).lower() not in text_lower]
        if missing:
            failed.append({"name": task.get("name", f"task-{idx}"), "missing_terms": missing})
    total = len(tasks)
    passed = total - len(failed)
    score = 0.8 if total == 0 else passed / total
    return score, {"path": str(fixture_path), "total": total, "passed": passed, "failed": failed}


def evaluate_candidate(
    *,
    registry: Path,
    skill_name: str,
    candidate_id: str,
    regression_fixture: Path | None = None,
) -> dict[str, Any]:
    """Run static/offline evaluation for a staged candidate skill."""
    stage_dir = registry / "staging" / _safe_name(skill_name) / candidate_id
    candidate = stage_dir / "candidate.md"
    manifest_path = stage_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    text = candidate.read_text()
    base_text = ""
    if manifest.get("base_skill_path") and Path(manifest["base_skill_path"]).exists():
        base_text = Path(manifest["base_skill_path"]).read_text()

    has_frontmatter = text.startswith("---\n") and "\n---\n" in text[4:]
    has_description = "description:" in text[:1000].lower()
    secret_hits = _secret_hits(text) + list(manifest.get("stage_sensitive_hits", []))
    dangerous_hits = _DANGEROUS_RE.findall(text)
    qmd_grounding, qmd_references = _score_qmd_grounding(text)
    regression_score, regression_findings = _score_regression_fixture(text, regression_fixture)
    scores = {
        "correctness": 1.0 if has_frontmatter and has_description and len(text.strip()) >= 200 else 0.4,
        "tool_grounding": _score_tool_grounding(text),
        "qmd_grounding": qmd_grounding,
        "regression": regression_score,
        "safety": 1.0 if not secret_hits and not dangerous_hits else 0.0,
        "user_style": 1.0 if ("繁體" in text or "Traditional Chinese" in text or "用途" in text) else 0.7,
        "operational_quality": 1.0 if ("Verify" in text or "驗證" in text or "Verification" in text) else 0.6,
        "cost_latency": 1.0 if ("cost" in text.lower() or "latency" in text.lower() or "成本" in text or "延遲" in text) else 0.7,
    }
    diff = ""
    if base_text:
        diff = "".join(
            difflib.unified_diff(
                base_text.splitlines(keepends=True),
                text.splitlines(keepends=True),
                fromfile="base.md",
                tofile="candidate.md",
            )
        )
        (stage_dir / "candidate.diff").write_text(diff)

    report = {
        "candidate_id": candidate_id,
        "evaluated_at": _utc_now(),
        "scores": scores,
        "findings": {
            "has_frontmatter": has_frontmatter,
            "has_description": has_description,
            "secret_hits": [REDACTED for _ in secret_hits],
            "dangerous_patterns": dangerous_hits,
            "qmd_references": qmd_references,
            "regression": regression_findings,
            "diff_path": str(stage_dir / "candidate.diff") if diff else None,
        },
    }
    _write_json(stage_dir / "eval.json", report)
    manifest["status"] = "evaluated"
    _write_json(manifest_path, manifest)
    return report


def gate_candidate(
    *,
    registry: Path,
    skill_name: str,
    candidate_id: str,
    min_score: float = 0.8,
) -> GateDecision:
    """Decide whether a candidate can advance to human review."""
    stage_dir = registry / "staging" / _safe_name(skill_name) / candidate_id
    eval_report = _read_json(stage_dir / "eval.json")
    scores = {str(k): float(v) for k, v in eval_report["scores"].items()}
    reasons = [f"{name}={score:.2f} < {min_score:.2f}" for name, score in scores.items() if score < min_score]
    passed = not reasons
    decision = GateDecision(passed=passed, reasons=reasons, scores=scores)
    _write_json(
        stage_dir / "gate.json",
        {
            "candidate_id": candidate_id,
            "gated_at": _utc_now(),
            "passed": passed,
            "reasons": reasons,
            "scores": scores,
            "next_state": "human_review" if passed else "rejected",
        },
    )
    manifest = _read_json(stage_dir / "manifest.json")
    manifest["status"] = "human_review" if passed else "rejected"
    _write_json(stage_dir / "manifest.json", manifest)
    if not passed:
        rejected_dir = registry / "rejected" / _safe_name(skill_name) / candidate_id
        rejected_dir.parent.mkdir(parents=True, exist_ok=True)
        if rejected_dir.exists():
            shutil.rmtree(rejected_dir)
        shutil.copytree(stage_dir, rejected_dir)
    return decision


def write_review_request(*, registry: Path, skill_name: str, candidate_id: str) -> Path:
    """Create a human-readable review request for the staged candidate."""
    stage_dir = registry / "staging" / _safe_name(skill_name) / candidate_id
    manifest = _read_json(stage_dir / "manifest.json")
    eval_report = _read_json(stage_dir / "eval.json")
    gate = _read_json(stage_dir / "gate.json")
    review_path = stage_dir / "human_review.md"
    review_path.write_text(
        "# Human review request\n\n"
        f"- skill: {skill_name}\n"
        f"- candidate_id: {candidate_id}\n"
        f"- status: {manifest.get('status')}\n"
        f"- gate_passed: {gate.get('passed')}\n"
        f"- scores: {json.dumps(eval_report.get('scores', {}), ensure_ascii=False)}\n"
        f"- candidate: {manifest.get('candidate_path')}\n"
        f"- diff: {eval_report.get('findings', {}).get('diff_path')}\n\n"
        "## Human review checklist\n\n"
        "- [ ] Read candidate.md, not just this summary.\n"
        "- [ ] Read candidate.diff when a base skill exists.\n"
        "- [ ] Confirm qmd grounding references are relevant and non-sensitive.\n"
        "- [ ] Confirm regression fixture failures are absent or intentionally accepted.\n"
        "- [ ] Confirm safety gate found no secrets or dangerous commands.\n"
        "- [ ] Confirm promotion dry-run output before writing the live skill.\n\n"
        "Promotion requires an explicit approver name.\n"
    )
    return review_path


def promote_candidate(
    *,
    registry: Path,
    skill_name: str,
    candidate_id: str,
    live_skill_path: Path,
    approved_by: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Promote a gated candidate into a live skill path after explicit approval."""
    if not approved_by.strip():
        raise ValueError("approved_by is required for promotion")
    stage_dir = registry / "staging" / _safe_name(skill_name) / candidate_id
    gate = _read_json(stage_dir / "gate.json")
    if not gate.get("passed"):
        raise ValueError("candidate did not pass gate")
    review_path = stage_dir / "human_review.md"
    if not dry_run and not review_path.exists():
        raise ValueError("human_review.md is required before promotion")
    candidate = stage_dir / "candidate.md"
    live_skill_path = live_skill_path.expanduser().resolve()
    if dry_run:
        return {
            "candidate_id": candidate_id,
            "skill_name": skill_name,
            "dry_run": True,
            "would_write": str(live_skill_path),
            "candidate_sha256": _sha256(candidate),
            "approved_by": approved_by,
        }
    archive_dir = registry / "archive" / _safe_name(skill_name) / candidate_id
    archive_dir.mkdir(parents=True, exist_ok=True)
    if live_skill_path.exists():
        shutil.copy2(live_skill_path, archive_dir / "previous_live.md")
    live_skill_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate, live_skill_path)
    record = {
        "candidate_id": candidate_id,
        "skill_name": skill_name,
        "dry_run": False,
        "promoted_at": _utc_now(),
        "approved_by": approved_by,
        "live_skill_path": str(live_skill_path),
        "live_sha256": _sha256(live_skill_path),
        "previous_live_archive": str(archive_dir / "previous_live.md") if (archive_dir / "previous_live.md").exists() else None,
    }
    _write_json(stage_dir / "promotion.json", record)
    manifest = _read_json(stage_dir / "manifest.json")
    manifest["status"] = "promoted"
    _write_json(stage_dir / "manifest.json", manifest)
    return record


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hermes skill evolution governance prototype")
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--registry", type=Path, required=True)
    common.add_argument("--skill-name", required=True)
    stage = sub.add_parser("stage", parents=[common])
    stage.add_argument("--candidate", type=Path, required=True)
    stage.add_argument("--base", type=Path)
    stage.add_argument("--source", default="skillopt")
    evaluate = sub.add_parser("eval", parents=[common])
    evaluate.add_argument("--candidate-id", required=True)
    evaluate.add_argument("--regression-fixture", type=Path)
    gate = sub.add_parser("gate", parents=[common])
    gate.add_argument("--candidate-id", required=True)
    gate.add_argument("--min-score", type=float, default=0.8)
    review = sub.add_parser("review", parents=[common])
    review.add_argument("--candidate-id", required=True)
    promote = sub.add_parser("promote", parents=[common])
    promote.add_argument("--candidate-id", required=True)
    promote.add_argument("--live-skill-path", type=Path, required=True)
    promote.add_argument("--approved-by", required=True)
    promote.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "stage":
        result = stage_candidate(
            registry=args.registry,
            skill_name=args.skill_name,
            candidate_path=args.candidate,
            base_skill_path=args.base,
            source=args.source,
        )
    elif args.command == "eval":
        result = evaluate_candidate(
            registry=args.registry,
            skill_name=args.skill_name,
            candidate_id=args.candidate_id,
            regression_fixture=args.regression_fixture,
        )
    elif args.command == "gate":
        decision = gate_candidate(
            registry=args.registry,
            skill_name=args.skill_name,
            candidate_id=args.candidate_id,
            min_score=args.min_score,
        )
        result = {"passed": decision.passed, "reasons": decision.reasons, "scores": decision.scores}
    elif args.command == "review":
        result = {"review_path": str(write_review_request(registry=args.registry, skill_name=args.skill_name, candidate_id=args.candidate_id))}
    elif args.command == "promote":
        result = promote_candidate(
            registry=args.registry,
            skill_name=args.skill_name,
            candidate_id=args.candidate_id,
            live_skill_path=args.live_skill_path,
            approved_by=args.approved_by,
            dry_run=args.dry_run,
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
