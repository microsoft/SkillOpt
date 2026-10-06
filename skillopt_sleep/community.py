"""Privacy-preserving community rule exchange for SkillOpt-Sleep."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Tuple

from skillopt_sleep.backend import Backend
from skillopt_sleep.gate import select_gate_score
from skillopt_sleep.memory import apply_edits_detailed
from skillopt_sleep.replay import aggregate_scores, replay_batch
from skillopt_sleep.staging import redact_secrets
from skillopt_sleep.types import EditRecord, ReplayResult, TaskRecord

RULE_MANIFEST_SCHEMA = "skillopt.community-rules"
RULE_MANIFEST_VERSION = 1
_MAX_RULES = 100
_MAX_RULE_CHARS = 4000
_MAX_RATIONALE_CHARS = 2000
_RULE_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


class CommunityRuleError(ValueError):
    """A community rule manifest or export source is invalid."""


@dataclass(frozen=True)
class ObservedEffect:
    metric: str
    baseline: float
    candidate: float
    delta: float
    sample_size: int
    scope: str = "candidate_set"


@dataclass(frozen=True)
class CommunityRule:
    id: str
    category: str
    rule: str
    rationale: str
    observed_effect: ObservedEffect


@dataclass(frozen=True)
class RuleManifest:
    license: str
    rules: Tuple[CommunityRule, ...]
    schema: str = RULE_MANIFEST_SCHEMA
    schema_version: int = RULE_MANIFEST_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "license": self.license,
            "rules": [asdict(rule) for rule in self.rules],
        }


@dataclass
class CommunityGateResult:
    accepted: bool
    baseline_score: float
    candidate_score: float
    new_skill: str
    accepted_edits: List[EditRecord]
    rejected_edits: List[EditRecord]
    unmatched_edits: List[EditRecord]
    trials: List[Dict[str, Any]]


def _object(value: Any, label: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise CommunityRuleError(f"{label} must be a JSON object")
    return value


def _exact_keys(value: Dict[str, Any], expected: set[str], label: str) -> None:
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unknown:
        raise CommunityRuleError(f"{label} has unknown fields: {', '.join(unknown)}")
    if missing:
        raise CommunityRuleError(f"{label} is missing fields: {', '.join(missing)}")


def _text(value: Any, label: str, max_chars: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CommunityRuleError(f"{label} must be non-empty text")
    text = value.strip()
    if len(text) > max_chars:
        raise CommunityRuleError(f"{label} exceeds {max_chars} characters")
    return text


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CommunityRuleError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise CommunityRuleError(f"{label} must be a finite number")
    return result


def _parse_effect(payload: Any, label: str) -> ObservedEffect:
    value = _object(payload, label)
    _exact_keys(
        value,
        {"metric", "baseline", "candidate", "delta", "sample_size", "scope"},
        label,
    )
    metric = _text(value["metric"], f"{label}.metric", 128)
    baseline = _number(value["baseline"], f"{label}.baseline")
    candidate = _number(value["candidate"], f"{label}.candidate")
    delta = _number(value["delta"], f"{label}.delta")
    sample_size = value["sample_size"]
    if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size < 1:
        raise CommunityRuleError(f"{label}.sample_size must be a positive integer")
    if value["scope"] != "candidate_set":
        raise CommunityRuleError(f"{label}.scope must be 'candidate_set'")
    if not math.isclose(candidate - baseline, delta, abs_tol=1e-9):
        raise CommunityRuleError(f"{label}.delta must equal candidate - baseline")
    return ObservedEffect(metric, baseline, candidate, delta, sample_size)


def parse_rule_manifest(payload: Any) -> RuleManifest:
    value = _object(payload, "manifest")
    _exact_keys(value, {"schema", "schema_version", "license", "rules"}, "manifest")
    if value["schema"] != RULE_MANIFEST_SCHEMA:
        raise CommunityRuleError("manifest has an unsupported schema")
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != RULE_MANIFEST_VERSION
    ):
        raise CommunityRuleError("manifest has an unsupported schema version")
    license_id = _text(value["license"], "manifest.license", 128)
    raw_rules = value["rules"]
    if not isinstance(raw_rules, list) or not raw_rules:
        raise CommunityRuleError("manifest.rules must be a non-empty array")
    if len(raw_rules) > _MAX_RULES:
        raise CommunityRuleError(f"manifest.rules exceeds {_MAX_RULES} entries")

    rules: List[CommunityRule] = []
    seen_ids: set[str] = set()
    for index, raw_rule in enumerate(raw_rules):
        label = f"manifest.rules[{index}]"
        row = _object(raw_rule, label)
        _exact_keys(
            row,
            {"id", "category", "rule", "rationale", "observed_effect"},
            label,
        )
        rule_id = _text(row["id"], f"{label}.id", 128)
        if not _RULE_ID_RE.fullmatch(rule_id):
            raise CommunityRuleError(f"{label}.id contains unsupported characters")
        if rule_id in seen_ids:
            raise CommunityRuleError(f"duplicate rule id: {rule_id}")
        seen_ids.add(rule_id)
        rules.append(CommunityRule(
            id=rule_id,
            category=_text(row["category"], f"{label}.category", 128),
            rule=_text(row["rule"], f"{label}.rule", _MAX_RULE_CHARS),
            rationale=_text(
                row["rationale"], f"{label}.rationale", _MAX_RATIONALE_CHARS
            ),
            observed_effect=_parse_effect(
                row["observed_effect"], f"{label}.observed_effect"
            ),
        ))
    return RuleManifest(license=license_id, rules=tuple(rules))


def load_rule_manifest(path: str) -> RuleManifest:
    source = os.path.abspath(os.path.expanduser(path))
    with open(source, encoding="utf-8") as handle:
        return parse_rule_manifest(json.load(handle))


def write_rule_manifest(path: str, manifest: RuleManifest) -> str:
    validated = parse_rule_manifest(manifest.to_dict())
    output = os.path.abspath(os.path.expanduser(path))
    parent = os.path.dirname(output)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(validated.to_dict(), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return output


def _rule_id(category: str, rule: str) -> str:
    digest = hashlib.sha256(f"{category}\0{rule}".encode("utf-8")).hexdigest()
    return f"rule-{digest[:16]}"


def export_rules_from_staging(
    staging_dir: str,
    output_path: str,
    *,
    category: str,
    license_id: str,
) -> tuple[str, RuleManifest]:
    """Export accepted skill additions without copying transcripts or task data."""
    category = _text(category, "category", 128)
    license_id = _text(license_id, "license", 128)
    report_path = os.path.join(os.path.abspath(staging_dir), "report.json")
    with open(report_path, encoding="utf-8") as handle:
        report = _object(json.load(handle), "staging report")
    if report.get("accepted") is not True:
        raise CommunityRuleError("staging report has no accepted candidate to export")
    baseline = _number(report.get("baseline_score"), "report.baseline_score")
    candidate = _number(report.get("candidate_score"), "report.candidate_score")
    sample_size = report.get("n_tasks")
    if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size < 1:
        raise CommunityRuleError("report.n_tasks must be a positive integer")
    raw_edits = report.get("edits")
    if not isinstance(raw_edits, list):
        raise CommunityRuleError("report.edits must be an array")

    effect = ObservedEffect(
        metric="local_gate_score",
        baseline=baseline,
        candidate=candidate,
        delta=candidate - baseline,
        sample_size=sample_size,
    )
    rules: List[CommunityRule] = []
    for index, raw_edit in enumerate(raw_edits):
        edit = _object(raw_edit, f"report.edits[{index}]")
        if edit.get("target") != "skill" or edit.get("op") != "add":
            continue
        rule = _text(edit.get("content"), f"report.edits[{index}].content", _MAX_RULE_CHARS)
        rationale = _text(
            edit.get("rationale"),
            f"report.edits[{index}].rationale",
            _MAX_RATIONALE_CHARS,
        )
        if redact_secrets(rule) != rule or redact_secrets(rationale) != rationale:
            raise CommunityRuleError(
                f"report.edits[{index}] contains secret-shaped text; review it manually"
            )
        rules.append(CommunityRule(
            id=_rule_id(category, rule),
            category=category,
            rule=rule,
            rationale=rationale,
            observed_effect=effect,
        ))
    if not rules:
        raise CommunityRuleError("staging report has no accepted skill additions to export")
    manifest = RuleManifest(license=license_id, rules=tuple(rules))
    return write_rule_manifest(output_path, manifest), manifest


def _task_deltas(
    tasks: List[TaskRecord],
    baseline_pairs: List[Tuple[TaskRecord, ReplayResult]],
    candidate_pairs: List[Tuple[TaskRecord, ReplayResult]],
    metric: str,
    mixed_weight: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for task, (_, baseline), (_, candidate) in zip(
        tasks, baseline_pairs, candidate_pairs
    ):
        before = select_gate_score(
            baseline.hard, baseline.soft, metric, mixed_weight
        )
        after = select_gate_score(
            candidate.hard, candidate.soft, metric, mixed_weight
        )
        scores_are_finite = math.isfinite(before) and math.isfinite(after)
        rows.append({
            "task_id": task.id,
            "baseline_score": before,
            "candidate_score": after,
            "status": (
                "regressed"
                if not scores_are_finite
                else "improved"
                if after > before
                else "regressed"
                if after < before
                else "unchanged"
            ),
            "scores_are_finite": scores_are_finite,
        })
    return rows


def gate_community_rules(
    backend: Backend,
    tasks: List[TaskRecord],
    skill: str,
    memory: str,
    manifest: RuleManifest,
    *,
    gate_metric: str = "mixed",
    gate_mixed_weight: float = 0.5,
) -> CommunityGateResult:
    """Put each imported rule through a strict, no-regression local gate."""
    if not tasks:
        raise CommunityRuleError("local gate requires at least one reviewed task")
    current_skill = skill
    current_pairs = replay_batch(backend, tasks, current_skill, memory)
    base_hard, base_soft = aggregate_scores(current_pairs)
    original_score = select_gate_score(
        base_hard, base_soft, gate_metric, gate_mixed_weight
    )
    if not math.isfinite(original_score):
        raise CommunityRuleError("local gate produced a non-finite baseline score")
    current_score = original_score
    accepted_edits: List[EditRecord] = []
    rejected_edits: List[EditRecord] = []
    unmatched_edits: List[EditRecord] = []
    trials: List[Dict[str, Any]] = []

    for rule in manifest.rules:
        edit = EditRecord(
            target="skill",
            op="add",
            content=rule.rule,
            rationale=rule.rationale,
        )
        candidate_skill, applied, unmatched = apply_edits_detailed(
            current_skill, [edit]
        )
        if unmatched:
            unmatched_edits.extend(unmatched)
            trials.append({
                "rule_id": rule.id,
                "category": rule.category,
                "baseline_score": current_score,
                "candidate_score": current_score,
                "accepted": False,
                "reason": "duplicate_or_empty",
                "task_deltas": [],
            })
            continue
        candidate_pairs = replay_batch(backend, tasks, candidate_skill, memory)
        cand_hard, cand_soft = aggregate_scores(candidate_pairs)
        candidate_score = select_gate_score(
            cand_hard, cand_soft, gate_metric, gate_mixed_weight
        )
        task_deltas = _task_deltas(
            tasks,
            current_pairs,
            candidate_pairs,
            gate_metric,
            gate_mixed_weight,
        )
        regressed = any(row["status"] == "regressed" for row in task_deltas)
        accepted = (
            math.isfinite(candidate_score)
            and candidate_score > current_score
            and not regressed
        )
        trials.append({
            "rule_id": rule.id,
            "category": rule.category,
            "baseline_score": current_score,
            "candidate_score": candidate_score,
            "accepted": accepted,
            "reason": "improved" if accepted else "no_strict_lift_or_regression",
            "task_deltas": task_deltas,
        })
        if accepted:
            current_skill = candidate_skill
            current_pairs = candidate_pairs
            current_score = candidate_score
            accepted_edits.extend(applied)
        else:
            rejected_edits.extend(applied)

    return CommunityGateResult(
        accepted=bool(accepted_edits),
        baseline_score=original_score,
        candidate_score=current_score,
        new_skill=current_skill,
        accepted_edits=accepted_edits,
        rejected_edits=rejected_edits,
        unmatched_edits=unmatched_edits,
        trials=trials,
    )
