"""Gitmoot SkillOpt exchange package models."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CONTRACT_VERSION = 1
TRAINING_PACKAGE_KIND = "gitmoot-skillopt-training-package"
CANDIDATE_PACKAGE_KIND = "gitmoot-skillopt-candidate-package"

_TEMPLATE_ID_RE = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")
_TEMPLATE_KIND = "agent-template"
_TEMPLATE_VERSION = 1
_VALID_CAPABILITIES = {"ask", "review", "implement"}
_VALID_RUNTIMES = {"codex", "claude", "shell"}
_ARTIFACT_REF_FIELDS = (
    "source_artifact_id",
    "baseline_artifact_id",
    "candidate_artifact_id",
    "preview_artifact_id",
    "diff_artifact_id",
)


class ContractError(ValueError):
    """Raised when a Gitmoot SkillOpt package violates the v1 contract."""


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be an object")
    return value


def _require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or value.strip() == "":
        raise ContractError(f"{label} is required")
    return value.strip()


def _require_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or value.strip() == "":
        raise ContractError(f"{label} is required")
    return value


def _optional_string(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ContractError("optional string field must be a string")
    return value.strip()


def _require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{label} must be an integer")
    return value


def _optional_number(value: Any, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ContractError(f"{label} must be numeric")
    return float(value)


def _optional_string_list(value: Any, label: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ContractError(f"{label} must be a list")
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ContractError(f"{label} must contain strings")
        normalized.append(item.strip())
    return normalized


def _optional_dimension_scores(value: Any) -> dict[str, float] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ContractError("evaluator_score.dimension_scores must be an object")
    scores: dict[str, float] = {}
    for key, score in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ContractError("evaluator_score.dimension_scores keys must be strings")
        if score is None:
            raise ContractError("evaluator_score.dimension_scores value must be numeric")
        scores[key.strip()] = _optional_number(score, "evaluator_score.dimension_scores value")
    return scores


def _optional_gate_scores(value: Any, label: str) -> "GateRejectionScores":
    if value is None:
        return GateRejectionScores()
    return GateRejectionScores.from_dict(value, label)


def _raw_json(value: Any) -> Any:
    if value is None:
        return None
    json.dumps(value)
    return value


def validate_template_id(template_id: str, label: str = "template id") -> str:
    template_id = _require_string(template_id, label)
    if not _TEMPLATE_ID_RE.match(template_id):
        raise ContractError(f"invalid {label} {template_id!r}; use lowercase letters, numbers, and single dashes")
    return template_id


def _normalize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(metadata)
    for key, value in list(normalized.items()):
        if isinstance(value, str):
            normalized[key] = value.strip()
        elif isinstance(value, list):
            if all(isinstance(item, str) and item.strip() for item in value):
                normalized[key] = [item.strip() for item in value]
    evaluation = normalized.get("evaluation")
    if isinstance(evaluation, dict):
        if all(isinstance(key, str) and key.strip() and isinstance(value, str) and value.strip() for key, value in evaluation.items()):
            normalized["evaluation"] = {key.strip(): value.strip() for key, value in evaluation.items()}
    return normalized


def _require_metadata_list(metadata: dict[str, Any], key: str) -> list[str]:
    value = metadata.get(key)
    if not isinstance(value, list) or len(value) == 0:
        raise ContractError(f"template frontmatter missing {key}")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ContractError(f"template frontmatter {key} must contain strings")
    return [item.strip() for item in value]


def _validate_known_values(label: str, values: list[str], allowed: set[str]) -> None:
    for value in values:
        if value not in allowed:
            raise ContractError(f"template frontmatter has invalid {label} {value!r}")


def _validate_template_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    metadata = _normalize_metadata(metadata)
    validate_template_id(str(metadata.get("id", "")), "template frontmatter id")
    for key in ("name", "description"):
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            raise ContractError(f"template frontmatter missing {key}")
    if metadata.get("kind") != _TEMPLATE_KIND:
        raise ContractError(f"template kind must be {_TEMPLATE_KIND!r}")
    version = metadata.get("version")
    if isinstance(version, bool) or version != _TEMPLATE_VERSION:
        raise ContractError(f"template version must be {_TEMPLATE_VERSION}")
    capabilities = _require_metadata_list(metadata, "capabilities")
    _validate_known_values("capability", capabilities, _VALID_CAPABILITIES)
    runtimes = _require_metadata_list(metadata, "runtime_compatibility")
    _validate_known_values("runtime_compatibility", runtimes, _VALID_RUNTIMES)
    _require_metadata_list(metadata, "tags")
    _require_metadata_list(metadata, "inputs")
    _require_metadata_list(metadata, "outputs")
    evaluation = metadata.get("evaluation")
    if evaluation is not None and not isinstance(evaluation, dict):
        raise ContractError("template frontmatter evaluation must be an object")
    if isinstance(evaluation, dict) and not all(
        isinstance(key, str) and key.strip() and isinstance(value, str) and value.strip()
        for key, value in evaluation.items()
    ):
        raise ContractError("template frontmatter evaluation must contain string keys and values")
    return metadata


def _split_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    content = content.replace("\r\n", "\n").strip()
    if not content:
        raise ContractError("candidate template content is empty")
    lines = content.split("\n")
    if len(lines) < 3 or lines[0].strip() != "---":
        raise ContractError("candidate template must start with YAML frontmatter")
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() != "---":
            continue
        frontmatter = "\n".join(lines[1:index])
        body = "\n".join(lines[index + 1 :]).lstrip("\n")
        parsed = yaml.safe_load(frontmatter) or {}
        if not isinstance(parsed, dict):
            raise ContractError("candidate template frontmatter must be an object")
        if body.strip() == "":
            raise ContractError("candidate template body is empty")
        return _validate_template_metadata(parsed), body
    raise ContractError("candidate template frontmatter is missing closing ---")


def validate_template_content_metadata(content: str, metadata: dict[str, Any], template_id: str) -> None:
    parsed_metadata, _body = _split_frontmatter(content)
    parsed_id = validate_template_id(str(parsed_metadata.get("id", "")), "candidate template id")
    if parsed_id != template_id:
        raise ContractError(f"candidate template id {parsed_id!r} does not match package template_id {template_id!r}")
    if _validate_template_metadata(metadata) != parsed_metadata:
        raise ContractError("candidate metadata does not match candidate template frontmatter")


def _validate_kind_and_version(kind: str, contract_version: int, expected_kind: str, label: str) -> None:
    if kind != expected_kind:
        raise ContractError(f"{label} kind must be {expected_kind!r}")
    if contract_version != CONTRACT_VERSION:
        raise ContractError(f"{label} contract_version must be {CONTRACT_VERSION}")


@dataclass(frozen=True)
class TemplateSnapshot:
    id: str
    version_id: str
    version_number: int
    version_state: str
    content_hash: str
    source_repo: str
    source_ref: str
    source_path: str
    resolved_commit: str
    metadata: dict[str, Any]
    content: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TemplateSnapshot:
        from .artifacts import content_hash, normalize_hash

        data = _require_mapping(data, "template")
        content = _require_text(data.get("content"), "template.content")
        expected_hash = normalize_hash(_require_string(data.get("content_hash"), "template.content_hash"))
        actual_hash = content_hash(content.encode())
        if actual_hash != expected_hash:
            raise ContractError(f"template.content_hash mismatch: got {actual_hash}, want {expected_hash}")
        snapshot = cls(
            id=validate_template_id(str(data.get("id", "")), "template id"),
            version_id=_require_string(data.get("version_id"), "template.version_id"),
            version_number=_require_int(data.get("version_number"), "template.version_number"),
            version_state=_require_string(data.get("version_state"), "template.version_state"),
            content_hash=expected_hash,
            source_repo=_optional_string(data.get("source_repo")),
            source_ref=_optional_string(data.get("source_ref")),
            source_path=_optional_string(data.get("source_path")),
            resolved_commit=_optional_string(data.get("resolved_commit")),
            metadata=_validate_template_metadata(_require_mapping(data.get("metadata"), "template.metadata")),
            content=content,
        )
        if str(snapshot.metadata.get("id", "")).strip() != snapshot.id:
            raise ContractError("template.metadata.id must match template.id")
        return snapshot

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "version_id": self.version_id,
            "version_number": self.version_number,
            "version_state": self.version_state,
            "content_hash": self.content_hash,
            "source_repo": self.source_repo,
            "source_ref": self.source_ref,
            "source_path": self.source_path,
            "resolved_commit": self.resolved_commit,
            "metadata": self.metadata,
            "content": self.content,
        }


@dataclass(frozen=True)
class ArtifactRef:
    id: str
    hash: str
    media_type: str
    size_bytes: int
    driver: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArtifactRef:
        from .artifacts import normalize_hash

        data = _require_mapping(data, "artifact")
        size_bytes = _require_int(data.get("size_bytes"), "artifact.size_bytes")
        if size_bytes < 0:
            raise ContractError("artifact.size_bytes cannot be negative")
        artifact = cls(
            id=_require_string(data.get("id"), "artifact.id"),
            hash=normalize_hash(_require_string(data.get("hash"), "artifact.hash")),
            media_type=_require_string(data.get("media_type"), "artifact.media_type"),
            size_bytes=size_bytes,
            driver=_require_string(data.get("driver"), "artifact.driver"),
        )
        return artifact

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "hash": self.hash,
            "media_type": self.media_type,
            "size_bytes": self.size_bytes,
            "driver": self.driver,
        }


@dataclass(frozen=True)
class EvaluatorCheckConfig:
    id: str = ""
    type: str = ""
    when: str = ""
    required: bool = False
    config: Any = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluatorCheckConfig:
        data = _require_mapping(data, "evaluator_profile.checks item")
        required = data.get("required", False)
        if not isinstance(required, bool):
            raise ContractError("evaluator_profile.checks.required must be a boolean")
        return cls(
            id=_optional_string(data.get("id")),
            type=_optional_string(data.get("type")),
            when=_optional_string(data.get("when")),
            required=required,
            config=_raw_json(data.get("config")),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if self.id:
            data["id"] = self.id
        if self.type:
            data["type"] = self.type
        if self.when:
            data["when"] = self.when
        if self.required:
            data["required"] = self.required
        if self.config is not None:
            data["config"] = self.config
        return data


@dataclass(frozen=True)
class EvaluatorJudgeConfig:
    type: str = ""
    when: str = ""
    model: str = ""
    config: Any = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> EvaluatorJudgeConfig | None:
        if data is None:
            return None
        data = _require_mapping(data, "evaluator_profile.judge")
        return cls(
            type=_optional_string(data.get("type")),
            when=_optional_string(data.get("when")),
            model=_optional_string(data.get("model")),
            config=_raw_json(data.get("config")),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if self.type:
            data["type"] = self.type
        if self.when:
            data["when"] = self.when
        if self.model:
            data["model"] = self.model
        if self.config is not None:
            data["config"] = self.config
        return data


@dataclass(frozen=True)
class EvaluatorProfile:
    profile_id: str = ""
    task_kind: str = ""
    artifact_contract: str = ""
    preview_adapter: str = ""
    checks: list[EvaluatorCheckConfig] = field(default_factory=list)
    judge: EvaluatorJudgeConfig | None = None
    metadata: Any = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> EvaluatorProfile | None:
        if data is None:
            return None
        data = _require_mapping(data, "evaluator_profile")
        checks = data.get("checks", [])
        if not isinstance(checks, list):
            raise ContractError("evaluator_profile.checks must be a list")
        return cls(
            profile_id=_optional_string(data.get("profile_id")),
            task_kind=_optional_string(data.get("task_kind")),
            artifact_contract=_optional_string(data.get("artifact_contract")),
            preview_adapter=_optional_string(data.get("preview_adapter")),
            checks=[EvaluatorCheckConfig.from_dict(check) for check in checks],
            judge=EvaluatorJudgeConfig.from_dict(data.get("judge")),
            metadata=_raw_json(data.get("metadata")),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if self.profile_id:
            data["profile_id"] = self.profile_id
        if self.task_kind:
            data["task_kind"] = self.task_kind
        if self.artifact_contract:
            data["artifact_contract"] = self.artifact_contract
        if self.preview_adapter:
            data["preview_adapter"] = self.preview_adapter
        if self.checks:
            data["checks"] = [check.to_dict() for check in self.checks]
        if self.judge is not None:
            data["judge"] = self.judge.to_dict()
        if self.metadata is not None:
            data["metadata"] = self.metadata
        return data


@dataclass(frozen=True)
class EvaluatorStageStatus:
    stage: str = ""
    status: str = ""
    started_at: str = ""
    finished_at: str = ""
    duration_ms: int = 0
    details: Any = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluatorStageStatus:
        data = _require_mapping(data, "evaluator stage status")
        duration_ms = data.get("duration_ms", 0)
        if duration_ms is None:
            duration_ms = 0
        if isinstance(duration_ms, bool) or not isinstance(duration_ms, int):
            raise ContractError("evaluator stage status duration_ms must be an integer")
        return cls(
            stage=_optional_string(data.get("stage")),
            status=_optional_string(data.get("status")),
            started_at=_optional_string(data.get("started_at")),
            finished_at=_optional_string(data.get("finished_at")),
            duration_ms=duration_ms,
            details=_raw_json(data.get("details")),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if self.stage:
            data["stage"] = self.stage
        if self.status:
            data["status"] = self.status
        if self.started_at:
            data["started_at"] = self.started_at
        if self.finished_at:
            data["finished_at"] = self.finished_at
        if self.duration_ms:
            data["duration_ms"] = self.duration_ms
        if self.details is not None:
            data["details"] = self.details
        return data


@dataclass(frozen=True)
class EvaluatorCheckResult:
    check: str = ""
    severity: str = ""
    reason: str = ""
    evidence: list[str] = field(default_factory=list)
    metadata: Any = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluatorCheckResult:
        data = _require_mapping(data, "evaluator failed check")
        return cls(
            check=_optional_string(data.get("check")),
            severity=_optional_string(data.get("severity")),
            reason=_optional_string(data.get("reason")),
            evidence=_optional_string_list(data.get("evidence"), "evaluator failed check evidence"),
            metadata=_raw_json(data.get("metadata")),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if self.check:
            data["check"] = self.check
        if self.severity:
            data["severity"] = self.severity
        if self.reason:
            data["reason"] = self.reason
        if self.evidence:
            data["evidence"] = self.evidence
        if self.metadata is not None:
            data["metadata"] = self.metadata
        return data


@dataclass(frozen=True)
class EvaluatorFailurePacket:
    primary_reason: str = ""
    human_reason: str = ""
    optimizer_hint: str = ""
    failed_checks: list[EvaluatorCheckResult] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    stage_status: list[EvaluatorStageStatus] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> EvaluatorFailurePacket | None:
        if data is None:
            return None
        data = _require_mapping(data, "evaluator failure")
        failed_checks = data.get("failed_checks", [])
        if not isinstance(failed_checks, list):
            raise ContractError("evaluator failure failed_checks must be a list")
        stage_status = data.get("stage_status", [])
        if not isinstance(stage_status, list):
            raise ContractError("evaluator failure stage_status must be a list")
        return cls(
            primary_reason=_optional_string(data.get("primary_reason")),
            human_reason=_optional_string(data.get("human_reason")),
            optimizer_hint=_optional_string(data.get("optimizer_hint")),
            failed_checks=[EvaluatorCheckResult.from_dict(check) for check in failed_checks],
            evidence=_optional_string_list(data.get("evidence"), "evaluator failure evidence"),
            stage_status=[EvaluatorStageStatus.from_dict(stage) for stage in stage_status],
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if self.primary_reason:
            data["primary_reason"] = self.primary_reason
        if self.human_reason:
            data["human_reason"] = self.human_reason
        if self.optimizer_hint:
            data["optimizer_hint"] = self.optimizer_hint
        if self.failed_checks:
            data["failed_checks"] = [check.to_dict() for check in self.failed_checks]
        if self.evidence:
            data["evidence"] = self.evidence
        if self.stage_status:
            data["stage_status"] = [stage.to_dict() for stage in self.stage_status]
        return data


@dataclass(frozen=True)
class GateRejectionScores:
    hard: float | None = None
    soft: float | None = None
    gate_score: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], label: str = "gate rejection scores") -> GateRejectionScores:
        data = _require_mapping(data, label)
        return cls(
            hard=_optional_number(data.get("hard"), f"{label}.hard"),
            soft=_optional_number(data.get("soft"), f"{label}.soft"),
            gate_score=_optional_number(data.get("gate_score"), f"{label}.gate_score"),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if self.hard is not None:
            data["hard"] = self.hard
        if self.soft is not None:
            data["soft"] = self.soft
        if self.gate_score is not None:
            data["gate_score"] = self.gate_score
        return data


@dataclass(frozen=True)
class GateRejectionPacket:
    rejection_type: str = ""
    retryable: bool = False
    baseline: GateRejectionScores = field(default_factory=GateRejectionScores)
    candidate: GateRejectionScores = field(default_factory=GateRejectionScores)
    primary_reason: str = ""
    human_reason: str = ""
    optimizer_hint: str = ""
    failed_dimensions: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    attempted_patch: str = ""
    retry_attempts: str = ""
    next_action: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> GateRejectionPacket | None:
        if data is None:
            return None
        data = _require_mapping(data, "gate_rejection")
        retryable = data.get("retryable", False)
        if not isinstance(retryable, bool):
            raise ContractError("gate_rejection.retryable must be a boolean")
        return cls(
            rejection_type=_optional_string(data.get("rejection_type")),
            retryable=retryable,
            baseline=_optional_gate_scores(data.get("baseline"), "gate_rejection.baseline"),
            candidate=_optional_gate_scores(data.get("candidate"), "gate_rejection.candidate"),
            primary_reason=_optional_string(data.get("primary_reason")),
            human_reason=_optional_string(data.get("human_reason")),
            optimizer_hint=_optional_string(data.get("optimizer_hint")),
            failed_dimensions=_optional_string_list(data.get("failed_dimensions"), "gate_rejection.failed_dimensions"),
            evidence=_optional_string_list(data.get("evidence"), "gate_rejection.evidence"),
            attempted_patch=_optional_string(data.get("attempted_patch")),
            retry_attempts=_optional_string(data.get("retry_attempts")),
            next_action=_optional_string(data.get("next_action")),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if self.rejection_type:
            data["rejection_type"] = self.rejection_type
        if self.retryable:
            data["retryable"] = self.retryable
        baseline = self.baseline.to_dict()
        if baseline:
            data["baseline"] = baseline
        candidate = self.candidate.to_dict()
        if candidate:
            data["candidate"] = candidate
        if self.primary_reason:
            data["primary_reason"] = self.primary_reason
        if self.human_reason:
            data["human_reason"] = self.human_reason
        if self.optimizer_hint:
            data["optimizer_hint"] = self.optimizer_hint
        if self.failed_dimensions:
            data["failed_dimensions"] = self.failed_dimensions
        if self.evidence:
            data["evidence"] = self.evidence
        if self.attempted_patch:
            data["attempted_patch"] = self.attempted_patch
        if self.retry_attempts:
            data["retry_attempts"] = self.retry_attempts
        if self.next_action:
            data["next_action"] = self.next_action
        return data


@dataclass(frozen=True)
class EvaluatorScore:
    profile_id: str = ""
    task_kind: str = ""
    contract_status: str = ""
    quality_status: str = ""
    human_feedback_alignment: Any = None
    hard: float | None = None
    soft: float | None = None
    dimension_scores: dict[str, float] | None = None
    fail_reason: str = ""
    failure: EvaluatorFailurePacket | None = None
    gate_rejection: GateRejectionPacket | None = None
    stage_status: list[EvaluatorStageStatus] = field(default_factory=list)
    metadata: Any = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> EvaluatorScore | None:
        if data is None:
            return None
        data = _require_mapping(data, "evaluator_score")
        stage_status = data.get("stage_status", [])
        if not isinstance(stage_status, list):
            raise ContractError("evaluator_score.stage_status must be a list")
        return cls(
            profile_id=_optional_string(data.get("profile_id")),
            task_kind=_optional_string(data.get("task_kind")),
            contract_status=_optional_string(data.get("contract_status")),
            quality_status=_optional_string(data.get("quality_status")),
            human_feedback_alignment=_raw_json(data.get("human_feedback_alignment")),
            hard=_optional_number(data.get("hard"), "evaluator_score.hard"),
            soft=_optional_number(data.get("soft"), "evaluator_score.soft"),
            dimension_scores=_optional_dimension_scores(data.get("dimension_scores")),
            fail_reason=_optional_string(data.get("fail_reason")),
            failure=EvaluatorFailurePacket.from_dict(data.get("failure")),
            gate_rejection=GateRejectionPacket.from_dict(data.get("gate_rejection")),
            stage_status=[EvaluatorStageStatus.from_dict(stage) for stage in stage_status],
            metadata=_raw_json(data.get("metadata")),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if self.profile_id:
            data["profile_id"] = self.profile_id
        if self.task_kind:
            data["task_kind"] = self.task_kind
        if self.contract_status:
            data["contract_status"] = self.contract_status
        if self.quality_status:
            data["quality_status"] = self.quality_status
        if self.human_feedback_alignment is not None:
            data["human_feedback_alignment"] = self.human_feedback_alignment
        if self.hard is not None:
            data["hard"] = self.hard
        if self.soft is not None:
            data["soft"] = self.soft
        if self.dimension_scores is not None:
            data["dimension_scores"] = self.dimension_scores
        if self.fail_reason:
            data["fail_reason"] = self.fail_reason
        if self.failure is not None:
            data["failure"] = self.failure.to_dict()
        if self.gate_rejection is not None:
            data["gate_rejection"] = self.gate_rejection.to_dict()
        if self.stage_status:
            data["stage_status"] = [stage.to_dict() for stage in self.stage_status]
        if self.metadata is not None:
            data["metadata"] = self.metadata
        return data


@dataclass(frozen=True)
class EvalRun:
    id: str
    template_id: str
    template_version_id: str
    target_repo: str
    state: str
    mode: str = ""
    exploration_level: str = ""
    options_count: int = 0
    metadata: Any = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalRun:
        data = _require_mapping(data, "eval_run")
        options_count = data.get("options_count", 0)
        if options_count is None:
            options_count = 0
        if isinstance(options_count, bool) or not isinstance(options_count, int):
            raise ContractError("eval_run.options_count must be an integer")
        if options_count < 0:
            raise ContractError("eval_run.options_count must be zero or greater")
        return cls(
            id=_require_string(data.get("id"), "eval_run.id"),
            template_id=validate_template_id(str(data.get("template_id", "")), "eval_run.template_id"),
            template_version_id=_optional_string(data.get("template_version_id")),
            target_repo=_optional_string(data.get("target_repo")),
            state=_require_string(data.get("state"), "eval_run.state"),
            mode=_optional_string(data.get("mode")),
            exploration_level=_optional_string(data.get("exploration_level")),
            options_count=options_count,
            metadata=_raw_json(data.get("metadata")),
        )

    def to_dict(self) -> dict[str, Any]:
        data = {
            "id": self.id,
            "template_id": self.template_id,
            "template_version_id": self.template_version_id,
            "target_repo": self.target_repo,
            "state": self.state,
        }
        if self.mode:
            data["mode"] = self.mode
        if self.exploration_level:
            data["exploration_level"] = self.exploration_level
        if self.options_count:
            data["options_count"] = self.options_count
        if self.metadata is not None:
            data["metadata"] = self.metadata
        return data


@dataclass(frozen=True)
class EvalReviewOption:
    label: str
    artifact_id: str
    role: str = ""
    metadata: Any = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalReviewOption:
        data = _require_mapping(data, "item option")
        return cls(
            label=_require_string(data.get("label"), "item.options.label"),
            artifact_id=_require_string(data.get("artifact_id"), "item.options.artifact_id"),
            role=_optional_string(data.get("role")),
            metadata=_raw_json(data.get("metadata")),
        )

    def to_dict(self) -> dict[str, Any]:
        data = {
            "label": self.label,
            "artifact_id": self.artifact_id,
        }
        if self.role:
            data["role"] = self.role
        if self.metadata is not None:
            data["metadata"] = self.metadata
        return data


@dataclass(frozen=True)
class EvalItem:
    id: str
    title: str = ""
    source_artifact_id: str = ""
    baseline_artifact_id: str = ""
    candidate_artifact_id: str = ""
    preview_artifact_id: str = ""
    diff_artifact_id: str = ""
    options: list[EvalReviewOption] = field(default_factory=list)
    metadata: Any = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalItem:
        data = _require_mapping(data, "item")
        options = data.get("options") or []
        if not isinstance(options, list):
            raise ContractError("item.options must be a list")
        return cls(
            id=_require_string(data.get("id"), "item.id"),
            title=_optional_string(data.get("title")),
            source_artifact_id=_optional_string(data.get("source_artifact_id")),
            baseline_artifact_id=_optional_string(data.get("baseline_artifact_id")),
            candidate_artifact_id=_optional_string(data.get("candidate_artifact_id")),
            preview_artifact_id=_optional_string(data.get("preview_artifact_id")),
            diff_artifact_id=_optional_string(data.get("diff_artifact_id")),
            options=[EvalReviewOption.from_dict(option) for option in options],
            metadata=_raw_json(data.get("metadata")),
        )

    def artifact_ids(self) -> list[str]:
        return [
            artifact_id
            for artifact_id in (
                self.source_artifact_id,
                self.baseline_artifact_id,
                self.candidate_artifact_id,
                self.preview_artifact_id,
                self.diff_artifact_id,
            )
            if artifact_id
        ] + [option.artifact_id for option in self.options if option.artifact_id]

    def to_dict(self) -> dict[str, Any]:
        data = {
            "id": self.id,
            "title": self.title,
            "source_artifact_id": self.source_artifact_id,
            "baseline_artifact_id": self.baseline_artifact_id,
            "candidate_artifact_id": self.candidate_artifact_id,
            "preview_artifact_id": self.preview_artifact_id,
            "diff_artifact_id": self.diff_artifact_id,
        }
        if self.options:
            data["options"] = [option.to_dict() for option in self.options]
        if self.metadata is not None:
            data["metadata"] = self.metadata
        return data


@dataclass(frozen=True)
class FeedbackEvent:
    run_id: str
    item_id: str
    choice: str
    reviewer: str
    source: str
    created_at: str
    reasoning: str = ""
    source_url: str = ""
    quality: str = ""
    continue_mode: str = ""
    promote: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeedbackEvent:
        data = _require_mapping(data, "feedback event")
        return cls(
            run_id=_require_string(data.get("run_id"), "feedback.run_id"),
            item_id=_require_string(data.get("item_id"), "feedback.item_id"),
            choice=_require_string(data.get("choice"), "feedback.choice"),
            reasoning=_optional_string(data.get("reasoning")),
            reviewer=_require_string(data.get("reviewer"), "feedback.reviewer"),
            source=_require_string(data.get("source"), "feedback.source"),
            source_url=_optional_string(data.get("source_url")),
            created_at=_require_string(data.get("created_at"), "feedback.created_at"),
            quality=_optional_string(data.get("quality")),
            continue_mode=_optional_string(data.get("continue_mode")),
            promote=_optional_string(data.get("promote")),
        )

    def to_dict(self) -> dict[str, Any]:
        data = {
            "run_id": self.run_id,
            "item_id": self.item_id,
            "choice": self.choice,
            "reasoning": self.reasoning,
            "reviewer": self.reviewer,
            "source": self.source,
            "source_url": self.source_url,
            "created_at": self.created_at,
        }
        if self.quality:
            data["quality"] = self.quality
        if self.continue_mode:
            data["continue_mode"] = self.continue_mode
        if self.promote:
            data["promote"] = self.promote
        return data


@dataclass(frozen=True)
class RankedFeedbackEvent:
    id: str
    run_id: str
    item_id: str
    ranking: list[str]
    reviewer: str
    source: str
    created_at: str
    winner: str = ""
    useful_traits: Any = None
    rejected_traits: Any = None
    required_improvements: Any = None
    quality: str = ""
    continue_mode: str = ""
    promote: str = ""
    reasoning: str = ""
    source_url: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RankedFeedbackEvent:
        data = _require_mapping(data, "ranked feedback event")
        ranking = data.get("ranking")
        if not ranking or not isinstance(ranking, list) or not all(isinstance(item, str) and item.strip() for item in ranking):
            raise ContractError("ranked feedback ranking must be a non-empty string list")
        normalized_ranking = [item.strip() for item in ranking]
        if len(set(normalized_ranking)) != len(normalized_ranking):
            raise ContractError("ranked feedback ranking labels must be unique")
        return cls(
            id=_optional_string(data.get("id")),
            run_id=_require_string(data.get("run_id"), "ranked_feedback.run_id"),
            item_id=_require_string(data.get("item_id"), "ranked_feedback.item_id"),
            ranking=normalized_ranking,
            winner=_optional_string(data.get("winner")),
            useful_traits=_raw_json(data.get("useful_traits")),
            rejected_traits=_raw_json(data.get("rejected_traits")),
            required_improvements=_raw_json(data.get("required_improvements")),
            quality=_optional_string(data.get("quality")),
            continue_mode=_optional_string(data.get("continue_mode")),
            promote=_optional_string(data.get("promote")),
            reasoning=_optional_string(data.get("reasoning")),
            reviewer=_require_string(data.get("reviewer"), "ranked_feedback.reviewer"),
            source=_require_string(data.get("source"), "ranked_feedback.source"),
            source_url=_optional_string(data.get("source_url")),
            created_at=_require_string(data.get("created_at"), "ranked_feedback.created_at"),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "run_id": self.run_id,
            "item_id": self.item_id,
            "ranking": self.ranking,
            "reviewer": self.reviewer,
            "source": self.source,
            "created_at": self.created_at,
        }
        if self.id:
            data["id"] = self.id
        if self.winner:
            data["winner"] = self.winner
        if self.useful_traits is not None:
            data["useful_traits"] = self.useful_traits
        if self.rejected_traits is not None:
            data["rejected_traits"] = self.rejected_traits
        if self.required_improvements is not None:
            data["required_improvements"] = self.required_improvements
        if self.quality:
            data["quality"] = self.quality
        if self.continue_mode:
            data["continue_mode"] = self.continue_mode
        if self.promote:
            data["promote"] = self.promote
        if self.reasoning:
            data["reasoning"] = self.reasoning
        if self.source_url:
            data["source_url"] = self.source_url
        return data


@dataclass(frozen=True)
class TrainingPackage:
    kind: str
    contract_version: int
    template: TemplateSnapshot
    eval_run: EvalRun
    items: list[EvalItem] = field(default_factory=list)
    artifacts: list[ArtifactRef] = field(default_factory=list)
    feedback_events: list[FeedbackEvent] = field(default_factory=list)
    ranked_feedback_events: list[RankedFeedbackEvent] = field(default_factory=list)
    evaluator_config: Any = None
    evaluator_profile: EvaluatorProfile | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingPackage:
        data = _require_mapping(data, "training package")
        kind = _require_string(data.get("kind"), "training package.kind")
        contract_version = _require_int(data.get("contract_version"), "training package.contract_version")
        _validate_kind_and_version(kind, contract_version, TRAINING_PACKAGE_KIND, "training package")
        package = cls(
            kind=kind,
            contract_version=contract_version,
            template=TemplateSnapshot.from_dict(data.get("template")),
            eval_run=EvalRun.from_dict(data.get("eval_run")),
            items=[EvalItem.from_dict(item) for item in data.get("items", [])],
            artifacts=[ArtifactRef.from_dict(artifact) for artifact in data.get("artifacts", [])],
            feedback_events=[FeedbackEvent.from_dict(event) for event in data.get("feedback_events", [])],
            ranked_feedback_events=[
                RankedFeedbackEvent.from_dict(event) for event in data.get("ranked_feedback_events", [])
            ],
            evaluator_config=_raw_json(data.get("evaluator_config")),
            evaluator_profile=EvaluatorProfile.from_dict(data.get("evaluator_profile")),
        )
        package.validate()
        return package

    @classmethod
    def load(cls, path: str | Path) -> TrainingPackage:
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def validate(self) -> None:
        if self.template.id != self.eval_run.template_id:
            raise ContractError("eval_run.template_id must match template.id")
        if self.eval_run.template_version_id and self.eval_run.template_version_id != self.template.version_id:
            raise ContractError("eval_run.template_version_id must match template.version_id")
        artifact_ids = {artifact.id for artifact in self.artifacts}
        if len(artifact_ids) != len(self.artifacts):
            raise ContractError("artifact ids must be unique")
        items_by_id = {item.id: item for item in self.items}
        if len(items_by_id) != len(self.items):
            raise ContractError("item ids must be unique")
        for item in self.items:
            option_labels = [option.label for option in item.options]
            if len(set(option_labels)) != len(option_labels):
                raise ContractError(f"item {item.id!r} option labels must be unique")
            for artifact_id in item.artifact_ids():
                if artifact_id not in artifact_ids:
                    raise ContractError(f"item {item.id!r} references missing artifact {artifact_id!r}")
        for event in self.feedback_events:
            if event.run_id != self.eval_run.id:
                raise ContractError(f"feedback event for item {event.item_id!r} has wrong run_id {event.run_id!r}")
            if event.item_id not in items_by_id:
                raise ContractError(f"feedback event references missing item {event.item_id!r}")
        for event in self.ranked_feedback_events:
            if event.run_id != self.eval_run.id:
                raise ContractError(f"ranked feedback event for item {event.item_id!r} has wrong run_id {event.run_id!r}")
            item = items_by_id.get(event.item_id)
            if item is None:
                raise ContractError(f"ranked feedback event references missing item {event.item_id!r}")
            option_labels = {option.label for option in item.options}
            if not option_labels:
                raise ContractError(f"ranked feedback event references item {event.item_id!r} without options")
            unknown_labels = [label for label in event.ranking if label not in option_labels]
            if unknown_labels:
                raise ContractError(
                    f"ranked feedback event references unknown option labels: {', '.join(unknown_labels)}"
                )
            if event.winner and event.winner not in option_labels:
                raise ContractError(f"ranked feedback winner references unknown option label {event.winner!r}")
            if event.winner and event.winner not in event.ranking:
                raise ContractError(f"ranked feedback winner {event.winner!r} must appear in ranking")

    def to_dict(self) -> dict[str, Any]:
        data = {
            "kind": self.kind,
            "contract_version": self.contract_version,
            "template": self.template.to_dict(),
            "eval_run": self.eval_run.to_dict(),
            "items": [item.to_dict() for item in self.items],
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "feedback_events": [event.to_dict() for event in self.feedback_events],
        }
        if self.ranked_feedback_events:
            data["ranked_feedback_events"] = [event.to_dict() for event in self.ranked_feedback_events]
        if self.evaluator_config is not None:
            data["evaluator_config"] = self.evaluator_config
        if self.evaluator_profile is not None:
            data["evaluator_profile"] = self.evaluator_profile.to_dict()
        return data

    def dump(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")


@dataclass(frozen=True)
class CandidateTemplate:
    content: str
    metadata: dict[str, Any]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CandidateTemplate:
        data = _require_mapping(data, "candidate")
        return cls(
            content=_require_text(data.get("content"), "candidate.content"),
            metadata=_require_mapping(data.get("metadata"), "candidate.metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"content": self.content, "metadata": self.metadata}


@dataclass(frozen=True)
class CandidateSummary:
    diff_artifact_id: str = ""
    score: float | None = None
    preference_summary: str = ""
    metadata: Any = None
    evaluator_score: EvaluatorScore | None = None
    failure: EvaluatorFailurePacket | None = None
    gate_rejection: GateRejectionPacket | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> CandidateSummary:
        if data is None:
            return cls()
        data = _require_mapping(data, "summary")
        score = data.get("score")
        if score is not None and (isinstance(score, bool) or not isinstance(score, int | float)):
            raise ContractError("summary.score must be numeric")
        return cls(
            diff_artifact_id=_optional_string(data.get("diff_artifact_id")),
            score=float(score) if score is not None else None,
            preference_summary=_optional_string(data.get("preference_summary")),
            metadata=_raw_json(data.get("metadata")),
            evaluator_score=EvaluatorScore.from_dict(data.get("evaluator_score")),
            failure=EvaluatorFailurePacket.from_dict(data.get("failure")),
            gate_rejection=GateRejectionPacket.from_dict(data.get("gate_rejection")),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "diff_artifact_id": self.diff_artifact_id,
            "preference_summary": self.preference_summary,
        }
        if self.score is not None:
            data["score"] = self.score
        if self.metadata is not None:
            data["metadata"] = self.metadata
        if self.evaluator_score is not None:
            data["evaluator_score"] = self.evaluator_score.to_dict()
        if self.failure is not None:
            data["failure"] = self.failure.to_dict()
        if self.gate_rejection is not None:
            data["gate_rejection"] = self.gate_rejection.to_dict()
        return data


@dataclass(frozen=True)
class CandidatePackage:
    kind: str
    contract_version: int
    template_id: str
    candidate: CandidateTemplate
    base_version_id: str = ""
    artifacts: list[Any] = field(default_factory=list)
    eval_report: Any = None
    summary: CandidateSummary = field(default_factory=CandidateSummary)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CandidatePackage:
        data = _require_mapping(data, "candidate package")
        kind = _require_string(data.get("kind"), "candidate package.kind")
        contract_version = _require_int(data.get("contract_version"), "candidate package.contract_version")
        _validate_kind_and_version(kind, contract_version, CANDIDATE_PACKAGE_KIND, "candidate package")
        from .artifacts import CandidateArtifactManifestEntry

        raw_artifacts = data["artifacts"] if "artifacts" in data else []
        if not isinstance(raw_artifacts, list):
            raise ContractError("candidate package.artifacts must be a list")
        package = cls(
            kind=kind,
            contract_version=contract_version,
            template_id=validate_template_id(str(data.get("template_id", "")), "candidate package.template_id"),
            base_version_id=_optional_string(data.get("base_version_id")),
            candidate=CandidateTemplate.from_dict(data.get("candidate")),
            artifacts=[
                CandidateArtifactManifestEntry.from_dict(artifact)
                for artifact in raw_artifacts
            ],
            eval_report=_raw_json(data.get("eval_report")),
            summary=CandidateSummary.from_dict(data.get("summary")),
        )
        package.validate()
        return package

    @classmethod
    def load(cls, path: str | Path) -> CandidatePackage:
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def validate(self) -> None:
        validate_template_content_metadata(self.candidate.content, self.candidate.metadata, self.template_id)
        artifact_ids: set[str] = set()
        for artifact in self.artifacts:
            artifact_id = getattr(artifact, "id", "")
            if not isinstance(artifact_id, str) or not artifact_id.strip():
                raise ContractError("candidate artifact id is required")
            if artifact_id in artifact_ids:
                raise ContractError(f"candidate artifact id is duplicated: {artifact_id}")
            artifact_ids.add(artifact_id)
        if artifact_ids and self.summary.diff_artifact_id not in artifact_ids:
            raise ContractError("candidate summary.diff_artifact_id must reference a candidate artifact")

    def to_dict(self) -> dict[str, Any]:
        data = {
            "kind": self.kind,
            "contract_version": self.contract_version,
            "template_id": self.template_id,
            "base_version_id": self.base_version_id,
            "candidate": self.candidate.to_dict(),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "summary": self.summary.to_dict(),
        }
        if self.eval_report is not None:
            data["eval_report"] = self.eval_report
        return data

    def dump(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, sort_keys=True)
            handle.write("\n")
