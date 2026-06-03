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
