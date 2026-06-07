"""Shared Gitmoot package helpers for the SkillOpt adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gitmoot_skillopt.contracts import (
    ArtifactRef,
    EvalItem,
    FeedbackEvent,
    RankedFeedbackEvent,
    TrainingPackage,
)


@dataclass(frozen=True)
class TemplateDocument:
    """A template split into frontmatter and optimizable Markdown body."""

    frontmatter: str
    body: str

    def compose(self, body: str | None = None) -> str:
        actual_body = self.body if body is None else body
        return f"---\n{self.frontmatter.strip()}\n---\n{actual_body.lstrip()}"


def split_template_document(content: str) -> TemplateDocument:
    """Split a full agent-template Markdown document without rewriting YAML."""
    normalized = content.replace("\r\n", "\n").strip()
    lines = normalized.split("\n")
    if len(lines) < 3 or lines[0].strip() != "---":
        return TemplateDocument(frontmatter="", body=content)
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return TemplateDocument(
                frontmatter="\n".join(lines[1:index]),
                body="\n".join(lines[index + 1 :]).lstrip("\n"),
            )
    return TemplateDocument(frontmatter="", body=content)


def feedback_events_for_item(package: TrainingPackage, item_id: str) -> list[FeedbackEvent]:
    return [event for event in package.feedback_events if event.item_id == item_id]


def ranked_feedback_events_for_item(package: TrainingPackage, item_id: str) -> list[RankedFeedbackEvent]:
    return [event for event in package.ranked_feedback_events if event.item_id == item_id]


_PREVIOUS_OUTPUT_FEEDBACK_TARGETS = {
    "baseline_review_outputs",
    "baseline_outputs",
    "previous_outputs",
    "previous_review_outputs",
    "prior_outputs",
}


def feedback_scope_from_event(event: Any) -> str:
    normalized_targets = set(feedback_target_values(event))
    if normalized_targets & _PREVIOUS_OUTPUT_FEEDBACK_TARGETS:
        return "previous_outputs"
    return "live_candidate" if normalized_targets else "unspecified"


def feedback_target_values(event: Any) -> list[str]:
    targets = _event_string_values(_event_field(event, "feedback_target"))
    normalized = [target.strip().lower().replace("-", "_") for target in targets if target.strip()]
    return list(dict.fromkeys(normalized))


def feedback_is_about_previous_outputs(event: Any) -> bool:
    return feedback_scope_from_event(event) == "previous_outputs"


def _event_field(event: Any, field: str) -> Any:
    if isinstance(event, dict):
        return event.get(field)
    return getattr(event, field, None)


def _event_string_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        values: list[str] = []
        for item in value:
            values.extend(_event_string_values(item))
        return values
    if isinstance(value, tuple | set):
        values = []
        for item in value:
            values.extend(_event_string_values(item))
        return values
    return [str(value)]


def artifact_refs_by_id(package: TrainingPackage) -> dict[str, ArtifactRef]:
    return {artifact.id: artifact for artifact in package.artifacts}


def artifact_ids_for_item(item: EvalItem) -> dict[str, str]:
    artifact_ids = {
        "source": item.source_artifact_id,
        "baseline": item.baseline_artifact_id,
        "candidate": item.candidate_artifact_id,
        "preview": item.preview_artifact_id,
        "diff": item.diff_artifact_id,
    }
    for option in item.options:
        artifact_ids[f"option:{option.label}"] = option.artifact_id
    return artifact_ids


def json_safe_metadata(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def safe_item_path_segment(item_id: str) -> str:
    """Return an item id that is safe to use as one filesystem path segment."""
    value = str(item_id).strip()
    if not value:
        raise ValueError("Gitmoot item id is required")
    if value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"Gitmoot item id {item_id!r} is not safe for filesystem output")
    return value
