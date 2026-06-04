"""Gitmoot training package dataloader."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from gitmoot_skillopt.artifacts import ArtifactError, GitmootArtifactResolver
from gitmoot_skillopt.contracts import ArtifactRef, EvalItem, TrainingPackage
from skillopt.datasets.base import BaseDataLoader, BatchSpec
from skillopt.envs.gitmoot.package import (
    artifact_ids_for_item,
    artifact_refs_by_id,
    feedback_events_for_item,
    json_safe_metadata,
    ranked_feedback_events_for_item,
    safe_item_path_segment,
    split_template_document,
)

_SPLIT_ALIASES = {
    "train": "train",
    "selection": "val",
    "valid_seen": "val",
    "val": "val",
    "test": "test",
    "valid_unseen": "test",
}

_SUPPORTED_TEXT_DRIVERS = {"text", "markdown", "text/markdown"}
_SUPPORTED_PREVIEW_DRIVERS = {"vue-vite"}


class GitmootDataLoader(BaseDataLoader):
    """Load Gitmoot SkillOpt training packages into SkillOpt batches."""

    def __init__(
        self,
        training_package: str = "",
        artifact_root: str = "",
        seed: int = 42,
        limit: int = 0,
        **kwargs,
    ) -> None:
        del kwargs
        self.training_package = training_package
        self.artifact_root = artifact_root
        self.seed = int(seed)
        self.limit = int(limit)
        self.package: TrainingPackage | None = None
        self.evaluator_config: dict[str, Any] = {}
        self._splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}

    def setup(self, cfg: dict) -> None:
        if not self.training_package:
            self.training_package = str(cfg.get("training_package") or "")
        if not self.artifact_root:
            self.artifact_root = str(cfg.get("artifact_root") or "")
        if not self.training_package:
            raise ValueError("GitmootDataLoader requires training_package")
        if not self.artifact_root:
            raise ValueError("GitmootDataLoader requires artifact_root")

        self.package = TrainingPackage.load(self.training_package)
        profile_config = _evaluator_profile_config(self.package)
        package_config = self.package.evaluator_config if isinstance(self.package.evaluator_config, dict) else {}
        override_config = cfg.get("evaluator_config") if isinstance(cfg.get("evaluator_config"), dict) else {}
        self.evaluator_config = {**profile_config, **package_config}
        _apply_evaluator_id_mode_override(self.evaluator_config, package_config)
        self.evaluator_config = {**self.evaluator_config, **override_config}
        resolver = GitmootArtifactResolver(self.artifact_root)
        items = [self._item_to_task(item, resolver) for item in self.package.items]
        self._splits = self._split_items(items)
        if self.limit:
            self._splits = {name: split_items[: self.limit] for name, split_items in self._splits.items()}

    @property
    def initial_skill_content(self) -> str:
        if self.package is None:
            return ""
        return self.package.template.content

    @property
    def initial_skill_body(self) -> str:
        return split_template_document(self.initial_skill_content).body

    def get_train_size(self) -> int:
        return len(self._splits.get("train", []))

    @property
    def train_items(self) -> list[dict[str, Any]]:
        return list(self._splits.get("train", []))

    @property
    def val_items(self) -> list[dict[str, Any]]:
        return list(self._splits.get("val", []))

    @property
    def test_items(self) -> list[dict[str, Any]]:
        return list(self._splits.get("test", []))

    def get_split_items(self, split: str) -> list[dict[str, Any]]:
        canonical = _SPLIT_ALIASES.get(str(split or "").strip(), "val")
        return list(self._splits.get(canonical, []))

    def plan_train_epoch(
        self,
        *,
        epoch: int,
        steps_per_epoch: int,
        accumulation: int,
        batch_size: int,
        seed: int,
        **kwargs,
    ) -> list[BatchSpec]:
        del kwargs
        epoch_rng = random.Random(seed + epoch * 1000)
        items = self.train_items
        epoch_rng.shuffle(items)

        total_batches = steps_per_epoch * accumulation
        if total_batches <= 0:
            return []

        batches: list[BatchSpec] = []
        cursor = 0
        for batch_idx in range(total_batches):
            batch_seed = seed + epoch * 1000 + batch_idx + 1
            batch_items = items[cursor : cursor + batch_size]
            cursor += len(batch_items)
            if not batch_items and items:
                refill_rng = random.Random(batch_seed)
                batch_items = list(items)
                refill_rng.shuffle(batch_items)
                batch_items = batch_items[:batch_size]
            batches.append(
                BatchSpec(
                    phase="train",
                    split="train",
                    seed=batch_seed,
                    batch_size=len(batch_items),
                    payload=batch_items,
                )
            )
        return batches

    def build_train_batch(self, batch_size: int, seed: int, **kwargs) -> BatchSpec:
        del kwargs
        items = self.get_split_items("train")
        rng = random.Random(seed)
        rng.shuffle(items)
        selected = items[:batch_size]
        return BatchSpec(
            phase="train",
            split="train",
            seed=seed,
            batch_size=len(selected),
            payload=selected,
        )

    def build_eval_batch(self, env_num: int, split: str, seed: int, **kwargs) -> BatchSpec:
        del kwargs
        items = self.get_split_items(split)
        if env_num and env_num < len(items):
            items = items[:env_num]
        return BatchSpec(
            phase="eval",
            split=split,
            seed=seed,
            batch_size=len(items),
            payload=items,
        )

    def _item_to_task(self, item: EvalItem, resolver: GitmootArtifactResolver) -> dict[str, Any]:
        assert self.package is not None
        safe_item_path_segment(item.id)
        metadata = json_safe_metadata(item.metadata)
        artifact_refs = artifact_refs_by_id(self.package)
        artifacts = self._resolve_item_artifacts(item, artifact_refs, resolver)
        feedback_events = feedback_events_for_item(self.package, item.id)
        ranked_feedback_events = ranked_feedback_events_for_item(self.package, item.id)
        prompt = build_task_prompt(
            package=self.package,
            item=item,
            artifacts=artifacts,
            feedback_events=[
                _feedback_event_prompt_context(event)
                for event in feedback_events
            ],
            ranked_feedback_events=[
                _ranked_feedback_event_prompt_context(event)
                for event in ranked_feedback_events
            ],
        )
        return {
            "id": item.id,
            "title": item.title,
            "task_type": "gitmoot-skillopt",
            "task_description": item.title or item.id,
            "metadata": metadata,
            "split": self._item_split(metadata),
            "prompt": prompt,
            "artifacts": artifacts,
            "feedback_events": [event.to_dict() for event in feedback_events],
            "ranked_feedback_events": [event.to_dict() for event in ranked_feedback_events],
            "evaluator_config": self.evaluator_config,
        }

    def _resolve_item_artifacts(
        self,
        item: EvalItem,
        artifact_refs: dict[str, ArtifactRef],
        resolver: GitmootArtifactResolver,
    ) -> dict[str, dict[str, Any]]:
        resolved: dict[str, dict[str, Any]] = {}
        for role, artifact_id in artifact_ids_for_item(item).items():
            if not artifact_id:
                continue
            ref = artifact_refs[artifact_id]
            blob = resolver.read(ref.hash, expected_size=ref.size_bytes)
            text = self._artifact_prompt_text(ref, blob.content)
            resolved[role] = {
                "id": ref.id,
                "hash": ref.hash,
                "media_type": ref.media_type,
                "driver": ref.driver,
                "text": text,
            }
        return resolved

    def _artifact_prompt_text(self, artifact: ArtifactRef, content: bytes) -> str:
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ArtifactError(f"artifact {artifact.id!r} is not valid UTF-8 text") from exc

        driver = artifact.driver.strip().lower()
        if driver in _SUPPORTED_TEXT_DRIVERS:
            self._validate_text_artifact(artifact)
            return text
        if driver in _SUPPORTED_PREVIEW_DRIVERS:
            self._validate_preview_artifact(artifact)
            return _preview_bundle_prompt_text(artifact, text)
        raise ArtifactError(f"artifact driver not supported yet: {artifact.driver}")

    def _validate_text_artifact(self, artifact: ArtifactRef) -> None:
        driver = artifact.driver.strip().lower()
        if driver not in _SUPPORTED_TEXT_DRIVERS:
            raise ArtifactError(f"artifact driver not supported yet: {artifact.driver}")
        media_type = artifact.media_type.strip().lower()
        if media_type and not (media_type.startswith("text/") or media_type in {"application/markdown"}):
            raise ArtifactError(f"artifact media type not supported yet: {artifact.media_type}")

    def _validate_preview_artifact(self, artifact: ArtifactRef) -> None:
        media_type = artifact.media_type.strip().lower()
        if media_type != "application/json":
            raise ArtifactError(f"preview artifact media type not supported yet: {artifact.media_type}")

    def _split_items(self, items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
        has_explicit_split = any(item.get("split") for item in items)
        if has_explicit_split:
            for item in items:
                splits[self._canonical_split(str(item.get("split") or ""))].append(item)
        else:
            splits = self._deterministic_holdout_split(items)
        self._validate_splits(splits, explicit=has_explicit_split)
        return splits

    def _item_split(self, metadata: dict[str, Any]) -> str:
        split = str(metadata.get("split") or "").strip()
        if not split:
            return ""
        return self._canonical_split(split)

    def _canonical_split(self, split: str) -> str:
        split = str(split or "").strip()
        if not split:
            return "train"
        if split not in _SPLIT_ALIASES:
            raise ValueError(f"Gitmoot metadata.split {split!r} is not supported")
        return _SPLIT_ALIASES[split]

    def _deterministic_holdout_split(self, items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        if len(items) < 3:
            return self._small_package_split(items)
        shuffled = list(items)
        random.Random(self.seed).shuffle(shuffled)
        test_count = max(1, len(shuffled) // 5)
        val_count = max(1, len(shuffled) // 5)
        train_count = len(shuffled) - val_count - test_count
        if train_count < 1:
            raise ValueError("Gitmoot split planning requires at least one training item")
        splits = {
            "train": [dict(item, split="train") for item in shuffled[:train_count]],
            "val": [dict(item, split="val") for item in shuffled[train_count : train_count + val_count]],
            "test": [dict(item, split="test") for item in shuffled[train_count + val_count :]],
        }
        return splits

    def _small_package_split(self, items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        if not items:
            raise ValueError("Gitmoot split planning requires at least one item")
        shuffled = list(items)
        random.Random(self.seed).shuffle(shuffled)
        train = dict(shuffled[0], split="train")
        val_source = shuffled[1] if len(shuffled) > 1 else shuffled[0]
        test_source = shuffled[1] if len(shuffled) > 1 else shuffled[0]
        return {
            "train": [train],
            "val": [dict(val_source, split="val")],
            "test": [dict(test_source, split="test")],
        }

    def _validate_splits(self, splits: dict[str, list[dict[str, Any]]], *, explicit: bool) -> None:
        missing = [name for name in ("train", "val", "test") if not splits.get(name)]
        if missing:
            source = "metadata.split" if explicit else "deterministic split"
            raise ValueError(f"Gitmoot {source} must provide non-empty train, val, and test splits")


def build_task_prompt(
    *,
    package: TrainingPackage,
    item: EvalItem,
    artifacts: dict[str, dict[str, Any]],
    feedback_events: list[dict[str, Any]],
    ranked_feedback_events: list[dict[str, Any]] | None = None,
) -> str:
    parts = [
        "# Gitmoot SkillOpt Item",
        f"Template: {package.template.id} ({package.template.version_id})",
        f"Eval run: {package.eval_run.id}",
        f"Item: {item.id}",
        f"Title: {item.title}",
    ]
    if item.metadata is not None:
        parts.extend(["", "## Item Metadata", json.dumps(item.metadata, indent=2, sort_keys=True)])
    preference_summary = _human_preference_summary(ranked_feedback_events or [])
    if preference_summary:
        parts.extend(["", "## Human Preference Summary", preference_summary])
    for role in ("source", "baseline", "candidate", "preview", "diff"):
        artifact = artifacts.get(role)
        if not artifact:
            continue
        parts.extend(
            [
                "",
                f"## {role.title()} Artifact",
                f"Artifact id: {artifact['id']}",
                artifact["text"],
            ]
        )
    option_parts: list[str] = []
    for option in item.options:
        artifact = artifacts.get(f"option:{option.label}")
        if not artifact:
            continue
        option_parts.extend(
            [
                "",
                f"### Option {option.label}",
                f"Artifact id: {artifact['id']}",
            ]
        )
        if option.role:
            option_parts.append(f"Role: {option.role}")
        if option.metadata is not None:
            option_parts.extend(["Metadata:", json.dumps(option.metadata, indent=2, sort_keys=True)])
        option_parts.append(artifact["text"])
    if option_parts:
        parts.extend(["", "## Ranked Option Artifacts", *option_parts])
    if feedback_events:
        parts.extend(["", "## Human Feedback Events", json.dumps(feedback_events, indent=2, sort_keys=True)])
    if ranked_feedback_events:
        parts.extend(
            [
                "",
                "## Ranked Human Feedback Events",
                json.dumps(ranked_feedback_events, indent=2, sort_keys=True),
            ]
        )
    parts.append(
        "\nUse the current skill to produce the requested improved response. "
        "Ground the response in the artifacts and feedback above."
    )
    return "\n".join(parts)


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
        config["mode"] = "landing_page_v1"
    return config


def _profile_requires_landing_page_mode(config: dict[str, Any]) -> bool:
    profile_id = str(config.get("profile_id") or "").strip().lower()
    task_kind = str(config.get("task_kind") or "").strip().lower()
    artifact_contract = str(config.get("artifact_contract") or "").strip().lower()
    return (
        profile_id in {"landing_page_v1", "vue_landing_page_v1"}
        or task_kind == "vue_landing_page"
        or artifact_contract in {"vue_vite_bundle", "vue-vite-bundle"}
    )


def _apply_evaluator_id_mode_override(config: dict[str, Any], override_config: dict[str, Any]) -> None:
    if "mode" in override_config:
        return
    driver = str(override_config.get("driver") or "").strip().lower().replace("-", "_")
    evaluator_id = _normal_evaluator_mode(override_config.get("evaluator_id") or override_config.get("id") or "")
    if not evaluator_id and driver and driver != "manual_review":
        evaluator_id = _normal_evaluator_mode(driver)
    if evaluator_id:
        config["mode"] = evaluator_id


def _normal_evaluator_mode(value: Any) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    if normalized in {"judge", "llm", "llmjudge", "manual_judge", "manual_review", "pairwise"}:
        return "llm_judge"
    if normalized == "landing_page":
        return "landing_page_v1"
    if normalized in {"deterministic", "mock"}:
        return "fixture"
    if normalized == "substring":
        return "contains"
    return normalized


def _feedback_event_prompt_context(event: Any) -> dict[str, str]:
    context = {
        "choice": event.choice,
        "reasoning": event.reasoning,
        "reviewer": event.reviewer,
        "source": event.source,
        "created_at": event.created_at,
    }
    for field in ("quality", "continue_mode", "promote"):
        value = str(getattr(event, field, "") or "").strip()
        if value:
            context[field] = value
    return context


def _ranked_feedback_event_prompt_context(event: Any) -> dict[str, Any]:
    context: dict[str, Any] = {
        "ranking": list(event.ranking),
        "reviewer": event.reviewer,
        "source": event.source,
        "created_at": event.created_at,
    }
    for field in ("winner", "quality", "continue_mode", "promote", "reasoning"):
        value = str(getattr(event, field, "") or "").strip()
        if value:
            context[field] = value
    for field in ("useful_traits", "rejected_traits", "required_improvements"):
        value = getattr(event, field, None)
        if value is not None:
            context[field] = value
    return context


def _human_preference_summary(ranked_feedback_events: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for index, event in enumerate(ranked_feedback_events, start=1):
        lines: list[str] = []
        reviewer = _summary_text(event.get("reviewer"))
        source = _summary_text(event.get("source"))
        label_parts = [part for part in (reviewer, source) if part]
        label = f" ({', '.join(label_parts)})" if label_parts else ""
        ranking = _summary_ranking(event.get("ranking"))
        headline_parts = [f"ranking {ranking}" if ranking else ""]
        winner = _summary_text(event.get("winner"))
        if winner:
            headline_parts.append(f"winner {winner}")
        for field in ("quality", "continue_mode", "promote"):
            value = _summary_text(event.get(field))
            if value:
                headline_parts.append(f"{field} {value}")
        headline = "; ".join(part for part in headline_parts if part) or "ranked feedback"
        lines.append(f"- Review {index}{label}: {headline}.")

        reasoning = _summary_text(event.get("reasoning"), limit=700)
        if reasoning:
            lines.append(f"  Reasoning: {reasoning}")
        useful_traits = _summary_traits(event.get("useful_traits"))
        if useful_traits:
            lines.append(f"  Useful traits: {useful_traits}")
        rejected_traits = _summary_traits(event.get("rejected_traits"))
        if rejected_traits:
            lines.append(f"  Rejected traits: {rejected_traits}")
        required_improvements = _summary_list(event.get("required_improvements"))
        if required_improvements:
            lines.append(f"  Required improvements: {required_improvements}")
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def _summary_ranking(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    labels = [_summary_text(item) for item in value]
    labels = [label for label in labels if label]
    return " > ".join(labels)


def _summary_traits(value: Any) -> str:
    if isinstance(value, dict):
        parts: list[str] = []
        for key in sorted(value):
            label = _summary_text(key)
            summary = _summary_list(value.get(key))
            if label and summary:
                parts.append(f"{label}: {summary}")
        return "; ".join(parts)
    return _summary_list(value)


def _summary_list(value: Any) -> str:
    if isinstance(value, list):
        items = [_summary_text(item, limit=180) for item in value]
        return "; ".join(item for item in items if item)
    return _summary_text(value)


def _summary_text(value: Any, *, limit: int = 220) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, sort_keys=True)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def _preview_bundle_prompt_text(artifact: ArtifactRef, text: str) -> str:
    try:
        bundle = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"preview artifact {artifact.id!r} is not valid JSON") from exc
    if not isinstance(bundle, dict):
        raise ArtifactError(f"preview artifact {artifact.id!r} must be a JSON object")
    files = bundle.get("files")
    if not isinstance(files, list) or not files:
        raise ArtifactError(f"preview artifact {artifact.id!r} must include files")

    parts = [
        f"Preview bundle renderer: {str(bundle.get('renderer') or '').strip()}",
        f"Build command: {str(bundle.get('build_command') or '').strip()}",
        f"Dist dir: {str(bundle.get('dist_dir') or '').strip()}",
    ]
    for index, file_entry in enumerate(files, start=1):
        if not isinstance(file_entry, dict):
            raise ArtifactError(f"preview artifact {artifact.id!r} file {index} must be an object")
        path = str(file_entry.get("path") or "").strip()
        if "content" not in file_entry:
            raise ArtifactError(f"preview artifact {artifact.id!r} file {index} must include path and content")
        content = str(file_entry["content"])
        if not path:
            raise ArtifactError(f"preview artifact {artifact.id!r} file {index} must include path and content")
        parts.extend(["", f"### {path}", content])
    return "\n".join(parts)


def load_package(path: str | Path) -> TrainingPackage:
    return TrainingPackage.load(path)
