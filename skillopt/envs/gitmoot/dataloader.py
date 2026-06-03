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
        prompt = build_task_prompt(
            package=self.package,
            item=item,
            artifacts=artifacts,
            feedback_events=[
                {
                    "choice": event.choice,
                    "reasoning": event.reasoning,
                    "reviewer": event.reviewer,
                    "source": event.source,
                    "created_at": event.created_at,
                }
                for event in feedback_events
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
            "evaluator_config": self.package.evaluator_config or {},
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
            self._validate_text_artifact(ref)
            blob = resolver.read(ref.hash, expected_size=ref.size_bytes)
            try:
                text = blob.content.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ArtifactError(f"artifact {ref.id!r} is not valid UTF-8 text") from exc
            resolved[role] = {
                "id": ref.id,
                "hash": ref.hash,
                "media_type": ref.media_type,
                "driver": ref.driver,
                "text": text,
            }
        return resolved

    def _validate_text_artifact(self, artifact: ArtifactRef) -> None:
        driver = artifact.driver.strip().lower()
        if driver not in _SUPPORTED_TEXT_DRIVERS:
            raise ArtifactError(f"artifact driver not supported yet: {artifact.driver}")
        media_type = artifact.media_type.strip().lower()
        if media_type and not (media_type.startswith("text/") or media_type in {"application/markdown"}):
            raise ArtifactError(f"artifact media type not supported yet: {artifact.media_type}")

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
    if feedback_events:
        parts.extend(["", "## Human Feedback Events", json.dumps(feedback_events, indent=2, sort_keys=True)])
    parts.append(
        "\nUse the current skill to produce the requested improved response. "
        "Ground the response in the artifacts and feedback above."
    )
    return "\n".join(parts)


def load_package(path: str | Path) -> TrainingPackage:
    return TrainingPackage.load(path)
