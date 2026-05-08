"""MMRB task dataloader."""
from __future__ import annotations

import glob
import json
import os
import re
from typing import Any

from skillopt.datasets.base import SplitDataLoader


# ── Raw data loading utilities (for preprocessing / standalone eval) ─────

def _load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _iter_data_files(data_path: str) -> list[str]:
    if not data_path:
        return []
    if os.path.isfile(data_path):
        return [data_path]
    if os.path.isdir(data_path):
        nested = glob.glob(os.path.join(data_path, "**", "*_human.json"), recursive=True)
        flat = glob.glob(os.path.join(data_path, "*_human.json"))
        return sorted(set(nested + flat))
    return []


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _normalize_item(item: dict, row_idx: int, source_path: str) -> dict | None:
    question = _normalize_space(item.get("question") or "")
    answer = _normalize_space(item.get("answer") or "")
    raw_image_paths = item.get("image_paths") or []
    if not question or not answer or not isinstance(raw_image_paths, list) or not raw_image_paths:
        return None

    base_dir = os.path.dirname(source_path)
    image_paths: list[str] = []
    for raw_path in raw_image_paths:
        rel = str(raw_path or "").strip()
        if not rel:
            continue
        abs_path = rel if os.path.isabs(rel) else os.path.abspath(os.path.join(base_dir, rel))
        if os.path.exists(abs_path):
            image_paths.append(abs_path)
    if not image_paths:
        return None

    options_raw = item.get("options") or []
    options = [_normalize_space(opt) for opt in options_raw if _normalize_space(opt)]
    source = _normalize_space(item.get("source") or "unknown")
    subtask = _normalize_space(item.get("subtask") or "unknown")
    item_index = item.get("index", row_idx)
    item_id = f"{source}:{subtask}:{item_index}"

    return {
        "id": item_id,
        "source": source,
        "subtask": subtask,
        "task_type": subtask,
        "question": question,
        "answer": answer,
        "options": options,
        "is_choice": bool(options),
        "image_paths": image_paths,
        "reasoning_steps": item.get("reasoning_steps") or [],
        "annotation_time": item.get("annotation_time"),
        "source_path": os.path.abspath(source_path),
    }


def load_items(data_path: str) -> list[dict]:
    """Load and normalise MMRB items from JSON files."""
    files = _iter_data_files(data_path)
    if not files:
        raise ValueError(
            "MMRB requires data_path to be a *_human.json file or a directory "
            "containing extracted MMRB subtask folders."
        )

    items: list[dict] = []
    for path in files:
        raw = _load_json(path)
        if not isinstance(raw, list):
            raise ValueError(f"Expected JSON array in {path}, got {type(raw).__name__}")
        for row_idx, item in enumerate(raw):
            if not isinstance(item, dict):
                continue
            norm = _normalize_item(item, row_idx=row_idx, source_path=path)
            if norm is not None:
                items.append(norm)

    if not items:
        raise ValueError(f"No valid MMRB items loaded from {data_path}")
    return items


# ── Dataloader ───────────────────────────────────────────────────────────

class MMRBDataLoader(SplitDataLoader):
    """MMRB dataloader."""

    def __init__(
        self,
        split_dir: str = "",
        data_path: str = "",
        split_mode: str = "ratio",
        split_ratio: str = "2:1:7",
        split_seed: int = 42,
        split_output_dir: str = "",
        seed: int = 42,
        limit: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(
            split_dir=split_dir,
            data_path=data_path,
            split_mode=split_mode,
            split_ratio=split_ratio,
            split_seed=split_seed,
            split_output_dir=split_output_dir,
            seed=seed,
            limit=limit,
        )
        self._task_types: list[str] = []

    def load_raw_items(self, data_path: str) -> list[dict]:
        return load_items(data_path)

    def setup(self, cfg: dict) -> None:
        super().setup(cfg)
        all_items = self.train_items + self.val_items + self.test_items
        task_types = {
            item.get("subtask") or item.get("task_type") or "unknown"
            for item in all_items
        }
        self._task_types = sorted(task_types)

    def get_task_types(self) -> list[str]:
        return list(self._task_types)
