"""LiveMathematicianBench task dataloader."""
from __future__ import annotations

import glob
import hashlib
import json
import os
import random
from typing import Any

from skillopt.datasets.base import BatchSpec, SplitDataLoader


# ── Raw data loading utilities (for preprocessing / standalone eval) ─────

_CHOICE_LABELS = ["A", "B", "C", "D", "E", "F", "G"]


def _load_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


def _iter_monthly_files(data_path: str) -> list[str]:
    if not data_path:
        return []
    if os.path.isfile(data_path):
        return [data_path]
    if os.path.isdir(data_path):
        nested = glob.glob(
            os.path.join(data_path, "**", "qa_*_final.json"),
            recursive=True,
        )
        flat = glob.glob(os.path.join(data_path, "qa_*_final.json"))
        return sorted(set(nested + flat))
    return []


def _coerce_choices(raw_choices: Any) -> list[dict]:
    if isinstance(raw_choices, list):
        choices: list[dict] = []
        for idx, item in enumerate(raw_choices):
            if isinstance(item, dict):
                label = str(item.get("label") or _CHOICE_LABELS[idx]).strip()
                text = str(item.get("text") or item.get("content") or "").strip()
            else:
                label = _CHOICE_LABELS[idx]
                text = str(item).strip()
            if text:
                choices.append({"label": label, "text": text})
        return choices

    if isinstance(raw_choices, dict):
        labels = sorted(raw_choices.keys())
        return [
            {"label": str(label).strip(), "text": str(raw_choices[label]).strip()}
            for label in labels
            if str(raw_choices[label]).strip()
        ]

    return []


def _coerce_theorem_types(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if raw is None:
        return []
    text = str(raw).strip()
    return [text] if text else []


def _normalize_label(text: str) -> str:
    return str(text).strip().upper().rstrip(".):")


def _normalize_item(item: dict, row_idx: int, source_path: str) -> dict:
    mcq = item.get("mcq", {}) if isinstance(item.get("mcq"), dict) else {}
    question = str(mcq.get("question") or item.get("question") or "").strip()
    choices = _coerce_choices(mcq.get("choices") or item.get("choices") or [])
    correct = mcq.get("correct_choice") or item.get("correct_choice") or {}

    if isinstance(correct, dict):
        correct_label = _normalize_label(correct.get("label", ""))
        correct_text = str(correct.get("text") or "").strip()
    else:
        correct_label = _normalize_label(correct)
        correct_text = ""

    choice_by_label = {
        _normalize_label(choice["label"]): choice["text"]
        for choice in choices
    }
    if correct_label and not correct_text:
        correct_text = choice_by_label.get(correct_label, "")
    if correct_label and correct_text and correct_label not in choice_by_label:
        choices.append({"label": correct_label, "text": correct_text})
        choices.sort(key=lambda choice: _CHOICE_LABELS.index(choice["label"]) if choice["label"] in _CHOICE_LABELS else len(_CHOICE_LABELS))
        choice_by_label[correct_label] = correct_text

    month = str(item.get("month") or "").strip()
    item_no = item.get("no", row_idx + 1)
    item_id = f"{month}:{item_no}" if month else str(item_no)

    return {
        "id": item_id,
        "month": month,
        "no": item_no,
        "paper_link": str(item.get("paper_link") or "").strip(),
        "theorem": str(item.get("theorem") or "").strip(),
        "sketch": str(item.get("sketch") or "").strip(),
        "theorem_type": _coerce_theorem_types(item.get("theorem_type")),
        "question": question,
        "choices": choices,
        "correct_choice": {
            "label": correct_label,
            "text": correct_text,
        },
        "source_path": source_path,
    }


def load_items(data_path: str) -> list[dict]:
    """Load and normalise LiveMathematicianBench items from JSON files."""
    files = _iter_monthly_files(data_path)
    if not files:
        raise ValueError(
            "LiveMathematicianBench requires data_path to be a qa_*_final.json file "
            "or a directory containing monthly qa_*_final.json files."
        )

    items: list[dict] = []
    for path in files:
        raw = _load_json(path)
        if not isinstance(raw, list):
            raise ValueError(f"Expected JSON array in {path}, got {type(raw).__name__}")
        for row_idx, item in enumerate(raw):
            norm = _normalize_item(item, row_idx=row_idx, source_path=path)
            if norm["question"] and norm["choices"] and norm["correct_choice"]["label"]:
                items.append(norm)
    if not items:
        raise ValueError(f"No valid LiveMathematicianBench items loaded from {data_path}")
    return items


# ── Dataloader ───────────────────────────────────────────────────────────

class LiveMathematicianBenchDataLoader(SplitDataLoader):
    """LiveMathematicianBench dataloader with per-seed choice shuffling."""

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
        shuffle_choices: bool = True,
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
        self.shuffle_choices = shuffle_choices
        self._task_types: list[str] = []

    def load_raw_items(self, data_path: str) -> list[dict]:
        return load_items(data_path)

    def setup(self, cfg: dict) -> None:
        super().setup(cfg)
        all_items = self.train_items + self.val_items + self.test_items
        task_types: set[str] = set()
        for item in all_items:
            for name in item.get("theorem_type", []):
                if name:
                    task_types.add(name)
        self._task_types = sorted(task_types)

    def get_task_types(self) -> list[str]:
        return list(self._task_types)

    # ── Choice shuffling ─────────────────────────────────────────────────

    @staticmethod
    def _item_shuffle_seed(item_id: str, seed: int) -> int:
        digest = hashlib.sha256(f"{seed}:{item_id}".encode("utf-8")).hexdigest()
        return int(digest[:16], 16)

    def _shuffle_item_choices(self, item: dict, seed: int) -> dict:
        if not self.shuffle_choices:
            return {
                **item,
                "choices": [dict(c) for c in item["choices"]],
                "correct_choice": dict(item["correct_choice"]),
            }

        shuffled_choices = [dict(c) for c in item["choices"]]
        rng = random.Random(self._item_shuffle_seed(str(item["id"]), seed))
        rng.shuffle(shuffled_choices)

        original_correct = _normalize_label(item["correct_choice"]["label"])
        remapped_choices: list[dict] = []
        new_correct_choice = dict(item["correct_choice"])

        for idx, choice in enumerate(shuffled_choices):
            new_label = _CHOICE_LABELS[idx]
            old_label = _normalize_label(choice["label"])
            remapped_choices.append({"label": new_label, "text": choice["text"]})
            if old_label == original_correct:
                new_correct_choice = {"label": new_label, "text": choice["text"]}

        transformed = dict(item)
        transformed["choices"] = remapped_choices
        transformed["correct_choice"] = new_correct_choice
        return transformed

    def _materialize_batch(self, items: list[dict], seed: int) -> list[dict]:
        return [self._shuffle_item_choices(item, seed) for item in items]

    # ── Batch construction (override for choice shuffling) ───────────────

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
        """Build a shuffled epoch while preserving per-batch choice shuffling."""
        epoch_rng = random.Random(seed + epoch * 1000)
        items = list(self.train_items)
        epoch_rng.shuffle(items)

        total_batches = steps_per_epoch * accumulation
        if total_batches <= 0:
            return []

        batches: list[BatchSpec] = []
        cursor = 0
        for batch_idx in range(total_batches):
            batch_seed = seed + epoch * 1000 + batch_idx + 1
            batch_items = items[cursor: cursor + batch_size]
            cursor += len(batch_items)

            if not batch_items and items:
                refill_rng = random.Random(batch_seed)
                batch_items = list(items)
                refill_rng.shuffle(batch_items)
                batch_items = batch_items[:batch_size]

            batch_items = self._materialize_batch(batch_items, batch_seed)
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
        rng = random.Random(seed)
        items = list(self.train_items)
        rng.shuffle(items)
        items = self._materialize_batch(items[:batch_size], seed)
        return BatchSpec(
            phase="train",
            split="train",
            seed=seed,
            batch_size=len(items),
            payload=items,
        )

    def build_eval_batch(
        self,
        env_num: int,
        split: str,
        seed: int,
        **kwargs,
    ) -> BatchSpec:
        items = self.get_split_items(split)
        if env_num and env_num < len(items):
            items = items[:env_num]
        items = self._materialize_batch(items, seed)
        return BatchSpec(
            phase="eval",
            split=split,
            seed=seed,
            batch_size=len(items),
            payload=items,
        )
