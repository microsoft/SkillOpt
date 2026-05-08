"""BabyVision task dataloader."""
from __future__ import annotations

import json
import os
from typing import Any

from skillopt.datasets.base import SplitDataLoader


# ── Raw data loading utilities (for preprocessing / standalone eval) ─────

_CHOICE_LABELS = ["A", "B", "C", "D", "E", "F", "G"]


def _iter_jsonl(path: str) -> list[dict]:
    items: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _normalize_ans_type(raw: Any, options: list[dict], choice_answer: Any) -> str:
    text = str(raw or "").strip().lower()
    if text in {"choice", "multiple_choice", "mcq", "option"}:
        return "choice"
    if text in {"blank", "open", "open_ended", "fill_blank", "short_answer"}:
        return "blank"
    if options or choice_answer not in (None, "", []):
        return "choice"
    return "blank"


def _coerce_options(raw: Any) -> list[dict]:
    options: list[dict] = []
    if isinstance(raw, list):
        for idx, item in enumerate(raw):
            if isinstance(item, dict):
                text = str(item.get("text") or item.get("content") or item.get("option") or "").strip()
                label = str(item.get("label") or _CHOICE_LABELS[idx]).strip()
            else:
                text = str(item).strip()
                label = _CHOICE_LABELS[idx]
            if text:
                options.append({"label": label, "text": text})
    elif isinstance(raw, dict):
        for idx, (key, value) in enumerate(raw.items()):
            text = str(value).strip()
            if text:
                options.append({"label": str(key).strip() or _CHOICE_LABELS[idx], "text": text})
    return options


def _normalize_choice_answer(choice_answer: Any, options: list[dict]) -> dict[str, str]:
    if not options:
        return {"label": "", "text": ""}

    if isinstance(choice_answer, dict):
        label = str(choice_answer.get("label") or "").strip().upper()
        text = str(choice_answer.get("text") or "").strip()
        for option in options:
            if label and option["label"].strip().upper() == label:
                return {"label": option["label"], "text": option["text"]}
            if text and option["text"] == text:
                return {"label": option["label"], "text": option["text"]}

    if isinstance(choice_answer, int):
        idx = choice_answer
        if 0 <= idx < len(options):
            return dict(options[idx])
        if 1 <= idx <= len(options):
            return dict(options[idx - 1])

    text = str(choice_answer or "").strip()
    label = text.upper().rstrip(".):")
    for option in options:
        if option["label"].strip().upper() == label:
            return dict(option)
        if option["text"] == text:
            return dict(option)

    return {"label": "", "text": ""}


def _coerce_blank_answers(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    if raw is None:
        return []
    text = str(raw).strip()
    return [text] if text else []


def load_items(data_path: str) -> list[dict]:
    """Load and normalise BabyVision items from a directory or JSONL file."""
    if not data_path:
        raise ValueError("BabyVision requires data_path pointing to a local dataset directory or meta_data.jsonl.")

    if os.path.isdir(data_path):
        meta_path = os.path.join(data_path, "meta_data.jsonl")
        image_root = os.path.join(data_path, "images")
    else:
        meta_path = data_path
        image_root = os.path.join(os.path.dirname(data_path), "images")

    if not os.path.exists(meta_path):
        raise ValueError(
            "BabyVision expected a meta_data.jsonl file. "
            f"Could not find: {meta_path}"
        )

    raw_items = _iter_jsonl(meta_path)
    items: list[dict] = []
    for idx, raw in enumerate(raw_items):
        options = _coerce_options(raw.get("options") or raw.get("choices") or raw.get("choiceOptions"))
        ans_type = _normalize_ans_type(raw.get("ansType"), options, raw.get("choiceAns"))
        correct_choice = _normalize_choice_answer(raw.get("choiceAns"), options)
        blank_answers = _coerce_blank_answers(raw.get("blankAns"))

        image_name = str(
            raw.get("image")
            or raw.get("image_path")
            or raw.get("image_file")
            or raw.get("img")
            or ""
        ).strip()
        if not image_name:
            continue
        image_path = image_name if os.path.isabs(image_name) else os.path.join(image_root, image_name)
        if not os.path.exists(image_path):
            alt = os.path.join(os.path.dirname(meta_path), image_name)
            if os.path.exists(alt):
                image_path = alt
            else:
                continue

        task_id = str(raw.get("taskId") or raw.get("id") or idx + 1)
        task_type = str(raw.get("type") or raw.get("taskType") or "unknown").strip() or "unknown"
        subtype = str(raw.get("subtype") or raw.get("subType") or task_type).strip() or task_type
        question = str(raw.get("question") or raw.get("query") or "").strip()
        if not question:
            continue

        if ans_type == "choice" and not correct_choice["label"]:
            continue
        if ans_type != "choice" and not blank_answers:
            continue

        items.append({
            "id": task_id,
            "task_type": task_type,
            "subtype": subtype,
            "question": question,
            "image_path": os.path.abspath(image_path),
            "ans_type": ans_type,
            "choices": options,
            "correct_choice": correct_choice,
            "blank_answers": blank_answers,
            "cot": str(raw.get("coT") or raw.get("cot") or "").strip(),
            "source_path": os.path.abspath(meta_path),
        })

    if not items:
        raise ValueError(f"No valid BabyVision items loaded from {data_path}")
    return items


# ── Dataloader ───────────────────────────────────────────────────────────

class BabyVisionDataLoader(SplitDataLoader):
    """BabyVision dataloader."""

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
            item.get("subtype") or item.get("task_type") or "unknown"
            for item in all_items
        }
        self._task_types = sorted(task_types)

    def get_task_types(self) -> list[str]:
        return list(self._task_types)
