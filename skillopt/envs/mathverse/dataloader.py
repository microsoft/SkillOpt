"""MathVerse task dataloader."""
from __future__ import annotations

import json
import os
import re
from typing import Any

from skillopt.datasets.base import SplitDataLoader


_CHOICE_LABELS = ["A", "B", "C", "D", "E", "F", "G"]
_CHOICE_BLOCK_RE = re.compile(r"\bChoices?\s*:\s*", re.IGNORECASE)
_CHOICE_ITEM_RE = re.compile(r"([A-G])\s*[:.)]\s*(.*?)(?=(?:\s+[A-G]\s*[:.)])|$)", re.DOTALL)


def _load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _normalize_space(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _resolve_image_path(raw_path: str, *, data_root: str, source_path: str) -> str:
    candidates = []
    if raw_path:
        if os.path.isabs(raw_path):
            candidates.append(raw_path)
        else:
            if data_root:
                candidates.append(os.path.join(data_root, raw_path))
                candidates.append(os.path.join(data_root, "images", raw_path))
            candidates.append(os.path.join(os.path.dirname(source_path), raw_path))
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return os.path.abspath(candidate)
    return ""


def _split_question_and_choices(question: str) -> tuple[str, list[dict]]:
    text = str(question or "").strip()
    match = _CHOICE_BLOCK_RE.search(text)
    if not match:
        return text, []

    stem = text[:match.start()].strip()
    choice_block = text[match.end():].strip()
    choices: list[dict] = []
    for idx, m in enumerate(_CHOICE_ITEM_RE.finditer(choice_block)):
        label = (m.group(1) or _CHOICE_LABELS[idx]).strip().upper()
        choice_text = _normalize_space(m.group(2))
        if choice_text:
            choices.append({"label": label, "text": choice_text})
    return stem or text, choices


def _build_text_dominant_map(data_root: str) -> dict[str, str]:
    if not data_root:
        return {}
    candidates = [
        os.path.join(data_root, "testmini.json"),
        os.path.join(data_root, "data", "testmini.json"),
    ]
    source_path = next((path for path in candidates if os.path.exists(path)), "")
    if not source_path:
        return {}

    raw = _load_json(source_path)
    if not isinstance(raw, list):
        return {}

    mapping: dict[str, str] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        if str(item.get("problem_version") or "").strip() != "Text Dominant":
            continue
        problem_index = str(item.get("problem_index") or "").strip()
        question = str(item.get("question") or "").strip()
        if problem_index and question:
            mapping[problem_index] = question
    return mapping


def _normalize_item(
    item: dict,
    *,
    row_idx: int,
    source_path: str,
    data_root: str,
    problem_version: str,
    text_dominant_map: dict[str, str],
) -> dict | None:
    raw_problem_version = str(item.get("problem_version") or "").strip()
    if problem_version and raw_problem_version and raw_problem_version != problem_version:
        return None

    question = str(item.get("question") or "").strip()
    question_type = str(item.get("question_type") or "").strip()
    answer = str(item.get("answer") or "").strip()
    image_rel = str(item.get("image") or "").strip()
    image_path = _resolve_image_path(image_rel, data_root=data_root, source_path=source_path)
    if not answer or not image_path:
        return None

    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    subject = str(metadata.get("subject") or "").strip()
    subfield = str(metadata.get("subfield") or "").strip()
    source = str(metadata.get("source") or "").strip()

    question_stem, choices = _split_question_and_choices(question)
    is_choice = question_type == "multi-choice" or bool(choices)

    correct_choice = {"label": "", "text": ""}
    if is_choice:
        label = str(answer).strip().upper().rstrip(".):")
        choice_text = ""
        for choice in choices:
            if choice["label"].upper() == label:
                choice_text = choice["text"]
                break
        correct_choice = {"label": label, "text": choice_text}

    problem_index = str(item.get("problem_index") or "").strip()
    sample_index = str(item.get("sample_index") or row_idx + 1).strip()
    item_id = problem_index or sample_index
    task_type = subfield or subject or question_type or "mathverse"

    return {
        "id": item_id,
        "sample_index": sample_index,
        "problem_index": problem_index,
        "problem_version": raw_problem_version or problem_version,
        "question": question,
        "question_stem": question_stem,
        "question_for_eval": str(item.get("question_for_eval") or question).strip(),
        "question_type": question_type or ("multi-choice" if is_choice else "free-form"),
        "is_choice": is_choice,
        "choices": choices,
        "correct_choice": correct_choice,
        "answer": answer,
        "gold_answers": [answer] if answer else [],
        "image_rel": image_rel,
        "image_path": image_path,
        "query_wo": str(item.get("query_wo") or "").strip(),
        "query_cot": str(item.get("query_cot") or "").strip(),
        "metadata": {
            "split": str(metadata.get("split") or "").strip(),
            "source": source,
            "subject": subject,
            "subfield": subfield,
        },
        "task_type": task_type,
        "source_path": os.path.abspath(source_path),
        "text_dominant_question": str(
            item.get("text_dominant_question")
            or text_dominant_map.get(problem_index, "")
        ).strip(),
    }


class MathVerseDataLoader(SplitDataLoader):
    """MathVerse dataloader."""

    def __init__(
        self,
        split_dir: str = "",
        seed: int = 42,
        limit: int = 0,
        data_root: str = "",
        problem_version: str = "Text Lite",
        **kwargs,
    ) -> None:
        super().__init__(split_dir=split_dir, seed=seed, limit=limit)
        self.data_root = data_root
        self.problem_version = problem_version
        self._task_types: list[str] = []
        self._text_dominant_map = _build_text_dominant_map(data_root)

    def setup(self, cfg: dict) -> None:
        if not self.data_root:
            self.data_root = str(cfg.get("data_root") or "")
        if not self.problem_version:
            self.problem_version = str(cfg.get("problem_version") or "Text Lite")
        self._text_dominant_map = _build_text_dominant_map(self.data_root)
        super().setup(cfg)
        all_items = self.train_items + self.val_items + self.test_items
        task_types = {
            item.get("task_type") or item.get("question_type") or "mathverse"
            for item in all_items
        }
        self._task_types = sorted(str(x) for x in task_types if str(x).strip())

    def get_task_types(self) -> list[str]:
        return list(self._task_types)

    def load_split_items(self, split_path: str) -> list[dict]:
        raw_items = super().load_split_items(split_path)
        source_path = next(
            (
                os.path.join(split_path, name)
                for name in sorted(os.listdir(split_path))
                if name.endswith(".json")
            ),
            split_path,
        )
        items: list[dict] = []
        for row_idx, item in enumerate(raw_items):
            if not isinstance(item, dict):
                continue
            norm = _normalize_item(
                item,
                row_idx=row_idx,
                source_path=source_path,
                data_root=self.data_root,
                problem_version=self.problem_version,
                text_dominant_map=self._text_dominant_map,
            )
            if norm is not None:
                items.append(norm)
        if not items:
            raise ValueError(
                f"No valid MathVerse items loaded from {split_path} "
                f"for problem_version={self.problem_version!r}"
            )
        return items
