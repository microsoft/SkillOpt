from __future__ import annotations

import ast
import json
from collections.abc import Iterable
from typing import Any

DEFAULT_ANLS_THRESHOLD = 0.5


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    return " ".join(text.split())


def _levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) > len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (char_a != char_b)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def _score_single_answer(predicted: Any, target: Any, threshold: float) -> float:
    predicted_norm = _normalize_text(predicted)
    target_norm = _normalize_text(target)
    if not predicted_norm and not target_norm:
        return 1.0
    if not predicted_norm or not target_norm:
        return 0.0
    distance = _levenshtein_distance(predicted_norm, target_norm)
    normalized_distance = distance / max(len(predicted_norm), len(target_norm))
    if normalized_distance >= threshold:
        return 0.0
    return 1.0 - normalized_distance


def _extract_answer_strings(raw: Any) -> list[str]:
    if raw is None:
        return [""]
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return [""]
        parsed = None
        if text[0] in "[{":
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(text)
                except (ValueError, SyntaxError):
                    parsed = None
        if parsed is None:
            return [text]
        return _extract_answer_strings(parsed)
    if isinstance(raw, dict):
        for key in ("answers", "ground_truth", "answer"):
            if key in raw:
                return _extract_answer_strings(raw[key])
        return [str(raw)]
    if isinstance(raw, Iterable) and not isinstance(raw, (bytes, bytearray)):
        answers: list[str] = []
        for item in raw:
            if isinstance(item, dict):
                for key in ("text", "answer", "value"):
                    if key in item:
                        answers.extend(_extract_answer_strings(item[key]))
                        break
                else:
                    answers.append(str(item))
                continue
            answers.append(str(item))
        return answers or [""]
    return [str(raw)]


def extract_answer(text: str) -> str:
    lower = text.lower()
    start = lower.rfind("<answer>")
    end = lower.rfind("</answer>")
    if start != -1 and end != -1 and end > start:
        return text[start + len("<answer>"):end].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text.strip()


def evaluate(prediction_text: str, gold_answers: Any) -> dict:
    answer = extract_answer(prediction_text)
    answers = _extract_answer_strings(gold_answers)
    score = 0.0
    for target in answers:
        score = max(score, _score_single_answer(answer, target, DEFAULT_ANLS_THRESHOLD))
    return {
        "anls": score,
        "predicted_answer": answer,
        "gold_answers": answers,
    }
