"""LiveMathematicianBench evaluation helpers."""
from __future__ import annotations

import re


def extract_answer(text: str) -> str:
    matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if lines:
        return lines[-1]
    return text.strip()


def normalize_label(text: str) -> str:
    return str(text).strip().upper().rstrip(".):")


def parse_choice_label(prediction_text: str, choices: list[dict]) -> str:
    answer = extract_answer(prediction_text)
    label = normalize_label(answer)
    valid_labels = {normalize_label(choice.get("label", "")) for choice in choices}
    if label in valid_labels:
        return label

    answer_lower = answer.lower()
    for choice in choices:
        choice_label = normalize_label(choice.get("label", ""))
        choice_text = str(choice.get("text", "")).strip()
        if choice_text and choice_text.lower() == answer_lower:
            return choice_label

    first_token = normalize_label(answer.split()[0]) if answer.split() else ""
    if first_token in valid_labels:
        return first_token
    return label


def evaluate(prediction_text: str, correct_choice: dict, choices: list[dict]) -> dict:
    predicted_label = parse_choice_label(prediction_text, choices)
    correct_label = normalize_label(correct_choice.get("label", ""))
    predicted_text = ""
    correct_text = str(correct_choice.get("text", "")).strip()

    for choice in choices:
        if normalize_label(choice.get("label", "")) == predicted_label:
            predicted_text = str(choice.get("text", "")).strip()
            break

    is_correct = float(predicted_label == correct_label)
    return {
        "em": is_correct,
        "f1": is_correct,
        "sub_em": is_correct,
        "predicted_answer": predicted_label or extract_answer(prediction_text),
        "predicted_label": predicted_label,
        "predicted_text": predicted_text,
        "correct_label": correct_label,
        "correct_text": correct_text,
    }
