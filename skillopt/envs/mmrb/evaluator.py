"""MMRB evaluation helpers."""
from __future__ import annotations

import re
import string


_EVAL_MODE = "mmrb_exact_match_v1"


def normalize_text(text: str) -> str:
    text = str(text or "").strip().lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def extract_answer(text: str | None) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""

    answer_tags = re.findall(r"<answer>\s*(.*?)\s*</answer>", raw, re.IGNORECASE | re.DOTALL)
    if answer_tags:
        return answer_tags[-1].strip()

    bracket = re.findall(r"Answer\s*\[\s*(.*?)\s*\]", raw, re.IGNORECASE | re.DOTALL)
    if bracket:
        return bracket[-1].strip()

    boxed = re.findall(r"\\boxed\{(.*?)\}", raw, re.IGNORECASE | re.DOTALL)
    if boxed:
        return boxed[-1].strip()

    single = raw.strip().rstrip(".):")
    if re.fullmatch(r"[A-Z]", single, re.IGNORECASE):
        return single.strip()

    patterns = [
        r"final answer\s*(?:is)?\s*[:：]?\s*(.+)",
        r"the answer is\s*[:：]?\s*(.+)",
        r"answer\s*[:：]?\s*(.+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw, re.IGNORECASE)
        if match:
            return match.group(1).strip().strip("*")

    return raw


def evaluate_item(*, item: dict, prediction_text: str) -> dict:
    predicted_answer = extract_answer(prediction_text)
    gold_answer = str(item.get("answer") or "").strip()
    predicted_norm = normalize_text(predicted_answer)
    gold_norm = normalize_text(gold_answer)

    hard = 0.0
    matched_gold = ""
    predicted_label = ""
    predicted_text = predicted_answer

    if item.get("is_choice"):
        predicted_label = str(predicted_answer).strip().upper().rstrip(".):")
        if predicted_label == str(gold_answer).strip().upper():
            hard = 1.0
            matched_gold = gold_answer
        else:
            for option in item.get("options") or []:
                label_match = re.match(r"\(?([A-Z])\)", option)
                if not label_match:
                    continue
                label = label_match.group(1).upper()
                option_text = option[label_match.end():].strip(" .:-")
                if predicted_norm and normalize_text(option_text) == predicted_norm:
                    predicted_label = label
                    predicted_text = option_text
                    break
            if predicted_label == str(gold_answer).strip().upper():
                hard = 1.0
                matched_gold = gold_answer
    else:
        if predicted_norm and gold_norm and (
            predicted_norm == gold_norm or predicted_norm in gold_norm or gold_norm in predicted_norm
        ):
            hard = 1.0
            matched_gold = gold_answer

    return {
        "evaluation_mode": _EVAL_MODE,
        "predicted_answer": predicted_answer,
        "predicted_label": predicted_label,
        "predicted_text": predicted_text,
        "em": hard,
        "f1": hard,
        "sub_em": hard,
        "matched_gold": matched_gold,
    }


def evaluation_mode() -> str:
    return _EVAL_MODE

