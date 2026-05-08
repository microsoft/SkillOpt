from __future__ import annotations

import re
import string
from collections import Counter


_NUMERIC_CHARS = set("0123456789.-")


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = text.replace(",", "")
    text = "".join(ch for ch in text if ch not in string.punctuation or ch in _NUMERIC_CHARS or ch == "%")
    text = re.sub(r"\b(million|millions|billion|billions|dollars|dollar|nominal)\b", " ", text)
    text = " ".join(text.split())
    return text


def exact_match(prediction: str, gold: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return 1.0 if pred_tokens == gold_tokens else 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(pred_tokens)
    recall = n_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate(prediction: str, gold: str) -> dict:
    em = exact_match(prediction, gold)
    f1 = token_f1(prediction, gold)
    return {
        "em": em,
        "f1": f1,
        "predicted_answer": prediction.strip(),
        "gold_answer": gold,
    }
