"""SearchQA evaluation metrics: Exact Match, F1, and Substring Match.

Normalization follows the SQuAD convention:
  - lowercase
  - remove punctuation
  - remove articles (a, an, the)
  - collapse whitespace

Answer extraction looks for <answer>...</answer> XML tags,
falling back to the last non-empty line of the response.
"""
from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    """Normalize answer string (SQuAD convention)."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s.strip()


def extract_answer(text: str) -> str:
    """Extract answer from <answer>...</answer> tags.

    Fallback: last non-empty line, then full response stripped.
    """
    matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if lines:
        return lines[-1]
    return text.strip()


def exact_match(prediction: str, gold_answers: list[str]) -> float:
    norm_pred = normalize_answer(prediction)
    for gold in gold_answers:
        if normalize_answer(gold) == norm_pred:
            return 1.0
    return 0.0


def f1_score(prediction: str, gold_answers: list[str]) -> float:
    """Token-level F1 (SQuAD-style), max across all gold answers."""
    norm_pred = normalize_answer(prediction)
    pred_tokens = norm_pred.split()

    if not pred_tokens:
        for gold in gold_answers:
            if not normalize_answer(gold).split():
                return 1.0
        return 0.0

    best_f1 = 0.0
    for gold in gold_answers:
        gold_tokens = normalize_answer(gold).split()
        if not gold_tokens:
            continue
        common = Counter(pred_tokens) & Counter(gold_tokens)
        n_common = sum(common.values())
        if n_common == 0:
            continue
        precision = n_common / len(pred_tokens)
        recall = n_common / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)

    return best_f1


def sub_em(prediction: str, gold_answers: list[str]) -> float:
    """1.0 if any normalized gold is a substring of prediction, or vice versa."""
    norm_pred = normalize_answer(prediction)
    for gold in gold_answers:
        norm_gold = normalize_answer(gold)
        if norm_gold in norm_pred or norm_pred in norm_gold:
            return 1.0
    return 0.0


def evaluate(prediction_text: str, gold_answers: list[str]) -> dict:
    """Evaluate a single QA prediction against gold answers.

    Returns dict with: em, f1, sub_em, predicted_answer, gold_answers.
    """
    answer = extract_answer(prediction_text)
    return {
        "em": exact_match(answer, gold_answers),
        "f1": f1_score(answer, gold_answers),
        "sub_em": sub_em(answer, gold_answers),
        "predicted_answer": answer,
        "gold_answers": gold_answers,
    }
