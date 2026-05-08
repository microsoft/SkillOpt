"""MathVerse evaluation helpers."""
from __future__ import annotations

import re
import string

from skillopt.model import chat_with_deployment
from skillopt.prompts import load_prompt


_EVAL_MODE = "mathverse_choice_or_judge_v1"


def normalize_text(text: str) -> str:
    text = str(text or "").strip().lower()
    text = text.replace("\\,", " ")
    text = text.replace("\\ ", " ")
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def normalize_math_text(text: str) -> str:
    text = str(text or "").strip()
    text = text.replace("$", "")
    text = text.replace("\\mathrm", "")
    text = text.replace("{", "")
    text = text.replace("}", "")
    text = text.replace("~", " ")
    text = text.replace("\\,", " ")
    text = text.replace("\\ ", " ")
    return " ".join(text.split()).lower()


def extract_answer(text: str | None) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""

    tags = re.findall(r"<answer>\s*(.*?)\s*</answer>", raw, re.IGNORECASE | re.DOTALL)
    if tags:
        return tags[-1].strip()

    boxed = re.findall(r"\\boxed\{(.*?)\}", raw, re.IGNORECASE | re.DOTALL)
    if boxed:
        return boxed[-1].strip()

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if lines:
        return lines[-1]
    return raw


def _judge_answer(
    *,
    item: dict,
    extracted_answer: str,
    judge_model: str,
    max_completion_tokens: int,
    retries: int,
) -> dict:
    question = str(item.get("question_for_eval") or item.get("question") or "").strip()
    ground_truth = str(item.get("answer") or "").strip()
    raw, _ = chat_with_deployment(
        deployment=judge_model,
        system="You are a careful and strict mathematical answer evaluator.",
        user=load_prompt("judge", env="mathverse").format(
            question=question,
            groundtruth=ground_truth,
            modeloutput=extracted_answer,
        ),
        max_completion_tokens=max_completion_tokens,
        retries=retries,
        stage="mathverse_judge",
    )
    response = str(raw).strip().lower()
    if "true" in response:
        correct = True
    elif "false" in response:
        correct = False
    else:
        correct = False
    return {
        "raw": raw,
        "correct": correct,
        "reason": response,
        "matched_gold": ground_truth if correct else "",
    }


def evaluate_item(
    *,
    item: dict,
    prediction_text: str,
    judge_model: str,
    max_completion_tokens: int = 256,
    retries: int = 5,
) -> dict:
    extracted = extract_answer(prediction_text)

    if item.get("is_choice"):
        predicted_label = str(extracted).strip().upper().rstrip(".):")
        correct_label = str(item["correct_choice"].get("label") or "").strip().upper()
        predicted_text = ""
        for choice in item.get("choices") or []:
            if str(choice.get("label") or "").strip().upper() == predicted_label:
                predicted_text = str(choice.get("text") or "").strip()
                break
        hard = 1.0 if predicted_label == correct_label else 0.0
        return {
            "evaluation_mode": _EVAL_MODE,
            "predicted_answer": extracted,
            "predicted_label": predicted_label,
            "predicted_text": predicted_text,
            "correct_label": correct_label,
            "correct_text": str(item["correct_choice"].get("text") or "").strip(),
            "em": hard,
            "f1": hard,
            "sub_em": hard,
            "judge_raw": "",
            "judge_reason": "exact_label_match" if hard else "label_mismatch",
            "matched_gold": correct_label if hard else "",
        }

    gold_answer = str(item.get("answer") or "").strip()
    pred_norm = normalize_math_text(extracted)
    gold_norm = normalize_math_text(gold_answer)
    if pred_norm and gold_norm and pred_norm == gold_norm:
        return {
            "evaluation_mode": _EVAL_MODE,
            "predicted_answer": extracted,
            "em": 1.0,
            "f1": 1.0,
            "sub_em": 1.0,
            "judge_raw": "",
            "judge_reason": "normalized_exact_match",
            "matched_gold": gold_answer,
            "string_f1": 1.0,
        }

    judge = _judge_answer(
        item=item,
        extracted_answer=extracted,
        judge_model=judge_model,
        max_completion_tokens=max_completion_tokens,
        retries=retries,
    )
    hard = 1.0 if judge["correct"] else 0.0
    pred_tokens = normalize_text(extracted).split()
    gold_tokens = normalize_text(gold_answer).split()
    overlap = 0
    gold_counts: dict[str, int] = {}
    for tok in gold_tokens:
        gold_counts[tok] = gold_counts.get(tok, 0) + 1
    for tok in pred_tokens:
        count = gold_counts.get(tok, 0)
        if count > 0:
            overlap += 1
            gold_counts[tok] = count - 1
    if pred_tokens and gold_tokens and overlap:
        precision = overlap / len(pred_tokens)
        recall = overlap / len(gold_tokens)
        string_f1 = 2 * precision * recall / (precision + recall)
    else:
        string_f1 = 0.0

    return {
        "evaluation_mode": _EVAL_MODE,
        "predicted_answer": extracted,
        "em": hard,
        "f1": hard,
        "sub_em": hard,
        "judge_raw": judge["raw"],
        "judge_reason": judge["reason"],
        "matched_gold": judge["matched_gold"],
        "string_f1": string_f1,
    }


def evaluation_mode() -> str:
    return _EVAL_MODE
