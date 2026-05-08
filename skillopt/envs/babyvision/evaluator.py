"""BabyVision evaluation helpers using the official-style LLM judge."""
from __future__ import annotations

import re
import string

import regex

from skillopt.model import chat_with_deployment
from skillopt.prompts import load_prompt

_EVAL_MODE = "babyvision_judge_v2_official_style"

def normalize_text(text: str) -> str:
    text = str(text).strip().lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


def extract_boxed_answer(text: str | None) -> str | None:
    """Extract the final answer using the official BabyVision rule."""
    if text is None:
        return None

    pattern = r'\\boxed\{((?:[^{}]|{(?:[^{}]|{.*})*})*)\}'
    matches = regex.findall(pattern, text)
    if matches:
        return matches[-1]

    pattern_alt = r'<\|begin_of_box\|>(.*?)<\|end_of_box\|>'
    matches_alt = regex.findall(pattern_alt, text)
    if matches_alt:
        return matches_alt[-1].strip()

    return None


def _token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_set = {}
    gold_set = {}
    for tok in pred_tokens:
        pred_set[tok] = pred_set.get(tok, 0) + 1
    for tok in gold_tokens:
        gold_set[tok] = gold_set.get(tok, 0) + 1
    common = 0
    for tok, count in pred_set.items():
        common += min(count, gold_set.get(tok, 0))
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _format_choices(choices: list[dict]) -> str:
    return "\n".join(f"{choice['label']}. {choice['text']}" for choice in choices)


def _judge_answer(
    *,
    item: dict,
    prediction_text: str,
    extracted_answer: str,
    judge_model: str,
    max_completion_tokens: int,
    retries: int,
) -> dict:
    if item["ans_type"] == "choice":
        ground_truth = str(item["correct_choice"]["label"])
    else:
        if len(item["blank_answers"]) == 1:
            ground_truth = item["blank_answers"][0]
        else:
            ground_truth = " | ".join(item["blank_answers"])

    question = str(item["question"])
    if item["ans_type"] == "choice" and item.get("choices"):
        question = f"{question}\nChoices:\n{_format_choices(item['choices'])}"

    raw, _ = chat_with_deployment(
        deployment=judge_model,
        system="You are a careful and strict evaluator.",
        user=load_prompt("judge", env="babyvision").format(
            question=question,
            groundtruth=ground_truth,
            modeloutput=extracted_answer,
        ),
        max_completion_tokens=max_completion_tokens,
        retries=retries,
        stage="babyvision_judge",
    )
    judge_response_clean = str(raw).strip().lower()
    if "true" in judge_response_clean:
        correct = True
    elif "false" in judge_response_clean:
        correct = False
    else:
        correct = False
    return {
        "raw": raw,
        "correct": correct,
        "reason": judge_response_clean,
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
    answer = extract_boxed_answer(prediction_text)
    judge = _judge_answer(
        item=item,
        prediction_text=prediction_text,
        extracted_answer=answer,
        judge_model=judge_model,
        max_completion_tokens=max_completion_tokens,
        retries=retries,
    )
    hard = 1.0 if judge["correct"] else 0.0

    result = {
        "evaluation_mode": _EVAL_MODE,
        "predicted_answer": answer,
        "em": hard,
        "f1": hard,
        "sub_em": hard,
        "judge_model": judge_model,
        "judge_raw": judge["raw"],
        "judge_reason": judge["reason"],
        "matched_gold": judge["matched_gold"],
    }

    if item["ans_type"] == "choice":
        result["predicted_label"] = str(answer or "").strip().upper().rstrip(".):")
        result["predicted_text"] = ""
        result["correct_label"] = str(item["correct_choice"].get("label") or "")
        result["correct_text"] = str(item["correct_choice"].get("text") or "")
    else:
        result["gold_answers"] = list(item["blank_answers"])
        best_f1 = 0.0
        for gold in item["blank_answers"]:
            best_f1 = max(best_f1, _token_f1(str(answer or ""), gold))
        result["string_f1"] = best_f1

    return result


def evaluation_mode() -> str:
    return _EVAL_MODE
