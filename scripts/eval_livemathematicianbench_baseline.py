#!/usr/bin/env python3
"""Evaluate LiveMathematicianBench under current or official-style prompts."""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from skillopt.envs.livemathematicianbench.dataloader import load_items
from skillopt.envs.livemathematicianbench.evaluator import evaluate as current_evaluate
from skillopt.envs.livemathematicianbench.rollout import _build_system, _build_user
from skillopt.model import (
    chat_with_deployment,
    configure_azure_openai,
    set_backend,
    set_reasoning_effort,
)

_LABELS = ["A", "B", "C", "D", "E"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--model", type=str, default="gpt-5.4")
    p.add_argument("--backend", type=str, choices=["azure_openai", "codex", "claude"], default="azure_openai")
    p.add_argument("--mode", type=str, choices=["current", "official"], required=True)
    p.add_argument("--reasoning_effort", type=str, default=None)
    p.add_argument("--azure_endpoint", type=str, default="")
    p.add_argument("--azure_api_version", type=str, default="")
    p.add_argument("--azure_api_key", type=str, default="")
    p.add_argument("--max_completion_tokens", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260227)
    p.add_argument("--skill_path", type=str, default="skillopt/envs/livemathematicianbench/skills/initial.md")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--output_json", type=str, required=True)
    return p.parse_args()

def read_skill(skill_path: str) -> str:
    with open(skill_path, encoding="utf-8") as f:
        return f.read()


def official_extract_answer(response_text: str) -> str | None:
    if not response_text:
        return None
    boxed_match = re.search(r"\\boxed\{([A-Ea-e])\}", response_text)
    if boxed_match:
        return boxed_match.group(1).upper()
    boxed_match = re.search(r"boxed\{([A-Ea-e])\}", response_text)
    if boxed_match:
        return boxed_match.group(1).upper()
    answer_match = re.search(r"answer is[:\s]*([A-Ea-e])", response_text, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).upper()
    answer_match = re.search(r"Answer[:\s]*\(?([A-Ea-e])\)?", response_text)
    if answer_match:
        return answer_match.group(1).upper()
    final_match = re.search(r"\b([A-Ea-e])\b\s*[.)]?\s*$", response_text.strip())
    if final_match:
        return final_match.group(1).upper()
    return None


def official_format_mcq_prompt(question: str, choices: list[dict]) -> str:
    lines = [
        "Answer the following multiple-choice question.",
        "Think carefully, then provide your final answer in the format: \\boxed{X} where X is A, B, C, D, or E.",
        "",
        "Question:",
        question,
        "",
        "Choices:",
    ]
    for choice in choices:
        lines.append(f"{choice['label']}. {choice['text']}")
    lines.append("")
    lines.append("Your answer:")
    return "\n".join(lines)


def shuffle_choices(item: dict, seed: int) -> tuple[list[dict], dict]:
    correct_choice = dict(item["correct_choice"])
    all_choices = [dict(choice) for choice in item["choices"]]
    rng = random.Random(f"{seed}:{item['id']}")
    rng.shuffle(all_choices)

    shuffled: list[dict] = []
    new_correct = dict(correct_choice)
    correct_text = correct_choice["text"]

    for idx, choice in enumerate(all_choices[: len(_LABELS)]):
        relabeled = {"label": _LABELS[idx], "text": choice["text"]}
        shuffled.append(relabeled)
        if choice["text"] == correct_text:
            new_correct = dict(relabeled)

    return shuffled, new_correct


def load_existing(output_path: Path) -> dict[str, dict]:
    if not output_path.exists():
        return {}
    with open(output_path, encoding="utf-8") as f:
        payload = json.load(f)
    existing = {}
    for row in payload.get("results", []):
        existing[str(row["id"])] = row
    return existing


def save_results(output_path: Path, meta: dict, results: list[dict]) -> None:
    correct = sum(1 for row in results if row.get("is_correct"))
    total = len(results)
    payload = {
        **meta,
        "summary": {
            "correct": correct,
            "total": total,
            "accuracy": (correct / total) if total else 0.0,
        },
        "results": sorted(results, key=lambda row: str(row["id"])),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def call_model(
    *,
    model: str,
    system: str,
    user: str,
    max_completion_tokens: int | None,
    reasoning_effort: str | None,
) -> str:
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            set_reasoning_effort(reasoning_effort)
            raw, _ = chat_with_deployment(
                deployment=model,
                system=system,
                user=user,
                max_completion_tokens=max_completion_tokens if max_completion_tokens and max_completion_tokens > 0 else 4096,
                retries=1,
                stage="rollout",
            )
            return str(raw or "")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == 4:
                break
            time.sleep(min(2 ** attempt, 10))
    raise RuntimeError(f"LLM call failed after retries: {last_error}")


def evaluate_one(
    item: dict,
    *,
    mode: str,
    model: str,
    skill_content: str,
    max_completion_tokens: int,
    reasoning_effort: str | None,
    seed: int,
) -> dict:
    shuffled_choices, correct_choice = shuffle_choices(item, seed)

    if mode == "official":
        system = "You are an expert mathematician. Answer accurately."
        user = official_format_mcq_prompt(item["question"], shuffled_choices)
        effective_max_completion_tokens = max_completion_tokens if max_completion_tokens > 0 else None
    else:
        materialized = dict(item)
        materialized["choices"] = shuffled_choices
        materialized["correct_choice"] = correct_choice
        system = _build_system(skill_content)
        user = _build_user(materialized, use_theorem=False, use_sketch=False)
        effective_max_completion_tokens = max_completion_tokens if max_completion_tokens > 0 else 768

    t0 = time.time()
    response = call_model(
        model=model,
        system=system,
        user=user,
        max_completion_tokens=effective_max_completion_tokens,
        reasoning_effort=reasoning_effort,
    )
    elapsed = time.time() - t0

    if mode == "official":
        predicted = official_extract_answer(response)
        predicted_text = ""
        for choice in shuffled_choices:
            if choice["label"] == predicted:
                predicted_text = choice["text"]
                break
        is_correct = predicted == correct_choice["label"]
        return {
            "id": item["id"],
            "question": item["question"],
            "correct_label": correct_choice["label"],
            "correct_text": correct_choice["text"],
            "predicted_label": predicted,
            "predicted_text": predicted_text,
            "is_correct": is_correct,
            "elapsed_seconds": elapsed,
            "response": response,
        }

    eval_result = current_evaluate(response, correct_choice, shuffled_choices)
    return {
        "id": item["id"],
        "question": item["question"],
        "correct_label": correct_choice["label"],
        "correct_text": correct_choice["text"],
        "predicted_label": eval_result["predicted_label"],
        "predicted_text": eval_result["predicted_text"],
        "is_correct": bool(eval_result["em"]),
        "elapsed_seconds": elapsed,
        "response": response,
    }


def main() -> None:
    args = parse_args()
    set_backend(args.backend)
    configure_azure_openai(
        endpoint=args.azure_endpoint or None,
        api_version=args.azure_api_version or None,
        api_key=args.azure_api_key or None,
    )
    set_reasoning_effort(args.reasoning_effort)
    output_path = Path(args.output_json).resolve()
    skill_content = read_skill(args.skill_path) if args.mode == "current" else ""

    items = load_items(args.data_path)
    if args.limit:
        items = items[:args.limit]

    existing = load_existing(output_path) if args.resume else {}
    pending = [item for item in items if str(item["id"]) not in existing]
    results = list(existing.values())

    print("=" * 72, flush=True)
    print("LiveMathematicianBench baseline eval", flush=True)
    print("=" * 72, flush=True)
    print(f"Mode: {args.mode}", flush=True)
    print(f"Model: {args.model}", flush=True)
    print(f"Reasoning effort: {args.reasoning_effort}", flush=True)
    print(f"Items: {len(items)} total, {len(pending)} pending, {len(existing)} resumed", flush=True)
    print(f"Output: {output_path}", flush=True)
    print("=" * 72, flush=True)

    meta = {
        "mode": args.mode,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "seed": args.seed,
        "max_completion_tokens": args.max_completion_tokens,
    }

    if not pending:
        save_results(output_path, meta, results)
        summary = json.loads(output_path.read_text(encoding="utf-8"))["summary"]
        print(f"Accuracy: {summary['correct']}/{summary['total']} = {summary['accuracy']:.4f}", flush=True)
        return

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(
                evaluate_one,
                item,
                mode=args.mode,
                model=args.model,
                skill_content=skill_content,
                max_completion_tokens=args.max_completion_tokens,
                reasoning_effort=args.reasoning_effort,
                seed=args.seed,
            ): item
            for item in pending
        }
        completed = 0
        for fut in as_completed(futs):
            item = futs[fut]
            try:
                row = fut.result()
            except Exception as exc:  # noqa: BLE001
                row = {
                    "id": item["id"],
                    "question": item["question"],
                    "correct_label": None,
                    "correct_text": item["correct_choice"]["text"],
                    "predicted_label": None,
                    "predicted_text": "",
                    "is_correct": False,
                    "elapsed_seconds": 0.0,
                    "response": "",
                    "error": str(exc),
                }
            results.append(row)
            completed += 1
            correct = sum(1 for result in results if result.get("is_correct"))
            total = len(results)
            print(
                f"[{completed}/{len(pending)}] id={row['id']} "
                f"pred={row['predicted_label']} gold={row['correct_label']} "
                f"acc={correct}/{total}={correct/total:.4f}",
                flush=True,
            )
            save_results(output_path, meta, results)

    summary = json.loads(output_path.read_text(encoding="utf-8"))["summary"]
    print("=" * 72, flush=True)
    print(f"Accuracy: {summary['correct']}/{summary['total']} = {summary['accuracy']:.4f}", flush=True)


if __name__ == "__main__":
    main()
