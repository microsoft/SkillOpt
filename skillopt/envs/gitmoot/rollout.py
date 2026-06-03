"""Gitmoot rollout execution."""

from __future__ import annotations

import json
import os
from typing import Any

from skillopt.envs.gitmoot.evaluator import evaluate_response
from skillopt.envs.gitmoot.package import safe_item_path_segment
from skillopt.envs.gitmoot.result_contract import (
    EVALUATOR_FAILED,
    EVALUATOR_NOT_RUN,
    TARGET_FAILED,
    TARGET_PASSED,
    make_unscored_evaluation,
    normalize_scored_evaluation,
)
from skillopt.model import chat_target, is_target_chat_backend


def run_batch(
    *,
    items: list[dict[str, Any]],
    skill_content: str,
    out_root: str,
    max_completion_tokens: int = 4096,
    evaluator_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    return [
        process_one(
            item=item,
            skill_content=skill_content,
            out_root=out_root,
            max_completion_tokens=max_completion_tokens,
            evaluator_config=evaluator_config,
        )
        for item in items
    ]


def process_one(
    *,
    item: dict[str, Any],
    skill_content: str,
    out_root: str,
    max_completion_tokens: int = 4096,
    evaluator_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    item_id = str(item["id"])
    prediction_id = safe_item_path_segment(item_id)
    pred_dir = os.path.join(out_root, "predictions", prediction_id)
    os.makedirs(pred_dir, exist_ok=True)
    target_trace_path = os.path.join(pred_dir, "conversation.json")
    evaluator_trace_path = os.path.join(pred_dir, "result.json")
    system_prompt = _build_system_prompt(skill_content)
    user_prompt = str(item.get("prompt") or "")
    response = ""
    fail_reason = ""
    agent_ok = False
    agent_failed = False

    try:
        response = _run_agent(item, system_prompt, user_prompt, max_completion_tokens)
        agent_ok = True
    except Exception as exc:  # noqa: BLE001 - rollout result records the failure.
        agent_failed = True
        fail_reason = str(exc) or "agent execution failed"

    if agent_failed:
        score = make_unscored_evaluation(
            fail_reason=fail_reason,
            target_status=TARGET_FAILED,
            evaluator_status=EVALUATOR_NOT_RUN,
            blocker="target_rollout_failed",
            target_trace_path=target_trace_path,
            evaluator_trace_path=evaluator_trace_path,
            metadata={"evaluator": "not_run", "agent_error": True},
        )
    else:
        config = evaluator_config if evaluator_config is not None else item.get("evaluator_config")
        try:
            raw_score = evaluate_response(item, response, config if isinstance(config, dict) else {})
            score = normalize_scored_evaluation(
                raw_score,
                target_trace_path=target_trace_path,
                evaluator_trace_path=evaluator_trace_path,
            )
        except Exception as exc:  # noqa: BLE001 - rollout result records the failure.
            fail_reason = str(exc) or "evaluator execution failed"
            score = make_unscored_evaluation(
                fail_reason=fail_reason,
                target_status=TARGET_PASSED,
                evaluator_status=EVALUATOR_FAILED,
                blocker="evaluator_failed",
                target_trace_path=target_trace_path,
                evaluator_trace_path=evaluator_trace_path,
                metadata={"evaluator": "unknown"},
            )
    result = {
        "id": item_id,
        "hard": score.get("hard"),
        "soft": score.get("soft"),
        "response": response,
        "fail_reason": str(score.get("fail_reason") or ""),
        "agent_ok": agent_ok,
        "n_turns": 1 if agent_ok else 0,
        "task_type": item.get("task_type", "gitmoot-skillopt"),
        "task_description": item.get("task_description", item_id),
        "metadata": {
            **(item.get("metadata") if isinstance(item.get("metadata"), dict) else {}),
            **(score.get("metadata") if isinstance(score.get("metadata"), dict) else {}),
        },
        "target_status": score.get("target_status"),
        "evaluator_status": score.get("evaluator_status"),
        "score_status": score.get("score_status"),
        "blocker": score.get("blocker", ""),
        "evaluator_id": score.get("evaluator_id", ""),
        "evaluator_version": score.get("evaluator_version", ""),
        "target_trace_path": score.get("target_trace_path", ""),
        "evaluator_trace_path": score.get("evaluator_trace_path", ""),
        "target_system_prompt": system_prompt,
        "target_user_prompt": user_prompt,
    }
    _write_prediction(pred_dir, result)
    return result


def _run_agent(
    item: dict[str, Any],
    system_prompt: str,
    user_prompt: str,
    max_completion_tokens: int,
) -> str:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    if metadata.get("mock_response") is not None:
        return str(metadata["mock_response"])
    if not is_target_chat_backend():
        raise RuntimeError("Gitmoot rollout currently requires a chat target backend or metadata.mock_response")
    response, _usage = chat_target(
        system=system_prompt,
        user=user_prompt,
        max_completion_tokens=max_completion_tokens,
        retries=2,
        stage="gitmoot_rollout",
    )
    return response


def _build_system_prompt(skill_content: str) -> str:
    return (
        "Use the following Gitmoot agent-template skill to answer the task. "
        "Preserve the intent of the skill and produce only the requested response.\n\n"
        f"## Skill\n{skill_content.strip()}"
    )


def _write_prediction(pred_dir: str, result: dict[str, Any]) -> None:
    with open(os.path.join(pred_dir, "result.json"), "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    conversation = [
        {"role": "system", "content": result["target_system_prompt"]},
        {"role": "user", "content": result["target_user_prompt"]},
        {"role": "assistant", "content": result["response"]},
    ]
    with open(os.path.join(pred_dir, "conversation.json"), "w", encoding="utf-8") as handle:
        json.dump(conversation, handle, indent=2)
        handle.write("\n")
    with open(os.path.join(pred_dir, "target_system_prompt.txt"), "w", encoding="utf-8") as handle:
        handle.write(result["target_system_prompt"])
    with open(os.path.join(pred_dir, "target_user_prompt.txt"), "w", encoding="utf-8") as handle:
        handle.write(result["target_user_prompt"])
