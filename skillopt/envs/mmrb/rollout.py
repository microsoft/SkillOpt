"""MMRB rollout."""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from skillopt.envs.mmrb.evaluator import evaluate_item, evaluation_mode
from skillopt.model import chat_student_messages, get_student_backend, is_student_exec_backend
from skillopt.model.codex_harness import prepare_workspace, render_skill_md, run_student_exec
from skillopt.prompts import load_prompt

_IMAGE_REF_RE = re.compile(r"\{image#(\d+)\}", re.IGNORECASE)


def _build_system(skill_content: str) -> str:
    if skill_content.strip():
        skill_section = f"## Skill\n{skill_content.strip()}\n\n"
    else:
        skill_section = ""
    return load_prompt("rollout_system", env="mmrb").format(skill_section=skill_section)


def _image_to_data_uri(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _build_user_content(
    item: dict,
    image_detail: str,
    *,
    diagnostic_mode: bool = False,
    diagnostic_instruction: str = "",
    diagnostic_trace_context: str = "",
) -> tuple[list[dict], str]:
    raw_question = str(item["question"])
    content: list[dict] = []
    text_parts: list[str] = []
    used_indices: set[int] = set()
    cursor = 0

    if diagnostic_trace_context.strip():
        prefix = (
            "## Previous Codex Trace Snapshot\n"
            "This is a partial transcript from an earlier attempt. Use it as your current reasoning context.\n\n"
            f"{diagnostic_trace_context.strip()}\n\n"
        )
        content.append({"type": "text", "text": prefix})
        text_parts.append(prefix)

    for match in _IMAGE_REF_RE.finditer(raw_question):
        if match.start() > cursor:
            chunk = raw_question[cursor:match.start()]
            if chunk:
                content.append({"type": "text", "text": chunk})
                text_parts.append(chunk)

        image_idx = int(match.group(1)) - 1
        marker = f"[Image #{image_idx + 1}]"
        text_parts.append(marker)
        if 0 <= image_idx < len(item["image_paths"]):
            image_url = {"url": _image_to_data_uri(item["image_paths"][image_idx])}
            if image_detail and image_detail != "auto":
                image_url["detail"] = image_detail
            content.append({"type": "image_url", "image_url": image_url})
            used_indices.add(image_idx)
        else:
            content.append({"type": "text", "text": marker})
        cursor = match.end()

    if cursor < len(raw_question):
        tail = raw_question[cursor:]
        if tail:
            content.append({"type": "text", "text": tail})
            text_parts.append(tail)

    for idx, path in enumerate(item["image_paths"]):
        if idx in used_indices:
            continue
        marker = f"\n[Additional Image #{idx + 1}]"
        text_parts.append(marker)
        content.append({"type": "text", "text": marker})
        image_url = {"url": _image_to_data_uri(path)}
        if image_detail and image_detail != "auto":
            image_url["detail"] = image_detail
        content.append({"type": "image_url", "image_url": image_url})

    answer_instruction = (
        "\n\nAnswer with the single correct option letter inside <answer>...</answer>."
        if item.get("is_choice")
        else "\n\nAnswer with the short final answer inside <answer>...</answer>."
    )
    content.append({"type": "text", "text": answer_instruction})
    text_parts.append(answer_instruction)

    if diagnostic_mode and diagnostic_instruction.strip():
        diag_block = f"\n\n## Training Readout\n{diagnostic_instruction.strip()}"
        content.append({"type": "text", "text": diag_block})
        text_parts.append(diag_block)

    return content, "".join(text_parts)


def _build_messages(
    item: dict,
    skill_content: str,
    image_detail: str,
    *,
    diagnostic_mode: bool = False,
    diagnostic_instruction: str = "",
) -> tuple[list[dict], str, str]:
    system = _build_system(skill_content)
    user_content, user_text = _build_user_content(
        item,
        image_detail,
        diagnostic_mode=diagnostic_mode,
        diagnostic_instruction=diagnostic_instruction,
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
    return messages, system, user_text


def _build_codex_skill(skill_content: str) -> str:
    return render_skill_md(
        skill_content,
        description="Dynamic ReflACT skill for solving the current MMRB multi-image reasoning question.",
        preamble=(
            "Use this skill when solving the current multi-image reasoning task.\n"
            "Inspect all attached images carefully and return the final answer inside <answer>...</answer>."
        ),
    )


def _run_codex_once(
    *,
    pred_dir: str,
    item: dict,
    skill_content: str,
    model: str,
    timeout: int,
    image_detail: str,
    diagnostic_mode: bool = False,
    diagnostic_instruction: str = "",
    diagnostic_trace_context: str = "",
    previous_response: str = "",
) -> tuple[str, str, str, str]:
    user_text = _build_user_content(
        item,
        image_detail,
        diagnostic_mode=diagnostic_mode,
        diagnostic_instruction=diagnostic_instruction,
        diagnostic_trace_context=diagnostic_trace_context,
    )[1]
    task_parts = [user_text]
    if previous_response:
        task_parts.append(
            "## Previous Attempt\n"
            f"{previous_response}\n\n"
            "Review the same images carefully and answer again."
        )
    task_text = "\n\n".join(task_parts)
    skill_md = _build_codex_skill(skill_content)
    work_dir = os.path.join(pred_dir, "codex_exec")
    prepare_workspace(
        work_dir=work_dir,
        skill_md=skill_md,
        task_text=task_text,
        images=item["image_paths"],
    )
    prompt = (
        "Use the `skillopt-student` skill available in this workspace.\n"
        "Read `task.md`, inspect all attached images, and answer the question.\n"
        "Keep the final answer inside <answer>...</answer>."
    )
    final_message, raw = run_student_exec(
        work_dir=work_dir,
        prompt=prompt,
        model=model,
        timeout=timeout,
        images=item["image_paths"],
    )
    return final_message or raw, raw, skill_md, task_text


def process_one(
    item: dict,
    out_root: str,
    skill_content: str,
    *,
    max_turns: int = 1,
    image_detail: str = "auto",
    diagnostic_mode: bool = False,
    diagnostic_instruction: str = "",
    diagnostic_trace_context: str = "",
) -> dict:
    item_id = str(item["id"])
    result = {
        "id": item_id,
        "question": item["question"],
        "task_type": item.get("subtask") or item.get("task_type") or "mmrb",
        "task_description": item["question"],
        "hard": 0,
        "soft": 0.0,
        "predicted_answer": "",
        "predicted_label": "",
        "predicted_text": "",
        "response": "",
        "fail_reason": "",
        "agent_ok": False,
        "n_turns": 0,
        "image_paths": item["image_paths"],
        "gold_answer": item["answer"],
        "evaluation_mode": evaluation_mode(),
    }

    try:
        pred_dir = os.path.join(out_root, "predictions", item_id)
        os.makedirs(pred_dir, exist_ok=True)

        if is_student_exec_backend():
            from skillopt.model import azure_openai as _llm

            response = ""
            conversation: list[dict] = [
                {
                    "role": "user",
                    "content": item["question"] + "\n\n" + "\n".join(
                        f"[image] {os.path.basename(path)}" for path in item["image_paths"]
                    ),
                }
            ]
            system_prompt = ""
            user_text = ""
            for turn in range(max_turns):
                response, raw, system_prompt, user_text = _run_codex_once(
                    pred_dir=pred_dir,
                    item=item,
                    skill_content=skill_content,
                    model=_llm.STUDENT_DEPLOYMENT,
                    timeout=120,
                    image_detail=image_detail,
                    diagnostic_mode=diagnostic_mode if turn == 0 else False,
                    diagnostic_instruction=diagnostic_instruction if turn == 0 else "",
                    diagnostic_trace_context=diagnostic_trace_context if turn == 0 else "",
                    previous_response=response if turn > 0 else "",
                )
                conversation.append({"type": "message", "turn": turn + 1, "content": response})
                if "<answer>" in response.lower():
                    break

            result["response"] = response
            result["agent_ok"] = True
            result["n_turns"] = len(conversation) - 1
            with open(os.path.join(pred_dir, "student_system_prompt.txt"), "w", encoding="utf-8") as f:
                f.write(system_prompt)
            with open(os.path.join(pred_dir, "student_user_prompt.txt"), "w", encoding="utf-8") as f:
                f.write(user_text)

            eval_result = evaluate_item(item=item, prediction_text=response)
            result["evaluation_mode"] = eval_result["evaluation_mode"]
            result["predicted_answer"] = eval_result["predicted_answer"]
            result["predicted_label"] = eval_result["predicted_label"]
            result["predicted_text"] = eval_result["predicted_text"]
            result["matched_gold"] = eval_result["matched_gold"]
            result["hard"] = int(eval_result["em"])
            result["soft"] = eval_result["f1"]
            if not result["hard"]:
                result["fail_reason"] = (
                    f"predicted '{eval_result['predicted_answer']}' but expected '{item['answer']}'"
                )
            eval_detail = (
                "[EVALUATION RESULT]\n"
                f"Question: {item['question']}\n"
                f"Predicted answer: {eval_result['predicted_answer']!r}\n"
                f"Predicted label: {eval_result['predicted_label']!r}\n"
                f"Gold answer: {item['answer']!r}\n"
                f"Correct: {eval_result['em']}\n"
            )
            conversation.append({"role": "system", "content": eval_detail})
            with open(os.path.join(pred_dir, "conversation.json"), "w", encoding="utf-8") as f:
                json.dump(conversation, f, ensure_ascii=False, indent=2)
            return result

        messages, system_prompt, user_text = _build_messages(
            item,
            skill_content,
            image_detail,
            diagnostic_mode=diagnostic_mode,
            diagnostic_instruction=diagnostic_instruction,
            diagnostic_trace_context=diagnostic_trace_context,
        )
        response = ""
        conversation: list[dict] = [
            {
                "role": "user",
                "content": user_text + "\n\n" + "\n".join(
                    f"[image] {os.path.basename(path)}" for path in item["image_paths"]
                ),
            }
        ]

        for turn in range(max_turns):
            if turn == 0:
                resp_text, _ = chat_student_messages(
                    messages=messages,
                    max_completion_tokens=768,
                    retries=5,
                    stage="rollout",
                )
            else:
                refinement_messages = [
                    messages[0],
                    messages[1],
                    {"role": "assistant", "content": response},
                    {
                        "role": "user",
                        "content": "Review the same images carefully and answer again. Keep the final answer inside <answer>...</answer>.",
                    },
                ]
                resp_text, _ = chat_student_messages(
                    messages=refinement_messages,
                    max_completion_tokens=512,
                    retries=5,
                    stage="rollout",
                )
            response = resp_text
            conversation.append({"type": "message", "turn": turn + 1, "content": resp_text})
            if "<answer>" in resp_text.lower():
                break

        result["response"] = response
        result["agent_ok"] = True
        result["n_turns"] = len(conversation) - 1

        with open(os.path.join(pred_dir, "student_system_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(system_prompt)
        with open(os.path.join(pred_dir, "student_user_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(user_text)

        eval_result = evaluate_item(item=item, prediction_text=response)
        result["evaluation_mode"] = eval_result["evaluation_mode"]
        result["predicted_answer"] = eval_result["predicted_answer"]
        result["predicted_label"] = eval_result["predicted_label"]
        result["predicted_text"] = eval_result["predicted_text"]
        result["matched_gold"] = eval_result["matched_gold"]
        result["hard"] = int(eval_result["em"])
        result["soft"] = eval_result["f1"]
        if not result["hard"]:
            result["fail_reason"] = (
                f"predicted '{eval_result['predicted_answer']}' but expected '{item['answer']}'"
            )

        eval_detail = (
            "[EVALUATION RESULT]\n"
            f"Question: {item['question']}\n"
            f"Predicted answer: {eval_result['predicted_answer']!r}\n"
            f"Predicted label: {eval_result['predicted_label']!r}\n"
            f"Gold answer: {item['answer']!r}\n"
            f"Correct: {eval_result['em']}\n"
        )
        conversation.append({"role": "system", "content": eval_detail})
        with open(os.path.join(pred_dir, "conversation.json"), "w", encoding="utf-8") as f:
            json.dump(conversation, f, ensure_ascii=False, indent=2)
    except Exception as e:  # noqa: BLE001
        result["fail_reason"] = f"error: {e}"
    return result


def run_batch(
    items: list[dict],
    out_root: str,
    skill_content: str,
    *,
    max_turns: int = 1,
    workers: int = 16,
    image_detail: str = "auto",
    diagnostic_mode: bool = False,
    diagnostic_instruction: str = "",
    diagnostic_trace_context_by_id: dict[str, str] | None = None,
) -> list[dict]:
    results_path = os.path.join(out_root, "results.jsonl")
    os.makedirs(out_root, exist_ok=True)

    expected_eval_mode = evaluation_mode()
    done_ids: set[str] = set()
    existing: list[dict] = []
    rewrite_results = False
    if os.path.exists(results_path):
        with open(results_path, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    if row.get("evaluation_mode") != expected_eval_mode:
                        rewrite_results = True
                        continue
                    done_ids.add(str(row["id"]))
                    existing.append(row)
                except Exception:
                    rewrite_results = True

    pending = [item for item in items if str(item["id"]) not in done_ids]
    if not pending and not rewrite_results:
        return existing

    results = list(existing)
    file_mode = "w" if rewrite_results else "a"
    with open(results_path, file_mode, encoding="utf-8") as outf, ThreadPoolExecutor(max_workers=workers) as ex:
        if rewrite_results:
            for row in existing:
                outf.write(json.dumps(row, ensure_ascii=False) + "\n")
        futs = {
            ex.submit(
                process_one,
                item,
                out_root,
                skill_content,
                max_turns=max_turns,
                image_detail=image_detail,
                diagnostic_mode=diagnostic_mode,
                diagnostic_instruction=diagnostic_instruction,
                diagnostic_trace_context=(diagnostic_trace_context_by_id or {}).get(str(item["id"]), ""),
            ): item
            for item in pending
        }
        for fut in as_completed(futs):
            row = fut.result()
            results.append(row)
            outf.write(json.dumps(row, ensure_ascii=False) + "\n")
            outf.flush()
    return results
