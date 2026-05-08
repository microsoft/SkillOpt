"""BabyVision rollout — multimodal visual QA with image input."""
from __future__ import annotations

import base64
import json
import mimetypes
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from skillopt.envs.babyvision.evaluator import evaluate_item, evaluation_mode, extract_boxed_answer
from skillopt.model import chat_student_messages, get_student_backend, is_student_exec_backend
from skillopt.model.codex_harness import prepare_workspace, render_skill_md, run_student_exec
from skillopt.prompts import load_prompt

def _build_system(skill_content: str) -> str:
    if skill_content.strip():
        skill_section = f"## Skill\n{skill_content.strip()}\n\n"
    else:
        skill_section = ""
    return load_prompt("rollout_system", env="babyvision").format(skill_section=skill_section)


def _format_choices(choices: list[dict]) -> str:
    return "\n".join(f"{choice['label']}. {choice['text']}" for choice in choices)


def _build_user_text(
    item: dict,
    *,
    diagnostic_mode: bool = False,
    diagnostic_instruction: str = "",
    diagnostic_trace_context: str = "",
) -> str:
    parts = []
    if diagnostic_trace_context.strip():
        parts.append(
            "## Previous Codex Trace Snapshot\n"
            "This is a partial transcript from an earlier attempt. Use it as your current reasoning context.\n\n"
            f"{diagnostic_trace_context.strip()}"
        )
    parts.append(f"## Question\n{item['question']}")
    if item["ans_type"] == "choice":
        parts.append(f"## Choices\n{_format_choices(item['choices'])}")
        parts.append("Answer using the single correct option label in \\boxed{...}.")
    else:
        parts.append("Answer with a short phrase in \\boxed{...}.")
    if diagnostic_mode and diagnostic_instruction.strip():
        parts.append(f"## Training Readout\n{diagnostic_instruction.strip()}")
    return "\n\n".join(parts)


def _image_to_data_uri(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _build_messages(
    item: dict,
    skill_content: str,
    image_detail: str,
    *,
    diagnostic_mode: bool = False,
    diagnostic_instruction: str = "",
    diagnostic_trace_context: str = "",
) -> tuple[list[dict], str, str]:
    system = _build_system(skill_content)
    user_text = _build_user_text(
        item,
        diagnostic_mode=diagnostic_mode,
        diagnostic_instruction=diagnostic_instruction,
        diagnostic_trace_context=diagnostic_trace_context,
    )
    image_url = {
        "url": _image_to_data_uri(item["image_path"]),
    }
    if image_detail and image_detail != "auto":
        image_url["detail"] = image_detail
    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": image_url},
            ],
        },
    ]
    return messages, system, user_text


def _build_codex_skill(skill_content: str) -> str:
    return render_skill_md(
        skill_content,
        description="Dynamic ReflACT skill for solving the current BabyVision visual reasoning question.",
        preamble=(
            "Use this skill when answering the current visual reasoning question.\n"
            "Inspect the attached image carefully and return the final answer in \\boxed{...}."
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
    user_text = _build_user_text(
        item,
        diagnostic_mode=diagnostic_mode,
        diagnostic_instruction=diagnostic_instruction,
        diagnostic_trace_context=diagnostic_trace_context,
    )
    task_parts = [user_text]
    if previous_response:
        task_parts.append(
            "## Previous Attempt\n"
            f"{previous_response}\n\n"
            "Review the same image and question carefully. If needed, correct the answer."
        )
    task_text = "\n\n".join(task_parts)
    skill_md = _build_codex_skill(skill_content)
    work_dir = os.path.join(pred_dir, "codex_exec")
    prepare_workspace(
        work_dir=work_dir,
        skill_md=skill_md,
        task_text=task_text,
        images=[item["image_path"]],
    )
    prompt = (
        "Use the `skillopt-student` skill available in this workspace.\n"
        "Read `task.md`, inspect the attached image, and answer the question.\n"
        "Return the final answer in \\boxed{...}."
    )
    final_message, raw = run_student_exec(
        work_dir=work_dir,
        prompt=prompt,
        model=model,
        timeout=timeout,
        images=[item["image_path"]],
    )
    return final_message or raw, raw, skill_md, task_text


def process_one(
    item: dict,
    out_root: str,
    skill_content: str,
    *,
    max_turns: int = 1,
    image_detail: str = "auto",
    judge_model: str = "gpt-5.4",
    judge_max_completion_tokens: int = 256,
    judge_retries: int = 5,
    diagnostic_mode: bool = False,
    diagnostic_instruction: str = "",
    diagnostic_trace_context: str = "",
) -> dict:
    item_id = str(item["id"])
    result = {
        "id": item_id,
        "question": item["question"],
        "task_type": item.get("subtype") or item.get("task_type") or "babyvision",
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
        "image_path": item["image_path"],
        "ans_type": item["ans_type"],
        "evaluation_mode": evaluation_mode(),
        "judge_model": judge_model,
    }
    if item["ans_type"] == "choice":
        result["correct_label"] = item["correct_choice"]["label"]
        result["correct_text"] = item["correct_choice"]["text"]
    else:
        result["gold_answers"] = item["blank_answers"]

    try:
        pred_dir = os.path.join(out_root, "predictions", item_id)
        os.makedirs(pred_dir, exist_ok=True)

        if is_student_exec_backend():
            from skillopt.model import azure_openai as _llm

            response = ""
            conversation: list[dict] = [
                {"role": "user", "content": f"{item['question']}\n\n[image] {os.path.basename(item['image_path'])}"}
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
                if extract_boxed_answer(response) is not None:
                    break

            result["response"] = response
            result["agent_ok"] = True
            result["n_turns"] = len(conversation) - 1
            with open(os.path.join(pred_dir, "student_system_prompt.txt"), "w", encoding="utf-8") as f:
                f.write(system_prompt)
            with open(os.path.join(pred_dir, "student_user_prompt.txt"), "w", encoding="utf-8") as f:
                f.write(user_text)

            eval_result = evaluate_item(
                item=item,
                prediction_text=response,
                judge_model=judge_model,
                max_completion_tokens=judge_max_completion_tokens,
                retries=judge_retries,
            )
            result["evaluation_mode"] = eval_result["evaluation_mode"]
            result["judge_raw"] = eval_result["judge_raw"]
            result["judge_reason"] = eval_result["judge_reason"]
            result["matched_gold"] = eval_result["matched_gold"]
            if item["ans_type"] == "choice":
                result["predicted_label"] = eval_result["predicted_label"]
                result["predicted_text"] = eval_result["predicted_text"]
                result["predicted_answer"] = eval_result["predicted_answer"]
                result["hard"] = int(eval_result["em"])
                result["soft"] = eval_result["f1"]
                if not result["hard"]:
                    result["fail_reason"] = (
                        f"judge=0: predicted '{eval_result['predicted_label'] or eval_result['predicted_answer']}' "
                        f"but expected '{eval_result['correct_label']}' ({eval_result['judge_reason']})"
                    )
                eval_detail = (
                    f"[EVALUATION RESULT]\n"
                    f"Question: {item['question']}\n"
                    f"Predicted label: {eval_result['predicted_label']!r}\n"
                    f"Predicted text: {eval_result['predicted_text']!r}\n"
                    f"Correct label: {eval_result['correct_label']!r}\n"
                    f"Correct text: {eval_result['correct_text']!r}\n"
                    f"Judge correct: {eval_result['em']}\n"
                    f"Judge reason: {eval_result['judge_reason']}"
                )
            else:
                result["predicted_answer"] = eval_result["predicted_answer"]
                result["hard"] = int(eval_result["em"])
                result["soft"] = eval_result["f1"]
                if not result["hard"]:
                    result["fail_reason"] = (
                        f"judge=0: predicted '{eval_result['predicted_answer']}' "
                        f"but expected {item['blank_answers']} ({eval_result['judge_reason']})"
                    )
                eval_detail = (
                    f"[EVALUATION RESULT]\n"
                    f"Question: {item['question']}\n"
                    f"Predicted answer: {eval_result['predicted_answer']!r}\n"
                    f"Gold answers: {item['blank_answers']!r}\n"
                    f"Judge correct: {eval_result['em']}\n"
                    f"Judge reason: {eval_result['judge_reason']}\n"
                    f"String F1: {eval_result.get('string_f1', 0.0):.4f}"
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
            {"role": "user", "content": f"{user_text}\n\n[image] {os.path.basename(item['image_path'])}"}
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
                refinement_text = (
                    f"Your previous answer was:\n{response}\n\n"
                    "Review the same image and question carefully. "
                    "If needed, correct your answer. Output the final answer in \\boxed{...}."
                )
                refinement_messages = [
                    messages[0],
                    messages[1],
                    {"role": "assistant", "content": response},
                    {"role": "user", "content": refinement_text},
                ]
                resp_text, _ = chat_student_messages(
                    messages=refinement_messages,
                    max_completion_tokens=512,
                    retries=5,
                    stage="rollout",
                )
            response = resp_text
            conversation.append({"type": "message", "turn": turn + 1, "content": resp_text})
            if extract_boxed_answer(resp_text) is not None:
                break

        result["response"] = response
        result["agent_ok"] = True
        result["n_turns"] = len(conversation) - 1

        with open(os.path.join(pred_dir, "student_system_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(system_prompt)
        with open(os.path.join(pred_dir, "student_user_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(user_text)

        eval_result = evaluate_item(
            item=item,
            prediction_text=response,
            judge_model=judge_model,
            max_completion_tokens=judge_max_completion_tokens,
            retries=judge_retries,
        )
        result["evaluation_mode"] = eval_result["evaluation_mode"]
        result["judge_raw"] = eval_result["judge_raw"]
        result["judge_reason"] = eval_result["judge_reason"]
        result["matched_gold"] = eval_result["matched_gold"]

        if item["ans_type"] == "choice":
            result["predicted_label"] = eval_result["predicted_label"]
            result["predicted_text"] = eval_result["predicted_text"]
            result["predicted_answer"] = eval_result["predicted_answer"]
            result["hard"] = int(eval_result["em"])
            result["soft"] = eval_result["f1"]
            if not result["hard"]:
                result["fail_reason"] = (
                    f"judge=0: predicted '{eval_result['predicted_label'] or eval_result['predicted_answer']}' "
                    f"but expected '{eval_result['correct_label']}' ({eval_result['judge_reason']})"
                )
            eval_detail = (
                f"[EVALUATION RESULT]\n"
                f"Question: {item['question']}\n"
                f"Predicted label: {eval_result['predicted_label']!r}\n"
                f"Predicted text: {eval_result['predicted_text']!r}\n"
                f"Correct label: {eval_result['correct_label']!r}\n"
                f"Correct text: {eval_result['correct_text']!r}\n"
                f"Judge correct: {eval_result['em']}\n"
                f"Judge reason: {eval_result['judge_reason']}"
            )
        else:
            result["predicted_answer"] = eval_result["predicted_answer"]
            result["hard"] = int(eval_result["em"])
            result["soft"] = eval_result["f1"]
            if not result["hard"]:
                result["fail_reason"] = (
                    f"judge=0: predicted '{eval_result['predicted_answer']}' "
                    f"but expected {item['blank_answers']} ({eval_result['judge_reason']})"
                )
            eval_detail = (
                f"[EVALUATION RESULT]\n"
                f"Question: {item['question']}\n"
                f"Predicted answer: {eval_result['predicted_answer']!r}\n"
                f"Gold answers: {item['blank_answers']!r}\n"
                f"Judge correct: {eval_result['em']}\n"
                f"Judge reason: {eval_result['judge_reason']}\n"
                f"String F1: {eval_result.get('string_f1', 0.0):.4f}"
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
    workers: int = 32,
    image_detail: str = "auto",
    judge_model: str = "gpt-5.4",
    judge_max_completion_tokens: int = 256,
    judge_retries: int = 5,
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
                judge_model=judge_model,
                judge_max_completion_tokens=judge_max_completion_tokens,
                judge_retries=judge_retries,
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
