from __future__ import annotations

import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from skillopt.envs.docvqa.evaluator import evaluate
from skillopt.model import chat_target_messages, is_target_exec_backend
from skillopt.model.codex_harness import prepare_workspace, render_skill_md, run_target_exec
from skillopt.prompts import load_prompt


def _build_system(skill_content: str) -> str:
    if skill_content.strip():
        skill_section = f"## Skill\n{skill_content.strip()}\n\n"
    else:
        skill_section = ""
    return load_prompt("rollout_system", env="docvqa").format(skill_section=skill_section)


def _image_to_data_uri(path: str) -> str:
    import base64
    import mimetypes

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
) -> tuple[list[dict], str, str]:
    system = _build_system(skill_content)
    user_text = item["question"] + "\n\nReturn the final answer inside <answer>...</answer>."
    if diagnostic_mode and diagnostic_instruction.strip():
        user_text += f"\n\n## Training Readout\n{diagnostic_instruction.strip()}"
    image_url = {"url": _image_to_data_uri(item["image_path"])}
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
        description="Dynamic ReflACT skill for solving the current DocVQA document-image question.",
        preamble=(
            "Use this skill when answering the current DocVQA question.\n"
            "Inspect the attached document image carefully and return the final answer inside <answer>...</answer>."
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
    previous_response: str = "",
) -> tuple[str, str, str, str]:
    _ = image_detail
    _messages, _system, user_text = _build_messages(
        item,
        skill_content,
        image_detail,
        diagnostic_mode=diagnostic_mode,
        diagnostic_instruction=diagnostic_instruction,
    )
    task_parts = [user_text]
    image_abs = os.path.abspath(item["image_path"])
    task_parts.append(
        "## Document Image\n"
        "The document image is available in this workspace via `ATTACHMENTS.md`.\n"
        f"Original image path: `{image_abs}`\n"
        "Open or inspect that image before answering; do not answer from memory."
    )
    if previous_response:
        task_parts.append(
            "## Previous Attempt\n"
            f"{previous_response}\n\n"
            "Review the same document image carefully and correct the answer if needed."
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
        "Use the `skillopt-target` skill available in this workspace.\n"
        "Read `task.md`, inspect the attached document image, and answer the DocVQA question.\n"
        "Return the final answer inside <answer>...</answer>."
    )
    final_message, raw = run_target_exec(
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
    exec_timeout: int = 120,
    image_detail: str = "auto",
    max_completion_tokens: int = 16384,
    diagnostic_mode: bool = False,
    diagnostic_instruction: str = "",
) -> dict:
    item_id = str(item["id"])
    result = {
        "id": item_id,
        "question": item["question"],
        "task_type": item.get("subtask") or item.get("task_type") or "docvqa",
        "task_description": item["question"],
        "hard": 0,
        "soft": 0.0,
        "predicted_answer": "",
        "response": "",
        "fail_reason": "",
        "agent_ok": False,
        "n_turns": 0,
        "image_paths": item.get("image_paths", []),
        "gold_answer": item.get("answers", []),
    }
    try:
        response = ""
        system_prompt = ""
        user_text = ""
        conversation: list[dict] = []
        if is_target_exec_backend():
            from skillopt.model import azure_openai as _llm

            conversation = [
                {
                    "role": "user",
                    "content": item["question"] + "\n\n" + f"[image] {os.path.basename(item['image_path'])}",
                }
            ]
            for turn in range(max_turns):
                response, _raw, system_prompt, user_text = _run_codex_once(
                    pred_dir=os.path.join(out_root, "predictions", item_id),
                    item=item,
                    skill_content=skill_content,
                    model=_llm.TARGET_DEPLOYMENT,
                    timeout=exec_timeout,
                    image_detail=image_detail,
                    diagnostic_mode=diagnostic_mode if turn == 0 else False,
                    diagnostic_instruction=diagnostic_instruction if turn == 0 else "",
                    previous_response=response if turn > 0 else "",
                )
                conversation.append({"type": "message", "turn": turn + 1, "content": response})
                if "<answer>" in response.lower():
                    break
        else:
            messages, system_prompt, user_text = _build_messages(
                item,
                skill_content,
                image_detail,
                diagnostic_mode=diagnostic_mode,
                diagnostic_instruction=diagnostic_instruction,
            )
            conversation = [
                {
                    "role": "user",
                    "content": user_text + "\n\n" + f"[image] {os.path.basename(item['image_path'])}",
                }
            ]
            for turn in range(max_turns):
                if turn == 0:
                    resp_text, _ = chat_target_messages(
                        messages=messages,
                        max_completion_tokens=max_completion_tokens,
                        retries=5,
                        stage="rollout",
                        timeout=exec_timeout,
                    )
                else:
                    refinement_messages = [
                        messages[0],
                        messages[1],
                        {"role": "assistant", "content": response},
                        {"role": "user", "content": "Review the same image carefully and answer again. Keep the final answer inside <answer>...</answer>."},
                    ]
                    resp_text, _ = chat_target_messages(
                        messages=refinement_messages,
                        max_completion_tokens=max_completion_tokens,
                        retries=5,
                        stage="rollout",
                        timeout=exec_timeout,
                    )
                response = resp_text
                conversation.append({"type": "message", "turn": turn + 1, "content": resp_text})
                if "<answer>" in resp_text.lower():
                    break

        result["response"] = response
        result["agent_ok"] = True
        result["n_turns"] = len(conversation) - 1

        pred_dir = os.path.join(out_root, "predictions", item_id)
        os.makedirs(pred_dir, exist_ok=True)
        with open(os.path.join(pred_dir, "target_system_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(system_prompt)
        with open(os.path.join(pred_dir, "target_user_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(user_text)

        eval_result = evaluate(response, item.get("answers", []))
        result["predicted_answer"] = eval_result["predicted_answer"]
        result["hard"] = int(eval_result["anls"] >= 0.999)
        result["soft"] = eval_result["anls"]
        if result["soft"] <= 0.0:
            result["fail_reason"] = f"predicted '{eval_result['predicted_answer']}' but expected one of {item.get('answers', [])}"

        eval_detail = (
            "[EVALUATION RESULT]\n"
            f"Question: {item['question']}\n"
            f"Predicted answer: {eval_result['predicted_answer']!r}\n"
            f"Gold answers: {item.get('answers', [])!r}\n"
            f"ANLS: {eval_result['anls']:.4f}"
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
    exec_timeout: int = 120,
    workers: int = 16,
    image_detail: str = "auto",
    max_completion_tokens: int = 16384,
    diagnostic_mode: bool = False,
    diagnostic_instruction: str = "",
    task_timeout: int = 600,
) -> list[dict]:
    task_timeout = max(int(task_timeout), int(exec_timeout) + 60)
    results_path = os.path.join(out_root, "results.jsonl")
    os.makedirs(out_root, exist_ok=True)

    done_ids: set[str] = set()
    existing: list[dict] = []
    if os.path.exists(results_path):
        with open(results_path, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                done_ids.add(str(row["id"]))
                existing.append(row)

    pending = [item for item in items if str(item["id"]) not in done_ids]
    if not pending:
        return existing

    def _timeout_result(item: dict) -> dict:
        return {
            "id": str(item["id"]),
            "question": item.get("question", ""),
            "task_type": item.get("subtask") or item.get("task_type") or "docvqa",
            "task_description": item.get("question", ""),
            "hard": 0,
            "soft": 0.0,
            "predicted_answer": "",
            "response": "",
            "fail_reason": f"task-timeout-{task_timeout}s",
            "agent_ok": False,
            "n_turns": 0,
            "image_paths": item.get("image_paths", []),
            "gold_answer": item.get("answers", []),
            "phase": "timeout",
        }

    def _error_result(item: dict, exc: Exception) -> dict:
        row = _timeout_result(item)
        row["phase"] = "error"
        row["fail_reason"] = f"unexpected: {type(exc).__name__}: {exc}"
        return row

    started_at: dict[str, float] = {}

    def _run_one(item: dict) -> dict:
        started_at[str(item["id"])] = time.time()
        return process_one(
            item,
            out_root,
            skill_content,
            max_turns=max_turns,
            exec_timeout=exec_timeout,
            image_detail=image_detail,
            max_completion_tokens=max_completion_tokens,
            diagnostic_mode=diagnostic_mode,
            diagnostic_instruction=diagnostic_instruction,
        )

    total = len(existing) + len(pending)
    completed = len(existing)
    correct = sum(1 for r in existing if r.get("hard", 0))
    if existing:
        print(f"    [rollout] resuming: {completed}/{total} already done", flush=True)

    results = list(existing)
    with open(results_path, "a", encoding="utf-8") as outf:
        ex = ThreadPoolExecutor(max_workers=workers)
        try:
            futs = {ex.submit(_run_one, item): item for item in pending}
            pending_futs = set(futs)
            while pending_futs:
                done, _ = wait(pending_futs, timeout=5, return_when=FIRST_COMPLETED)
                now = time.time()
                timed_out = [
                    fut for fut in pending_futs - done
                    if str(futs[fut]["id"]) in started_at
                    and now - started_at[str(futs[fut]["id"])] >= task_timeout
                ]
                for fut in done:
                    pending_futs.remove(fut)
                    item = futs[fut]
                    try:
                        res = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        res = _error_result(item, exc)
                    results.append(res)
                    completed += 1
                    if res.get("hard", 0):
                        correct += 1
                    acc = correct / completed if completed else 0
                    print(
                        f"    [rollout] {completed}/{total} "
                        f"(acc={acc:.3f}) id={res['id']} "
                        f"hard={res.get('hard', '?')}",
                        flush=True,
                    )
                    outf.write(json.dumps(res, ensure_ascii=False) + "\n")
                    outf.flush()
                for fut in timed_out:
                    pending_futs.remove(fut)
                    fut.cancel()
                    res = _timeout_result(futs[fut])
                    results.append(res)
                    completed += 1
                    acc = correct / completed if completed else 0
                    print(
                        f"    [rollout] {completed}/{total} "
                        f"(acc={acc:.3f}) id={res['id']} TIMEOUT",
                        flush=True,
                    )
                    outf.write(json.dumps(res, ensure_ascii=False) + "\n")
                    outf.flush()
        finally:
            ex.shutdown(wait=False, cancel_futures=True)
    return results
