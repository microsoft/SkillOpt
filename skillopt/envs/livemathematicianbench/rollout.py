"""LiveMathematicianBench rollout — theorem-grounded math MCQ agent."""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from skillopt.envs.livemathematicianbench.evaluator import evaluate
from skillopt.model import chat_target, is_target_exec_backend
from skillopt.model.codex_harness import prepare_workspace, render_skill_md, run_target_exec
from skillopt.prompts import load_prompt


def _build_system(skill_content: str) -> str:
    if skill_content.strip():
        skill_section = f"## Skill\n{skill_content.strip()}\n\n"
    else:
        skill_section = ""
    return load_prompt("rollout_system", env="livemathematicianbench").format(skill_section=skill_section)


def _format_choices(choices: list[dict]) -> str:
    return "\n".join(
        f"{choice['label']}. {choice['text']}"
        for choice in choices
    )


def _build_user(
    item: dict,
    *,
    use_theorem: bool = False,
    use_sketch: bool = False,
    diagnostic_mode: bool = False,
    diagnostic_instruction: str = "",
    diagnostic_trace_context: str = "",
) -> str:
    parts = [f"## Question\n{item['question']}", f"## Choices\n{_format_choices(item['choices'])}"]
    if use_theorem and item.get("theorem"):
        parts.append(f"## Theorem\n{item['theorem']}")
    if use_sketch and item.get("sketch"):
        parts.append(f"## Proof Sketch\n{item['sketch']}")
    if diagnostic_trace_context.strip():
        parts.append(
            "## Previous Codex Trace Snapshot\n"
            "This is a partial transcript from an earlier attempt. Use it as your current reasoning context.\n\n"
            f"{diagnostic_trace_context.strip()}"
        )
    if diagnostic_mode and diagnostic_instruction.strip():
        parts.append(f"## Training Readout\n{diagnostic_instruction.strip()}")
    return "\n\n".join(parts)


def _build_codex_skill(skill_content: str) -> str:
    return render_skill_md(
        skill_content,
        description="Dynamic ReflACT skill for solving the current LiveMathematicianBench multiple-choice question.",
        preamble=(
            "Use this skill when solving the current math multiple-choice question.\n"
            "Inspect the option wording carefully and output only the final choice label inside <answer>...</answer>."
        ),
    )

def _run_codex_once(
    *,
    pred_dir: str,
    skill_content: str,
    item: dict,
    model: str,
    timeout: int,
    use_theorem: bool = False,
    use_sketch: bool = False,
    diagnostic_mode: bool = False,
    diagnostic_instruction: str = "",
    diagnostic_trace_context: str = "",
    previous_response: str = "",
) -> tuple[str, str, str, str]:
    user = _build_user(
        item,
        use_theorem=use_theorem,
        use_sketch=use_sketch,
        diagnostic_mode=diagnostic_mode,
        diagnostic_instruction=diagnostic_instruction,
        diagnostic_trace_context=diagnostic_trace_context,
    )
    task_parts = [user]
    if previous_response:
        task_parts.append(
            "## Previous Attempt\n"
            f"{previous_response}\n\n"
            "Re-evaluate the exact option wording. If needed, correct it."
        )
    task_text = "\n\n".join(task_parts)
    skill_md = _build_codex_skill(skill_content)
    work_dir = os.path.join(pred_dir, "codex_exec")
    prepare_workspace(work_dir=work_dir, skill_md=skill_md, task_text=task_text)
    prompt = (
        "Use the `skillopt-target` skill available in this workspace.\n"
        "Read `task.md` and solve the multiple-choice problem.\n"
        "Output only the final choice label inside <answer>...</answer>."
    )
    final_message, raw = run_target_exec(
        work_dir=work_dir,
        prompt=prompt,
        model=model,
        timeout=timeout,
    )
    return final_message or raw, raw, skill_md, task_text


def process_one(
    item: dict,
    out_root: str,
    skill_content: str,
    *,
    max_turns: int = 1,
    use_theorem: bool = False,
    use_sketch: bool = False,
    diagnostic_mode: bool = False,
    diagnostic_instruction: str = "",
    diagnostic_trace_context: str = "",
    exec_timeout: int | None = 300,
    max_completion_tokens: int = 16384,
) -> dict:
    item_id = str(item["id"])
    result = {
        "id": item_id,
        "question": item["question"],
        "task_type": item.get("theorem_type", ["math_mcq"])[0] if item.get("theorem_type") else "math_mcq",
        "hard": 0,
        "soft": 0.0,
        "predicted_answer": "",
        "predicted_label": "",
        "predicted_text": "",
        "correct_label": item["correct_choice"]["label"],
        "correct_text": item["correct_choice"]["text"],
        "response": "",
        "fail_reason": "",
        "agent_ok": False,
        "n_turns": 0,
    }

    try:
        pred_dir = os.path.join(out_root, "predictions", item_id)
        os.makedirs(pred_dir, exist_ok=True)
        llm_timeout = int(exec_timeout) if exec_timeout and int(exec_timeout) > 0 else None

        if is_target_exec_backend():
            from skillopt.model import azure_openai as _llm

            conversation: list[dict] = []
            response = ""
            system = ""
            user = ""
            for turn in range(max_turns):
                response, raw, system, user = _run_codex_once(
                    pred_dir=pred_dir,
                    skill_content=skill_content,
                    item=item,
                    model=_llm.TARGET_DEPLOYMENT,
                    timeout=llm_timeout,
                    use_theorem=use_theorem,
                    use_sketch=use_sketch,
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
            result["n_turns"] = len(conversation)

            with open(os.path.join(pred_dir, "target_system_prompt.txt"), "w", encoding="utf-8") as f:
                f.write(system)
            with open(os.path.join(pred_dir, "target_user_prompt.txt"), "w", encoding="utf-8") as f:
                f.write(user)

            eval_result = evaluate(response, item["correct_choice"], item["choices"])
            result["hard"] = int(eval_result["em"])
            result["soft"] = eval_result["f1"]
            result["predicted_answer"] = eval_result["predicted_answer"]
            result["predicted_label"] = eval_result["predicted_label"]
            result["predicted_text"] = eval_result["predicted_text"]
            if not result["hard"]:
                result["fail_reason"] = (
                    f"MCQ=0: predicted '{eval_result['predicted_label'] or eval_result['predicted_answer']}' "
                    f"but expected '{eval_result['correct_label']}'"
                )
            eval_detail = (
                f"[EVALUATION RESULT]\n"
                f"Question: {item['question']}\n"
                f"Predicted label: {eval_result['predicted_label']!r}\n"
                f"Predicted text: {eval_result['predicted_text']!r}\n"
                f"Correct label: {eval_result['correct_label']!r}\n"
                f"Correct text: {eval_result['correct_text']!r}\n"
                f"Exact Match: {eval_result['em']}"
            )
            conversation.append({"role": "system", "content": eval_detail})
            with open(os.path.join(pred_dir, "conversation.json"), "w") as f:
                json.dump(conversation, f, ensure_ascii=False, indent=2)
            return result

        system = _build_system(skill_content)
        user = _build_user(
            item,
            use_theorem=use_theorem,
            use_sketch=use_sketch,
            diagnostic_mode=diagnostic_mode,
            diagnostic_instruction=diagnostic_instruction,
            diagnostic_trace_context=diagnostic_trace_context,
        )
        conversation: list[dict] = []
        response = ""

        for turn in range(max_turns):
            if turn == 0:
                resp_text, _ = chat_target(
                    system=system,
                    user=user,
                    max_completion_tokens=max_completion_tokens,
                    retries=5,
                    stage="rollout",
                    timeout=llm_timeout,
                )
            else:
                refinement = (
                    f"Your previous answer was:\n{response}\n\n"
                    "Re-evaluate the exact option wording. If needed, correct it. "
                    "Output only the final choice label inside <answer>...</answer>."
                )
                resp_text, _ = chat_target(
                    system=system,
                    user=refinement,
                    max_completion_tokens=max_completion_tokens,
                    retries=5,
                    stage="rollout",
                    timeout=llm_timeout,
                )
            response = resp_text
            conversation.append({"type": "message", "turn": turn + 1, "content": resp_text})
            if "<answer>" in resp_text.lower():
                break

        result["response"] = response
        result["agent_ok"] = True
        result["n_turns"] = len(conversation)

        with open(os.path.join(pred_dir, "target_system_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(system)
        with open(os.path.join(pred_dir, "target_user_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(user)

        eval_result = evaluate(response, item["correct_choice"], item["choices"])
        result["hard"] = int(eval_result["em"])
        result["soft"] = eval_result["f1"]
        result["predicted_answer"] = eval_result["predicted_answer"]
        result["predicted_label"] = eval_result["predicted_label"]
        result["predicted_text"] = eval_result["predicted_text"]

        if not result["hard"]:
            result["fail_reason"] = (
                f"MCQ=0: predicted '{eval_result['predicted_label'] or eval_result['predicted_answer']}' "
                f"but expected '{eval_result['correct_label']}'"
            )

        eval_detail = (
            f"[EVALUATION RESULT]\n"
            f"Question: {item['question']}\n"
            f"Predicted label: {eval_result['predicted_label']!r}\n"
            f"Predicted text: {eval_result['predicted_text']!r}\n"
            f"Correct label: {eval_result['correct_label']!r}\n"
            f"Correct text: {eval_result['correct_text']!r}\n"
            f"Exact Match: {eval_result['em']}"
        )
        conversation.append({"role": "system", "content": eval_detail})

        with open(os.path.join(pred_dir, "conversation.json"), "w") as f:
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
    exec_timeout: int | None = 300,
    workers: int = 64,
    max_completion_tokens: int = 16384,
    use_theorem: bool = False,
    use_sketch: bool = False,
    diagnostic_mode: bool = False,
    diagnostic_instruction: str = "",
    diagnostic_trace_context_by_id: dict[str, str] | None = None,
    task_timeout: int | None = 600,
) -> list[dict]:
    exec_timeout_value = int(exec_timeout) if exec_timeout and int(exec_timeout) > 0 else 0
    task_timeout_value = int(task_timeout) if task_timeout and int(task_timeout) > 0 else 0
    if exec_timeout_value <= 0 or task_timeout_value <= 0:
        task_timeout = None
    else:
        task_timeout = max(task_timeout_value, exec_timeout_value + 60)
    results_path = os.path.join(out_root, "results.jsonl")
    os.makedirs(out_root, exist_ok=True)

    done_ids: set[str] = set()
    existing: list[dict] = []
    if os.path.exists(results_path):
        with open(results_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done_ids.add(str(r["id"]))
                    existing.append(r)
                except Exception:
                    pass

    pending = [it for it in items if str(it["id"]) not in done_ids]
    if not pending:
        return existing

    total = len(existing) + len(pending)
    completed = len(existing)
    correct_count = sum(1 for r in existing if r.get("hard", 0))
    if existing:
        print(f"    [rollout] resuming: {completed}/{total} already done", flush=True)

    results = list(existing)

    started_at: dict[str, float] = {}

    def _run_one(it: dict) -> dict:
        started_at[str(it["id"])] = time.time()
        return process_one(
            it,
            out_root,
            skill_content,
            max_turns=max_turns,
            exec_timeout=exec_timeout,
            max_completion_tokens=max_completion_tokens,
            use_theorem=use_theorem,
            use_sketch=use_sketch,
            diagnostic_mode=diagnostic_mode,
            diagnostic_instruction=diagnostic_instruction,
            diagnostic_trace_context=(diagnostic_trace_context_by_id or {}).get(str(it["id"]), ""),
        )

    def _timeout_result(it: dict) -> dict:
        correct = it.get("correct_choice") or {}
        return {
            "id": str(it["id"]),
            "question": it.get("question", ""),
            "task_type": it.get("theorem_type", ["math_mcq"])[0] if it.get("theorem_type") else "math_mcq",
            "hard": 0,
            "soft": 0.0,
            "predicted_answer": "",
            "predicted_label": "",
            "predicted_text": "",
            "correct_label": correct.get("label", ""),
            "correct_text": correct.get("text", ""),
            "response": "",
            "fail_reason": f"task-timeout-{task_timeout}s",
            "agent_ok": False,
            "n_turns": 0,
        }

    def _error_result(it: dict, exc: Exception) -> dict:
        res = _timeout_result(it)
        res["fail_reason"] = f"error: {type(exc).__name__}: {exc}"
        return res

    with open(results_path, "a") as outf:
        ex = ThreadPoolExecutor(max_workers=workers)
        try:
            futs = {
                ex.submit(_run_one, it): it
                for it in pending
            }
            pending_futs = set(futs)
            while pending_futs:
                done, _ = wait(pending_futs, timeout=5, return_when=FIRST_COMPLETED)
                now = time.time()
                timed_out = [
                    fut for fut in pending_futs - done
                    if task_timeout is not None
                    if str(futs[fut]["id"]) in started_at
                    and now - started_at[str(futs[fut]["id"])] >= task_timeout
                ]
                for fut in done:
                    pending_futs.remove(fut)
                    item = futs[fut]
                    try:
                        res = fut.result()
                    except Exception as e:  # noqa: BLE001
                        res = _error_result(item, e)
                    results.append(res)
                    completed += 1
                    if res.get("hard", 0):
                        correct_count += 1
                    acc = correct_count / completed if completed else 0
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
                    res = _timeout_result(futs[fut])
                    results.append(res)
                    completed += 1
                    acc = correct_count / completed if completed else 0
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
