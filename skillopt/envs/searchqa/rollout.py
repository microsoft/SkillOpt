"""SearchQA rollout — single-turn QA agent + batch execution.

The QA agent receives a skill document, question, and context passages,
then produces an answer in <answer>...</answer> tags.

Public API
----------
- :func:`process_one`  — run + evaluate one QA item
- :func:`run_batch`    — parallel execution of a list of items
"""
from __future__ import annotations

import json
import os
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from skillopt.model import chat_student, get_student_backend, is_student_exec_backend
from skillopt.model.codex_harness import prepare_workspace, render_skill_md, run_student_exec
from skillopt.prompts import load_prompt
from skillopt.envs.searchqa.evaluator import evaluate


# ── Prompt templates ─────────────────────────────────────────────────────────

_MAX_CONTEXT_CHARS = 6000


def _truncate_context(context: str, max_chars: int = _MAX_CONTEXT_CHARS) -> str:
    """Truncate context at [DOC] boundaries to stay within budget."""
    if len(context) <= max_chars:
        return context
    docs = context.split("[DOC]")
    result = ""
    for doc in docs:
        candidate = result + "[DOC]" + doc if result else doc
        if len(candidate) > max_chars:
            break
        result = candidate
    if not result:
        result = context[:max_chars] + "\n...[truncated]"
    return result


def _build_system(skill_content: str) -> str:
    if skill_content.strip():
        skill_section = f"## Skill\n{skill_content.strip()}\n\n"
    else:
        skill_section = ""
    return load_prompt("rollout_system", env="searchqa").format(skill_section=skill_section)


def _build_user(
    question: str,
    context: str,
    *,
    diagnostic_mode: bool = False,
    diagnostic_instruction: str = "",
    diagnostic_trace_context: str = "",
) -> str:
    context = _truncate_context(context)
    parts = [
        f"## Context\n{context}",
        f"## Question\n{question}",
    ]
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
        description="Dynamic ReflACT skill for solving the current SearchQA example.",
        preamble=(
            "Use this skill when solving the current SearchQA task.\n"
            "Read the provided context carefully, ground the answer in that context,\n"
            "and return the final answer inside <answer>...</answer>."
        ),
    )


def _run_codex_once(
    *,
    pred_dir: str,
    skill_content: str,
    question: str,
    context: str,
    model: str,
    timeout: int,
    diagnostic_mode: bool = False,
    diagnostic_instruction: str = "",
    diagnostic_trace_context: str = "",
    previous_response: str = "",
) -> tuple[str, str, str, str]:
    user = _build_user(
        question,
        context,
        diagnostic_mode=diagnostic_mode,
        diagnostic_instruction=diagnostic_instruction,
        diagnostic_trace_context=diagnostic_trace_context,
    )
    task_parts = [user]
    if previous_response:
        task_parts.append(
            "## Previous Attempt\n"
            f"{previous_response}\n\n"
            "Review it against the same context and question. If needed, correct it."
        )
    task_text = "\n\n".join(task_parts)
    skill_md = _build_codex_skill(skill_content)
    work_dir = os.path.join(pred_dir, "codex_exec")
    prepare_workspace(
        work_dir=work_dir,
        skill_md=skill_md,
        task_text=task_text,
    )
    prompt = (
        "Use the `skillopt-student` skill available in this workspace.\n"
        "Read `task.md` and answer the SearchQA question.\n"
        "Return the final answer inside <answer>...</answer>."
    )
    final_message, raw = run_student_exec(
        work_dir=work_dir,
        prompt=prompt,
        model=model,
        timeout=timeout,
    )
    return final_message or raw, raw, skill_md, task_text


# ── Single-item execution ───────────────────────────────────────────────────


def process_one(
    item: dict,
    out_root: str,
    skill_content: str,
    max_turns: int = 1,
    diagnostic_mode: bool = False,
    diagnostic_instruction: str = "",
    diagnostic_trace_context: str = "",
    exec_timeout: int = 120,
) -> dict:
    """Process a single QA item: run agent + evaluate.

    Parameters
    ----------
    item : dict
        Must have keys: ``id``, ``question``, ``context``, ``answers``.
    out_root : str
        Output directory (predictions saved under ``predictions/<id>/``).
    skill_content : str
        Current skill document text.
    max_turns : int
        Max reasoning turns (1 = single-turn QA).

    Returns
    -------
    dict
        Result with ``hard`` (EM as int), ``soft`` (F1), etc.
    """
    item_id = str(item["id"])
    question = item["question"]
    context = item.get("context", "")
    gold_answers = item.get("answers", [])

    result = {
        "id": item_id,
        "question": question,
        "em": 0.0,
        "f1": 0.0,
        "sub_em": 0.0,
        "hard": 0,
        "soft": 0.0,
        "predicted_answer": "",
        "gold_answers": gold_answers,
        "response": "",
        "fail_reason": "",
        "agent_ok": False,
        "n_turns": 0,
    }

    try:
        pred_dir = os.path.join(out_root, "predictions", item_id)
        os.makedirs(pred_dir, exist_ok=True)

        if is_student_exec_backend():
            from skillopt.model import azure_openai as _llm

            conversation: list[dict] = []
            response = ""
            system = ""
            user = ""
            for turn in range(max_turns):
                response, raw, system, user = _run_codex_once(
                    pred_dir=pred_dir,
                    skill_content=skill_content,
                    question=question,
                    context=context,
                    model=_llm.STUDENT_DEPLOYMENT,
                    timeout=exec_timeout,
                    diagnostic_mode=diagnostic_mode if turn == 0 else False,
                    diagnostic_instruction=diagnostic_instruction if turn == 0 else "",
                    diagnostic_trace_context=diagnostic_trace_context if turn == 0 else "",
                    previous_response=response if turn > 0 else "",
                )
                conversation.append({"type": "message", "turn": turn + 1, "content": response})
                if turn > 0 and "<answer>" in response.lower():
                    break

            result["response"] = response
            result["agent_ok"] = True
            result["n_turns"] = len(conversation)

            with open(os.path.join(pred_dir, "student_system_prompt.txt"), "w") as f:
                f.write(system)
            with open(os.path.join(pred_dir, "student_user_prompt.txt"), "w") as f:
                f.write(user)
            with open(os.path.join(pred_dir, "conversation.json"), "w") as f:
                json.dump(conversation, f, ensure_ascii=False, indent=2)

            eval_result = evaluate(response, gold_answers)
            result["em"] = eval_result["em"]
            result["f1"] = eval_result["f1"]
            result["sub_em"] = eval_result["sub_em"]
            result["predicted_answer"] = eval_result["predicted_answer"]
            result["hard"] = int(eval_result["em"])
            result["soft"] = eval_result["f1"]
            if eval_result["em"] < 1.0:
                result["fail_reason"] = (
                    f"EM=0: predicted '{eval_result['predicted_answer']}' "
                    f"but expected {gold_answers}"
                )
            eval_detail = (
                f"[EVALUATION RESULT]\n"
                f"Question: {question}\n"
                f"Predicted answer: {eval_result['predicted_answer']!r}\n"
                f"Gold answers: {gold_answers!r}\n"
                f"Exact Match: {eval_result['em']}\n"
                f"F1: {eval_result['f1']:.4f}"
            )
            conversation.append({"role": "system", "content": eval_detail})
            with open(os.path.join(pred_dir, "conversation.json"), "w") as f:
                json.dump(conversation, f, ensure_ascii=False, indent=2)
            return result

        system = _build_system(skill_content)
        user = _build_user(
            question,
            context,
            diagnostic_mode=diagnostic_mode,
            diagnostic_instruction=diagnostic_instruction,
            diagnostic_trace_context=diagnostic_trace_context,
        )

        conversation: list[dict] = []
        response = ""

        for turn in range(max_turns):
            if turn == 0:
                resp_text, _ = chat_student(
                    system=system, user=user,
                    max_completion_tokens=512,
                    retries=5, stage="rollout",
                    timeout=exec_timeout,
                )
            else:
                refinement = (
                    f"Your previous answer was:\n{response}\n\n"
                    f"Review it against the context and question. "
                    f"If correct, repeat it. If wrong, provide a corrected answer.\n"
                    f"Use <answer>...</answer> tags for your final answer."
                )
                resp_text, _ = chat_student(
                    system=system, user=refinement,
                    max_completion_tokens=512,
                    retries=5, stage="rollout",
                    timeout=exec_timeout,
                )

            response = resp_text
            conversation.append({"type": "message", "turn": turn + 1, "content": resp_text})

            if turn > 0 and "<answer>" in resp_text.lower():
                break

        result["response"] = response
        result["agent_ok"] = True
        result["n_turns"] = len(conversation)

        # Save conversation
        with open(os.path.join(pred_dir, "student_system_prompt.txt"), "w") as f:
            f.write(system)
        with open(os.path.join(pred_dir, "student_user_prompt.txt"), "w") as f:
            f.write(user)
        with open(os.path.join(pred_dir, "conversation.json"), "w") as f:
            json.dump(conversation, f, ensure_ascii=False, indent=2)

        # Evaluate
        eval_result = evaluate(response, gold_answers)
        result["em"] = eval_result["em"]
        result["f1"] = eval_result["f1"]
        result["sub_em"] = eval_result["sub_em"]
        result["predicted_answer"] = eval_result["predicted_answer"]
        result["hard"] = int(eval_result["em"])
        result["soft"] = eval_result["f1"]

        if eval_result["em"] < 1.0:
            result["fail_reason"] = (
                f"EM=0: predicted '{eval_result['predicted_answer']}' "
                f"but expected {gold_answers}"
            )

        # Append eval details to conversation for the analyst
        eval_detail = (
            f"[EVALUATION RESULT]\n"
            f"Question: {question}\n"
            f"Predicted answer: {eval_result['predicted_answer']!r}\n"
            f"Gold answers: {gold_answers!r}\n"
            f"Exact Match: {eval_result['em']}\n"
            f"F1: {eval_result['f1']:.4f}"
        )
        conversation.append({
            "role": "system",
            "content": eval_detail,
        })
        # Re-save enriched conversation
        with open(os.path.join(pred_dir, "conversation.json"), "w") as f:
            json.dump(conversation, f, ensure_ascii=False, indent=2)

    except Exception as e:  # noqa: BLE001
        result["fail_reason"] = f"error: {e}"

    return result


# ── Batch execution ──────────────────────────────────────────────────────────


def run_batch(
    items: list[dict],
    out_root: str,
    skill_content: str,
    max_turns: int = 1,
    exec_timeout: int = 120,
    workers: int = 64,
    diagnostic_mode: bool = False,
    diagnostic_instruction: str = "",
    diagnostic_trace_context_by_id: dict[str, str] | None = None,
    task_timeout: int = 600,
) -> list[dict]:
    """Run QA agent on all items with ThreadPoolExecutor. Resume-aware."""
    task_timeout = max(int(task_timeout), int(exec_timeout) + 60)
    results_path = os.path.join(out_root, "results.jsonl")
    os.makedirs(out_root, exist_ok=True)

    # Resume: load already-done
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

    results = list(existing)

    def _timeout_result(item: dict) -> dict:
        return {
            "id": str(item["id"]),
            "question": item.get("question", ""),
            "task_description": item.get("question", ""),
            "task_type": item.get("task_type") or "searchqa",
            "hard": 0,
            "soft": 0.0,
            "predicted_answer": "",
            "response": "",
            "fail_reason": f"task-timeout-{task_timeout}s",
            "agent_ok": False,
            "n_turns": 0,
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
            max_turns,
            diagnostic_mode,
            diagnostic_instruction,
            (diagnostic_trace_context_by_id or {}).get(str(item["id"]), ""),
            exec_timeout,
        )

    with open(results_path, "a") as outf:
        ex = ThreadPoolExecutor(max_workers=workers)
        try:
            futs = {ex.submit(_run_one, it): it for it in pending}
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
                    outf.write(json.dumps(res, ensure_ascii=False) + "\n")
                    outf.flush()
                for fut in timed_out:
                    pending_futs.remove(fut)
                    fut.cancel()
                    res = _timeout_result(futs[fut])
                    results.append(res)
                    outf.write(json.dumps(res, ensure_ascii=False) + "\n")
                    outf.flush()
        finally:
            ex.shutdown(wait=False, cancel_futures=True)

    return results
