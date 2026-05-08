from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from skillopt.envs.officeqa.evaluator import evaluate
from skillopt.envs.officeqa.tool_runtime import resolve_candidate_files, resolve_docs_roots, run_tool
from skillopt.model import chat_student_messages, get_student_backend, is_student_exec_backend
from skillopt.model.codex_harness import prepare_workspace, render_skill_md, run_student_exec
from skillopt.prompts import load_prompt

_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find candidate local document files by filename or relative-path glob pattern.",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a local text document excerpt by path and line window.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start": {"type": "integer"},
                    "limit": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search a local text document for a literal pattern and return matching lines.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["pattern", "path"],
            },
        },
    },
]

_FINAL_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)


def _build_system(skill_content: str) -> str:
    if skill_content.strip():
        skill_section = f"## Skill\n{skill_content.strip()}\n\n"
    else:
        skill_section = ""
    return load_prompt("rollout_system", env="officeqa").format(skill_section=skill_section)


def _build_user(
    item: dict,
    candidate_files: list[str],
    *,
    diagnostic_mode: bool = False,
    diagnostic_instruction: str = "",
    corpus_note: str = "",
) -> str:
    file_block = "\n".join(f"- {path}" for path in candidate_files[:20]) or "- none resolved"
    parts = [f"## Question\n{item['question']}"]
    if corpus_note.strip():
        parts.append(f"## Document Corpus\n{corpus_note.strip()}")
    parts.append(f"## Candidate Files\n{file_block}")
    if item.get("source_docs"):
        parts.append("## Source Hints\n" + "\n".join(f"- {hint}" for hint in item["source_docs"]))
    if diagnostic_mode and diagnostic_instruction.strip():
        parts.append(f"## Training Readout\n{diagnostic_instruction.strip()}")
    return "\n\n".join(parts)


def _extract_answer(text: str) -> str:
    match = _FINAL_RE.search(text)
    if match:
        return match.group(1).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text.strip()


def _docs_link_targets(docs_roots: list[str]) -> list[tuple[str, str]]:
    return [(root, os.path.join("docs", f"root_{idx}")) for idx, root in enumerate(docs_roots, start=1)]


def _workspace_doc_path(path: str, docs_roots: list[str]) -> str:
    resolved_path = os.path.realpath(path)
    for idx, root in enumerate(docs_roots, start=1):
        resolved_root = os.path.realpath(root)
        if resolved_path == resolved_root or resolved_path.startswith(resolved_root + os.sep):
            rel_path = os.path.relpath(resolved_path, resolved_root)
            return os.path.join("docs", f"root_{idx}", rel_path)
    return path


def _build_codex_skill(skill_content: str) -> str:
    return render_skill_md(
        skill_content,
        description="Dynamic ReflACT skill for solving the current OfficeQA local-document question.",
        preamble=(
            "Use this skill when answering the current OfficeQA question.\n"
            "Inspect the provided local document excerpts or files, ground the answer in the evidence,\n"
            "and return the final answer inside <answer>...</answer>."
        ),
    )


def _run_codex_once(
    *,
    pred_dir: str,
    item: dict,
    skill_content: str,
    candidate_files: list[str],
    docs_roots: list[str],
    model: str,
    timeout: int,
    diagnostic_mode: bool = False,
    diagnostic_instruction: str = "",
    previous_response: str = "",
) -> tuple[str, str, str, str]:
    rel_files = [_workspace_doc_path(path, docs_roots) for path in candidate_files[:20]]
    corpus_note = (
        "The full OfficeQA document corpus is available under `docs/`. "
        "The candidate files below are source hints or likely starting points; search the full corpus if needed."
    )
    user = _build_user(
        item,
        rel_files,
        diagnostic_mode=diagnostic_mode,
        diagnostic_instruction=diagnostic_instruction,
        corpus_note=corpus_note,
    )
    task_parts = [user]
    if previous_response:
        task_parts.append(
            "## Previous Attempt\n"
            f"{previous_response}\n\n"
            "Review the local documents again and correct the answer if needed."
        )
    task_text = "\n\n".join(task_parts)
    skill_md = _build_codex_skill(skill_content)
    work_dir = os.path.join(pred_dir, "codex_exec")
    prepare_workspace(
        work_dir=work_dir,
        skill_md=skill_md,
        task_text=task_text,
        link_dirs=_docs_link_targets(docs_roots),
    )
    prompt = (
        "Use the `skillopt-student` skill available in this workspace.\n"
        "Read `task.md`, inspect or search the full OfficeQA corpus under `docs/`, and answer the question.\n"
        "Treat candidate files in `task.md` as hints, not an access limit.\n"
        "Return the final answer inside <answer>...</answer>."
    )
    final_message, raw = run_student_exec(
        work_dir=work_dir,
        prompt=prompt,
        model=model,
        timeout=timeout,
        data_dirs=docs_roots,
    )
    return final_message or raw, raw, skill_md, task_text


def process_one(
    item: dict,
    out_root: str,
    skill_content: str,
    *,
    max_tool_turns: int = 12,
    data_dirs: list[str] | str | None = None,
    diagnostic_mode: bool = False,
    diagnostic_instruction: str = "",
) -> dict:
    item_id = str(item["id"])
    pred_dir = os.path.join(out_root, "predictions", item_id)
    os.makedirs(pred_dir, exist_ok=True)

    docs_roots = resolve_docs_roots(data_dirs)
    candidate_files = resolve_candidate_files(item.get("source_files", []), docs_roots)
    system = _build_system(skill_content)
    user = _build_user(item, candidate_files, diagnostic_mode=diagnostic_mode, diagnostic_instruction=diagnostic_instruction)

    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    conversation: list[dict] = [{"role": "user", "content": user}]
    final_response = ""
    final_answer = ""
    fail_reason = ""

    allowed_files = [os.path.basename(path) for path in candidate_files]

    try:
        if is_student_exec_backend():
            from skillopt.model import azure_openai as _llm

            response = ""
            system = ""
            user = ""
            for turn in range(1, max_tool_turns + 1):
                response, _raw, system, user = _run_codex_once(
                    pred_dir=pred_dir,
                    item=item,
                    skill_content=skill_content,
                    candidate_files=candidate_files,
                    docs_roots=docs_roots,
                    model=_llm.STUDENT_DEPLOYMENT,
                    timeout=180,
                    diagnostic_mode=diagnostic_mode if turn == 1 else False,
                    diagnostic_instruction=diagnostic_instruction if turn == 1 else "",
                    previous_response=response if turn > 1 else "",
                )
                final_response = response
                conversation.append({"type": "message", "turn": turn, "content": response})
                if "<answer>" in response.lower():
                    final_answer = _extract_answer(response)
                    break
            if not final_answer:
                fail_reason = f"Exceeded codex turn budget ({max_tool_turns})"
            system = system or _build_codex_skill(skill_content)
            user = user or _build_user(item, [_workspace_doc_path(path, docs_roots) for path in candidate_files])
        else:
            for turn in range(1, max_tool_turns + 1):
                message, _ = chat_student_messages(
                    messages=messages,
                    max_completion_tokens=768,
                    retries=5,
                    stage="rollout",
                    tools=_TOOL_SCHEMAS,
                    tool_choice="auto",
                    return_message=True,
                )
                response = message.content or ""
                final_response = response
                assistant_message = {"role": "assistant", "content": response}
                if getattr(message, "tool_calls", None):
                    assistant_message["tool_calls"] = [tool_call.model_dump(mode="json") for tool_call in message.tool_calls]
                messages.append(assistant_message)
                conversation.append({"type": "message", "content": response})

                if getattr(message, "tool_calls", None):
                    for tool_call in message.tool_calls:
                        tool_name = tool_call.function.name
                        arguments = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
                        cmd, obs = run_tool(tool_name, arguments, allowed_roots=docs_roots, allowed_files=allowed_files)
                        conversation.append({"type": "tool_call", "cmd": cmd, "obs": obs})
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": obs,
                        })
                    continue

                if "<answer>" in response.lower():
                    final_answer = _extract_answer(response)
                    break
                if turn == max_tool_turns:
                    fail_reason = f"Exceeded tool-turn budget ({max_tool_turns})"
                else:
                    fail_reason = "Model neither produced a tool request nor a final answer"
                    break
    except Exception as e:  # noqa: BLE001
        fail_reason = f"error: {e}"

    with open(os.path.join(pred_dir, "student_system_prompt.txt"), "w", encoding="utf-8") as f:
        f.write(system)
    with open(os.path.join(pred_dir, "student_user_prompt.txt"), "w", encoding="utf-8") as f:
        f.write(user)
    with open(os.path.join(pred_dir, "conversation.json"), "w", encoding="utf-8") as f:
        json.dump(conversation, f, ensure_ascii=False, indent=2)

    eval_result = evaluate(final_answer, item.get("ground_truth", "")) if final_answer else {"em": 0.0, "f1": 0.0, "predicted_answer": "", "gold_answer": item.get("ground_truth", "")}
    result = {
        "id": item_id,
        "question": item.get("question", ""),
        "task_type": item.get("task_type", "officeqa"),
        "task_description": item.get("question", ""),
        "predicted_answer": eval_result["predicted_answer"],
        "response": final_response,
        "ground_truth": item.get("ground_truth", ""),
        "source_files": item.get("source_files", []),
        "resolved_source_paths": candidate_files,
        "hard": int(eval_result["em"]),
        "soft": eval_result["f1"],
        "fail_reason": fail_reason or ("" if eval_result["em"] else f"predicted '{eval_result['predicted_answer']}' but expected '{item.get('ground_truth', '')}'"),
        "agent_ok": not fail_reason,
        "n_turns": len(conversation),
        "student_system_prompt": system,
        "student_user_prompt": user,
    }
    return result


def run_batch(
    items: list[dict],
    out_root: str,
    skill_content: str,
    *,
    workers: int = 8,
    max_tool_turns: int = 12,
    data_dirs: list[str] | str | None = None,
    diagnostic_mode: bool = False,
    diagnostic_instruction: str = "",
) -> list[dict]:
    results_path = os.path.join(out_root, "results.jsonl")
    os.makedirs(out_root, exist_ok=True)

    done_ids: set[str] = set()
    existing: list[dict] = []
    if os.path.exists(results_path):
        with open(results_path, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done_ids.add(str(row.get("id")))
                existing.append(row)

    pending = [item for item in items if str(item["id"]) not in done_ids]
    if not pending:
        return existing

    results = list(existing)
    with open(results_path, "a", encoding="utf-8") as outf, ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(
                process_one,
                item,
                out_root,
                skill_content,
                max_tool_turns=max_tool_turns,
                data_dirs=data_dirs,
                diagnostic_mode=diagnostic_mode,
                diagnostic_instruction=diagnostic_instruction,
            ): item
            for item in pending
        }
        for fut in as_completed(futs):
            res = fut.result()
            results.append(res)
            outf.write(json.dumps(res, ensure_ascii=False) + "\n")
            outf.flush()
    return results
