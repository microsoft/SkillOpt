from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from skillopt.envs.sealqa.evaluator import score_sealqa
from skillopt.envs.sealqa.tool_runtime import web_fetch
from skillopt.model import chat_student, get_student_backend, is_student_exec_backend
from skillopt.model.codex_harness import prepare_workspace, render_skill_md, run_student_exec
from skillopt.prompts import load_prompt

_FINAL_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)


def _build_system(skill_content: str) -> str:
    if skill_content.strip():
        skill_section = f"## Skill\n{skill_content.strip()}\n\n"
    else:
        skill_section = ""
    return load_prompt("rollout_system", env="sealqa").format(skill_section=skill_section)


def _build_user(item: dict, *, diagnostic_mode: bool = False, diagnostic_instruction: str = '') -> str:
    parts = [f"## Question\n{item['question']}"]
    if item.get('search_results'):
        parts.append(f"## Search Results\n{item['search_results']}")
    if item.get('urls'):
        parts.append(f"## URL Hints\n{item['urls']}")
    if item.get('freshness'):
        parts.append(f"## Freshness\n{item['freshness']}")
    if item.get('question_types'):
        parts.append(f"## Question Types\n{item['question_types']}")
    if diagnostic_mode and diagnostic_instruction.strip():
        parts.append(f"## Training Readout\n{diagnostic_instruction.strip()}")
    parts.append('Use the provided search evidence as your primary context. Do not rely on external tool use.')
    return "\n\n".join(parts)


def _extract_answer(text: str) -> str:
    match = _FINAL_RE.search(text)
    if match:
        return match.group(1).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text.strip()


def _build_codex_skill(skill_content: str) -> str:
    return render_skill_md(
        skill_content,
        description="Dynamic ReflACT skill for solving the current SealQA evidence-grounded question.",
        preamble=(
            "Use this skill when answering the current SealQA question.\n"
            "Use the provided search evidence first, reconcile conflicts carefully,\n"
            "and return the final answer inside <answer>...</answer>."
        ),
    )


def _run_codex_once(
    *,
    pred_dir: str,
    skill_content: str,
    task_text: str,
    model: str,
    timeout: int,
    previous_response: str = '',
) -> tuple[str, str, str, str]:
    task_parts = [task_text]
    if previous_response:
        task_parts.append(
            "## Previous Attempt\n"
            f"{previous_response}\n\n"
            "Review the evidence again and correct the final answer if needed."
        )
    final_task_text = "\n\n".join(task_parts)
    skill_md = _build_codex_skill(skill_content)
    work_dir = os.path.join(pred_dir, 'codex_exec')
    prepare_workspace(
        work_dir=work_dir,
        skill_md=skill_md,
        task_text=final_task_text,
    )
    prompt = (
        "Use the `skillopt-student` skill available in this workspace.\n"
        "Read `task.md`, answer the SealQA question using the provided evidence,\n"
        "and return the final answer inside <answer>...</answer>."
    )
    final_message, raw = run_student_exec(
        work_dir=work_dir,
        prompt=prompt,
        model=model,
        timeout=timeout,
    )
    return final_message or raw, raw, skill_md, final_task_text


def process_one(
    item: dict,
    out_root: str,
    skill_content: str,
    *,
    max_tool_turns: int = 12,
    diagnostic_mode: bool = False,
    diagnostic_instruction: str = '',
) -> dict:
    item_id = str(item['id'])
    pred_dir = os.path.join(out_root, 'predictions', item_id)
    os.makedirs(pred_dir, exist_ok=True)

    system = _build_system(skill_content)
    user = _build_user(
        item,
        diagnostic_mode=diagnostic_mode,
        diagnostic_instruction=diagnostic_instruction,
    )
    conversation: list[dict] = [{'role': 'user', 'content': user}]
    final_response = ''
    final_answer = ''
    fail_reason = ''

    try:
        if is_student_exec_backend():
            from skillopt.model import azure_openai as _llm

            response, _raw, system, user_for_save = _run_codex_once(
                pred_dir=pred_dir,
                skill_content=skill_content,
                task_text=user,
                model=_llm.STUDENT_DEPLOYMENT,
                timeout=120,
            )
            final_response = response
            conversation.append({'type': 'message', 'content': response})
            if '<answer>' in response.lower():
                final_answer = _extract_answer(response)
            else:
                user = user_for_save
        else:
            response, _ = chat_student(
                system=system,
                user=user,
                max_completion_tokens=768,
                retries=5,
                stage='rollout',
            )
            final_response = response
            conversation.append({'type': 'message', 'content': response})
            if '<answer>' in response.lower():
                final_answer = _extract_answer(response)

        if not final_answer:
            urls_text = str(item.get('urls') or '').strip()
            fetched_blocks = []
            for raw_url in re.findall(r'https?://[^\s\]\[\'\",]+', urls_text)[:2]:
                try:
                    fetched = web_fetch(raw_url)
                except Exception as fetch_error:  # noqa: BLE001
                    fetched = f'URL: {raw_url}\n\n[fetch error: {fetch_error}]'
                fetched_blocks.append(fetched)
                conversation.append({'type': 'tool_call', 'cmd': f'web_fetch({raw_url!r})', 'obs': fetched})
            if fetched_blocks:
                retry_user = user + '\n\n## Fetched URL Content\n' + '\n\n'.join(fetched_blocks)
                if is_student_exec_backend():
                    retry_response, _raw, system, retry_user = _run_codex_once(
                        pred_dir=pred_dir,
                        skill_content=skill_content,
                        task_text=retry_user,
                        model=_llm.STUDENT_DEPLOYMENT,
                        timeout=120,
                        previous_response=final_response,
                    )
                else:
                    retry_response, _ = chat_student(
                        system=system,
                        user=retry_user,
                        max_completion_tokens=768,
                        retries=5,
                        stage='rollout',
                    )
                final_response = retry_response
                conversation.append({'type': 'message', 'content': retry_response})
                if '<answer>' in retry_response.lower():
                    final_answer = _extract_answer(retry_response)
                else:
                    fail_reason = 'Model did not produce a final answer'
            else:
                fail_reason = 'Model did not produce a final answer'
    except Exception as e:  # noqa: BLE001
        fail_reason = f'error: {e}'

    with open(os.path.join(pred_dir, 'student_system_prompt.txt'), 'w', encoding='utf-8') as f:
        f.write(system)
    with open(os.path.join(pred_dir, 'student_user_prompt.txt'), 'w', encoding='utf-8') as f:
        f.write(user)
    with open(os.path.join(pred_dir, 'conversation.json'), 'w', encoding='utf-8') as f:
        json.dump(conversation, f, ensure_ascii=False, indent=2)

    score = score_sealqa(item.get('question', ''), item.get('ground_truth', ''), final_answer) if final_answer else 0.0
    result = {
        'id': item_id,
        'question': item.get('question', ''),
        'task_type': item.get('task_type', 'sealqa'),
        'task_description': item.get('question', ''),
        'predicted_answer': final_answer,
        'response': final_response,
        'ground_truth': item.get('ground_truth', ''),
        'hard': int(score >= 1.0),
        'soft': float(score),
        'fail_reason': fail_reason or ('' if score >= 1.0 else f"predicted '{final_answer}' but expected '{item.get('ground_truth', '')}'"),
        'agent_ok': not fail_reason,
        'n_turns': len(conversation),
        'student_system_prompt': system,
        'student_user_prompt': user,
    }
    return result


def run_batch(
    items: list[dict],
    out_root: str,
    skill_content: str,
    *,
    workers: int = 4,
    max_tool_turns: int = 12,
    diagnostic_mode: bool = False,
    diagnostic_instruction: str = '',
) -> list[dict]:
    results_path = os.path.join(out_root, 'results.jsonl')
    os.makedirs(out_root, exist_ok=True)

    done_ids: set[str] = set()
    existing: list[dict] = []
    if os.path.exists(results_path):
        with open(results_path, encoding='utf-8') as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done_ids.add(str(row.get('id')))
                existing.append(row)

    pending = [item for item in items if str(item['id']) not in done_ids]
    if not pending:
        return existing

    results = list(existing)
    with open(results_path, 'a', encoding='utf-8') as outf, ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(
                process_one,
                item,
                out_root,
                skill_content,
                max_tool_turns=max_tool_turns,
                diagnostic_mode=diagnostic_mode,
                diagnostic_instruction=diagnostic_instruction,
            ): item
            for item in pending
        }
        for fut in as_completed(futs):
            res = fut.result()
            results.append(res)
            outf.write(json.dumps(res, ensure_ascii=False) + '\n')
            outf.flush()
    return results
