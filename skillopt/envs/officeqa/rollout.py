from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from skillopt.envs.officeqa.evaluator import evaluate
from skillopt.envs.officeqa.tool_runtime import (
    build_oracle_parsed_pages_context,
    custom_search,
    resolve_candidate_files,
    resolve_docs_roots,
    run_tool,
)
from skillopt.model import chat_target_messages, get_target_backend, is_target_exec_backend
from skillopt.model.codex_harness import prepare_workspace, render_skill_md, run_target_exec
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
_SEARCH_RE = re.compile(r"<search_queries>(.*?)</search_queries>", re.IGNORECASE | re.DOTALL)
_DEFAULT_SEARCH_MODE = "offline"
_CUSTOM_SEARCH_MODE = "custom_search"
_AZURE_SEARCH_MODE = "azure_search"
def _normalize_search_mode(search_mode: str | None) -> str:
    normalized = str(search_mode or _DEFAULT_SEARCH_MODE).strip().lower()
    if normalized in {"custom", _CUSTOM_SEARCH_MODE}:
        return _CUSTOM_SEARCH_MODE
    if normalized in {"azure", _AZURE_SEARCH_MODE}:
        return _AZURE_SEARCH_MODE
    return _DEFAULT_SEARCH_MODE
def _build_system(
    skill_content: str,
    *,
    search_mode: str = _DEFAULT_SEARCH_MODE,
    use_local_tools: bool = True,
    max_tool_turns: int = 12,
    max_queries_per_turn: int = 4,
) -> str:
    if skill_content.strip():
        skill_section = f"## Skill\n{skill_content.strip()}\n\n"
    else:
        skill_section = ""
    normalized_search_mode = _normalize_search_mode(search_mode)
    if normalized_search_mode == _AZURE_SEARCH_MODE:
        return (
            "You are an expert OfficeQA research assistant. Solve the question using the model's built-in web "
            "search tool when needed, keep the answer grounded in authoritative evidence, and return the final "
            "answer inside <answer>...</answer>.\n\n"
            + skill_section
        ).rstrip()
    if normalized_search_mode == _CUSTOM_SEARCH_MODE:
        protocol = (
            "You are an expert OfficeQA research assistant. Solve the question using the provided oracle parsed "
            "OfficeQA page(s) and evidence returned by the controller-managed custom search loop.\n\n"
            "Search protocol:\n"
            f"- You have at most {max_tool_turns} model rounds total.\n"
            f"- On any non-final round, you may either return `<search_queries>[\"query 1\", \"query 2\"]</search_queries>` "
            f"with up to {max_queries_per_turn} queries, or return `<answer>...</answer>` if you are ready.\n"
            "- If you request search, do not include an answer in the same response.\n"
            "- On the final round, you must return `<answer>...</answer>` and must not request more search.\n"
            "- Base your answer on the returned evidence, reconcile conflicting snippets carefully, and stay concise.\n\n"
        )
        return protocol + skill_section + "Return the final answer inside <answer>...</answer> when you are ready."
    if not use_local_tools:
        return (
            "You are an expert OfficeQA research assistant. Solve the question using the provided oracle parsed "
            "OfficeQA page(s) and source hints. Do not request or assume access to any external search or local "
            "function tools. Return the final answer inside <answer>...</answer>.\n\n"
            + skill_section
        ).rstrip()
    return load_prompt("rollout_system", env="officeqa").format(skill_section=skill_section)
def _build_round_instruction(
    *,
    turn: int,
    max_tool_turns: int,
    max_queries_per_turn: int,
) -> str:
    if turn >= max_tool_turns:
        return (
            "## Round Policy\n"
            f"This is the final round ({turn}/{max_tool_turns}). You must return `<answer>...</answer>` now. "
            "Do not output `<search_queries>`."
        )
    remaining_rounds = max_tool_turns - turn
    return (
        "## Round Policy\n"
        f"This is round {turn}/{max_tool_turns}. "
        f"You may either return `<answer>...</answer>` now, or request up to {max_queries_per_turn} search queries "
        f"inside `<search_queries>...</search_queries>`. "
        f"After this response, at most {remaining_rounds} model rounds remain."
    )
def _message_debug_metadata(message: object) -> dict:
    metadata = getattr(message, "metadata", None)
    if isinstance(metadata, dict):
        return metadata
    return {}
def _build_user(
    item: dict,
    candidate_files: list[str] | None = None,
    *,
    diagnostic_mode: bool = False,
    diagnostic_instruction: str = "",
    corpus_note: str = "",
    search_mode: str = _DEFAULT_SEARCH_MODE,
    turn: int = 1,
    max_tool_turns: int = 12,
    max_queries_per_turn: int = 4,
    oracle_context: str = "",
) -> str:
    normalized_search_mode = _normalize_search_mode(search_mode)
    parts = [f"## Question\n{item['question']}"]
    if oracle_context.strip():
        parts.append(f"## Oracle Parsed Pages\n{oracle_context.strip()}")
    if normalized_search_mode == _DEFAULT_SEARCH_MODE:
        file_block = "\n".join(f"- {path}" for path in (candidate_files or [])[:20]) or "- none resolved"
        if corpus_note.strip():
            parts.append(f"## Document Corpus\n{corpus_note.strip()}")
        parts.append(f"## Candidate Files\n{file_block}")
    if item.get("source_docs"):
        parts.append("## Source Hints\n" + "\n".join(f"- {hint}" for hint in item["source_docs"]))
    if normalized_search_mode != _DEFAULT_SEARCH_MODE and item.get("source_files"):
        parts.append("## File Hints\n" + "\n".join(f"- {hint}" for hint in item["source_files"]))
    if diagnostic_mode and diagnostic_instruction.strip():
        parts.append(f"## Training Readout\n{diagnostic_instruction.strip()}")
    if normalized_search_mode == _CUSTOM_SEARCH_MODE:
        parts.append(
            _build_round_instruction(
                turn=turn,
                max_tool_turns=max_tool_turns,
                max_queries_per_turn=max_queries_per_turn,
            )
        )
        parts.append(
            "## Output Format\n"
            "If you need more evidence, return only `<search_queries>[...]</search_queries>`.\n"
            "If you are ready to answer, return only `<answer>...</answer>`."
        )
        parts.append(
            "Use only the provided oracle parsed pages and controller-provided custom search evidence. "
            "Do not rely on any built-in web search capability."
        )
    elif normalized_search_mode == _AZURE_SEARCH_MODE:
        parts.append("Use the model's built-in web search tool when needed. Return the final answer inside <answer>...</answer>.")
    return "\n\n".join(parts)
def _extract_answer(text: str) -> str:
    match = _FINAL_RE.search(text)
    if match:
        return match.group(1).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text.strip()
def _extract_search_queries(text: str) -> list[str]:
    match = _SEARCH_RE.search(text or "")
    if not match:
        return []
    raw = match.group(1).strip()
    if not raw:
        return []
    parsed_queries: list[str] = []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        for key in ("queries", "search_queries", "query"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                parsed_queries = [value.strip()]
                break
            if isinstance(value, list):
                parsed_queries = [str(item).strip() for item in value if str(item).strip()]
                break
    elif isinstance(parsed, list):
        parsed_queries = [str(item).strip() for item in parsed if str(item).strip()]
    elif isinstance(parsed, str) and parsed.strip():
        parsed_queries = [parsed.strip()]
    if not parsed_queries:
        raw_lines = [line.strip(" -*\t\r\n\"'") for line in raw.splitlines()]
        parsed_queries = [line for line in raw_lines if line]
    if len(parsed_queries) <= 1 and parsed_queries:
        multi = [part.strip(" \"'") for part in re.split(r"[;,]", parsed_queries[0]) if part.strip(" \"'")]
        if len(multi) > 1:
            parsed_queries = multi
    deduped: list[str] = []
    seen: set[str] = set()
    for query in parsed_queries:
        normalized = query.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped
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
    oracle_context: str = "",
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
        oracle_context=oracle_context,
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
        "Use the `skillopt-target` skill available in this workspace.\n"
        "Read `task.md`, inspect or search the full OfficeQA corpus under `docs/`, and answer the question.\n"
        "Treat candidate files in `task.md` as hints, not an access limit.\n"
        "Return the final answer inside <answer>...</answer>."
    )
    final_message, raw = run_target_exec(
        work_dir=work_dir,
        prompt=prompt,
        model=model,
        timeout=timeout,
        data_dirs=docs_roots,
    )
    return final_message or raw, raw, skill_md, task_text
def _execute_custom_search_round(
    queries: list[str],
    *,
    api_url: str,
    auth_env: str,
    provider: str,
    max_num_results: int,
    timeout: int,
) -> str:
    blocks = []
    for index, query in enumerate(queries, start=1):
        try:
            result = custom_search(
                query,
                api_url=api_url,
                auth_env=auth_env,
                provider=provider,
                max_num_results=max_num_results,
                timeout=timeout,
            )
        except Exception as search_error:  # noqa: BLE001
            result = f"Query: {query}\n\n[search error: {search_error}]"
        blocks.append(f"## Query {index}\n{result}")
    return "\n\n".join(blocks)
def _run_custom_search_process(
    item: dict,
    skill_content: str,
    *,
    max_tool_turns: int,
    max_completion_tokens: int,
    max_queries_per_turn: int,
    diagnostic_mode: bool,
    diagnostic_instruction: str,
    search_api_url: str,
    search_auth_env: str,
    search_provider: str,
    search_max_num_results: int,
    search_timeout_seconds: int,
    oracle_context: str = "",
) -> tuple[str, str, str, str, list[dict], str, dict]:
    if not str(search_api_url or "").strip():
        raise ValueError("custom_search mode requires a non-empty search_api_url")
    if not os.environ.get(search_auth_env, "").strip():
        raise ValueError(f"custom_search mode requires auth token env var {search_auth_env}")
    if get_target_backend() not in {"openai_chat", "qwen_chat"}:
        raise ValueError("custom_search mode is only supported with target_backend='openai_chat' or 'qwen_chat'")
    system = _build_system(
        skill_content,
        search_mode=_CUSTOM_SEARCH_MODE,
        max_tool_turns=max_tool_turns,
        max_queries_per_turn=max_queries_per_turn,
    )
    initial_user = _build_user(
        item,
        diagnostic_mode=diagnostic_mode,
        diagnostic_instruction=diagnostic_instruction,
        search_mode=_CUSTOM_SEARCH_MODE,
        turn=1,
        max_tool_turns=max_tool_turns,
        max_queries_per_turn=max_queries_per_turn,
        oracle_context=oracle_context,
    )
    latest_user = initial_user
    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": initial_user},
    ]
    conversation: list[dict] = [{"role": "user", "content": initial_user}]
    final_response = ""
    final_answer = ""
    fail_reason = ""
    last_response_metadata: dict = {}
    for turn in range(1, max_tool_turns + 1):
        message, _ = chat_target_messages(
            messages=messages,
            max_completion_tokens=max_completion_tokens,
            retries=5,
            stage="rollout",
            return_message=True,
        )
        response = message.content or ""
        final_response = response
        last_response_metadata = _message_debug_metadata(message)
        messages.append({"role": "assistant", "content": response})
        message_event = {"type": "message", "turn": turn, "content": response}
        if last_response_metadata:
            message_event["response_metadata"] = last_response_metadata
        conversation.append(message_event)
        if "<answer>" in response.lower():
            final_answer = _extract_answer(response)
            return system, latest_user, final_response, final_answer, conversation, "", last_response_metadata
        if turn == max_tool_turns:
            fail_reason = f"Final round ({max_tool_turns}) ended without <answer>...</answer>"
            break
        queries = _extract_search_queries(response)[:max_queries_per_turn]
        if not queries:
            fail_reason = "Model neither produced search queries nor a final answer"
            break
        results_text = _execute_custom_search_round(
            queries,
            api_url=search_api_url,
            auth_env=search_auth_env,
            provider=search_provider,
            max_num_results=search_max_num_results,
            timeout=search_timeout_seconds,
        )
        conversation.append({"type": "tool_call", "turn": turn, "cmd": f"custom_search({queries!r})", "obs": results_text})
        latest_user = (
            f"## Search Results Round {turn}\n{results_text}\n\n"
            + _build_round_instruction(
                turn=turn + 1,
                max_tool_turns=max_tool_turns,
                max_queries_per_turn=max_queries_per_turn,
            )
            + "\n\nFollow the round policy above exactly."
        )
        messages.append({"role": "user", "content": latest_user})
        conversation.append({"role": "user", "turn": turn + 1, "content": latest_user})
    return system, latest_user, final_response, final_answer, conversation, fail_reason, last_response_metadata
def _run_azure_search_process(
    item: dict,
    skill_content: str,
    *,
    max_completion_tokens: int,
    diagnostic_mode: bool,
    diagnostic_instruction: str,
) -> tuple[str, str, str, str, list[dict], str, dict]:
    if get_target_backend() != "openai_chat":
        raise ValueError("azure_search mode is only supported with target_backend='openai_chat'")
    system = _build_system(skill_content, search_mode=_AZURE_SEARCH_MODE)
    user = _build_user(
        item,
        diagnostic_mode=diagnostic_mode,
        diagnostic_instruction=diagnostic_instruction,
        search_mode=_AZURE_SEARCH_MODE,
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    conversation: list[dict] = [{"role": "user", "content": user}]
    message, _ = chat_target_messages(
        messages=messages,
        max_completion_tokens=max_completion_tokens,
        retries=5,
        stage="rollout",
        return_message=True,
        tools=[{"type": "web_search"}],
    )
    response = message.content or ""
    last_response_metadata = _message_debug_metadata(message)
    message_event = {"type": "message", "content": response}
    if last_response_metadata:
        message_event["response_metadata"] = last_response_metadata
    conversation.append(message_event)
    if "<answer>" in response.lower():
        return system, user, response, _extract_answer(response), conversation, "", last_response_metadata
    return system, user, response, "", conversation, "Model did not produce a final answer", last_response_metadata
def _run_offline_no_tools_process(
    item: dict,
    skill_content: str,
    *,
    max_completion_tokens: int,
    diagnostic_mode: bool,
    diagnostic_instruction: str,
    candidate_files: list[str],
    oracle_context: str = "",
) -> tuple[str, str, str, str, list[dict], str, dict]:
    system = _build_system(skill_content, search_mode=_DEFAULT_SEARCH_MODE, use_local_tools=False)
    user = _build_user(
        item,
        candidate_files,
        diagnostic_mode=diagnostic_mode,
        diagnostic_instruction=diagnostic_instruction,
        search_mode=_DEFAULT_SEARCH_MODE,
        oracle_context=oracle_context,
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    conversation: list[dict] = [{"role": "user", "content": user}]
    message, _ = chat_target_messages(
        messages=messages,
        max_completion_tokens=max_completion_tokens,
        retries=5,
        stage="rollout",
        return_message=True,
    )
    response = message.content or ""
    last_response_metadata = _message_debug_metadata(message)
    message_event = {"type": "message", "content": response}
    if last_response_metadata:
        message_event["response_metadata"] = last_response_metadata
    conversation.append(message_event)
    if "<answer>" in response.lower():
        return system, user, response, _extract_answer(response), conversation, "", last_response_metadata
    return system, user, response, "", conversation, "Model did not produce a final answer", last_response_metadata
def process_one(
    item: dict,
    out_root: str,
    skill_content: str,
    *,
    max_tool_turns: int = 12,
    max_completion_tokens: int = 16384,
    search_mode: str = _DEFAULT_SEARCH_MODE,
    max_queries_per_turn: int = 4,
    search_api_url: str = "",
    search_auth_env: str = "OFFICEQA_CUSTOM_SEARCH_AUTH",
    search_provider: str = "duckduckgo",
    search_max_num_results: int = 4,
    search_timeout_seconds: int = 20,
    use_local_tools: bool = True,
    data_dirs: list[str] | str | None = None,
    diagnostic_mode: bool = False,
    diagnostic_instruction: str = "",
) -> dict:
    item_id = str(item["id"])
    pred_dir = os.path.join(out_root, "predictions", item_id)
    os.makedirs(pred_dir, exist_ok=True)
    normalized_search_mode = _normalize_search_mode(search_mode)
    docs_roots: list[str] = []
    candidate_files: list[str] = []
    oracle_context = ""
    if normalized_search_mode == _DEFAULT_SEARCH_MODE:
        docs_roots = resolve_docs_roots(data_dirs)
        candidate_files = resolve_candidate_files(item.get("source_files", []), docs_roots)
        oracle_context = build_oracle_parsed_pages_context(
            item.get("source_files", []),
            item.get("source_docs", []),
            docs_roots,
            evidence_note=(
                "Treat it as primary document evidence and combine it with local document tool evidence when useful."
                if use_local_tools
                else "Treat it as primary document evidence for answering the question."
            ),
        )
    elif normalized_search_mode == _CUSTOM_SEARCH_MODE:
        docs_roots = resolve_docs_roots(data_dirs)
        if item.get("source_files"):
            candidate_files = resolve_candidate_files(item.get("source_files", []), docs_roots)
        oracle_context = build_oracle_parsed_pages_context(
            item.get("source_files", []),
            item.get("source_docs", []),
            docs_roots,
        )
    system = _build_system(
        skill_content,
        search_mode=normalized_search_mode,
        use_local_tools=use_local_tools,
        max_tool_turns=max_tool_turns,
        max_queries_per_turn=max_queries_per_turn,
    )
    user = _build_user(
        item,
        candidate_files if normalized_search_mode == _DEFAULT_SEARCH_MODE else None,
        diagnostic_mode=diagnostic_mode,
        diagnostic_instruction=diagnostic_instruction,
        search_mode=normalized_search_mode,
        max_tool_turns=max_tool_turns,
        max_queries_per_turn=max_queries_per_turn,
        oracle_context=oracle_context,
    )
    conversation: list[dict] = [{"role": "user", "content": user}]
    final_response = ""
    final_answer = ""
    fail_reason = ""
    last_response_metadata: dict = {}
    allowed_files = [os.path.basename(path) for path in candidate_files]
    try:
        if normalized_search_mode == _CUSTOM_SEARCH_MODE:
            system, user, final_response, final_answer, conversation, fail_reason, last_response_metadata = _run_custom_search_process(
                item,
                skill_content,
                max_tool_turns=max_tool_turns,
                max_completion_tokens=max_completion_tokens,
                max_queries_per_turn=max_queries_per_turn,
                diagnostic_mode=diagnostic_mode,
                diagnostic_instruction=diagnostic_instruction,
                search_api_url=search_api_url,
                search_auth_env=search_auth_env,
                search_provider=search_provider,
                search_max_num_results=search_max_num_results,
                search_timeout_seconds=search_timeout_seconds,
                oracle_context=oracle_context,
            )
        elif normalized_search_mode == _AZURE_SEARCH_MODE:
            system, user, final_response, final_answer, conversation, fail_reason, last_response_metadata = _run_azure_search_process(
                item,
                skill_content,
                max_completion_tokens=max_completion_tokens,
                diagnostic_mode=diagnostic_mode,
                diagnostic_instruction=diagnostic_instruction,
            )
        elif not use_local_tools:
            system, user, final_response, final_answer, conversation, fail_reason, last_response_metadata = _run_offline_no_tools_process(
                item,
                skill_content,
                max_completion_tokens=max_completion_tokens,
                diagnostic_mode=diagnostic_mode,
                diagnostic_instruction=diagnostic_instruction,
                candidate_files=candidate_files,
                oracle_context=oracle_context,
            )
        elif is_target_exec_backend():
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
                    model=_llm.TARGET_DEPLOYMENT,
                    timeout=180,
                    diagnostic_mode=diagnostic_mode if turn == 1 else False,
                    diagnostic_instruction=diagnostic_instruction if turn == 1 else "",
                    previous_response=response if turn > 1 else "",
                    oracle_context=oracle_context,
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
            messages: list[dict] = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
            for turn in range(1, max_tool_turns + 1):
                message, _ = chat_target_messages(
                    messages=messages,
                    max_completion_tokens=max_completion_tokens,
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
    with open(os.path.join(pred_dir, "target_system_prompt.txt"), "w", encoding="utf-8") as f:
        f.write(system)
    with open(os.path.join(pred_dir, "target_user_prompt.txt"), "w", encoding="utf-8") as f:
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
        "oracle_parsed_pages_included": bool(oracle_context),
        "oracle_parsed_pages_chars": len(oracle_context),
        "use_local_tools": bool(use_local_tools),
        "hard": int(eval_result["em"]),
        "soft": eval_result["f1"],
        "fail_reason": fail_reason or ("" if eval_result["em"] else f"predicted '{eval_result['predicted_answer']}' but expected '{item.get('ground_truth', '')}'"),
        "agent_ok": not fail_reason,
        "n_turns": len(conversation),
        "last_finish_reason": last_response_metadata.get("finish_reason", ""),
        "target_system_prompt": system,
        "target_user_prompt": user,
    }
    return result
def run_batch(
    items: list[dict],
    out_root: str,
    skill_content: str,
    *,
    workers: int = 8,
    max_tool_turns: int = 12,
    max_completion_tokens: int = 16384,
    search_mode: str = _DEFAULT_SEARCH_MODE,
    max_queries_per_turn: int = 4,
    search_api_url: str = "",
    search_auth_env: str = "OFFICEQA_CUSTOM_SEARCH_AUTH",
    search_provider: str = "duckduckgo",
    search_max_num_results: int = 4,
    search_timeout_seconds: int = 20,
    use_local_tools: bool = True,
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
    total = len(existing) + len(pending)
    completed = len(existing)
    correct_count = sum(1 for r in existing if r.get("hard", 0))
    if existing:
        print(f"    [rollout] resuming: {completed}/{total} already done", flush=True)

    results = list(existing)
    with open(results_path, "a", encoding="utf-8") as outf, ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(
                process_one,
                item,
                out_root,
                skill_content,
                max_tool_turns=max_tool_turns,
                max_completion_tokens=max_completion_tokens,
                search_mode=search_mode,
                max_queries_per_turn=max_queries_per_turn,
                search_api_url=search_api_url,
                search_auth_env=search_auth_env,
                search_provider=search_provider,
                search_max_num_results=search_max_num_results,
                search_timeout_seconds=search_timeout_seconds,
                use_local_tools=use_local_tools,
                data_dirs=data_dirs,
                diagnostic_mode=diagnostic_mode,
                diagnostic_instruction=diagnostic_instruction,
            ): item
            for item in pending
        }
        for fut in as_completed(futs):
            res = fut.result()
            results.append(res)
            completed += 1
            if res.get("hard", 0):
                correct_count += 1
            acc = correct_count / completed if completed else 0
            print(
                f"    [rollout] {completed}/{total} "
                f"(acc={acc:.3f}) id={res.get('id', '?')} "
                f"hard={res.get('hard', '?')}",
                flush=True,
            )
            outf.write(json.dumps(res, ensure_ascii=False) + "\n")
            outf.flush()
    return results
