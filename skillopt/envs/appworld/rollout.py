"""AppWorld rollout for SkillOpt.

The episode itself follows AppWorld's react_code protocol, implemented in ``episode.py``:
AppWorld's official react_code few-shot prompt, one ``execute_code`` tool, 50 executed calls,
reward = fraction of the task's no_op_fail tests passed, TGC = all tests pass.

What SkillOpt adds around it:

* the target model call goes through ``skillopt.model.chat_target_messages`` (whatever
  target backend is configured, with native tool calls);
* the skill is appended to the system prompt under ``## Skill Knowledge`` (same wording as
  the ALFWorld env). An empty skill sends the protocol's prompt unchanged;
* an optional per-episode generated-token wall, ``episode_token_wall``; 0 (default) turns it
  off so only the 50-call budget and the per-call ``max_completion_tokens`` limit an episode
  (AppWorld's own protocol has no token wall).

Scores: ``hard`` = official TGC (0/1), ``soft`` = the no_op_fail pass fraction. A task whose
episode crashed on infrastructure is kept and counted as 0, with ``agent_ok=False`` and
the error in ``fail_reason``.
"""
from __future__ import annotations

import atexit
import concurrent.futures
import hashlib
import json
import os
import queue
import socket
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from skillopt.envs.appworld.episode import (
    REACT_PROMPT_PATH,
    AppWorldServer,
    ChatModel,
    ContextOverflow,
    load_app_descriptions,
    run_episode,
)
from skillopt.envs.task_output import confined_task_output_dir
from skillopt.model import chat_target_messages

UNLIMITED_TOKEN_WALL = 10**9


def build_skill_suffix(skill_content: str) -> str:
    """Skill section appended to the system prompt (ALFWorld env wording)."""
    if not skill_content or not skill_content.strip():
        return ""
    return (
        "\n\n## Skill Knowledge\n"
        "Below is a skill document with learned strategies. "
        "Use these guidelines to inform your decisions:\n\n"
        f"{skill_content}\n"
    )


def _is_context_overflow(error: BaseException) -> bool:
    text = str(error).lower()
    return any(key in text for key in (
        "maximum context length", "context length", "context_length_exceeded",
        "prompt is too long", "input is too long", "too many tokens"))


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _tool_calls_as_dicts(message: Any) -> list[dict]:
    calls = []
    for index, call in enumerate(_field(message, "tool_calls") or [], start=1):
        function = _field(call, "function") or {}
        arguments = _field(function, "arguments") or "{}"
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments)
        calls.append({
            "id": str(_field(call, "id") or f"call_{index}"),
            "type": "function",
            "function": {"name": str(_field(function, "name") or ""), "arguments": arguments},
        })
    return calls


class SkillOptChat(ChatModel):
    """One target call through SkillOpt's configured target backend, skill in the system prompt."""

    def __init__(self, skill_suffix: str, max_completion_tokens: int, timeout: int | None) -> None:
        self.skill_suffix = skill_suffix
        self.max_completion_tokens = int(max_completion_tokens)
        self.timeout = timeout

    def complete(self, messages: list[dict], tools: list[dict], max_tokens: int) -> tuple[dict, dict]:
        sent = [dict(m) for m in messages]
        if self.skill_suffix and sent and sent[0].get("role") == "system":
            sent[0]["content"] = str(sent[0].get("content") or "") + self.skill_suffix
        limit = max(1, min(int(max_tokens), self.max_completion_tokens))
        try:
            message, usage = chat_target_messages(
                messages=sent,
                max_completion_tokens=limit,
                retries=5,
                stage="rollout",
                tools=tools,
                return_message=True,
                timeout=self.timeout,
            )
        except Exception as error:  # noqa: BLE001
            if _is_context_overflow(error):
                raise ContextOverflow(str(error)[:1000]) from error
            raise
        completion_tokens = int((usage or {}).get("completion_tokens") or 0)
        metadata = _field(message, "metadata") or {}
        finish = metadata.get("finish_reason") if isinstance(metadata, dict) else None
        if finish is None and completion_tokens >= limit:
            finish = "length"
        choice_message = ((metadata.get("choice0") or {}).get("message") or {}) if isinstance(metadata, dict) else {}
        reasoning = choice_message.get("reasoning") or choice_message.get("reasoning_content") or ""
        reply = {"content": _field(message, "content") or "", "tool_calls": _tool_calls_as_dicts(message)}
        info = {
            "completion_tokens": completion_tokens,
            "prompt_tokens": int((usage or {}).get("prompt_tokens") or 0),
            "finish_reason": finish,
            "max_tokens": limit,
            "reasoning": reasoning,
        }
        return reply, info


class ServerDead(RuntimeError):
    """The AppWorld server stopped answering; the episode is an infrastructure failure."""


class GuardedServer(AppWorldServer):
    """AppWorldServer that fails fast once a request has timed out.

    Policy code can wedge a server permanently (e.g. by rebinding ``os``); every later
    /execute would then cost the full client timeout. After the first timeout every request
    raises at once, so the episode ends quickly and the server is replaced.
    """

    def __init__(self, *args: Any, request_timeout: float = 300.0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.request_timeout = request_timeout
        self.dead = False

    def post(self, route: str, payload: dict, timeout: float | None = None) -> Any:
        if self.dead:
            raise ServerDead(f"appworld server on port {self.port} is unresponsive")
        try:
            return super().post(route, payload, timeout=timeout or self.request_timeout)
        except (socket.timeout, TimeoutError) as error:
            self.dead = True
            raise ServerDead(f"appworld server on port {self.port} timed out on {route}") from error
        except OSError as error:
            if "timed out" in str(error):
                self.dead = True
                raise ServerDead(f"appworld server on port {self.port} timed out on {route}") from error
            raise


_PORT_LOCK = threading.Lock()
_PORTS_IN_USE: set[int] = set()
_LIVE_SERVERS: set[GuardedServer] = set()


def _stop_all_live_servers() -> None:
    for server in list(_LIVE_SERVERS):
        try:
            server.stop()
        except Exception:  # noqa: BLE001
            pass


atexit.register(_stop_all_live_servers)


def _port_is_free(port: int) -> bool:
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


class ServerPool:
    """``size`` AppWorld servers on free ports of [port_base, port_base + port_span)."""

    def __init__(self, size: int, *, root: Path, venv: Path, log_dir: Path, port_base: int,
                 port_span: int, request_timeout: float) -> None:
        self.root, self.venv, self.log_dir = root, venv, log_dir
        self.port_base, self.port_span = port_base, port_span
        self.request_timeout = request_timeout
        self.idle: queue.Queue[GuardedServer] = queue.Queue()
        self.ports: list[int] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=size) as pool:
            servers = list(pool.map(lambda _: self._new_server(), range(size)))
        for server in servers:
            self.idle.put(server)

    def _claim_port(self) -> int:
        with _PORT_LOCK:
            for port in range(self.port_base, self.port_base + self.port_span):
                if port not in _PORTS_IN_USE and _port_is_free(port):
                    _PORTS_IN_USE.add(port)
                    return port
        raise RuntimeError(
            f"no free AppWorld port in [{self.port_base}, {self.port_base + self.port_span}); "
            "stale servers from an earlier run, or two processes sharing one port range")

    def _release_port(self, port: int) -> None:
        with _PORT_LOCK:
            _PORTS_IN_USE.discard(port)

    def _new_server(self) -> GuardedServer:
        port = self._claim_port()
        server = GuardedServer(port, self.root, self.venv, self.log_dir / f"{port}.log",
                               request_timeout=self.request_timeout)
        try:
            server.start()
        except Exception:
            server.stop()
            self._release_port(port)
            raise
        _LIVE_SERVERS.add(server)
        return server

    def _retire(self, server: GuardedServer) -> None:
        try:
            server.stop()
        finally:
            _LIVE_SERVERS.discard(server)
            self._release_port(server.port)

    def acquire(self) -> GuardedServer:
        return self.idle.get()

    def release(self, server: GuardedServer, healthy: bool) -> None:
        if healthy and not server.dead and server.proc is not None and server.proc.poll() is None:
            self.idle.put(server)
            return
        self._retire(server)
        self.idle.put(self._new_server())

    def close(self) -> None:
        while True:
            try:
                server = self.idle.get_nowait()
            except queue.Empty:
                break
            self._retire(server)


def _conversation_for_reflection(item: dict, episode: Any, result: dict,
                                 fail_reason_style: str = "checks") -> list[dict]:
    """SkillOpt reflection format: the task, assistant turns, tool_call/obs records, outcome."""
    conversation: list[dict] = [{"role": "user", "content": f"Task: {item.get('instruction', '')}"}]
    pending_results = [m for m in episode.messages[2:] if m.get("role") == "tool"]
    tool_iter = iter(pending_results)
    for message in episode.messages[2:]:
        if message.get("role") != "assistant":
            continue
        content = str(message.get("content") or "").strip()
        if content:
            conversation.append({"role": "assistant", "content": content})
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            try:
                args = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            code = args.get("code") if isinstance(args, dict) and "code" in args else function.get("arguments")
            obs = next(tool_iter, {}).get("content", "")
            conversation.append({"type": "tool_call", "cmd": str(code or ""), "obs": str(obs or "")})
    if fail_reason_style == "legacy":
        outcome = (f"Outcome: stop_reason={result['stop_reason']}, task_goal_completion={bool(result['hard'])}, "
                   f"score={result['soft']:.3f}, tool_calls={result['n_turns']}. {result['fail_reason']}")
    else:
        outcome = (f"Outcome: task_goal_completion={bool(result['hard'])}, score={result['soft']:.3f}, "
                   f"tool_calls={result['n_turns']}, episode end: {_episode_end(episode)}. {result['fail_reason']}")
    conversation.append({"role": "system", "content": outcome.strip()})
    return conversation


FAIL_REASON_STYLES = ("checks", "legacy")

# How an episode stopped, in words. ``no_tool_call`` is how EVERY episode normally ends (the
# model replies without a tool call after complete_task), so it is not a failure cause by itself.
_STOP_WORDS = {
    "budget_exhausted": "the tool-call budget ran out",
    "truncated": "the generation token limit was hit",
    "context_overflow": "the conversation exceeded the model context",
}


def _episode_end(episode: Any) -> str:
    stop = episode.stop_reason
    if stop == "no_tool_call":
        if episode.task_completed:
            return "normal (the agent called apis.supervisor.complete_task, then replied without a tool call)"
        return "the agent replied without a tool call before apis.supervisor.complete_task succeeded"
    return f"cut off, {_STOP_WORDS.get(stop, f'stop_reason={stop}')}"


def _fail_reason(episode: Any, style: str = "checks") -> str:
    """Failure text shown to the analysts and the slow update (empty for a passed task).

    ``checks`` (default) leads with AppWorld's failed checks and names the stop reason only when
    it explains the failure. ``legacy`` is the original text, which opened every failure with
    "Episode ended with stop_reason=no_tool_call", the normal ending of every episode; once a skill
    held an "emit no text after complete_task" rule, analysts kept citing it as the cause. Kept to
    reproduce earlier runs.
    """
    if episode.tgc:
        return ""
    if style == "legacy":
        return _fail_reason_legacy(episode)
    failed = (episode.verifier_components or {}).get("failed_requirements") or []
    parts = []
    if failed:
        parts.append("Failed AppWorld checks: " + "; ".join(str(f).rstrip(". ") for f in failed[:8]))
    if not episode.task_completed or episode.stop_reason != "no_tool_call":
        parts.append(f"Episode end: {_episode_end(episode)}")
    elif failed:
        parts.append("complete_task was called and the episode ended normally, "
                     "so the failed checks concern the submitted answer or the final app state")
    if episode.reward is None:
        parts.append(f"grader: {episode.verifier_status}")
    return ". ".join(parts)


def _fail_reason_legacy(episode: Any) -> str:
    parts = [f"Episode ended with stop_reason={episode.stop_reason}"]
    if not episode.task_completed:
        parts.append("apis.supervisor.complete_task was never called successfully")
    failed = (episode.verifier_components or {}).get("failed_requirements") or []
    if failed:
        parts.append("Failed AppWorld checks: " + "; ".join(str(f) for f in failed[:8]))
    if episode.reward is None:
        parts.append(f"grader: {episode.verifier_status}")
    return ". ".join(parts)


def _run_one(item: dict, *, pool: ServerPool, chat: SkillOptChat, experiment: str, template: str,
             max_steps: int, token_wall: int, out_root: Path,
             fail_reason_style: str = "checks", app_descriptions: str | None = None) -> tuple[dict, dict]:
    task_id = str(item["task_id"])
    server = pool.acquire()
    healthy = True
    started = time.monotonic()
    try:
        episode = run_episode(chat, server, task_id, experiment, template=template,
                              max_steps=max_steps, token_wall=token_wall, app_descriptions=app_descriptions)
    except Exception as error:  # noqa: BLE001 - kept as a failed task, annotated
        healthy = not isinstance(error, (ServerDead, OSError))
        result = {
            "id": str(item["id"]),
            "hard": 0,
            "soft": 0.0,
            "n_turns": 0,
            "fail_reason": f"infrastructure failure: {type(error).__name__}: {str(error)[:300]}",
            "agent_ok": False,
            "infra_failure": True,
            "task_type": item.get("task_type", "appworld"),
            "task_description": item.get("instruction", ""),
            "source_split": item.get("source_split", ""),
            "stop_reason": "infra_failure",
            "elapsed_s": round(time.monotonic() - started, 2),
        }
        record = {"task_id": task_id, "error": traceback.format_exc()[-4000:]}
        return result, record
    finally:
        pool.release(server, healthy)

    result = {
        "id": str(item["id"]),
        "hard": 1 if episode.tgc else 0,
        "soft": float(episode.reward or 0.0),
        "n_turns": len(episode.dispatches),
        "fail_reason": _fail_reason(episode, fail_reason_style),
        "agent_ok": True,
        "infra_failure": False,
        "task_type": item.get("task_type", "appworld"),
        "task_description": item.get("instruction", ""),
        "source_split": item.get("source_split", ""),
        "stop_reason": episode.stop_reason,
        "task_completed": episode.task_completed,
        "reward": episode.reward,
        "verifier_status": episode.verifier_status,
        "completion_tokens": episode.completion_tokens,
        "elapsed_s": episode.elapsed_s,
    }
    task_dir = Path(confined_task_output_dir(str(out_root), str(item["id"]), map_unsafe=True))
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "conversation.json").write_text(
        json.dumps(_conversation_for_reflection(item, episode, result, fail_reason_style),
                   ensure_ascii=False, indent=2),
        encoding="utf-8")
    record = {
        "task_id": task_id,
        "experiment": experiment,
        "skill_sha256": hashlib.sha256(chat.skill_suffix.encode()).hexdigest(),
        "messages": episode.messages,
        "turns": episode.turns,
        "dispatches": episode.dispatches,
        "stop_reason": episode.stop_reason,
        "completion_tokens": episode.completion_tokens,
        "task_completed": episode.task_completed,
        "reward": episode.reward,
        "tgc": episode.tgc,
        "verifier_status": episode.verifier_status,
        "verifier_components": episode.verifier_components,
        "tracker": episode.tracker,
        "elapsed_s": episode.elapsed_s,
    }
    return result, record


def run_batch(
    *,
    items: list[dict],
    skill_content: str,
    out_root: str,
    appworld_root: str,
    appworld_venv: str,
    workers: int,
    port_base: int,
    port_span: int,
    max_steps: int,
    max_completion_tokens: int,
    episode_token_wall: int,
    request_timeout: float,
    call_timeout: int | None,
    fail_reason_style: str = "checks",
    fill_app_descriptions: bool = True,
) -> list[dict]:
    if fail_reason_style not in FAIL_REASON_STYLES:
        raise ValueError(f"fail_reason_style must be one of {FAIL_REASON_STYLES}, got {fail_reason_style!r}")
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    if not items:
        return []
    root = Path(os.path.expanduser(appworld_root)).resolve()
    venv = Path(os.path.expanduser(appworld_venv))
    if not (venv / "bin" / "appworld").exists():
        raise FileNotFoundError(f"AppWorld CLI missing: {venv / 'bin' / 'appworld'}")
    if not (root / "data" / "tasks").is_dir():
        raise FileNotFoundError(f"AppWorld data missing under {root}")
    # one AppWorld experiment per rollout directory, so repeated tasks never share state
    experiment = "skillopt-" + hashlib.sha1(str(out.resolve()).encode()).hexdigest()[:16]
    template = REACT_PROMPT_PATH.read_text()
    app_descriptions = load_app_descriptions(venv) if fill_app_descriptions else None
    chat = SkillOptChat(build_skill_suffix(skill_content), max_completion_tokens, call_timeout)
    token_wall = int(episode_token_wall) if int(episode_token_wall or 0) > 0 else UNLIMITED_TOKEN_WALL
    size = max(1, min(int(workers), len(items)))
    print(f"    [appworld rollout] tasks={len(items)} servers={size} experiment={experiment}", flush=True)

    pool = ServerPool(size, root=root, venv=venv, log_dir=out / "appworld_servers",
                      port_base=port_base, port_span=port_span, request_timeout=request_timeout)
    results: dict[str, dict] = {}
    lock = threading.Lock()
    done = 0
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=size) as executor, \
                (out / "episodes.jsonl").open("a", encoding="utf-8") as episodes:
            futures = {
                executor.submit(_run_one, item, pool=pool, chat=chat, experiment=experiment,
                                template=template, max_steps=max_steps, token_wall=token_wall,
                                out_root=out, fail_reason_style=fail_reason_style,
                                app_descriptions=app_descriptions): item
                for item in items
            }
            for future in concurrent.futures.as_completed(futures):
                result, record = future.result()
                with lock:
                    results[result["id"]] = result
                    episodes.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                    episodes.flush()
                    done += 1
                    tgc = sum(r["hard"] for r in results.values())
                    print(f"    [appworld rollout] {done}/{len(items)} {result['id']} "
                          f"tgc={result['hard']} score={result['soft']:.2f} stop={result['stop_reason']} "
                          f"(running TGC {tgc}/{done})", flush=True)
    finally:
        pool.close()
    return [results[str(item["id"])] for item in items]
