"""The engine's own Claude replay sessions are recognized by who wrote them.

The Claude backend stamps ``CLAUDE_REPLAY_SENTINEL`` at the start of every
prompt it sends. The harvester drops a single-turn session only when its prompt
starts with that marker. A session that merely quotes an engine prompt, or the
marker itself, is an ordinary user session and is kept.

Only ``subprocess.run`` (the Claude child process) is faked. Every session goes
through the real ``harvest(..., scope="all")``.
"""
from __future__ import annotations

import json
import re
import subprocess
from unittest import mock

import pytest

from skillopt_sleep import prompts
from skillopt_sleep.backend import ClaudeCliBackend
from skillopt_sleep.harvest import CLAUDE_REPLAY_SENTINEL, harvest
from skillopt_sleep.harvest_codex import CODEX_REPLAY_SENTINEL
from skillopt_sleep.llm_miner import make_llm_miner
from skillopt_sleep.types import ReplayResult, SessionDigest, TaskRecord

_OPERATIONS = ("attempt", "judge", "reflect", "tool_attempt", "custom_system", "miner")
_CUSTOM_JUDGE = (
    "Grade this answer for the nightly run.\n"
    "Criteria: __RUBRIC__\nAnswer: __RESPONSE__\n" + "Be strict. " * 30
)


@pytest.fixture(autouse=True)
def _isolated_prompt_registry(tmp_path, monkeypatch):
    # No user override file unless a test writes one, and a test API key so
    # the backend takes its --bare route without probing a real CLI.
    monkeypatch.setenv("SKILLOPT_SLEEP_PROMPTS_PATH", str(tmp_path / "prompts.json"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")


def _write_session(folder, name: str, prompt: str, *, followup: str = "",
                   tool_result: bool = False) -> None:
    """One Claude Code transcript. Nine seconds long, so the short-session
    duration heuristic cannot be what drops it."""
    folder.mkdir(parents=True, exist_ok=True)
    records = [
        {"type": "user", "cwd": f"/work/{name}", "timestamp": "2026-11-01T00:00:00Z",
         "message": {"role": "user", "content": prompt}},
    ]
    if tool_result:
        records += [
            {"type": "assistant", "timestamp": "2026-11-01T00:00:03Z",
             "message": {"role": "assistant", "content": [
                 {"type": "tool_use", "id": "tu_1", "name": "Bash",
                  "input": {"command": "./search q"}}]}},
            {"type": "user", "timestamp": "2026-11-01T00:00:05Z",
             "message": {"role": "user", "content": [
                 {"type": "tool_result", "tool_use_id": "tu_1", "content": "3 notes"}]}},
        ]
    records.append(
        {"type": "assistant", "timestamp": "2026-11-01T00:00:09Z",
         "message": {"role": "assistant", "content": [{"type": "text", "text": "ok"}]}}
    )
    if followup:
        records.append(
            {"type": "user", "timestamp": "2026-11-01T00:01:00Z",
             "message": {"role": "user", "content": followup}}
        )
    (folder / f"{name}.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records), encoding="utf-8"
    )


def _engine_prompts() -> dict:
    """Exactly what the shipped Claude backend hands to the CLI, per operation."""
    sent: list[str] = []

    def capture(cmd, **kwargs):
        prompt = kwargs["input"]
        sent.append(prompt)
        if "You are SkillOpt's optimizer" in prompt:
            out = '[{"op": "add", "content": "Check the answer."}]'
        elif "You are mining" in prompt:
            out = "[]"
        else:
            out = '{"score": 0.5, "reason": "synthetic"}'
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    backend = ClaudeCliBackend(claude_path="claude")
    task = TaskRecord(
        id="t", project="/repo", intent="Answer the question", reference_kind="rubric"
    )
    with mock.patch("skillopt_sleep.backend.subprocess.run", side_effect=capture):
        backend.attempt(task, "skill", "memory")
        backend.judge(task, "answer")
        backend.reflect(
            [(task, ReplayResult(id="t", response="wrong", fail_reason="incorrect"))],
            [], "skill", "memory", edit_budget=1, evolve_skill=True, evolve_memory=False,
        )
        backend.attempt_with_tools(task, "skill", "memory", ["search"])
        backend.attempt(
            TaskRecord(id="custom", project="/repo", intent="task", system="custom system"),
            "", "",
        )
        make_llm_miner(backend)([
            SessionDigest(
                session_id="s", project="/repo", user_prompts=["do the thing"],
                assistant_finals=["done"],
            )
        ])
    assert len(sent) == len(_OPERATIONS), sent
    return dict(zip(_OPERATIONS, sent))


def test_sentinel_has_the_same_shape_as_the_codex_producer_marker() -> None:
    shape = re.compile(r"\[skillopt-sleep:[a-z]+-engine:v1\]")
    assert shape.fullmatch(CLAUDE_REPLAY_SENTINEL)
    assert shape.fullmatch(CODEX_REPLAY_SENTINEL)
    assert CLAUDE_REPLAY_SENTINEL != CODEX_REPLAY_SENTINEL


@pytest.mark.parametrize("operation", _OPERATIONS)
def test_engine_session_from_the_real_backend_prompt_path_is_dropped(
    tmp_path, operation
) -> None:
    prompt = _engine_prompts()[operation]
    assert prompt.startswith(CLAUDE_REPLAY_SENTINEL + "\n\n")
    _write_session(
        tmp_path / "-work-engine", "engine", prompt,
        tool_result=(operation == "tool_attempt"),
    )
    _write_session(tmp_path / "-work-user", "user", "Refactor the billing module")
    harvested = harvest(str(tmp_path), scope="all")
    assert [digest.session_id for digest in harvested] == ["user"]


def test_engine_session_recorded_with_crlf_line_breaks_is_dropped(tmp_path) -> None:
    prompt = _engine_prompts()["judge"].replace("\n", "\r\n")
    _write_session(tmp_path / "-work-engine", "engine", prompt)
    assert harvest(str(tmp_path), scope="all") == []


def test_custom_template_engine_session_is_dropped(tmp_path) -> None:
    """An overridden template shares no wording with the defaults; the marker
    is added by the backend after rendering, so the session is still dropped."""
    (tmp_path / "prompts.json").write_text(
        json.dumps({"judge": _CUSTOM_JUDGE}), encoding="utf-8"
    )
    prompt = _engine_prompts()["judge"]
    assert prompt.startswith(CLAUDE_REPLAY_SENTINEL + "\n\nGrade this answer")
    transcripts = tmp_path / "transcripts"
    _write_session(transcripts / "-work-engine", "engine", prompt)
    assert harvest(str(transcripts), scope="all") == []


@pytest.mark.parametrize("template", ["default", "custom"])
def test_user_session_quoting_the_grader_prompt_is_kept(tmp_path, template) -> None:
    """The look-alike: a person pastes the full current grader prompt and asks
    about it. Matching prompt text drops this session; provenance keeps it."""
    if template == "custom":
        (tmp_path / "prompts.json").write_text(
            json.dumps({"judge": _CUSTOM_JUDGE}), encoding="utf-8"
        )
    grader = prompts.render(
        "judge", {"__RUBRIC__": "Mentions the risk", "__RESPONSE__": "It is fine."}
    )
    for name, prompt in {
        "quoted-after-question": "Why does this grader prompt score my answer so low?\n\n" + grader,
        "quoted-verbatim": grader,
    }.items():
        transcripts = tmp_path / name
        _write_session(transcripts / "-work-user", "user", prompt)
        harvested = harvest(str(transcripts), scope="all")
        assert [digest.session_id for digest in harvested] == ["user"], name
        assert harvested[0].user_prompts == [prompt.strip()]


def test_user_session_quoting_the_sentinel_is_kept(tmp_path) -> None:
    engine_prompt = _engine_prompts()["judge"]
    quotes = {
        "marker-mid-text": f"What does {CLAUDE_REPLAY_SENTINEL} at the top of a transcript mean?",
        "stamped-prompt-after-intro": "I found this in my transcripts, what is it?\n\n" + engine_prompt,
        "marker-without-the-blank-line": CLAUDE_REPLAY_SENTINEL + " is this yours?",
    }
    for name, prompt in quotes.items():
        transcripts = tmp_path / name
        _write_session(transcripts / "-work-user", "user", prompt)
        harvested = harvest(str(transcripts), scope="all")
        assert [digest.session_id for digest in harvested] == ["user"], name


def test_single_turn_session_that_begins_with_the_marker_is_engine_produced(
    tmp_path,
) -> None:
    """The boundary of the rule, stated as a test: position is the only signal.
    A one-message session that begins with the marker line is indistinguishable
    from an engine session and is dropped, whoever typed it."""
    pasted = _engine_prompts()["judge"]
    _write_session(tmp_path / "-work-pasted", "pasted", pasted)
    assert harvest(str(tmp_path), scope="all") == []


def test_multi_turn_sessions_are_kept_as_before(tmp_path) -> None:
    """Multi-turn sessions are interactive by definition and were always kept.
    That holds even when the first prompt starts with the marker, for example a
    person resuming an engine session and typing into it."""
    _write_session(
        tmp_path / "-work-resumed", "resumed", _engine_prompts()["attempt"],
        followup="now explain that answer",
    )
    _write_session(
        tmp_path / "-work-chat", "chat", "Refactor the billing module",
        followup="thanks, also add a test",
    )
    harvested = harvest(str(tmp_path), scope="all")
    assert sorted(digest.session_id for digest in harvested) == ["chat", "resumed"]
    assert {digest.n_user_turns for digest in harvested} == {2}


def test_static_markers_for_older_transcripts_are_unchanged(tmp_path) -> None:
    """Transcripts written before the marker existed are still recognized by
    the fixed markers, exactly as on the previous release."""
    _write_session(
        tmp_path / "-work-old", "old",
        "You are a strict grader. Score the response 0.0-1.0.\n" + "x" * 300,
    )
    assert harvest(str(tmp_path), scope="all") == []


def test_stamping_does_not_change_the_attempt_cache_or_token_accounting() -> None:
    """The marker is added to the text handed to the CLI only. The cache key
    and the cost estimate are computed from the unstamped prompt, so a repeated
    attempt is still one call and the reported tokens are what they were."""
    sent: list[str] = []

    def capture(cmd, **kwargs):
        sent.append(kwargs["input"])
        return subprocess.CompletedProcess(cmd, 0, stdout="answer", stderr="")

    backend = ClaudeCliBackend(claude_path="claude")
    task = TaskRecord(id="t", project="/repo", intent="Answer the question")
    with mock.patch("skillopt_sleep.backend.subprocess.run", side_effect=capture):
        first = backend.attempt(task, "skill", "memory")
        second = backend.attempt(task, "skill", "memory")
    assert (first, second) == ("answer", "answer")
    assert len(sent) == 1
    unstamped = sent[0][len(CLAUDE_REPLAY_SENTINEL) + 2:]
    assert backend.tokens_used() == len(unstamped) // 4 + len("answer") // 4
