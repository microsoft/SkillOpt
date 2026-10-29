"""The engine's own replay sessions are never harvested as user tasks.

``_is_headless_replay`` recognised engine sessions by static prompt markers
that no longer matched the current prompt registry, and the duration fallback
only applies to prompts under 200 characters. With ``projects: "all"``, the
engine's attempt, judge, reflect and miner prompts were mined back as user
tasks. Markers are now derived from the registry itself, so rewording a prompt
cannot silently reopen the gap.
"""
from __future__ import annotations

import json
import subprocess
from unittest import mock

import pytest

from skillopt_sleep import prompts
from skillopt_sleep.backend import ClaudeCliBackend, CodexCliBackend, CopilotCliBackend
from skillopt_sleep.harvest import _is_headless_replay, harvest
from skillopt_sleep.types import SessionDigest, TaskRecord

RENDER_ARGS = {
    "attempt": {"__SKILL__": "Use concise answers.", "__MEMORY__": "(none)",
                "__INTENT__": "Summarize the PR", "__CONTEXT__": ""},
    "judge": {"__RUBRIC__": "Mentions the risk", "__RESPONSE__": "It is fine."},
    "reflect": {"__EDIT_BUDGET__": "3", "__TARGET__": "skill", "__CUR_DOC__": "x",
                "__GUARD__": "", "__CRITERIA__": "", "__PREFS__": "",
                "__FAILURES__": "- wanted: a\n  got: b"},
    "miner": {"__PROJECT__": "/repo", "__PROMPTS__": "  - do x", "__FINAL__": "ok",
              "__FEEDBACK__": "(none)"},
}


def _digest(prompt: str) -> SessionDigest:
    return SessionDigest(
        session_id="s",
        project="/tmp/replay",
        started_at="2026-10-01T00:00:00",
        ended_at="2026-10-01T00:00:09",  # too slow for the duration fallback
        user_prompts=[prompt],
        assistant_finals=["reply"],
        n_user_turns=1,
        n_assistant_turns=1,
    )


@pytest.fixture(autouse=True)
def _no_user_prompt_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("SKILLOPT_SLEEP_PROMPTS_PATH", str(tmp_path / "prompts.json"))


def test_every_registry_prompt_has_render_args() -> None:
    assert set(RENDER_ARGS) == set(prompts.DEFAULTS)


@pytest.mark.parametrize("name", sorted(RENDER_ARGS))
def test_rendered_engine_prompts_are_filtered(name) -> None:
    prompt = prompts.render(name, RENDER_ARGS[name])
    # The session took nine seconds, so only a prompt marker can catch it.
    assert _is_headless_replay(_digest(prompt)) is True


def test_overridden_engine_prompt_is_filtered(tmp_path) -> None:
    override = "Grade this answer for the nightly SkillOpt run.\n__RESPONSE__\n" + "x" * 300
    (tmp_path / "prompts.json").write_text(json.dumps({"judge": override}), encoding="utf-8")
    prompt = prompts.render("judge", {"__RUBRIC__": "r", "__RESPONSE__": "resp"})
    assert prompt.startswith("Grade this answer")
    assert _is_headless_replay(_digest(prompt)) is True


def _captured_tool_attempt_prompt(backend) -> str:
    seen: list[str] = []

    def fake_run(cmd, **kwargs):
        texts = [kwargs.get("input") or ""] + [str(part) for part in cmd]
        seen.extend(text for text in texts if "# Task" in text or "Summarize the PR" in text)
        return subprocess.CompletedProcess(cmd, 0, stdout="answer", stderr="")

    task = TaskRecord(
        id="t", project="/repo", intent="Summarize the PR", reference_kind="rule",
        judge={"kind": "rule", "checks": [{"op": "tool_called", "arg": "search"}]},
    )
    with mock.patch("skillopt_sleep.backend.subprocess.run", side_effect=fake_run):
        try:
            backend.attempt_with_tools(task, "Use concise answers.", "(none)", ["search"])
        except Exception:
            pass
    assert seen, f"{type(backend).__name__} sent no capturable prompt"
    return max(seen, key=len)


@pytest.mark.parametrize(
    "backend",
    [ClaudeCliBackend(claude_path="claude"), CodexCliBackend(), CopilotCliBackend()],
    ids=["claude", "codex", "copilot"],
)
def test_tool_attempt_prompts_are_filtered(backend) -> None:
    prompt = _captured_tool_attempt_prompt(backend)
    assert _is_headless_replay(_digest(prompt)) is True


@pytest.mark.parametrize(
    "prompt",
    [
        "Complete the following task: fix the flaky login test and explain the root cause",
        "Please score how well my essay answers the prompt, and suggest edits " + "x" * 200,
        "You are mining data from logs today, find the failing deploy " + "y" * 200,
        "Treat this as urgent: the release branch is broken " + "z" * 200,
    ],
)
def test_real_user_prompts_are_kept(prompt) -> None:
    assert _is_headless_replay(_digest(prompt)) is False


def test_harvest_with_all_projects_skips_engine_sessions_and_keeps_user_work(tmp_path) -> None:
    sessions = {name: prompts.render(name, args) for name, args in RENDER_ARGS.items()}
    sessions["user"] = "Refactor the billing module so invoices round to cents " + "q" * 200
    for name, prompt in sessions.items():
        folder = tmp_path / f"-tmp-{name}"
        folder.mkdir()
        records = [
            {"type": "user", "cwd": f"/tmp/{name}", "timestamp": "2026-10-01T00:00:00Z",
             "message": {"role": "user", "content": prompt}},
            {"type": "assistant", "timestamp": "2026-10-01T00:00:09Z",
             "message": {"role": "assistant", "content": [{"type": "text", "text": "ok"}]}},
        ]
        (folder / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(record) for record in records), encoding="utf-8"
        )

    harvested = harvest(str(tmp_path), scope="all")
    assert [digest.user_prompts[0][:20] for digest in harvested] == [sessions["user"][:20]]
