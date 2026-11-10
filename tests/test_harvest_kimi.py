"""Offline coverage for the Kimi Code wire.jsonl transcript source."""

from __future__ import annotations

import json
import os
import time

from skillopt_sleep.harvest_kimi import (
    _kimi_project_matches,
    _workspace_slug,
    digest_kimi_session,
    harvest_kimi,
)
from skillopt_sleep.harvest_sources import harvest_for_config


def _wire_event(etype: str, role: str, text: str, *, origin_kind="user",
                ms: int = 1_767_225_600_000, envelope: str = "context") -> dict:
    msg = {"role": role, "content": [{"type": "text", "text": text}]}
    if envelope == "context":
        wrapper = dict(msg)
        wrapper["origin"] = {"kind": origin_kind}
        return {"type": "context.append_message", "message": wrapper, "time": ms}
    return {
        "type": "agent.message.appended",
        "message": {"message": msg, "meta": {"origin": {"kind": origin_kind}}},
        "time": ms,
    }


def _write_session(root: str, wd: str, session: str, events: list[dict]) -> str:
    d = os.path.join(root, wd, session, "agents", "main")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "wire.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    return path


_NOW = int(time.time() * 1000)


def _multi_turn_events() -> list[dict]:
    return [
        _wire_event("context.append_message", "user", "Help me reverse a string in Python", ms=_NOW - 4000),
        _wire_event("agent.message.appended", "assistant", "Here is the function ...", origin_kind=None, ms=_NOW - 3000),
        _wire_event("agent.message.appended", "user", "thanks, works perfectly", ms=_NOW - 2000),
        _wire_event("agent.message.appended", "assistant", "Glad it helped!", origin_kind=None, ms=_NOW - 1000),
    ]


def test_workspace_slug() -> None:
    assert _workspace_slug(r"C:\x\.kimi-code\sessions\wd_358_7051f95cba3a\s1") == "358"
    assert _workspace_slug(r"C:\x\wd_ pte_8d327c3178e2\s1") == " pte"


def test_kimi_project_matches_invoked_by_segment() -> None:
    inv = r"C:\Users\83773\Downloads\Compressed\notes\358"
    assert _kimi_project_matches("358", "invoked", inv)
    assert not _kimi_project_matches("pte", "invoked", inv)
    assert _kimi_project_matches("pte", "all", inv)


def test_digest_parses_both_envelopes(tmp_path) -> None:
    root = str(tmp_path)
    path = _write_session(root, "wd_358_7051f95cba3a", "session_a", _multi_turn_events())
    d = digest_kimi_session(path)
    assert d is not None
    assert d.project == "358"
    assert d.n_user_turns == 2
    assert d.n_assistant_turns == 2
    assert "reverse a string" in d.user_prompts[0]
    assert d.ended_at and d.ended_at.startswith("2026-")


def test_digest_skips_injection_and_reminders(tmp_path) -> None:
    root = str(tmp_path)
    events = [
        _wire_event("context.append_message", "user",
                    "<system-reminder>date</system-reminder>", origin_kind="injection"),
        _wire_event("context.append_message", "user", "real question here", ms=_NOW - 1000),
    ]
    path = _write_session(root, "wd_358_7051f95cba3a", "session_b", events)
    d = digest_kimi_session(path)
    assert d is not None
    assert d.user_prompts == ["real question here"]
    assert d.n_user_turns == 1


def test_digest_returns_none_for_empty(tmp_path) -> None:
    root = str(tmp_path)
    path = _write_session(root, "wd_358_7051f95cba3a", "session_c",
                          [_wire_event("context.append_message", "user",
                                       "<system-reminder>x</system-reminder>",
                                       origin_kind="injection")])
    assert digest_kimi_session(path) is None


def test_harvest_kimi_scope_and_since(tmp_path) -> None:
    root = str(tmp_path)
    _write_session(root, "wd_358_7051f95cba3a", "session_new", _multi_turn_events())
    old_events = [_wire_event("context.append_message", "user", "old task", ms=1_600_000_000_000)]
    _write_session(root, "wd_pte_8d327c3178e2", "session_old", old_events)
    _write_session(root, "wd_pte_8d327c3178e2", "session_pte_new", _multi_turn_events())

    all_digests = harvest_kimi(root, scope="all", since_iso="2026-01-01T00:00:00.000Z")
    # session_old (2020 timestamps) is correctly dropped by the since window
    assert len(all_digests) == 2

    invoked = harvest_kimi(root, scope="invoked",
                           invoked_project=r"C:\Users\83773\proj\358",
                           since_iso="2026-01-01T00:00:00.000Z")
    assert {d.project for d in invoked} == {"358"}

    recent = harvest_kimi(root, scope="all", since_iso="2026-01-01T00:00:00.000Z",
                          limit=2)
    assert len(recent) == 2


def test_harvest_for_config_kimi_source(tmp_path, monkeypatch) -> None:
    root = str(tmp_path)
    _write_session(root, "wd_358_7051f95cba3a", "session_x", _multi_turn_events())

    class Cfg(dict):
        kimi_sessions_dir = root

    digests = harvest_for_config(Cfg(transcript_source="kimi", projects="all"),
                                 since_iso="2026-01-01T00:00:00.000Z")
    assert len(digests) == 1
    assert digests[0].project == "358"


def test_engine_replay_sessions_are_filtered(tmp_path):
    from skillopt_sleep.harvest_kimi import _is_kimi_replay

    class _D:
        def __init__(self, prompts):
            self.user_prompts = prompts

    # CLI replay attempt prompt
    assert _is_kimi_replay(_D(["Complete the following task for the user. Follow the skill and memory ..."]))
    # programmatic session with no user intent
    assert _is_kimi_replay(_D([]))
    # real user session
    assert not _is_kimi_replay(_D(["SkillOpt 你全局装一下这个"]))


def test_harvest_skips_replay_sessions(tmp_path):
    root = str(tmp_path)
    replay = [
        _wire_event("context.append_message", "user",
                    "Complete the following task for the user. Follow the skill and memory X",
                    ms=_NOW - 3000),
    ]
    _write_session(root, "wd_83773_e9c3dd2644d1", "session_replay", replay)
    _write_session(root, "wd_358_7051f95cba3a", "session_real", _multi_turn_events())
    ds = harvest_kimi(root, scope="all", since_iso="2026-01-01T00:00:00.000Z")
    assert len(ds) == 1
    assert ds[0].session_id == "session_real"
