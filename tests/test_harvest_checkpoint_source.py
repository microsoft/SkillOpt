"""Harvest checkpoints are per (project, transcript source) — issue #312.

``skillopt-sleep run`` used to store one harvest cursor per project, whatever
``--source`` the run used. When one project is fed from several sources, every
source after the first only saw sessions newer than the first run's start, so
its older, never-harvested sessions were skipped for good.

Offline: mock backend, no model calls, no network.
Run:  python -m pytest tests/test_harvest_checkpoint_source.py
"""
from __future__ import annotations

import contextlib
import io
import json
import os
from types import SimpleNamespace
from unittest import mock

import pytest

from skillopt_sleep.config import load_config
from skillopt_sleep.state import SleepState, harvest_cursor_key

PROJECT = "/work/proj"


def _write_claude_session(transcripts_dir: str, session_id: str, project: str) -> None:
    """Two-turn interactive session, ended a few minutes ago (inside lookback)."""
    from datetime import datetime, timedelta

    folder = os.path.join(transcripts_dir, "projects", "demo")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, session_id + ".jsonl")
    base = datetime.now() - timedelta(minutes=5)
    with open(path, "w", encoding="utf-8") as f:
        for i, (role, text) in enumerate((
            ("user", "Fix the failing date parser test in module B."),
            ("assistant", "Fixed."),
            ("user", "Thanks, now add a regression test."),
            ("assistant", "Added."),
        )):
            timestamp = (base + timedelta(minutes=i)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
            f.write(json.dumps({
                "type": role, "cwd": project, "sessionId": session_id,
                "timestamp": timestamp,
                "message": {"role": role, "content": text},
            }) + "\n")


# ── state level ─────────────────────────────────────────────────────────────

def test_cursors_are_keyed_per_project_and_source(tmp_path):
    state = SleepState.load(str(tmp_path / "state.json"))
    state.set_last_harvest(PROJECT, "2026-01-02T03:00:00", "codex")

    assert state.last_harvest_for(PROJECT, "codex") == "2026-01-02T03:00:00"
    # another source of the same project has no cursor of its own
    assert state.last_harvest_for(PROJECT, "cursor") is None
    # and no project-wide cursor was written for it to inherit
    assert state.last_harvest_for(PROJECT) is None
    assert state.data["last_harvest"] == {harvest_cursor_key(PROJECT, "codex"):
                                          "2026-01-02T03:00:00"}


def test_sources_do_not_share_one_cursor(tmp_path):
    state = SleepState.load(str(tmp_path / "state.json"))
    state.set_last_harvest(PROJECT, "2026-01-02T03:00:00", "codex")
    state.set_last_harvest(PROJECT, "2026-01-05T03:00:00", "cursor")

    assert state.last_harvest_for(PROJECT, "codex") == "2026-01-02T03:00:00"
    assert state.last_harvest_for(PROJECT, "cursor") == "2026-01-05T03:00:00"


def test_legacy_bare_project_cursor_is_still_read_and_kept(tmp_path):
    # Older state files key the cursor by project alone; an upgrade must not
    # re-scan the whole transcript history, so it is read as the fallback.
    state = SleepState.load(str(tmp_path / "state.json"))
    state.set_last_harvest(PROJECT, "2026-01-02T03:00:00")

    assert state.last_harvest_for(PROJECT, "codex") == "2026-01-02T03:00:00"
    assert state.last_harvest_for(PROJECT) == "2026-01-02T03:00:00"

    # A per-source write must not overwrite the legacy cursor: other sources
    # still have only that one to fall back on.
    state.set_last_harvest(PROJECT, "2026-01-05T03:00:00", "codex")
    assert state.last_harvest_for(PROJECT, "cursor") == "2026-01-02T03:00:00"
    assert state.last_harvest_for(PROJECT, "codex") == "2026-01-05T03:00:00"
    assert state.last_harvest_for(PROJECT) == "2026-01-02T03:00:00"


# ── end to end (issue #312 repro) ───────────────────────────────────────────

def _cfg(project: str, claude_home: str, cursor_home: str, source: str):
    return load_config(
        invoked_project=project, projects="all", backend="mock",
        claude_home=claude_home, cursor_home=cursor_home,
        transcript_source=source, lookback_hours=24,
    )


def test_second_source_run_does_not_hide_the_first_sources_sessions(tmp_path):
    from skillopt_sleep.cycle import run_sleep_cycle

    proj = str(tmp_path / "proj")
    os.makedirs(proj)
    claude_home = str(tmp_path / ".claude")
    os.makedirs(claude_home)
    # an empty Cursor store: that run harvests nothing but persists state
    cursor_home = str(tmp_path / ".cursor")
    os.makedirs(os.path.join(cursor_home, "projects"))
    _write_claude_session(os.path.join(claude_home, "projects"), "real-1", proj)
    _write_claude_session(os.path.join(claude_home, "projects"), "real-2", proj)

    claude = _cfg(proj, claude_home, cursor_home, "claude")
    cursor = _cfg(proj, claude_home, cursor_home, "cursor")

    first = run_sleep_cycle(claude, dry_run=True)
    assert first.report.n_sessions == 2

    # stateful run of a DIFFERENT source in the same project: it persists a
    # checkpoint for itself only (#312: this used to move the project cursor)
    other = run_sleep_cycle(cursor, dry_run=False)
    assert other.report.n_sessions == 0

    # the claude sessions were never harvested, so they must still be there
    again = run_sleep_cycle(claude, dry_run=True)
    assert again.report.n_sessions == 2

    state = SleepState.load(claude.state_path)
    assert state.last_harvest_for(proj, "cursor") is not None
    assert state.last_harvest_for(proj, "claude") is None


def test_a_source_still_skips_what_it_already_harvested(tmp_path):
    from skillopt_sleep.cycle import run_sleep_cycle

    proj = str(tmp_path / "proj")
    os.makedirs(proj)
    claude_home = str(tmp_path / ".claude")
    os.makedirs(claude_home)
    cursor_home = str(tmp_path / ".cursor")
    os.makedirs(os.path.join(cursor_home, "projects"))
    _write_claude_session(os.path.join(claude_home, "projects"), "real-1", proj)

    claude = _cfg(proj, claude_home, cursor_home, "claude")
    first = run_sleep_cycle(claude, dry_run=False)
    assert first.report.n_sessions == 1

    # its own cursor advanced: the same session is not harvested twice
    assert run_sleep_cycle(claude, dry_run=True).report.n_sessions == 0


# ── source spelling (follow-up to #354) ─────────────────────────────────────
# harvest_for_config accepts "Codex" or " codex " as Codex, so the cursor has
# to be the Codex one too. A "<project>::Codex" cursor may hold a time written
# while that spelling still silently harvested Claude; reading it would skip
# older Codex sessions for good, the #312 failure again.

STALE = "2026-01-09T03:00:00"     # written under the raw spelling
CODEX_CURSOR = "2026-01-05T03:00:00"


def _codex_cfg(tmp_path, source):
    proj = str(tmp_path / "proj")
    os.makedirs(proj, exist_ok=True)
    os.makedirs(tmp_path / ".claude", exist_ok=True)
    return proj, load_config(
        invoked_project=proj, projects="all", backend="mock",
        claude_home=str(tmp_path / ".claude"),
        codex_home=str(tmp_path / ".codex"),
        transcript_source=source, lookback_hours=24,
    )


def _seed_cursors(cfg, proj, spelling):
    state = SleepState.load(cfg.state_path)
    state.set_last_harvest(proj, CODEX_CURSOR, "codex")
    state.set_last_harvest(proj, STALE, spelling)
    state.save()


@pytest.mark.parametrize("spelling", ["Codex", " codex ", "CODEX"])
def test_source_spelling_uses_the_canonical_cursor_and_harvester(tmp_path, spelling):
    from skillopt_sleep.cycle import run_sleep_cycle

    proj, cfg = _codex_cfg(tmp_path, spelling)
    _seed_cursors(cfg, proj, spelling)

    with mock.patch("skillopt_sleep.harvest_sources.harvest_codex",
                    return_value=[]) as codex, \
            mock.patch("skillopt_sleep.harvest_sources.harvest") as claude:
        run_sleep_cycle(cfg, dry_run=False)

    claude.assert_not_called()
    assert codex.call_args.kwargs["since_iso"] == CODEX_CURSOR
    cursors = SleepState.load(cfg.state_path).data["last_harvest"]
    # the Codex cursor advanced; the stale one is neither read nor rewritten
    assert set(cursors) == {harvest_cursor_key(proj, "codex"),
                            harvest_cursor_key(proj, spelling)}
    assert cursors[harvest_cursor_key(proj, "codex")] != CODEX_CURSOR
    assert cursors[harvest_cursor_key(proj, spelling)] == STALE


def test_handoff_pinning_uses_the_canonical_cursor(tmp_path):
    from skillopt_sleep.__main__ import _handoff_mine_and_pin
    from skillopt_sleep.handoff_backend import HandoffBackend

    proj, cfg = _codex_cfg(tmp_path, "Codex")
    _seed_cursors(cfg, proj, "Codex")
    backend = HandoffBackend(handoff_dir=str(tmp_path / "handoff"))

    with mock.patch("skillopt_sleep.harvest_sources.harvest_codex",
                    return_value=[]) as codex, \
            contextlib.redirect_stdout(io.StringIO()):
        rc, tasks = _handoff_mine_and_pin(
            cfg, SimpleNamespace(json=False), backend,
            str(tmp_path / "tasks.json"), False,
        )

    assert (rc, tasks) == (0, None)
    assert codex.call_args.kwargs["since_iso"] == CODEX_CURSOR
    cursors = SleepState.load(cfg.state_path).data["last_harvest"]
    assert cursors[harvest_cursor_key(proj, "codex")] != CODEX_CURSOR
    assert cursors[harvest_cursor_key(proj, "Codex")] == STALE


def test_harvest_command_records_the_canonical_source(tmp_path):
    from skillopt_sleep.__main__ import cmd_harvest

    _, cfg = _codex_cfg(tmp_path, " Codex ")
    out = io.StringIO()
    with mock.patch("skillopt_sleep.__main__._cfg_from_args", return_value=cfg), \
            mock.patch("skillopt_sleep.harvest_sources.harvest_codex",
                       return_value=[]), \
            contextlib.redirect_stdout(out):
        assert cmd_harvest(SimpleNamespace(json=True, output="")) == 0

    assert json.loads(out.getvalue())["transcript_source"] == "codex"


@pytest.mark.parametrize("value", [1, ["codex"], {"name": "codex"}, True])
def test_non_string_source_is_a_clear_value_error(tmp_path, value):
    from skillopt_sleep.harvest_sources import (
        harvest_for_config,
        normalize_transcript_source,
    )

    for call in (lambda: normalize_transcript_source(value),
                 lambda: harvest_for_config(_codex_cfg(tmp_path, value)[1])):
        with pytest.raises(ValueError, match="transcript_source must be a string"):
            call()


@pytest.mark.parametrize("value", [["codex"], "Codexx"])
def test_invalid_source_fails_before_the_night_writes_anything(tmp_path, value):
    from skillopt_sleep.cycle import run_sleep_cycle

    proj, cfg = _codex_cfg(tmp_path, value)
    with pytest.raises(ValueError):
        run_sleep_cycle(cfg, dry_run=False)

    # no staging dir for a night that never started, and no state change
    assert os.listdir(proj) == []
    assert not os.path.exists(cfg.state_path)


def test_canonical_sources_keep_their_cursor_keys():
    from skillopt_sleep.harvest_sources import (
        SUPPORTED_SOURCES,
        normalize_transcript_source,
    )

    for source in SUPPORTED_SOURCES:
        assert normalize_transcript_source(source) == source
        assert harvest_cursor_key(PROJECT, normalize_transcript_source(source)) \
            == f"{PROJECT}::{source}"
    assert normalize_transcript_source("") == "claude"
    assert normalize_transcript_source(None) == "claude"


def test_default_claude_source_still_reads_the_legacy_cursor(tmp_path):
    from skillopt_sleep.cycle import run_sleep_cycle

    proj, cfg = _codex_cfg(tmp_path, "claude")
    state = SleepState.load(cfg.state_path)
    state.set_last_harvest(proj, STALE)  # pre-#322 bare project key
    state.save()

    with mock.patch("skillopt_sleep.harvest_sources.harvest",
                    return_value=[]) as claude:
        run_sleep_cycle(cfg, dry_run=False)

    assert claude.call_args.kwargs["since_iso"] == STALE
    cursors = SleepState.load(cfg.state_path).data["last_harvest"]
    assert set(cursors) == {proj, f"{proj}::claude"}
    assert cursors[proj] == STALE
