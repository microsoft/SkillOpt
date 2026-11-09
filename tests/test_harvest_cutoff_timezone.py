"""Regression tests: the harvest cutoff is local time, transcripts are UTC.

``state._now_iso`` stores the checkpoint as local time without an offset,
while Claude Code, Codex and Pi record ``...Z`` timestamps. Comparing the two
as strings shifted the cutoff by the host's UTC offset, so east of UTC the
sessions that ended just after a harvest were skipped for good.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone

import pytest

from skillopt_sleep.harvest import harvest
from skillopt_sleep.harvest_codex import harvest_codex
from skillopt_sleep.harvest_pi import harvest_pi
from skillopt_sleep.state import _now_iso

_CUTOFF_EPOCH = 1_790_000_000  # 2026-09-21T14:13:20Z


@pytest.fixture
def east_of_utc(monkeypatch):
    """Run with a host timezone of UTC+5 (or skip if it cannot be set)."""
    if hasattr(time, "tzset"):
        monkeypatch.setenv("TZ", "PKT-5")
        time.tzset()
        yield
        monkeypatch.undo()
        time.tzset()
    elif time.localtime(_CUTOFF_EPOCH).tm_gmtoff > 0:
        yield
    else:
        pytest.skip("needs time.tzset() or a host timezone east of UTC")


def _utc(offset_seconds: int) -> str:
    moment = datetime.fromtimestamp(_CUTOFF_EPOCH + offset_seconds, tz=timezone.utc)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _write(path, records) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


# Each session starts ten minutes before it ends, so it is not mistaken for a
# sub-3-second headless replay.
def _claude(start: str, ts: str) -> list:
    return [
        {"timestamp": start, "cwd": "/repo", "message": {"role": "user", "content": "Fix the parser bug"}},
        {"timestamp": ts, "cwd": "/repo", "message": {"role": "assistant", "content": "Fixed."}},
    ]


def _codex(start: str, ts: str) -> list:
    return [
        {"timestamp": start, "payload": {"type": "user_message", "message": "Fix the parser bug", "cwd": "/repo"}},
        {"timestamp": ts, "payload": {"type": "agent_message", "message": "Fixed."}},
    ]


def _pi(start: str, ts: str) -> list:
    return [
        {"type": "session", "version": 3, "id": "s", "timestamp": start, "cwd": "/repo"},
        {"type": "message", "id": "u", "parentId": None, "timestamp": start,
         "message": {"role": "user", "content": "Fix the parser bug"}},
        {"type": "message", "id": "a", "parentId": "u", "timestamp": ts,
         "message": {"role": "assistant", "content": [{"type": "text", "text": "Fixed."}]}},
    ]


@pytest.mark.parametrize("harvester, records", [
    (harvest, _claude),
    (harvest_codex, _codex),
    (harvest_pi, _pi),
])
def test_cutoff_compares_instants_not_strings(tmp_path, east_of_utc, harvester, records):
    since = _now_iso(_CUTOFF_EPOCH)
    _write(str(tmp_path / "after.jsonl"), records(_utc(-540), _utc(60)))
    _write(str(tmp_path / "before.jsonl"), records(_utc(-660), _utc(-60)))

    digests = harvester(str(tmp_path), scope="all", since_iso=since)

    assert [d.session_id for d in digests] == ["after"]
