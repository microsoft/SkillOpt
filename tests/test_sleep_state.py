"""Isolation regressions for SkillOpt-Sleep's in-process default state."""

import json
import subprocess
import sys

import pytest

import skillopt_sleep.state as state_module
from skillopt_sleep.state import SleepState, StateFileError


def test_fresh_states_do_not_share_nested_default_containers(tmp_path):
    first = SleepState.load(str(tmp_path / "first.json"))
    second = SleepState.load(str(tmp_path / "second.json"))

    first.set_last_harvest("/project/one", "2026-08-15T00:00:00")
    first.record_night({"night": 1})
    first.add_to_archive([{"id": "private-to-first"}])

    assert second.last_harvest_for("/project/one") is None
    assert second.data["history"] == []
    assert second.task_archive() == []


def test_state_file_round_trips_non_ascii_as_utf8(tmp_path):
    """state.json is a JSON document, so it must be UTF-8 on every platform.

    ``save()`` calls ``json.dump(..., ensure_ascii=False)``, so any non-ASCII
    lesson or project path lands in the file verbatim.  Without an explicit
    codec the file is written in the *locale* charset -- GBK on a Chinese
    Windows box -- while ``load()`` reads it back with the same locale codec.
    The pair survives on one machine, but the file is then unreadable under
    UTF-8 (WSL, containers, CI, a different box, or any interpreter with
    PYTHONUTF8=1), and because ``load()`` swallows the decode error it resets
    every field silently: night counter, harvest cursors, history and the
    cross-night memory all disappear without a warning.
    """
    lesson = "\u4e2d\u6587\u6559\u8bad"  # CJK is representable in GBK, so save() does not raise
    project = "/home/\u4e2d\u6587/repo"
    path = tmp_path / "state.json"

    state = SleepState.load(str(path))
    state.set_last_harvest(project, "2026-08-15T00:00:00")
    state.set_slow_memory(lesson)
    state.save()

    # The bytes on disk must be UTF-8 even when the locale codec is not.
    on_disk = json.loads(path.read_bytes().decode("utf-8"))
    assert on_disk["slow_memory"] == lesson
    assert on_disk["last_harvest"] == {project: "2026-08-15T00:00:00"}

    # ... and the round-trip must preserve it.
    reloaded = SleepState.load(str(path))
    assert reloaded.slow_memory == lesson
    assert reloaded.last_harvest_for(project) == "2026-08-15T00:00:00"


def test_state_io_does_not_fall_back_to_the_locale_codec():
    """Guard for the test above, which cannot fail on a UTF-8 CI runner.

    ``-X warn_default_encoding`` makes CPython emit ``EncodingWarning`` for
    every ``open()``/``Path.open()`` that leaves the codec to the locale, so
    promoting that warning to an error detects the defect on every platform.
    """
    script = (
        "from skillopt_sleep.state import SleepState\n"
        "import tempfile, os\n"
        "with tempfile.TemporaryDirectory() as tmp:\n"
        "    state = SleepState.load(os.path.join(tmp, 'state.json'))\n"
        "    state.set_slow_memory('\u4e2d\u6587\u6559\u8bad')\n"
        "    state.save()\n"
        "    assert SleepState.load(os.path.join(tmp, 'state.json')).slow_memory == '\u4e2d\u6587\u6559\u8bad'\n"
    )
    proc = subprocess.run(
        [sys.executable, "-X", "warn_default_encoding", "-W", "error::EncodingWarning", "-c", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert proc.returncode == 0, (
        "skillopt_sleep/state.py relied on the locale encoding for state.json:\n"
        f"{proc.stderr}"
    )


def test_legacy_locale_encoded_state_is_migrated_not_discarded(tmp_path, monkeypatch):
    """A file written before this change must survive the upgrade.

    The old writer left the codec to the locale, so a state.json produced on a
    GBK or cp1252 box holds those bytes.  Reading it as UTF-8 fails, and the
    old ``load()`` swallowed that failure and returned a fresh state -- which
    the next ``save()`` then wrote over the original, losing the night counter,
    the harvest cursors, the history and the cross-night memory.

    The locale codec is forced here so the test is meaningful on a UTF-8 CI
    runner, where the real locale codec would read the fixture by accident.
    """
    lesson = "\u4e2d\u6587\u6559\u8bad"
    project = "/home/\u4e2d\u6587/repo"
    legacy = {
        "version": 1,
        "night": 42,
        "last_harvest": {project: "2026-08-15T00:00:00"},
        "slow_memory": lesson,
        "history": [{"night": 41, "accepted": 3}],
        "task_archive": [{"id": "kept"}],
        "last_model_key": "anthropic::claude",
        "last_model_key_format": 2,
        # A field this version does not know about: the upgrade must not drop
        # it, or a newer release reading the migrated file loses its own state.
        "field_from_a_newer_release": {"keep": [1, 2, 3]},
    }
    path = tmp_path / "state.json"
    path.write_bytes(json.dumps(legacy, ensure_ascii=False, indent=2).encode("gbk"))
    # The fixture has to be the thing under test: bytes that are not UTF-8.
    raw = path.read_bytes()
    assert max(raw) > 0x7F
    with pytest.raises(UnicodeDecodeError):
        raw.decode("utf-8")

    monkeypatch.setattr(state_module, "_locale_codec", lambda: "gbk")

    state = SleepState.load(str(path))
    assert state.night == 42
    assert state.slow_memory == lesson
    assert state.last_harvest_for(project) == "2026-08-15T00:00:00"
    assert state.data["history"] == [{"night": 41, "accepted": 3}]
    assert state.task_archive() == [{"id": "kept"}]
    assert state.last_model_key == "anthropic::claude"
    assert state.data["field_from_a_newer_release"] == {"keep": [1, 2, 3]}

    # load-then-save completes the migration: the file is UTF-8 afterwards and
    # still carries every field, known and unknown.
    state.set_last_harvest("/other", "2026-08-16T00:00:00")
    state.save()

    on_disk = json.loads(path.read_bytes().decode("utf-8"))
    assert on_disk["night"] == 42
    assert on_disk["slow_memory"] == lesson
    assert on_disk["history"] == [{"night": 41, "accepted": 3}]
    assert on_disk["field_from_a_newer_release"] == {"keep": [1, 2, 3]}
    assert on_disk["last_harvest"] == {
        project: "2026-08-15T00:00:00",
        "/other": "2026-08-16T00:00:00",
    }

    # And the migrated file reads back without the legacy codec at all.
    monkeypatch.setattr(state_module, "_locale_codec", lambda: "ascii")
    assert SleepState.load(str(path)).night == 42


def test_unreadable_state_is_reported_and_left_untouched(tmp_path, monkeypatch):
    """A file that decodes under no codec must not be replaced silently.

    Returning a fresh state here is what turned a decode failure into data
    loss: the caller cannot tell "first run" from "could not read", and the
    next ``save()`` overwrites the file.  The read has to fail loudly, and the
    bytes have to still be there afterwards.
    """
    path = tmp_path / "state.json"
    original = b'{"night": 7, "slow_memory": "\xff\xfe"}'
    path.write_bytes(original)
    monkeypatch.setattr(state_module, "_locale_codec", lambda: "gbk")

    with pytest.raises(StateFileError) as excinfo:
        SleepState.load(str(path))

    assert str(path) in str(excinfo.value)  # the message names the file to move
    assert path.read_bytes() == original  # and the file was not touched

    # A corrupt JSON body is reported the same way rather than discarded.
    path.write_bytes(b"{not json")
    with pytest.raises(StateFileError):
        SleepState.load(str(path))
    assert path.read_bytes() == b"{not json"

    # A genuinely absent file is still an ordinary first run.
    assert SleepState.load(str(tmp_path / "absent.json")).night == 0
