"""Isolation regressions for SkillOpt-Sleep's in-process default state."""

import json
import subprocess
import sys

from skillopt_sleep.state import SleepState


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
