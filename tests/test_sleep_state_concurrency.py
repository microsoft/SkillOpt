"""Only one cycle may read/update/save a shared state at a time."""
import multiprocessing
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from skillopt_sleep import state as state_module
from skillopt_sleep.state import SleepState, harvest_cursor_key


def update_project(path, project, ready, release):
    try:
        with SleepState.lock(path):
            state = SleepState.load(path)
            ready.set()
            assert release.wait(10)
            state.set_last_harvest(project, "2026-01-01T00:00:00", "codex")
            state.record_night({"project": project})
            state.save()
    except RuntimeError:
        raise SystemExit(2)


def exit_holding_lock(path):
    with SleepState.lock(path):
        os._exit(3)


def test_process_exit_releases_lock_without_changing_state(tmp_path):
    path = tmp_path / "state.json"
    original = b'{"night": 4}'
    path.write_bytes(original)
    process = multiprocessing.get_context("spawn").Process(
        target=exit_holding_lock, args=(str(path),)
    )
    try:
        process.start()
        process.join(10)
        assert process.exitcode == 3
        with SleepState.lock(str(path)):
            assert path.read_bytes() == original
    finally:
        if process.pid is not None:
            if process.is_alive():
                process.terminate()
            process.join(5)


def test_two_projects_refuse_overlap_then_preserve_both_updates(tmp_path):
    context = multiprocessing.get_context("spawn")
    path = str(tmp_path / "state.json")
    ready, release = context.Event(), context.Event()
    other_ready = context.Event()
    first = context.Process(target=update_project, args=(path, "first", ready, release))
    children = [first]
    try:
        first.start()
        assert ready.wait(10)
        second = context.Process(target=update_project, args=(path, "second", other_ready, release))
        children.append(second)
        second.start()
        second.join(10)
        assert second.exitcode == 2
        release.set()
        first.join(10)
        assert first.exitcode == 0
        retry = context.Process(target=update_project, args=(path, "second", other_ready, release))
        children.append(retry)
        retry.start()
        retry.join(10)
        assert retry.exitcode == 0
        saved = SleepState.load(path)
        assert set(saved.data["last_harvest"]) == {
            harvest_cursor_key("first", "codex"), harvest_cursor_key("second", "codex"),
        }
        assert [row["project"] for row in saved.data["history"]] == ["first", "second"]
    finally:
        release.set()
        for child in children:
            if child.pid is not None:
                if child.is_alive():
                    child.terminate()
                child.join(5)


def test_lock_releases_after_exception(tmp_path):
    path = str(tmp_path / "state.json")
    with pytest.raises(ValueError):
        with SleepState.lock(path):
            raise ValueError("synthetic cycle failure")
    with SleepState.lock(path):
        SleepState.load(path).save()


@pytest.mark.parametrize("failure", ["dump", "replace"])
def test_interrupted_save_preserves_prior_state_and_other_tempfiles(tmp_path, monkeypatch, failure):
    path = tmp_path / "state.json"
    path.write_text('{"night": 4}', encoding="utf-8")
    other = tmp_path / "state.json.tmp"
    other.write_bytes(b"another writer's file")
    state = SleepState.load(str(path))
    state.begin_night()

    def fail(*args, **kwargs):
        raise OSError("synthetic interrupted save")

    monkeypatch.setattr(state_module.json if failure == "dump" else state_module.os, failure, fail)
    with pytest.raises(OSError, match="interrupted save"):
        state.save()
    assert path.read_text(encoding="utf-8") == '{"night": 4}'
    assert other.read_bytes() == b"another writer's file"
    assert set(tmp_path.iterdir()) == {path, other}


def test_cycle_and_handoff_share_the_transaction_lock(tmp_path, monkeypatch):
    from skillopt_sleep import __main__ as cli
    from skillopt_sleep import cycle

    cfg = SimpleNamespace(state_path=str(tmp_path / "state.json"))
    marker = object()

    def locked(*args, **kwargs):
        with pytest.raises(RuntimeError, match="busy"):
            with SleepState.lock(cfg.state_path):
                pytest.fail("writer started without the shared lock")
        return marker

    monkeypatch.setattr(cycle, "_run_sleep_cycle", locked)
    monkeypatch.setattr(cli, "_handoff_mine_and_pin_locked", locked)
    assert cycle.run_sleep_cycle(cfg) is marker
    assert cli._handoff_mine_and_pin(cfg, None, None, "synthetic", False) is marker


def test_both_writer_routes_keep_source_checkpoints_and_prior_nights(tmp_path):
    """Each locked writer route advances its own (project, source) cursor and
    keeps the legacy bare cursor and earlier nights, including a budget-capped
    night that recorded no scores (#317, #322)."""
    from skillopt_sleep.__main__ import main
    from skillopt_sleep.config import load_config
    from skillopt_sleep.cycle import run_sleep_cycle

    proj = str(tmp_path / "proj")
    os.makedirs(proj)
    claude_home = str(tmp_path / ".claude")
    os.makedirs(os.path.join(claude_home, "projects"))
    source = "claude"
    cfg = load_config(
        invoked_project=proj, claude_home=claude_home, backend="mock",
        transcript_source=source,
    )
    legacy = "2026-01-01T00:00:00"
    budget_night = {"night": 1, "accepted": False, "n_tasks": 3, "staging": "s1"}
    seeded = SleepState(cfg.state_path)
    seeded.set_last_harvest(proj, legacy)
    seeded.begin_night()
    seeded.record_night(budget_night)
    seeded.save()

    def assert_kept():
        saved = SleepState.load(cfg.state_path)
        assert saved.data["last_harvest"][proj] == legacy
        assert saved.data["last_harvest"][harvest_cursor_key(proj, source)] > legacy
        assert saved.data["history"][0] == budget_night
        assert not list(Path(cfg.state_path).parent.glob(".sleep-state-*"))
        return saved

    run_sleep_cycle(cfg, dry_run=False)
    saved = assert_kept()
    assert saved.night == 2
    assert [row["night"] for row in saved.data["history"]] == [1, 2]

    # Clear the cycle's cursor so the handoff route must write its own.
    saved.data["last_harvest"].pop(harvest_cursor_key(proj, source))
    with SleepState.lock(cfg.state_path):
        saved.save()
    assert main([
        "run", "--backend", "handoff", "--project", proj,
        "--claude-home", claude_home, "--source", source,
    ]) == 0
    assert assert_kept().data["history"] == saved.data["history"]
