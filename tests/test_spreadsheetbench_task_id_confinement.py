"""Task-identifier confinement at the SpreadsheetBench production entry points.

An item id becomes one path segment under ``<out_root>/predictions``. The
reviewed defect: the security PR sanitized only the ``mkdtemp`` prefix, so an
id like ``../../escape`` stopped failing loudly there and instead continued on
to the destination derived from the raw id — reaching the prompt writes, the
agent call and the code-execution path.

These regressions drive the real entry points (``process_one`` for the ReAct
setting, ``process_one_codegen`` for the official codegen setting) and assert:

* an unsafe id is refused before the data root is even read — the fail reason is
  ``unsafe-task-id``, not ``no-test-cases`` — and leaves no directory behind;
* the agent / code-execution entry points are never reached for that id;
* a destination resolving outside ``predictions`` (a pre-existing symlink) is
  refused as well;
* normal ids still proceed (positive control).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from skillopt.envs.spreadsheetbench import rollout

ENTRIES = ["process_one", "process_one_codegen"]


def _write_case(data_root: Path, task_id: str) -> None:
    task_dir = data_root / task_id
    task_dir.mkdir(parents=True)
    (task_dir / "1_input.xlsx").write_bytes(b"")
    (task_dir / "1_answer.xlsx").write_bytes(b"")


def _item(task_id: str) -> dict:
    return {
        "id": task_id,
        "instruction": "put 1 in A1",
        "instruction_type": "cell",
        "answer_position": "A1",
        "answer_sheet": "Sheet1",
        "spreadsheet_path": task_id,
    }


def _call_entry(entry: str, item: dict, data_root: Path, out_root: Path) -> dict:
    """Invoke the production entry with its full required signature."""
    if entry == "process_one":
        return rollout.process_one(
            item, str(data_root), str(out_root), "", max_turns=5
        )
    return rollout.process_one_codegen(item, str(data_root), str(out_root), "")


def _block_agent(monkeypatch) -> list:
    """Make every agent / code-exec entry point a tripwire."""
    reached: list = []

    def _bomb(*args, **kwargs):
        reached.append((args, kwargs))
        raise AssertionError("agent / code-exec reached for an unsafe task id")

    import skillopt.envs.spreadsheetbench.codegen_agent as codegen_agent

    monkeypatch.setattr(rollout, "run_react", _bomb)
    monkeypatch.setattr(rollout, "run_generated_code", _bomb)
    monkeypatch.setattr(codegen_agent, "run_single", _bomb)
    monkeypatch.setattr(codegen_agent, "run_multi", _bomb)
    return reached


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize(
    "hostile_id",
    ["../../escape", "..", "a/b", "a\\b", "", ".", "../"],
)
def test_unsafe_task_id_stops_before_any_filesystem_access(
    tmp_path, monkeypatch, entry, hostile_id
):
    """Reach the entry point with a *loadable* task dir, as production does.

    The data root holds a real case directory, so without the identifier gate
    the pre-fix code finds the cases, derives the destination from the raw id
    and continues into the prompt writes / agent call — for ``../../escape``
    that means writing outside ``out_root``. Everything asserted here is
    confined to ``tmp_path``, so the regression is safe to run.
    """
    reached = _block_agent(monkeypatch)
    data_root = tmp_path / "data"
    _write_case(data_root, "1-1")
    out_root = tmp_path / "out"
    item = _item(hostile_id)
    item["spreadsheet_path"] = "1-1"

    result = _call_entry(entry, item, data_root, out_root)

    # Not "no-test-cases" (the valid task dir was never globbed) and not an
    # agent-error: the identifier gate ran first.
    assert result["fail_reason"] == "unsafe-task-id", result
    assert result["ok"] is False
    assert reached == []
    assert not out_root.exists(), "out_root was created for an unsafe id"
    assert not (tmp_path / "escape").exists()
    assert not (tmp_path / "1-1").exists()


@pytest.mark.parametrize(
    "absolute_id", ["/etc/passwd", "C:\\Windows", "\\\\server\\share", "//host/share"]
)
def test_absolute_shaped_ids_are_rejected_by_the_validator(absolute_id):
    """Absolute / UNC shapes never pass the predicate.

    Asserted against the pure predicate rather than the entry point: driving an
    absolute path through ``os.makedirs`` on the pre-fix code would try to
    create a real system path, which a regression test must never risk.
    """
    assert rollout._is_safe_task_id(absolute_id) is False


@pytest.mark.parametrize("entry", ENTRIES)
def test_symlinked_destination_is_refused(tmp_path, monkeypatch, entry):
    data_root = tmp_path / "data"
    _write_case(data_root, "1-1")
    out_root = tmp_path / "out"
    predictions = out_root / "predictions"
    predictions.mkdir(parents=True)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    try:
        os.symlink(elsewhere, predictions / "1-1", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are not available on this host")

    reached = _block_agent(monkeypatch)

    result = _call_entry(entry, _item("1-1"), data_root, out_root)

    assert result["fail_reason"] == "unsafe-task-id", result
    assert reached == []
    assert not (elsewhere / "1_pred.xlsx").exists()
    assert not (elsewhere / "target_user_prompt.txt").exists()


def test_codegen_normal_id_reaches_llm_phase(tmp_path, monkeypatch):
    """Positive control: an ordinary id still runs the normal flow."""
    import skillopt.envs.spreadsheetbench.codegen_agent as codegen_agent

    data_root = tmp_path / "data"
    _write_case(data_root, "1-1")
    out_root = tmp_path / "out"
    calls: list = []

    def _fake_run_single(**kwargs):
        calls.append(kwargs)
        return {"code": "", "raw": "no code block", "n_turns": 1}

    monkeypatch.setattr(codegen_agent, "run_single", _fake_run_single)

    result = rollout.process_one_codegen(
        _item("1-1"), str(data_root), str(out_root), ""
    )

    assert result["fail_reason"] == "empty-code-block", result
    assert result["llm_ok"] is True
    assert len(calls) == 1
    assert (out_root / "predictions" / "1-1").is_dir()


def test_react_normal_id_reaches_agent_phase(tmp_path, monkeypatch):
    """Positive control for the ReAct entry point."""
    data_root = tmp_path / "data"
    _write_case(data_root, "80-42")
    out_root = tmp_path / "out"

    def _fake_run_react(**kwargs):
        raise RuntimeError("stop-after-agent-entry")

    monkeypatch.setattr(rollout, "run_react", _fake_run_react)

    result = _call_entry("process_one", _item("80-42"), data_root, out_root)

    assert "stop-after-agent-entry" in result["fail_reason"], result
    assert (out_root / "predictions" / "80-42").is_dir()
