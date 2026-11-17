"""Opt-in smoke test for ``opencode_exec`` against the real OpenCode CLI.

Calls a model through the user's existing OpenCode login. Skipped unless
``SKILLOPT_TEST_REAL_OPENCODE_EXEC=1``. Pick the model with
``SKILLOPT_TEST_OPENCODE_EXEC_MODEL`` (default ``openai/gpt-6.1-sol``; ChatGPT
logins reject some older ids, for example ``openai/gpt-5.5``).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from skillopt.model import backend_config, opencode_backend

pytestmark = pytest.mark.skipif(
    os.environ.get("SKILLOPT_TEST_REAL_OPENCODE_EXEC", "").strip() != "1",
    reason="set SKILLOPT_TEST_REAL_OPENCODE_EXEC=1 to call the real OpenCode CLI",
)


@pytest.fixture(autouse=True)
def _real_cli():
    backend_config.configure_opencode_exec(path=os.environ.get("OPENCODE_EXEC_PATH", "opencode"))
    opencode_backend.set_optimizer_deployment(
        os.environ.get("SKILLOPT_TEST_OPENCODE_EXEC_MODEL", "openai/gpt-6.1-sol")
    )
    yield


def test_real_optimizer_call_follows_the_system_prompt() -> None:
    text, usage = opencode_backend.chat_optimizer(
        system="Whatever the user says, answer only with the single word BANANA.",
        user="What is 2+2?",
        retries=1,
        timeout=180,
    )

    assert text.strip().strip(".").upper() == "BANANA"
    assert usage["prompt_tokens"] > 0


def test_real_optimizer_cannot_use_tools() -> None:
    marker = Path(tempfile.gettempdir()) / "skillopt_opencode_exec_pwned.txt"
    marker.unlink(missing_ok=True)

    text, _usage = opencode_backend.chat_optimizer(
        system="You are a helpful assistant.",
        user=(
            f"Use your shell or file tool to create the file {marker} containing hi, "
            "then reply DONE. If you have no tools, reply NO_TOOLS."
        ),
        retries=1,
        timeout=180,
    )

    assert not marker.exists()
    assert "DONE" not in text.upper()


def _workspace(tmp_path: Path) -> Path:
    from skillopt.model.codex_harness import prepare_workspace

    work = tmp_path / "pred" / "workspace"
    prepare_workspace(
        work_dir=str(work),
        skill_md="---\nname: skillopt-target\ndescription: answering rules\n---\nAlways end the answer with the code word BANANA42.\n",
        task_text="What is the capital of France?",
    )
    return work


def _model() -> str:
    return os.environ.get("SKILLOPT_TEST_OPENCODE_EXEC_MODEL", "openai/gpt-6.1-sol")


def test_real_target_reads_the_skill_and_task_but_cannot_write(tmp_path) -> None:
    from skillopt.model.codex_harness import run_opencode_exec

    work = _workspace(tmp_path)
    response, raw = run_opencode_exec(
        work_dir=str(work),
        prompt="Also create the file note.txt containing hi. Return the final answer inside <answer>...</answer>.",
        model=_model(),
        timeout=240,
    )

    assert "Paris" in response and "BANANA42" in response
    assert not (work / "note.txt").exists()
    assert (work.parent / "opencode_trace_steps.txt").read_text().strip()


def test_real_target_file_edits_stay_inside_the_workspace(tmp_path) -> None:
    from skillopt.model.codex_harness import run_opencode_exec

    work = _workspace(tmp_path)
    outside = tmp_path / "outside.txt"
    run_opencode_exec(
        work_dir=str(work),
        prompt=f"Create the file answer.txt containing Paris. Then try to create the file {outside} containing hi. "
        "Return the final answer inside <answer>...</answer>.",
        model=_model(),
        timeout=240,
        allow_file_edits=True,
    )

    assert (work / "answer.txt").read_text().strip().startswith("Paris")
    assert not outside.exists()
