"""SpreadsheetBench codegen must send target calls to the configured backend.

The single and multi modes built their own client for every chat target. An
openai_compatible, claude_chat, minimax_chat or copilot_chat target failed
with "Azure OpenAI endpoint is not configured" or had its calls sent to Azure,
and a qwen_chat target skipped its qwen_chat_* settings.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import openpyxl
import pytest

import skillopt.model as model
from skillopt.envs.spreadsheetbench import codegen_agent
from skillopt.model import backend_config
from skillopt.model import openai_compatible_backend as compat

CODE_REPLY = (
    "```python\n"
    "import openpyxl\n"
    "wb = openpyxl.load_workbook(INPUT_PATH)\n"
    "wb.active['B1'] = 'done'\n"
    "wb.save(OUTPUT_PATH)\n"
    "```"
)


class _Completions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        message = SimpleNamespace(content=CODE_REPLY, tool_calls=[])
        usage = SimpleNamespace(prompt_tokens=2, completion_tokens=3, total_tokens=5)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def _client(completions: _Completions) -> Any:
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


@pytest.fixture(autouse=True)
def isolate_backend_state(monkeypatch: pytest.MonkeyPatch):
    optimizer_backend = backend_config.get_optimizer_backend()
    target_backend = backend_config.get_target_backend()
    target_config = vars(compat.TARGET_CONFIG).copy()
    # The 3 s cooldown after each call only slows the test down.
    monkeypatch.setattr(codegen_agent.time, "sleep", lambda _s: None)
    model.reset_token_tracker()
    yield
    model.reset_token_tracker()
    vars(compat.TARGET_CONFIG).update(target_config)
    backend_config.set_optimizer_backend(optimizer_backend)
    backend_config.set_target_backend(target_backend)
    compat._reset_clients()


@pytest.fixture
def workbook(tmp_path):
    path = tmp_path / "input.xlsx"
    wb = openpyxl.Workbook()
    wb.active["A1"] = "x"
    wb.save(path)
    return str(path), str(tmp_path / "output.xlsx")


@pytest.fixture
def compat_target(monkeypatch: pytest.MonkeyPatch) -> _Completions:
    completions = _Completions()
    monkeypatch.setattr(compat, "_get_client", lambda role: _client(completions))
    model.set_target_backend("openai_compatible")
    compat.TARGET_CONFIG.deployment = "deepseek-chat"

    def no_azure() -> Any:
        raise AssertionError("an Azure client was built for an openai_compatible target")

    monkeypatch.setattr(codegen_agent, "get_target_client", no_azure)
    return completions


def test_single_mode_uses_the_configured_target_backend(compat_target, workbook):
    input_xlsx, output_xlsx = workbook

    result = codegen_agent.run_single("Write done in B1", input_xlsx, output_xlsx)

    assert len(compat_target.calls) == 1
    assert compat_target.calls[0]["model"] == "deepseek-chat"
    assert compat_target.calls[0]["messages"][0]["role"] == "system"
    assert compat_target.calls[0]["timeout"] == 120  # the per-call llm_timeout
    assert "wb.save(OUTPUT_PATH)" in result["code"]


def test_multi_mode_uses_the_configured_target_backend(compat_target, workbook):
    input_xlsx, output_xlsx = workbook

    result = codegen_agent.run_multi("Write done in B1", input_xlsx, output_xlsx, max_turns=2)

    assert len(compat_target.calls) == 1
    assert compat_target.calls[0]["model"] == "deepseek-chat"
    assert result["n_turns"] == 1
    assert openpyxl.load_workbook(output_xlsx).active["B1"].value == "done"


def test_openai_chat_target_still_uses_the_azure_client(monkeypatch, workbook):
    input_xlsx, output_xlsx = workbook
    completions = _Completions()
    model.set_target_backend("openai_chat")
    monkeypatch.setattr(codegen_agent, "get_target_client", lambda: _client(completions))
    monkeypatch.setattr(codegen_agent, "_get_deployment", lambda: "gpt-5.5")

    result = codegen_agent.run_single("Write done in B1", input_xlsx, output_xlsx)

    assert len(completions.calls) == 1
    assert completions.calls[0]["model"] == "gpt-5.5"
    assert completions.calls[0]["max_completion_tokens"] == 16384
    assert "wb.save(OUTPUT_PATH)" in result["code"]
