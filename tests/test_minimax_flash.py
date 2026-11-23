"""Flash configuration reaches the serialized HTTP request without inference."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from skillopt.model import minimax_backend as backend


@pytest.fixture
def requests(monkeypatch):
    calls = []
    monkeypatch.setattr(backend, "TARGET_DEPLOYMENT", "MiniMax-M3.1-Flash")
    monkeypatch.setattr(backend, "OPTIMIZER_DEPLOYMENT", "MiniMax-M3.1-Flash")
    monkeypatch.setattr(backend, "ENABLE_THINKING", False)
    monkeypatch.setattr(backend, "REASONING_EFFORT", None)
    monkeypatch.setattr(backend, "API_KEY", "")

    def urlopen(request, timeout=None):
        calls.append(json.loads(request.data))
        return io.BytesIO(json.dumps({
            "choices": [{"message": {"content": "answer"}}],
        }).encode())

    monkeypatch.setattr(backend.urllib.request, "urlopen", urlopen)
    return calls


@pytest.mark.parametrize("method", [
    "chat_target", "chat_optimizer", "chat_target_messages", "chat_optimizer_messages",
])
@pytest.mark.parametrize("effort", [None, "", "low", "medium", "high", "xhigh", "max"])
def test_flash_effort_at_http_boundary(requests, method, effort):
    args = ([{"role": "user", "content": "hello"}],) if method.endswith("messages") else ("system", "hello")
    getattr(backend, method)(*args, reasoning_effort=effort, retries=1)
    payload = requests[0]
    assert payload["model"] == "MiniMax-M3.1-Flash"
    assert payload["thinking"] == {"type": "adaptive"}
    assert payload.get("reasoning_effort") == (effort or None)
    assert not {"service_tier", "reasoning_split", "chat_template_kwargs"} & payload.keys()


def test_shared_effort_and_per_call_override(requests):
    backend.set_reasoning_effort("high")
    backend.chat_target("system", "hello", retries=1)
    backend.chat_target("system", "hello", reasoning_effort="low", retries=1)
    backend.chat_target("system", "hello", reasoning_effort="", retries=1)
    assert requests[0]["reasoning_effort"] == "high"
    assert requests[1]["reasoning_effort"] == "low"
    assert "reasoning_effort" not in requests[2]


@pytest.mark.parametrize("effort", ["none", "minimal", "invalid"])
def test_flash_invalid_effort_fails_before_http(requests, effort):
    with pytest.raises(ValueError, match="reasoning effort"):
        backend.chat_target("system", "hello", reasoning_effort=effort, retries=1)
    assert requests == []


@pytest.mark.parametrize("model,thinking", [
    ("MiniMax-M3", "disabled"),
    ("MiniMax-M2.7", "adaptive"),
    ("saved-custom-model", "disabled"),
])
def test_saved_models_keep_their_thinking_and_ignore_effort(requests, monkeypatch, model, thinking):
    monkeypatch.setattr(backend, "TARGET_DEPLOYMENT", model)
    backend.chat_target("system", "hello", reasoning_effort="none", retries=1)
    assert requests[0]["model"] == model
    assert requests[0]["thinking"] == {"type": thinking}
    assert "reasoning_effort" not in requests[0]


@pytest.mark.parametrize("loop", ["spreadsheet", "office"])
def test_tool_loop_replays_flash_reasoning_at_http_boundary(requests, monkeypatch, tmp_path, loop):
    calls = []
    reasoning = "Preserve this reasoning exactly."
    tool_call = {"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}

    def urlopen(request, timeout=None):
        calls.append(json.loads(request.data))
        message = {"content": "", "reasoning_content": reasoning, "tool_calls": [tool_call]}
        if len(calls) > 1:
            message = {"content": "<answer>done</answer>"}
        return io.BytesIO(json.dumps({"choices": [{"message": message}]}).encode())

    monkeypatch.setattr(backend.urllib.request, "urlopen", urlopen)
    if loop == "spreadsheet":
        from skillopt.envs.spreadsheetbench import react_agent

        monkeypatch.setattr(react_agent, "chat_target_messages", backend.chat_target_messages)
        monkeypatch.setattr(react_agent, "get_target_backend", lambda: "minimax_chat")
        monkeypatch.setattr(react_agent, "_run_bash", lambda *args: "result")
        react_agent._react_loop("system", "user", str(tmp_path), 2, 8000)
    else:
        from skillopt.envs.officeqa import rollout

        monkeypatch.setattr(rollout, "chat_target_messages", backend.chat_target_messages)
        monkeypatch.setattr(rollout, "get_target_backend", lambda: "minimax_chat")
        monkeypatch.setattr(rollout, "run_tool", lambda *args, **kwargs: ("lookup", "result"))
        rollout.process_one({"id": "case", "question": "question", "answer": "done"},
                            str(tmp_path), "skill", data_dirs=[str(tmp_path)], max_tool_turns=2)
    assert len(calls) == 2
    assistant = next(message for message in calls[1]["messages"] if message["role"] == "assistant")
    assert assistant["reasoning_content"] == reasoning
    assert assistant["tool_calls"] == [tool_call]
    assert calls[1]["messages"][-1]["tool_call_id"] == "call-1"


def test_compat_message_dump_preserves_reasoning(requests, monkeypatch):
    message = {"content": "answer", "reasoning_content": "reasoning"}
    monkeypatch.setattr(backend.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(
        json.dumps({"choices": [{"message": message}]}).encode()))
    response, _ = backend.chat_target_messages([], return_message=True, retries=1)
    assert response.model_dump() == {"role": "assistant", **message}


# Exercise the actual entry-point setup in a fresh process: its global backend
# setters deliberately affect every adapter and must not leak into other tests.
_ENTRYPOINT_PROBE = r'''
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from scripts import eval_only, train
import skillopt.model as model
from skillopt.engine.trainer import ReflACTTrainer
from skillopt.model import minimax_backend as backend

entry, config, output, region, expected, explicit = sys.argv[1:]
module = train if entry == "train" else eval_only
sys.argv = [entry, "--config", config, "--backend", "minimax",
            "--optimizer_backend", "minimax_chat", "--out_root", output,
            "--minimax_region", region]
if entry == "eval":
    sys.argv += ["--skill", config]
if explicit == "cli":
    sys.argv += ["--target_model", expected, "--optimizer_model", expected]
args = module.parse_args()
cfg = module.load_config(args)
assert cfg["target_model"] == expected, cfg["target_model"]
assert cfg["optimizer_model"] == (expected if explicit in {"cli", "saved"} else "MiniMax-M3.1-Flash")

messages = [
    {"role": "user", "content": [
        {"type": "text", "text": "describe"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}},
    ]},
    {"role": "assistant", "content": "", "reasoning_content": "prior",
     "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}]},
    {"role": "tool", "tool_call_id": "call-1", "content": "result"},
]
tools = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]
calls = []
def urlopen(request, timeout=None):
    assert request.full_url == backend.REGION_BASE_URLS[region] + "/chat/completions"
    assert request.get_header("Authorization") is None
    calls.append(json.loads(request.data))
    return io.BytesIO(b'{"choices":[{"message":{"content":"answer"}}]}')
backend.urllib.request.urlopen = urlopen
backend.API_KEY = ""
class Configured(Exception):
    pass

def probe(*args):
    model.chat_target_messages(messages, tools=tools, tool_choice="auto", retries=1)
    model.chat_optimizer("system", "hello", retries=1)
    assert calls[0]["model"] == expected
    assert calls[0]["messages"] == messages
    assert calls[0]["tools"] == tools
    assert calls[0]["tool_choice"] == "auto"
    assert calls[0]["max_tokens"] == 8000
    for payload in calls:
        if payload["model"] == "MiniMax-M3.1-Flash":
            assert payload["thinking"] == {"type": "adaptive"}
            assert payload["reasoning_effort"] == "medium"
        else:
            assert "reasoning_effort" not in payload
    raise Configured

try:
    if entry == "train":
        adapter = SimpleNamespace(setup=lambda cfg: None, get_dataloader=lambda: None, requires_ray=probe)
        ReflACTTrainer(cfg, adapter).train()
    else:
        eval_only.get_adapter = probe
        eval_only.main()
except Configured:
    pass
else:
    raise AssertionError("model configuration probe was not reached")
assert len(calls) == 2
'''


@pytest.mark.parametrize("entry", ["train", "eval"])
@pytest.mark.parametrize("region", ["global_en", "cn_zh"])
@pytest.mark.parametrize("explicit,expected", [
    ("default", "MiniMax-M3.1-Flash"),
    ("provider", "MiniMax-M3"),
    ("saved", "MiniMax-M2.7"),
    ("cli", "saved-custom-model"),
])
def test_entrypoint_config_to_http(tmp_path, entry, region, explicit, expected):
    base = Path("configs/_base_/default.yaml").resolve()
    config = tmp_path / "config.yaml"
    extra = ""
    if explicit == "provider":
        extra = f"model:\n  minimax_model: {expected}\n"
    elif explicit == "saved":
        extra = f"model:\n  target: {expected}\n  optimizer: {expected}\n"
    config.write_text(f"_base_: {base}\n{extra}")
    result = subprocess.run(
        [sys.executable, "-c", _ENTRYPOINT_PROBE, entry, str(config), str(tmp_path / "output"),
         region, expected, explicit],
        capture_output=True, text=True, timeout=30,
        env={key: os.environ[key] for key in ("PATH", "HOME", "LANG", "SYSTEMROOT", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE") if key in os.environ},
    )
    assert result.returncode == 0, result.stdout + result.stderr
