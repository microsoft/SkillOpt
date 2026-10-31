"""Library hook for routing optimizer calls through a caller-supplied function."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

import skillopt.model as model
from skillopt.engine.trainer import ReflACTTrainer
from skillopt.envs.base import EnvAdapter
from skillopt.gradient.aggregate import merge_patches
from skillopt.gradient.reflect import run_error_analyst_minibatch
from skillopt.model import azure_openai, backend_config
from skillopt.optimizer.clip import rank_and_select
from skillopt.optimizer.lr_autonomous import decide_autonomous_learning_rate
from skillopt.optimizer.meta_skill import run_meta_skill
from skillopt.optimizer.rewrite import rewrite_skill_from_suggestions
from skillopt.optimizer.skill_aware import consolidate_appendix_notes
from skillopt.optimizer.slow_update import run_slow_update

# One canned optimizer reply per stage of a patch-mode run.
_STAGE_REPLIES: dict[str, dict[str, Any]] = {
    "analyst": {"patch": {"reasoning": "r", "edits": [{"op": "append", "content": "Check units."}]}},
    "merge": {
        "reasoning": "m",
        "edits": [
            {"op": "append", "content": "Check units."},
            {"op": "append", "content": "Cite the source."},
        ],
    },
    "lr_autonomous": {"learning_rate": 1, "reasoning": "one edit"},
    "ranking": {"selected_indices": [0]},
    "slow_update": {"reasoning": "s", "slow_update_content": "Prefer verified answers."},
    "meta_skill": {"reasoning": "k", "meta_skill_content": "Keep edits small."},
}


class _RecordingChat:
    """Optimizer function with the ``chat_optimizer_messages`` signature."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, messages: list[dict[str, Any]], **kwargs: Any) -> tuple[str, dict[str, int]]:
        self.calls.append({"messages": messages, **kwargs})
        reply = json.dumps(_STAGE_REPLIES[kwargs["stage"]])
        return reply, {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}

    @property
    def stages(self) -> set[str]:
        return {call["stage"] for call in self.calls}


def _failing_chat(messages: list[dict[str, Any]], **kwargs: Any) -> tuple[str, dict[str, int]]:
    raise RuntimeError("optimizer client unavailable")


class _SyntheticAdapter(EnvAdapter):
    """Two-item environment: item 1 always passes; item 0 passes once the
    skill contains the analyst's "Check units." edit."""

    def setup(self, cfg: dict) -> None:
        super().setup(cfg)
        self.analyst_workers = 1
        self.failure_only = False
        self.minibatch_size = 8
        self.edit_budget = cfg["edit_budget"]

    def build_train_env(self, batch_size: int, seed: int, **kwargs: Any) -> list[dict]:
        return [{"id": f"item{i}", "task_description": f"task {i}"} for i in range(2)]

    def build_eval_env(self, env_num: int, split: str, seed: int, **kwargs: Any) -> list[dict]:
        return self.build_train_env(env_num, seed)

    def rollout(self, env_manager: list[dict], skill_content: str, out_dir: str, **kwargs: Any) -> list[dict]:
        results = []
        for index, item in enumerate(env_manager):
            prediction_dir = os.path.join(out_dir, "predictions", item["id"])
            os.makedirs(prediction_dir, exist_ok=True)
            with open(os.path.join(prediction_dir, "conversation.json"), "w") as f:
                json.dump([{"role": "assistant", "content": f"answer {index}"}], f)
            passed = index == 1 or "Check units." in skill_content
            results.append({**item, "hard": int(passed), "soft": float(passed)})
        return results

    def get_task_types(self) -> list[str]:
        return ["synthetic"]


@pytest.fixture(autouse=True)
def isolated_model_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    saved_environ = os.environ.copy()
    for name in ("OPTIMIZER_BACKEND", "TARGET_BACKEND"):
        monkeypatch.setattr(backend_config, name, getattr(backend_config, name))
    model.reset_token_tracker()
    yield
    model.reset_optimizer_chat()
    model.set_strict_optimizer_errors(False)
    model.reset_token_tracker()
    os.environ.clear()
    os.environ.update(saved_environ)


@pytest.fixture
def provider_clients(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record any attempt to build (and authenticate) a built-in Azure/OpenAI client."""
    built: list[str] = []

    def record_client(role: str) -> None:
        built.append(role)
        raise AssertionError(f"built-in {role} client must not be created")

    monkeypatch.setattr(azure_openai, "_make_client", record_client)
    return built


def _train_cfg(out_root: str, skill_path: str, **overrides: Any) -> dict[str, Any]:
    cfg = {
        "optimizer_backend": "custom",
        "skill_init": skill_path,
        "num_epochs": 2,
        "train_size": 2,
        "batch_size": 2,
        "accumulation": 1,
        "seed": 7,
        "merge_batch_size": 2,
        "edit_budget": 2,
        "lr_control_mode": "autonomous",
        "use_slow_update": True,
        "slow_update_samples": 2,
        "use_meta_skill": True,
        "analyst_workers": 1,
        "sel_env_num": 2,
        "test_env_num": 2,
        "eval_test": False,
        "out_root": out_root,
    }
    cfg.update(overrides)
    return cfg


@pytest.fixture
def skill_path(tmp_path: Path) -> str:
    path = tmp_path / "skill.md"
    path.write_text("# Synthetic skill\n", encoding="utf-8")
    return str(path)


def test_chat_optimizer_sends_system_and_user_messages_to_registered_function() -> None:
    chat = _RecordingChat()
    model.set_optimizer_chat(chat)

    text, usage = model.chat_optimizer(
        "system prompt", "user prompt", max_completion_tokens=99, retries=2, stage="ranking", timeout=5,
    )

    assert json.loads(text) == _STAGE_REPLIES["ranking"]
    assert usage["total_tokens"] == 10
    assert chat.calls == [
        {
            "messages": [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "user prompt"},
            ],
            "max_completion_tokens": 99,
            "retries": 2,
            "stage": "ranking",
            "reasoning_effort": None,
            "tools": None,
            "tool_choice": None,
            "return_message": False,
            "timeout": 5,
        }
    ]
    assert model.get_token_summary()["ranking"] == {
        "calls": 1,
        "prompt_tokens": 7,
        "completion_tokens": 3,
        "total_tokens": 10,
    }


def test_chat_optimizer_messages_passes_tool_options_to_registered_function() -> None:
    chat = _RecordingChat()
    model.set_optimizer_chat(chat)
    messages = [{"role": "user", "content": "hi"}]
    tools = [{"type": "function", "function": {"name": "lookup"}}]

    model.chat_optimizer_messages(
        messages, stage="merge", tools=tools, tool_choice="auto", return_message=True,
    )

    assert chat.calls[0]["messages"] is messages
    assert chat.calls[0]["tools"] is tools
    assert chat.calls[0]["tool_choice"] == "auto"
    assert chat.calls[0]["return_message"] is True


def test_modules_that_imported_chat_optimizer_by_name_use_registered_function() -> None:
    chat = _RecordingChat()
    model.set_optimizer_chat(chat)
    patch = {"reasoning": "", "edits": [{"op": "append", "content": f"edit {i}"} for i in range(3)]}

    ranked = rank_and_select("# Skill\n", patch, max_edits=1)

    assert chat.stages == {"ranking"}
    assert ranked["edits"] == [{"op": "append", "content": "edit 0"}]


def test_builtin_backend_dispatch_is_unchanged_without_registered_function(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builtin_calls: list[dict[str, Any]] = []

    def fake_azure_chat(**kwargs: Any) -> tuple[str, dict[str, int]]:
        builtin_calls.append(kwargs)
        return "azure reply", {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}

    monkeypatch.setattr(azure_openai, "chat_optimizer", fake_azure_chat)
    backend_config.set_optimizer_backend("openai_chat")
    model.set_optimizer_chat(_RecordingChat())
    model.reset_optimizer_chat()

    text, _ = model.chat_optimizer("system", "user", stage="analyst")

    assert text == "azure reply"
    assert builtin_calls[0]["system"] == "system"
    assert builtin_calls[0]["stage"] == "analyst"


def test_custom_backend_without_registered_function_fails_clearly(skill_path: str, tmp_path: Path) -> None:
    backend_config.set_optimizer_backend("custom")

    with pytest.raises(RuntimeError, match="set_optimizer_chat"):
        model.chat_optimizer("system", "user")
    with pytest.raises(ValueError, match="set_optimizer_chat"):
        ReflACTTrainer(_train_cfg(str(tmp_path / "run"), skill_path), _SyntheticAdapter()).train()


def test_training_run_sends_every_optimizer_stage_to_registered_function(
    provider_clients: list[str], skill_path: str, tmp_path: Path,
) -> None:
    chat = _RecordingChat()
    model.set_optimizer_chat(chat)
    out_root = tmp_path / "run"

    summary = ReflACTTrainer(_train_cfg(str(out_root), skill_path), _SyntheticAdapter()).train()

    assert chat.stages == {"analyst", "merge", "lr_autonomous", "ranking", "slow_update", "meta_skill"}
    assert provider_clients == []
    assert summary["config"]["optimizer_backend"] == "custom"
    assert summary["token_summary"]["_total"]["calls"] == len(chat.calls)
    assert "Check units." in (out_root / "skills" / "skill_v0001.md").read_text()


@pytest.mark.parametrize(
    "run_stage",
    [
        pytest.param(
            lambda: run_error_analyst_minibatch("# Skill\n", [{"id": "a"}], "unused"),
            id="reflect",
        ),
        pytest.param(
            lambda: merge_patches(
                "# Skill\n",
                [{"edits": [{"op": "append", "content": "a"}]}],
                [{"edits": [{"op": "append", "content": "b"}]}],
                verbose=False,
            ),
            id="merge",
        ),
        pytest.param(
            lambda: rank_and_select(
                "# Skill\n", {"edits": [{"op": "append", "content": "a"}] * 2}, max_edits=1,
            ),
            id="rank",
        ),
        pytest.param(
            lambda: decide_autonomous_learning_rate(
                skill_content="# Skill\n",
                merged_patch={"edits": [{"op": "append", "content": "a"}]},
                update_mode="patch",
                rollout_hard=0.0,
                rollout_soft=0.0,
                rollout_n=1,
            ),
            id="lr_autonomous",
        ),
        pytest.param(
            lambda: rewrite_skill_from_suggestions("# Skill\n", {"revise_suggestions": [{"suggestion": "a"}]}),
            id="rewrite",
        ),
        pytest.param(lambda: run_slow_update("# Skill\n", [], [], [], comparison_pairs=[]), id="slow_update"),
        pytest.param(lambda: run_meta_skill("# Before\n", "# After\n", []), id="meta_skill"),
        pytest.param(
            lambda: consolidate_appendix_notes(["note a", "note b"], chat_fn=model.chat_optimizer),
            id="appendix_consolidate",
        ),
    ],
)
def test_strict_errors_propagate_optimizer_failures(
    monkeypatch: pytest.MonkeyPatch, run_stage: Callable[[], object],
) -> None:
    # The reflect stage only calls the optimizer when it has a trajectory to show.
    monkeypatch.setattr("skillopt.gradient.reflect.fmt_minibatch_trajectories", lambda *_: "trajectory")
    model.set_optimizer_chat(_failing_chat)
    model.set_strict_optimizer_errors(True)

    with pytest.raises(RuntimeError, match="optimizer client unavailable"):
        run_stage()


def test_optimizer_failures_fall_back_by_default() -> None:
    model.set_optimizer_chat(_failing_chat)
    patch = {"reasoning": "", "edits": [{"op": "append", "content": "a"}] * 2}

    ranked = rank_and_select("# Skill\n", patch, max_edits=1)

    assert "fallback truncated" in ranked["reasoning"]


def test_strict_errors_config_stops_training_on_optimizer_failure(
    provider_clients: list[str], skill_path: str, tmp_path: Path,
) -> None:
    model.set_optimizer_chat(_failing_chat)
    cfg = _train_cfg(str(tmp_path / "run"), skill_path, optimizer_strict_errors=True)

    with pytest.raises(RuntimeError, match="optimizer client unavailable"):
        ReflACTTrainer(cfg, _SyntheticAdapter()).train()

    assert provider_clients == []
