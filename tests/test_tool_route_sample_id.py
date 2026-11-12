"""Repeated rollouts of a tool task must reach the backend as separate samples.

``replay_one`` forwarded ``sample_id`` on the text route only. On the tool
route it called ``attempt_with_tools`` without it, and the inherited marker
fallback called ``attempt`` at sample zero, so K rollouts of a ``tool_called``
task were served from one attempt-cache entry.

Only the provider child process (``subprocess.run``) is faked. No model is
called.
"""
from __future__ import annotations

import inspect
import itertools
import subprocess
from unittest import mock

import pytest

from skillopt_sleep import backend as backend_module
from skillopt_sleep.backend import DualBackend, MockBackend, PiCliBackend
from skillopt_sleep.replay import replay_one
from skillopt_sleep.rollout import multi_rollout
from skillopt_sleep.types import TaskRecord

_TOOL_CHECK = {"op": "tool_called", "arg": "search"}
_TEXT_CHECK = {"op": "contains", "arg": "TOOL_CALL"}


def _rule_task(check: dict, task_id: str = "tool-task") -> TaskRecord:
    return TaskRecord(
        id=task_id,
        project="/project",
        intent="Please look it up",
        reference_kind="rule",
        judge={"kind": "rule", "checks": [check]},
    )


def _alternating_provider():
    """Fake Pi child: calls the tool on even calls, answers without it on odd."""
    counter = itertools.count()

    def run(cmd, *, input, **kwargs):
        if next(counter) % 2 == 0:
            out = "TOOL_CALL: search\nanswer"
        else:
            out = "answer without the tool"
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    return run


@pytest.mark.parametrize(
    "check", [_TOOL_CHECK, _TEXT_CHECK], ids=["tool-route", "text-route-control"]
)
def test_dream_rollouts_reach_the_provider_as_separate_samples(check) -> None:
    """Four rollouts are four provider calls on both routes.

    The text route already behaved this way and is the control. The tool route
    made one provider call and reported four identical scores.
    """
    with mock.patch(
        "skillopt_sleep.backend.subprocess.run", side_effect=_alternating_provider()
    ) as run:
        rollout_set = multi_rollout(
            PiCliBackend(), _rule_task(check), "skill", "", k=4, workers=1
        )
    assert run.call_count == 4
    assert [attempt.hard for attempt in rollout_set.attempts] == [1.0, 0.0, 1.0, 0.0]


def test_inherited_tool_fallback_keys_the_attempt_cache_by_sample_id() -> None:
    """Three distinct sample ids are three provider calls; a repeated id is a
    cache hit, so re-scoring an already-sampled rollout stays free."""
    backend = PiCliBackend()

    def run(cmd, *, input, **kwargs):
        return subprocess.CompletedProcess(
            cmd, 0, stdout="TOOL_CALL: search\nanswer", stderr=""
        )

    with mock.patch("skillopt_sleep.backend.subprocess.run", side_effect=run) as run_mock:
        for sample_id in (0, 1, 2, 0, 1):
            result = replay_one(
                backend, _rule_task(_TOOL_CHECK), "skill", "", sample_id=sample_id
            )
            assert result.tools_called == ["search"]
            assert result.hard == 1.0
    assert run_mock.call_count == 3


def test_mock_tool_model_forwards_sample_id_to_attempt() -> None:
    seen: list[int] = []

    class _Recording(MockBackend):
        def attempt(self, task, skill, memory, sample_id=0):
            seen.append(sample_id)
            return "answer"

    backend = _Recording()
    for sample_id in range(3):
        replay_one(backend, _rule_task(_TOOL_CHECK), "use search", "", sample_id=sample_id)
    assert seen == [0, 1, 2]


def test_dual_backend_forwards_tool_sample_ids_to_the_target() -> None:
    seen: list[int] = []

    class _Target(MockBackend):
        def attempt(self, task, skill, memory, sample_id=0):
            seen.append(sample_id)
            return "TOOL_CALL: search"

    dual = DualBackend(_Target(), MockBackend())
    for sample_id in range(3):
        replay_one(dual, _rule_task(_TOOL_CHECK), "", "", sample_id=sample_id)
    assert seen == [0, 1, 2]


def test_sample_zero_keeps_the_four_argument_tool_call() -> None:
    """Sample zero passes no ``sample_id`` keyword, so the default gate path
    calls an override exactly as it did before; later samples pass it."""
    calls: list[dict] = []

    class _Recording(MockBackend):
        def attempt_with_tools(self, task, skill, memory, tools, **kwargs):
            calls.append(dict(kwargs))
            return "answer", list(tools)

    backend = _Recording()
    for sample_id in (0, 2):
        replay_one(backend, _rule_task(_TOOL_CHECK), "", "", sample_id=sample_id)
    assert calls == [{}, {"sample_id": 2}]


def test_tool_override_without_sample_id_parameter_is_still_called() -> None:
    """A custom backend whose tool override predates ``sample_id`` keeps
    working for every sample. It is called with its original four arguments,
    so its repeated rollouts are whatever that override returns."""
    calls: list[int] = []

    class _Legacy(MockBackend):
        def attempt_with_tools(self, task, skill, memory, tools):
            calls.append(len(calls))
            return "answer", list(tools)

    backend = _Legacy()
    for sample_id in range(3):
        result = replay_one(
            backend, _rule_task(_TOOL_CHECK), "", "", sample_id=sample_id
        )
        assert result.hard == 1.0
    assert calls == [0, 1, 2]

    dual = DualBackend(_Legacy(), MockBackend())
    assert replay_one(dual, _rule_task(_TOOL_CHECK), "", "", sample_id=1).hard == 1.0


def test_attempt_override_without_sample_id_parameter_is_still_called() -> None:
    """The inherited tool fallback does not pass ``sample_id`` to an
    ``attempt`` override that cannot accept it."""

    class _LegacyAttempt(backend_module.Backend):
        def attempt(self, task, skill, memory):  # no sample_id parameter
            return "TOOL_CALL: search"

    response, called = _LegacyAttempt().attempt_with_tools(
        _rule_task(_TOOL_CHECK), "", "", ["search"], sample_id=3
    )
    assert (response, called) == ("TOOL_CALL: search", ["search"])


@pytest.mark.parametrize(
    "cls",
    [
        backend_module.Backend,
        backend_module.MockBackend,
        backend_module.PiCliBackend,
        backend_module.ClaudeCliBackend,
        backend_module.OpenCodeCliBackend,
        backend_module.CodexCliBackend,
        backend_module.CopilotCliBackend,
        backend_module.CursorCliBackend,
        backend_module.DualBackend,
        backend_module.AzureOpenAIBackend,
        backend_module.AzureResponsesBackend,
    ],
    ids=lambda cls: cls.__name__,
)
def test_every_shipped_tool_route_accepts_sample_id(cls) -> None:
    """Signature contract only: no CLI, network, or credential is touched."""
    parameter = inspect.signature(cls.attempt_with_tools).parameters["sample_id"]
    assert parameter.default == 0
