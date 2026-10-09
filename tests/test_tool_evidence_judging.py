"""A measured tool route must not be overridden by a self-reported marker.

Tool-loop backends detect real calls (for example from a shim's call log) and
return an empty ``tools_called`` when the agent never ran the tool. The rule
judge then fell back to a ``TOOL_CALL: <name>`` regex over the response text
and passed the check anyway, so a skill earned full credit for claiming a tool
call it never made.
"""
from __future__ import annotations

import subprocess
from unittest import mock

from skillopt_sleep.backend import Backend, ClaudeCliBackend, MockBackend, _optimizer_feedback
from skillopt_sleep.judges import score_rule_judge
from skillopt_sleep.replay import replay_one
from skillopt_sleep.types import TaskRecord

TOOL_JUDGE = {"kind": "rule", "checks": [{"op": "tool_called", "arg": "search"}]}
SELF_REPORT = "TOOL_CALL: search\nWe decided on Redis."


def _tool_task() -> TaskRecord:
    return TaskRecord(
        id="tool-task",
        project="/project",
        intent="What did we decide about caching?",
        reference_kind="rule",
        judge=dict(TOOL_JUDGE),
    )


class _ToolLoopBackend(MockBackend):
    """A real tool loop: it reports exactly the calls it observed."""

    def __init__(self, observed):
        self.observed = list(observed)

    def attempt_with_tools(self, task, skill, memory, tools):
        return SELF_REPORT, list(self.observed)


class _MarkerConventionBackend(Backend):
    """No tool loop: inherits the default marker-to-call conversion."""

    name = "marker"

    def attempt(self, task, skill, memory, sample_id=0):
        return SELF_REPORT

    def judge(self, task, response):  # pragma: no cover - rule tasks score locally
        raise AssertionError("rule judges are scored locally")


def test_claude_tool_route_does_not_credit_a_self_reported_call() -> None:
    def fake_run(cmd, **kwargs):
        # The agent prints the marker but never executes ./search, so the
        # shim's call log stays empty.
        return subprocess.CompletedProcess(cmd, 0, stdout=SELF_REPORT, stderr="")

    with mock.patch("skillopt_sleep.backend.subprocess.run", side_effect=fake_run):
        result = replay_one(ClaudeCliBackend(claude_path="claude"), _tool_task(), "skill", "")
    assert result.tools_called == []
    assert result.hard == 0.0
    assert "tool_called=search" in result.judge_rationale


def test_measured_empty_call_list_fails_the_tool_check() -> None:
    result = replay_one(_ToolLoopBackend(observed=[]), _tool_task(), "", "")
    assert result.hard == 0.0
    assert result.optimizer_feedback == (
        "Actually call the 'search' tool while completing the task."
    )


def test_measured_call_still_passes() -> None:
    result = replay_one(_ToolLoopBackend(observed=["search"]), _tool_task(), "", "")
    assert result.hard == 1.0


def test_marker_convention_backends_keep_scoring_through_the_default_route() -> None:
    # The inherited attempt_with_tools converts the marker into a call, so
    # single-shot backends that rely on the documented convention are unchanged.
    result = replay_one(_MarkerConventionBackend(), _tool_task(), "", "")
    assert result.tools_called == ["search"]
    assert result.hard == 1.0


def test_unmeasured_single_shot_judging_keeps_the_marker_approximation() -> None:
    assert score_rule_judge(TOOL_JUDGE, SELF_REPORT)[0] == 1.0
    assert score_rule_judge(TOOL_JUDGE, SELF_REPORT, [], verified_tools=True)[0] == 0.0
    assert score_rule_judge(
        TOOL_JUDGE, SELF_REPORT, ["search"], verified_tools=True
    )[0] == 1.0


def test_optimizer_feedback_for_a_self_report_asks_for_a_real_call() -> None:
    from skillopt_sleep.types import ReplayResult

    result = ReplayResult(
        id="tool-task", hard=0.0, soft=0.0, response=SELF_REPORT, tools_called=[]
    )
    assert _optimizer_feedback(_tool_task(), result) == (
        "Actually call the 'search' tool while completing the task."
    )
