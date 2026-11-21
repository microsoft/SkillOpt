"""Tests for optimizer edit application, gradient clipping, and autonomous LR robustness."""
from __future__ import annotations

import pytest

from skillopt.optimizer.clip import rank_and_select
from skillopt.optimizer.lr_autonomous import decide_autonomous_learning_rate
from skillopt.optimizer.skill import (
    _apply_edit_with_report,
    _edit_fields,
    apply_edit,
    apply_patch,
    apply_patch_with_report,
)
from skillopt.types import Edit, Patch


class TestSkillEditRobustness:
    """Test robustness of edit and patch application in skill.py."""

    def test_edit_fields_none_handling(self) -> None:
        assert _edit_fields(None) == ("", "", "")
        assert _edit_fields({"op": None, "content": None, "target": None}) == ("", "", "")
        assert _edit_fields({"op": "delete", "target": "section", "content": None}) == (
            "delete",
            "",
            "section",
        )

    def test_apply_edit_with_report_none_edit(self) -> None:
        skill = "Initial skill text"
        new_skill, report = _apply_edit_with_report(skill, None)
        assert new_skill == skill
        assert report["status"] == "skipped_null_edit"

    def test_apply_edit_with_none_content_delete(self) -> None:
        skill = "Line 1\nLine 2 to remove\nLine 3"
        edit = {"op": "delete", "target": "Line 2 to remove\n", "content": None}
        new_skill = apply_edit(skill, edit)
        assert new_skill == "Line 1\nLine 3"

    def test_apply_edit_with_object_none_content(self) -> None:
        skill = "Heading\nTarget text"
        edit = Edit(op="delete", target="Target text", content=None)  # type: ignore[arg-type]
        new_skill = apply_edit(skill, edit)
        assert new_skill == "Heading\n"

    def test_apply_patch_with_report_none_patch(self) -> None:
        skill = "Base skill"
        new_skill, reports = apply_patch_with_report(skill, None)
        assert new_skill == skill
        assert reports == []

    def test_apply_patch_with_report_none_edits(self) -> None:
        skill = "Base skill"
        new_skill, reports = apply_patch_with_report(skill, {"edits": None})
        assert new_skill == skill
        assert reports == []

    def test_apply_patch_handles_none_elements_in_edits(self) -> None:
        skill = "Base skill"
        patch = {
            "edits": [
                None,
                {"op": "append", "content": "Added note"},
            ]
        }
        new_skill, reports = apply_patch_with_report(skill, patch)
        assert "Added note" in new_skill
        assert len(reports) == 2
        assert reports[0]["status"] == "skipped_null_edit"
        assert reports[1]["status"] == "applied_append"

    def test_apply_patch_with_patch_instance(self) -> None:
        skill = "Start"
        patch = Patch(edits=[Edit(op="append", content="Next")])
        new_skill = apply_patch(skill, patch)
        assert "Next" in new_skill


class TestClipRobustness:
    """Test robustness of gradient clipping (rank_and_select) in clip.py."""

    def test_rank_and_select_zero_budget_no_llm_call(self) -> None:
        patch = {
            "reasoning": "Initial batch",
            "edits": [{"op": "append", "content": "1"}, {"op": "append", "content": "2"}],
        }
        res = rank_and_select("skill doc", patch, max_edits=0)
        assert res["edits"] == []
        assert "truncated 2->0 edits" in res["reasoning"]

    def test_rank_and_select_negative_budget(self) -> None:
        patch = {
            "reasoning": "Initial batch",
            "edits": [{"op": "append", "content": "1"}, {"op": "append", "content": "2"}],
        }
        res = rank_and_select("skill doc", patch, max_edits=-2)
        assert res["edits"] == []
        assert "budget=-2" in res["reasoning"]

    def test_rank_and_select_none_patch(self) -> None:
        res = rank_and_select("skill doc", None, max_edits=5)
        assert res == {}

    def test_rank_and_select_duck_typed_patch(self) -> None:
        patch = Patch(
            edits=[Edit(op="append", content="1"), Edit(op="append", content="2")],
            reasoning="Testing Patch dataclass",
        )
        res = rank_and_select("skill doc", patch, max_edits=0)
        assert res["edits"] == []

    def test_rank_and_select_within_budget(self) -> None:
        patch = {
            "reasoning": "keep all",
            "edits": [{"op": "append", "content": "1"}],
        }
        res = rank_and_select("skill doc", patch, max_edits=5)
        assert res == patch

    def test_rank_and_select_fallback_negative_budget(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def mock_chat(*args: object, **kwargs: object) -> tuple[str, dict]:
            raise RuntimeError("LLM unavailable")

        monkeypatch.setattr("skillopt.optimizer.clip.chat_optimizer", mock_chat)

        patch = {
            "reasoning": "failover",
            "edits": [{"op": "append", "content": "1"}, {"op": "append", "content": "2"}],
        }
        # Even if max_edits is negative, fallback should safely clamp to 0 rather than negative slice
        res = rank_and_select("skill doc", patch, max_edits=-1)
        assert res["edits"] == []
        assert "truncated 2->0" in res["reasoning"]

    def test_rank_and_select_rejects_boolean_indices(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import json

        def mock_chat(*args: object, **kwargs: object) -> tuple[str, dict]:
            # LLM returns boolean values in selected_indices
            return json.dumps({"selected_indices": [True, False, 1]}), {}

        monkeypatch.setattr("skillopt.optimizer.clip.chat_optimizer", mock_chat)

        patch = {
            "reasoning": "boolean test",
            "edits": [
                {"op": "append", "content": "item0"},
                {"op": "append", "content": "item1"},
            ],
        }
        res = rank_and_select("skill doc", patch, max_edits=1)
        # item at index 1 is selected; booleans True and False are rejected
        assert res["edits"] == [{"op": "append", "content": "item1"}]


class TestLrAutonomousRobustness:
    """Test robustness of decide_autonomous_learning_rate with None values."""

    def test_decide_autonomous_learning_rate_none_metrics(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured_user = {}

        def mock_chat(*args: object, **kwargs: object) -> tuple[str, dict]:
            captured_user["prompt"] = kwargs.get("user", "")
            return '{"learning_rate": 2}', {}

        monkeypatch.setattr("skillopt.optimizer.lr_autonomous.chat_optimizer", mock_chat)

        res = decide_autonomous_learning_rate(
            skill_content="My skill",
            merged_patch={"edits": [{"op": "append", "content": "e1"}]},
            update_mode="patch",
            rollout_hard=None,  # type: ignore[arg-type]
            rollout_soft=None,  # type: ignore[arg-type]
            rollout_n=None,  # type: ignore[arg-type]
        )
        assert res["learning_rate"] == 1  # clamped to available item count (1)
        assert "rollout_hard=0.000000" in captured_user["prompt"]
        assert "rollout_soft=0.000000" in captured_user["prompt"]
        assert "rollout_n=0" in captured_user["prompt"]
