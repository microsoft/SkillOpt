"""Tests for skillopt.types — Edit and Patch dataclass serialization."""
from __future__ import annotations

from skillopt.types import Edit, Patch

# ── Edit ────────────────────────────────────────────────────────────────────


class TestEditCreation:
    """Edit dataclass construction."""

    def test_minimal_edit(self) -> None:
        e = Edit(op="append")
        assert e.op == "append"
        assert e.content == ""
        assert e.target == ""
        assert e.support_count is None
        assert e.source_type is None
        assert e.merge_level is None
        assert e.update_origin == ""
        assert e.update_target == ""

    def test_full_edit(self) -> None:
        e = Edit(
            op="replace",
            content="new content",
            target="old content",
            support_count=5,
            source_type="failure",
            merge_level=2,
            update_origin="reflect",
            update_target="skill",
        )
        assert e.op == "replace"
        assert e.content == "new content"
        assert e.target == "old content"
        assert e.support_count == 5
        assert e.source_type == "failure"
        assert e.merge_level == 2
        assert e.update_origin == "reflect"
        assert e.update_target == "skill"

    def test_insert_after_op(self) -> None:
        e = Edit(op="insert_after", content="insertion", target="anchor")
        assert e.op == "insert_after"
        assert e.content == "insertion"
        assert e.target == "anchor"

    def test_delete_op(self) -> None:
        e = Edit(op="delete", target="thing_to_remove")
        assert e.op == "delete"
        assert e.target == "thing_to_remove"


class TestEditRoundTrip:
    """Edit.to_dict() / Edit.from_dict() round-trip."""

    def test_round_trip_minimal(self) -> None:
        e = Edit(op="append")
        d = e.to_dict()
        restored = Edit.from_dict(d)
        assert restored == e

    def test_round_trip_full(self) -> None:
        e = Edit(
            op="replace",
            content="new content",
            target="old content",
            support_count=3,
            source_type="success",
            merge_level=1,
            update_origin="meta_reflect",
            update_target="system_prompt",
        )
        d = e.to_dict()
        restored = Edit.from_dict(d)
        assert restored == e

    def test_round_trip_delete_without_content(self) -> None:
        e = Edit(op="delete", target="obsolete_line")
        d = e.to_dict()
        restored = Edit.from_dict(d)
        assert restored == e

    def test_optional_fields_omitted_when_default(self) -> None:
        e = Edit(op="append")
        d = e.to_dict()
        assert d == {"op": "append", "content": ""}
        # support_count, source_type, etc. should be absent
        assert "support_count" not in d
        assert "source_type" not in d
        assert "merge_level" not in d
        assert "target" not in d
        assert "update_origin" not in d
        assert "update_target" not in d

    def test_from_dict_with_defaults(self) -> None:
        d = {"op": "replace", "content": "abc"}
        e = Edit.from_dict(d)
        assert e.op == "replace"
        assert e.content == "abc"
        assert e.target == ""
        assert e.support_count is None
        assert e.source_type is None

    def test_from_dict_with_extra_keys(self) -> None:
        """Extra keys in dict should be ignored."""
        d = {"op": "append", "content": "", "unknown_field": 42}
        e = Edit.from_dict(d)
        assert e.op == "append"
        assert not hasattr(e, "unknown_field")


class TestEditEdgeCases:
    """Edge cases around Edit."""

    def test_support_count_zero(self) -> None:
        """0 is a valid support_count and should be serialized."""
        e = Edit(op="append", support_count=0)
        d = e.to_dict()
        assert d["support_count"] == 0
        restored = Edit.from_dict(d)
        assert restored.support_count == 0

    def test_merge_level_zero(self) -> None:
        e = Edit(op="replace", merge_level=0)
        d = e.to_dict()
        assert d["merge_level"] == 0
        restored = Edit.from_dict(d)
        assert restored.merge_level == 0

    def test_empty_target_stays_empty(self) -> None:
        e = Edit(op="append", target="")
        d = e.to_dict()
        assert "target" not in d


# ── Patch ───────────────────────────────────────────────────────────────────


class TestPatchCreation:
    """Patch dataclass construction."""

    def test_empty_patch(self) -> None:
        p = Patch()
        assert p.edits == []
        assert p.reasoning == ""
        assert p.ranking_details is None

    def test_patch_with_edits(self) -> None:
        edits = [
            Edit(op="append", content="step 1"),
            Edit(op="append", content="step 2"),
        ]
        p = Patch(edits=edits, reasoning="Added two steps")
        assert len(p.edits) == 2
        assert p.reasoning == "Added two steps"

    def test_patch_with_ranking_details(self) -> None:
        p = Patch(ranking_details={"score": 0.95, "rank": 1})
        assert p.ranking_details == {"score": 0.95, "rank": 1}


class TestPatchRoundTrip:
    """Patch.to_dict() / Patch.from_dict() round-trip."""

    def test_round_trip_empty(self) -> None:
        p = Patch()
        d = p.to_dict()
        restored = Patch.from_dict(d)
        assert restored.edits == []
        assert restored.reasoning == ""
        assert restored.ranking_details is None

    def test_round_trip_with_edits(self) -> None:
        edits = [
            Edit(op="insert_after", content="new step", target="existing step"),
            Edit(op="replace", content="updated", target="old"),
        ]
        p = Patch(edits=edits, reasoning="Batch update")
        d = p.to_dict()
        restored = Patch.from_dict(d)
        assert len(restored.edits) == 2
        for original, restored_edit in zip(p.edits, restored.edits):
            assert isinstance(restored_edit, Edit)
            assert original == restored_edit
        assert restored.reasoning == "Batch update"
        assert restored.ranking_details is None

    def test_round_trip_with_ranking_details(self) -> None:
        details = {"strategy": "rouge", "scores": [0.9, 0.8, 0.7]}
        p = Patch(
            edits=[Edit(op="append", content="a")],
            reasoning="selected best",
            ranking_details=details,
        )
        d = p.to_dict()
        restored = Patch.from_dict(d)
        assert restored.ranking_details == details

    def test_to_dict_contains_reasoning_and_edits(self) -> None:
        p = Patch(edits=[Edit(op="append", content="test")], reasoning="reason")
        d = p.to_dict()
        assert "reasoning" in d
        assert "edits" in d
        assert isinstance(d["edits"], list)

    def test_from_dict_preserves_edit_order(self) -> None:
        edits = [
            Edit(op="append", content="first"),
            Edit(op="insert_after", content="second", target="first"),
            Edit(op="append", content="third"),
        ]
        p = Patch(edits=edits, reasoning="ordered")
        d = p.to_dict()
        restored = Patch.from_dict(d)
        assert restored.edits[0].content == "first"
        assert restored.edits[1].content == "second"
        assert restored.edits[2].content == "third"


class TestPatchEdgeCases:
    """Edge cases around Patch."""

    def test_reasoning_empty_string(self) -> None:
        p = Patch(reasoning="")
        d = p.to_dict()
        assert d["reasoning"] == ""

    def test_zero_edits(self) -> None:
        """Patch with explicitly empty edit list."""
        p = Patch(edits=[])
        d = p.to_dict()
        assert d["edits"] == []

    def test_nested_edit_from_dict_handles_dicts(self) -> None:
        """from_dict should accept dicts in the 'edits' list."""
        d = {
            "reasoning": "test",
            "edits": [{"op": "append", "content": "hello"}],
        }
        p = Patch.from_dict(d)
        assert len(p.edits) == 1
        assert isinstance(p.edits[0], Edit)
        assert p.edits[0].op == "append"
        assert p.edits[0].content == "hello"
