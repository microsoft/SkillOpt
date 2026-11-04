"""Unmatched edits must reach the user-visible report, not just the evidence log.

Collecting them in ConsolidationResult is not enough: the nightly artifact a
user actually reads is report.md, so an edit that changed nothing has to be
visible there too.
"""
import unittest

from skillopt_sleep.config import load_config
from skillopt_sleep.cycle import _render_report_md
from skillopt_sleep.types import EditRecord, SleepReport


def _report(**kw) -> SleepReport:
    base = dict(night=1, project="/tmp/proj", n_sessions=0, n_tasks=3, n_replayed=3,
                baseline_score=0.5, candidate_score=0.5, accepted=False, gate_action="reject")
    base.update(kw)
    return SleepReport(**base)


class TestUnmatchedEditsInReport(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_config()

    def test_unmatched_section_is_rendered(self) -> None:
        rep = _report(unmatched_edits=[
            EditRecord(target="skill", op="replace", content="new wording",
                       anchor="a line that is not there")])
        md = _render_report_md(rep, self.cfg)
        self.assertIn("changed nothing", md)
        self.assertIn("new wording", md)
        self.assertIn("a line that is not there", md)

    def test_section_absent_when_nothing_is_unmatched(self) -> None:
        md = _render_report_md(_report(), self.cfg)
        self.assertNotIn("changed nothing", md)

    def test_unmatched_does_not_masquerade_as_accepted(self) -> None:
        rep = _report(
            edits=[EditRecord(target="skill", op="add", content="a real rule")],
            unmatched_edits=[EditRecord(target="skill", op="delete", anchor="ghost")])
        md = _render_report_md(rep, self.cfg)
        accepted_at = md.index("## Accepted edits")
        unmatched_at = md.index("## Proposed but changed nothing")
        self.assertLess(accepted_at, unmatched_at)
        # the real rule is listed once, under Accepted
        self.assertEqual(md.count("a real rule"), 1)

    def test_report_dict_round_trips_unmatched(self) -> None:
        rep = _report(unmatched_edits=[EditRecord(target="memory", op="add", content="x")])
        d = rep.to_dict()
        self.assertEqual(len(d["unmatched_edits"]), 1)
        self.assertEqual(d["unmatched_edits"][0]["target"], "memory")

    def test_staged_report_renders_and_serializes_edit_provenance(self) -> None:
        edit = EditRecord(
            target="skill", op="add", content="new rule",
            source_task_ids=["task-1", "task-2"],
            source_session_ids=["session-a"],
        )
        rep = _report(edits=[edit], rejected_edits=[edit])

        md = _render_report_md(rep, self.cfg)
        payload = rep.to_dict()

        self.assertIn("source task IDs: `task-1`, `task-2`", md)
        self.assertIn("source session IDs: `session-a`", md)
        self.assertEqual(payload["edits"][0]["source_task_ids"], ["task-1", "task-2"])
        self.assertEqual(payload["rejected_edits"][0]["source_session_ids"], ["session-a"])


if __name__ == "__main__":
    unittest.main()
