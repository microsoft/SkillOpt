"""SkillOpt-Sleep — persistent cross-night state.

state.json lives in ~/.skillopt-sleep and is the "long-term" store that
turns nightly episodes into durable competence (the Agent-Sleep paper's
short-term -> long-term transfer). It records:

  - night counter
  - last harvest timestamp per project (so each night only sees new data)
  - cross-night "slow/meta" memory (lessons that persisted across nights)
  - per-night history (scores, accept/reject) for trend reporting
"""
from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict, Optional

from skillopt_sleep.types import TaskRecord


def _now_iso(clock: Optional[float] = None) -> str:
    # caller passes a timestamp; we avoid importing time at module import
    import time as _t
    return _t.strftime("%Y-%m-%dT%H:%M:%S", _t.localtime(clock if clock is not None else _t.time()))


DEFAULT_STATE: Dict[str, Any] = {
    "version": 1,
    "night": 0,
    "last_harvest": {},     # project -> iso timestamp of last harvested record
    "slow_memory": "",      # cross-night consolidated lessons (meta-skill analogue)
    "history": [],          # list of per-night summaries
    "task_archive": [],     # capped list of past mined tasks (for associative recall)
    "split_exposure": {},   # uncapped project -> task id -> least-held-out split
    "split_lineage": {},    # uncapped project -> child id -> parent ids
    "last_model_key": "",   # "backend::model" string used in the last successful night (F16)
    "last_model_key_format": 1,  # v1=config text; v2=resolved backend/model
}


class SleepState:
    def __init__(self, path: str, data: Optional[Dict[str, Any]] = None) -> None:
        self.path = path
        # DEFAULT_STATE contains mutable lists and dicts. A shallow copy makes
        # independent projects in one Python process share history, harvest
        # cursors, and recalled tasks until they are persisted.
        self.data = data if data is not None else copy.deepcopy(DEFAULT_STATE)

    # io ---------------------------------------------------------------------
    @classmethod
    def load(cls, path: str) -> "SleepState":
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                merged = copy.deepcopy(DEFAULT_STATE)
                merged.update(data if isinstance(data, dict) else {})
                return cls(path, merged)
            except Exception:
                pass
        return cls(path, copy.deepcopy(DEFAULT_STATE))

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    # accessors --------------------------------------------------------------
    @property
    def night(self) -> int:
        return int(self.data.get("night", 0))

    def last_harvest_for(self, project: str) -> Optional[str]:
        return self.data.get("last_harvest", {}).get(project)

    def set_last_harvest(self, project: str, iso_ts: str) -> None:
        self.data.setdefault("last_harvest", {})[project] = iso_ts

    @property
    def slow_memory(self) -> str:
        return str(self.data.get("slow_memory", ""))

    def set_slow_memory(self, content: str) -> None:
        self.data["slow_memory"] = content

    def begin_night(self, clock: Optional[float] = None) -> int:
        self.data["night"] = self.night + 1
        return self.night

    def record_night(self, summary: Dict[str, Any]) -> None:
        self.data.setdefault("history", []).append(summary)

    # ── task archive (associative-recall memory) ──────────────────────────
    def task_archive(self) -> list:
        """Past mined tasks as plain dicts (newest last)."""
        return list(self.data.get("task_archive", []))

    def add_to_archive(self, task_dicts: list, cap: int = 300) -> None:
        """Append tonight's tasks; keep only the most recent ``cap``."""
        arc = self.data.setdefault("task_archive", [])
        arc.extend(task_dicts)
        if len(arc) > cap:
            self.data["task_archive"] = arc[-cap:]

    def protect_splits(self, tasks: list[TaskRecord]) -> None:
        """Never promote previously exposed tasks or their lineage to holdout.

        Keep provenance independently of the capped recall archive. Import old
        archives on first use. Exposure is saved with the successful night's
        state, preserving the cycle's existing abort/checkpoint semantics.
        """
        import logging

        ranks = {"train": 0, "val": 1, "test": 2}
        aliases = {"replay": "train", "holdout": "val"}
        exposure = self.data.setdefault("split_exposure", {})
        lineage = self.data.setdefault("split_lineage", {})
        records = self.task_archive() + [t.to_dict() for t in tasks]
        for row in records:
            if row.get("derived_from"):
                parents = lineage.setdefault(row.get("project", ""), {}).setdefault(row["id"], [])
                if row["derived_from"] not in parents:
                    parents.append(row["derived_from"])
        # Propagate exposure through recorded parent/child links to a fixed
        # point, including siblings that occur earlier in this night's pool.
        changed = True
        while changed:
            changed = False
            for row in records:
                project = exposure.setdefault(row.get("project", ""), {})
                ids = [row["id"]] + ([row["derived_from"]] if row.get("derived_from") else [])
                split = aliases.get(row.get("split", "train"), row.get("split", "train"))
                rank = min([ranks[split]] + [ranks[project.get(i, "test")] for i in ids])
                if rank == 2:
                    continue
                for task_id in ids:
                    if rank < ranks[project.get(task_id, "test")]:
                        project[task_id] = ("train", "val")[rank]
                        changed = True
            for project_name, children in lineage.items():
                project = exposure.setdefault(project_name, {})
                for child, parents in children.items():
                    ids = [child] + parents
                    rank = min(ranks[project.get(i, "test")] for i in ids)
                    for task_id in ids:
                        if rank < ranks[project.get(task_id, "test")]:
                            project[task_id] = ("train", "val")[rank]
                            changed = True
        for task in tasks:
            previous = aliases.get(task.split, task.split)
            protected = exposure.get(task.project, {}).get(task.id, previous)
            if ranks[protected] < ranks[previous]:
                logging.getLogger("skillopt_sleep").warning(
                    "task %s retained as %s instead of %s due to persistent exposure; "
                    "held-out coverage was reduced", task.id, protected, previous,
                )
                task.split = protected
        # Check after restoring provenance: a test-hash task may now supply
        # train. Only a genuinely missing train pool exposes val to reflect.
        if tasks and not any(aliases.get(t.split, t.split) == "train" for t in tasks):
            fallback = [t for t in tasks if aliases.get(t.split, t.split) == "val"]
            if fallback:
                for task in fallback:
                    task.split = "train"
                self.protect_splits(tasks)

    # ── model-swap tracking (F16) ─────────────────────────────────────────
    @property
    def last_model_key(self) -> str:
        return str(self.data.get("last_model_key", ""))

    def set_last_model_key(self, key: str) -> None:
        self.data["last_model_key"] = key
        self.data["last_model_key_format"] = 2

    @property
    def last_model_key_format(self) -> int:
        try:
            return int(self.data.get("last_model_key_format", 1))
        except (TypeError, ValueError):
            return 1
