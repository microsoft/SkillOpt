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
import locale
import os
from typing import Any, Dict, Optional


class StateFileError(RuntimeError):
    """An existing state.json could not be read, so it must not be overwritten.

    ``load()`` used to swallow every failure and hand back a fresh state.  The
    caller then ran a night against a reset night counter and empty harvest
    cursors, and the first ``save()`` replaced the file -- the original history
    and cross-night memory were gone with no warning.  Reporting the failure is
    what keeps the file intact; the message names the path and the fix.
    """


def _locale_codec() -> str:
    """The codec ``open()`` picks when no encoding is given.

    Writers before this change left the codec to the locale, so a state file
    produced on a GBK or cp1252 box holds those bytes.  Kept behind a function
    so the migration can be tested on a runner whose locale is UTF-8.
    """
    return locale.getpreferredencoding(False)


def _read_state(path: str) -> Dict[str, Any]:
    """Read state.json, migrating files written under the old locale codec.

    UTF-8 is tried first, so files written since this change are read directly.
    A file that is not valid UTF-8 falls back to the locale codec -- exactly
    what the old writer used on the machine that produced it -- and the next
    ``save()`` rewrites it as UTF-8, completing the migration on first write.

    Anything that decodes under neither is reported rather than discarded.
    """
    with open(path, "rb") as f:
        raw = f.read()

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as utf8_error:
        codec = _locale_codec()
        try:
            text = raw.decode(codec)
        except (UnicodeDecodeError, LookupError):
            raise StateFileError(
                f"{path} is not valid UTF-8 and does not decode as {codec!r} either, so it "
                f"cannot be read. It has been left untouched. Move or delete it to start "
                f"from a fresh state; it will not be overwritten while it is unreadable."
            ) from utf8_error

    try:
        data = json.loads(text)
    except ValueError as json_error:
        raise StateFileError(
            f"{path} could not be parsed as JSON ({json_error}). It has been left untouched. "
            f"Move or delete it to start from a fresh state; it will not be overwritten while "
            f"it is unreadable."
        ) from json_error

    return data if isinstance(data, dict) else {}


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
        if not os.path.exists(path):
            return cls(path, copy.deepcopy(DEFAULT_STATE))
        # Raises StateFileError rather than returning a fresh state: a state
        # that silently resets is indistinguishable from a first run, and the
        # save() that follows would overwrite the file it failed to read.
        data = _read_state(path)
        merged = copy.deepcopy(DEFAULT_STATE)
        merged.update(data)
        return cls(path, merged)

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
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
