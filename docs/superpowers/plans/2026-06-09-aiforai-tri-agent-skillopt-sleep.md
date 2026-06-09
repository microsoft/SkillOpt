# AIForAI Tri-Agent SkillOpt-Sleep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Phase 1/2 of AIForAI tri-agent SkillOpt-Sleep: read Codex, Claude, and CodeWhale trajectories; mine checkable AIForAI tasks; run a mock gated protected-block update; stage or adopt proposals safely.

**Architecture:** Add an `skillopt_sleep.aiforai` extension package that reuses existing SkillOpt-Sleep primitives where possible and keeps AIForAI-specific behavior isolated. Phase 1 delivers read-only audit. Phase 2 adds protected-block skill editing, curated regression checks, mock replay/gate, staging, and explicit adopt.

**Tech Stack:** Python 3.10+ stdlib, `unittest`, `sqlite3`, existing `skillopt_sleep` dataclasses/backends/judges/gate patterns.

---

## File Structure

Create:

- `skillopt_sleep/aiforai/__init__.py`: package marker and public version string.
- `skillopt_sleep/aiforai/__main__.py`: `python -m skillopt_sleep.aiforai` entrypoint.
- `skillopt_sleep/aiforai/types.py`: AIForAI-specific dataclasses and JSON helpers.
- `skillopt_sleep/aiforai/config.py`: CLI/config dataclass with default local paths.
- `skillopt_sleep/aiforai/harvesters/__init__.py`: exports harvester registry.
- `skillopt_sleep/aiforai/harvesters/base.py`: common JSON/JSONL/time/redaction utilities and abstract interface.
- `skillopt_sleep/aiforai/harvesters/codex.py`: parses Codex `state_5.sqlite` and rollout JSONL.
- `skillopt_sleep/aiforai/harvesters/claude.py`: wraps existing Claude transcript parser into AIForAI session schema.
- `skillopt_sleep/aiforai/harvesters/codewhale.py`: parses CodeWhale runtime files and DeepSeek session/log files.
- `skillopt_sleep/aiforai/mine.py`: AIForAI relevance classification, checkable task generation, source-stratified splitting.
- `skillopt_sleep/aiforai/regression_suite.py`: curated AIForAI behavior tasks.
- `skillopt_sleep/aiforai/skill_adapter.py`: protected learned block, AIForAI validators, staging/adopt helpers.
- `skillopt_sleep/aiforai/replay.py`: mock replay, rule scoring, slice metrics, gate decision.
- `skillopt_sleep/aiforai/report.py`: audit and run report writers.
- `skillopt_sleep/aiforai/run.py`: orchestration for `audit`, `run`, and `adopt`.
- `skillopt_sleep/aiforai/cli.py`: argument parser.
- `tests/test_aiforai_types.py`
- `tests/test_aiforai_harvesters.py`
- `tests/test_aiforai_mine.py`
- `tests/test_aiforai_skill_adapter.py`
- `tests/test_aiforai_run.py`

Modify:

- No existing production files outside the new package should be required.
- No AIForAI repo files should be modified by tests except inside temporary directories.

---

### Task 1: AIForAI Types And Config

**Files:**
- Create: `skillopt_sleep/aiforai/__init__.py`
- Create: `skillopt_sleep/aiforai/__main__.py`
- Create: `skillopt_sleep/aiforai/types.py`
- Create: `skillopt_sleep/aiforai/config.py`
- Create: `tests/test_aiforai_types.py`

- [ ] **Step 1: Write failing tests for dataclass round-trips**

Create `tests/test_aiforai_types.py`:

```python
from __future__ import annotations

import unittest

from skillopt_sleep.aiforai.config import AiforaiConfig
from skillopt_sleep.aiforai.types import (
    AiforaiSessionDigest,
    AiforaiTaskRecord,
    AiforaiRunResult,
)


class AiforaiTypesTests(unittest.TestCase):
    def test_session_round_trip_preserves_source(self) -> None:
        session = AiforaiSessionDigest(
            source_agent="codex",
            session_id="s1",
            raw_path="/tmp/session.jsonl",
            cwd="/repo",
            user_prompts=["train a model"],
            assistant_finals=["done"],
            tools_used=["exec"],
            skill_mentions=["ai-model-rd-protocol"],
            parse_warnings=["ignored record"],
        )

        restored = AiforaiSessionDigest.from_dict(session.to_dict())

        self.assertEqual(restored.source_agent, "codex")
        self.assertEqual(restored.session_id, "s1")
        self.assertEqual(restored.user_prompts, ["train a model"])
        self.assertEqual(restored.parse_warnings, ["ignored record"])

    def test_task_round_trip_preserves_judge(self) -> None:
        task = AiforaiTaskRecord(
            id="t1",
            source_agent="claude",
            source_sessions=["s1"],
            project="/repo",
            intent="prepare training contract",
            context_excerpt="ctx",
            task_family="training_contract",
            outcome="fail",
            split="val",
            judge={"kind": "rule", "checks": [{"op": "contains", "arg": "training contract"}]},
        )

        restored = AiforaiTaskRecord.from_dict(task.to_dict())

        self.assertEqual(restored.source_agent, "claude")
        self.assertEqual(restored.split, "val")
        self.assertEqual(restored.judge["checks"][0]["arg"], "training contract")

    def test_config_defaults_include_three_sources(self) -> None:
        cfg = AiforaiConfig(target_skill_repo="/tmp/AIForAI")

        self.assertEqual(cfg.sources, ("codex", "claude", "codewhale"))
        self.assertEqual(cfg.skill_rel_path, "ai-model-rd-protocol/SKILL.md")
        self.assertFalse(cfg.auto_adopt)

    def test_run_result_serializes_counts(self) -> None:
        result = AiforaiRunResult(
            mode="audit",
            staging_dir="/tmp/staging",
            sessions_by_source={"codex": 2, "claude": 1, "codewhale": 3},
            tasks_by_source={"codex": 1},
            accepted=False,
            notes=["audit only"],
        )

        data = result.to_dict()

        self.assertEqual(data["mode"], "audit")
        self.assertEqual(data["sessions_by_source"]["codewhale"], 3)
        self.assertEqual(data["notes"], ["audit only"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the new test to confirm it fails**

Run:

```bash
python3 -m unittest tests.test_aiforai_types -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'skillopt_sleep.aiforai'`.

- [ ] **Step 3: Add package entry files**

Create `skillopt_sleep/aiforai/__init__.py`:

```python
"""AIForAI-specific SkillOpt-Sleep extension."""

__all__ = ["__version__"]
__version__ = "0.1.0"
```

Create `skillopt_sleep/aiforai/__main__.py`:

```python
"""Entry point for python -m skillopt_sleep.aiforai."""

from __future__ import annotations

from skillopt_sleep.aiforai.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Implement dataclasses**

Create `skillopt_sleep/aiforai/types.py`:

```python
"""Data types for AIForAI tri-agent SkillOpt-Sleep."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Literal


SourceAgent = Literal["codex", "claude", "codewhale"]
TaskSplit = Literal["train", "val", "test"]
TaskOutcome = Literal["success", "fail", "mixed", "unknown"]
TaskOrigin = Literal["real", "curated"]


@dataclass(slots=True)
class AiforaiSessionDigest:
    source_agent: SourceAgent
    session_id: str
    raw_path: str
    cwd: str
    git_branch: str = ""
    started_at: str = ""
    ended_at: str = ""
    user_prompts: list[str] = field(default_factory=list)
    assistant_finals: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    files_touched: list[str] = field(default_factory=list)
    feedback_signals: list[str] = field(default_factory=list)
    skill_mentions: list[str] = field(default_factory=list)
    event_count: int = 0
    parse_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AiforaiSessionDigest":
        fields = cls.__dataclass_fields__  # type: ignore[attr-defined]
        return cls(**{key: value for key, value in data.items() if key in fields})


@dataclass(slots=True)
class AiforaiTaskRecord:
    id: str
    source_agent: str
    source_sessions: list[str]
    project: str
    intent: str
    context_excerpt: str
    task_family: str
    outcome: TaskOutcome
    split: TaskSplit = "train"
    reference_kind: str = "rule"
    judge: Dict[str, Any] = field(default_factory=dict)
    origin: TaskOrigin = "real"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AiforaiTaskRecord":
        fields = cls.__dataclass_fields__  # type: ignore[attr-defined]
        return cls(**{key: value for key, value in data.items() if key in fields})


@dataclass(slots=True)
class AiforaiReplayResult:
    task_id: str
    source_agent: str
    task_family: str
    hard: float = 0.0
    soft: float = 0.0
    response: str = ""
    fail_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AiforaiRunResult:
    mode: str
    staging_dir: str = ""
    sessions_by_source: Dict[str, int] = field(default_factory=dict)
    tasks_by_source: Dict[str, int] = field(default_factory=dict)
    checkable_tasks: int = 0
    uncheckable_candidates: int = 0
    accepted: bool = False
    baseline_score: float = 0.0
    candidate_score: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
```

- [ ] **Step 5: Implement config**

Create `skillopt_sleep/aiforai/config.py`:

```python
"""Configuration for AIForAI SkillOpt-Sleep."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AiforaiConfig:
    target_skill_repo: str
    sources: tuple[str, ...] = ("codex", "claude", "codewhale")
    skill_rel_path: str = "ai-model-rd-protocol/SKILL.md"
    lookback_days: int = 30
    max_tasks_per_source: int = 40
    val_fraction: float = 0.25
    test_fraction: float = 0.0
    seed: int = 42
    backend: str = "mock"
    gate: str = "on"
    auto_adopt: bool = False
    codex_home: str = os.path.expanduser("~/.codex")
    claude_home: str = os.path.expanduser("~/.claude")
    codewhale_home: str = os.path.expanduser("~/.codewhale")
    deepseek_home: str = os.path.expanduser("~/.deepseek")

    @property
    def skill_path(self) -> str:
        return os.path.join(self.target_skill_repo, self.skill_rel_path)

    @property
    def staging_root(self) -> str:
        return os.path.join(self.target_skill_repo, ".skillopt-sleep", "staging")
```

- [ ] **Step 6: Add temporary CLI stub so imports work**

Create `skillopt_sleep/aiforai/cli.py`:

```python
"""Command-line entrypoint for AIForAI SkillOpt-Sleep."""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    return 2
```

- [ ] **Step 7: Run the dataclass test**

Run:

```bash
python3 -m unittest tests.test_aiforai_types -v
```

Expected: PASS all tests in `tests.test_aiforai_types`.

- [ ] **Step 8: Commit**

Run:

```bash
git add skillopt_sleep/aiforai tests/test_aiforai_types.py
git commit -m "feat: add AIForAI sleep data types"
```

---

### Task 2: Shared Harvester Utilities

**Files:**
- Create: `skillopt_sleep/aiforai/harvesters/__init__.py`
- Create: `skillopt_sleep/aiforai/harvesters/base.py`
- Modify: `tests/test_aiforai_harvesters.py`

- [ ] **Step 1: Write failing tests for shared helpers**

Create `tests/test_aiforai_harvesters.py`:

```python
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from skillopt_sleep.aiforai.harvesters.base import (
    detect_feedback,
    detect_skill_mentions,
    iter_jsonl,
    redact_text,
    within_lookback,
)


class HarvesterBaseTests(unittest.TestCase):
    def test_iter_jsonl_skips_bad_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            path.write_text('{"a": 1}\nnot-json\n{"b": 2}\n', encoding="utf-8")

            rows = list(iter_jsonl(str(path)))

            self.assertEqual(rows, [{"a": 1}, {"b": 2}])

    def test_detect_feedback_supports_chinese_and_english(self) -> None:
        signals = detect_feedback("这个还是不对, please fix it")

        self.assertIn("neg:还是不对", signals)
        self.assertIn("neg:fix it", signals)

    def test_detect_skill_mentions(self) -> None:
        mentions = detect_skill_mentions("Use $ai-model-rd-protocol for this training run.")

        self.assertEqual(mentions, ["ai-model-rd-protocol"])

    def test_redact_text_masks_secret_like_values(self) -> None:
        redacted = redact_text("OPENAI_API_KEY=sk-abcdef1234567890 token=abc123")

        self.assertIn("OPENAI_API_KEY=<redacted>", redacted)
        self.assertIn("token=<redacted>", redacted)

    def test_within_lookback_accepts_recent_epoch_ms(self) -> None:
        now_ms = 1_800_000_000_000
        recent_ms = now_ms - 60_000
        old_ms = now_ms - 10 * 24 * 3600 * 1000

        self.assertTrue(within_lookback(recent_ms, lookback_days=1, now_ms=now_ms))
        self.assertFalse(within_lookback(old_ms, lookback_days=1, now_ms=now_ms))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run helper tests to confirm failure**

Run:

```bash
python3 -m unittest tests.test_aiforai_harvesters -v
```

Expected: FAIL because `skillopt_sleep.aiforai.harvesters.base` does not exist.

- [ ] **Step 3: Create harvester exports**

Create `skillopt_sleep/aiforai/harvesters/__init__.py`:

```python
"""Harvester implementations for AIForAI SkillOpt-Sleep."""

from __future__ import annotations

from skillopt_sleep.aiforai.harvesters.base import Harvester

__all__ = ["Harvester"]
```

- [ ] **Step 4: Implement base helpers**

Create `skillopt_sleep/aiforai/harvesters/base.py`:

```python
"""Shared helpers for AIForAI trajectory harvesters."""

from __future__ import annotations

import json
import os
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable

from skillopt_sleep.aiforai.config import AiforaiConfig
from skillopt_sleep.aiforai.types import AiforaiSessionDigest


SECRET_PATTERNS = (
    re.compile(r"(?i)(OPENAI_API_KEY|ANTHROPIC_API_KEY|AZURE_OPENAI_API_KEY|token|api_key)=([^\s]+)"),
    re.compile(r"sk-[A-Za-z0-9_-]{12,}"),
)

NEGATIVE_PHRASES = (
    "still broken",
    "still wrong",
    "not working",
    "fix it",
    "wrong",
    "还是不对",
    "不对",
    "没修好",
)

POSITIVE_PHRASES = (
    "thanks",
    "works now",
    "perfect",
    "lgtm",
    "很好",
    "可以了",
)

SKILL_PATTERNS = (
    re.compile(r"\bai-model-rd-protocol\b", re.IGNORECASE),
    re.compile(r"\bAI Model R&D Protocol\b", re.IGNORECASE),
    re.compile(r"\$ai-model-rd-protocol\b", re.IGNORECASE),
)


class Harvester(ABC):
    source_agent: str

    @abstractmethod
    def harvest(self, cfg: AiforaiConfig) -> list[AiforaiSessionDigest]:
        """Return normalized session digests for this source."""


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    yield obj
    except (FileNotFoundError, PermissionError, IsADirectoryError):
        return


def read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as handle:
            obj = json.load(handle)
        return obj if isinstance(obj, dict) else {}
    except (FileNotFoundError, PermissionError, IsADirectoryError, json.JSONDecodeError):
        return {}


def flatten_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                if item.get("type") in {"text", "input_text", "output_text"} and item.get("text"):
                    parts.append(str(item["text"]))
                elif item.get("content"):
                    parts.append(flatten_text(item["content"]))
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        if "text" in value:
            return str(value["text"])
        if "content" in value:
            return flatten_text(value["content"])
    return ""


def detect_feedback(text: str) -> list[str]:
    low = (text or "").lower()
    signals: list[str] = []
    for phrase in NEGATIVE_PHRASES:
        if phrase.lower() in low:
            signals.append(f"neg:{phrase}")
    for phrase in POSITIVE_PHRASES:
        if phrase.lower() in low:
            signals.append(f"pos:{phrase}")
    return list(dict.fromkeys(signals))


def detect_skill_mentions(text: str) -> list[str]:
    mentions: list[str] = []
    for pattern in SKILL_PATTERNS:
        if pattern.search(text or ""):
            mentions.append("ai-model-rd-protocol")
    return list(dict.fromkeys(mentions))


def redact_text(text: str) -> str:
    redacted = text or ""
    for pattern in SECRET_PATTERNS:
        if pattern.pattern.startswith("(?i)("):
            redacted = pattern.sub(lambda match: f"{match.group(1)}=<redacted>", redacted)
        else:
            redacted = pattern.sub("<redacted-secret>", redacted)
    return redacted


def within_lookback(ts_ms: int | float | None, *, lookback_days: int, now_ms: int | None = None) -> bool:
    if ts_ms is None:
        return True
    now = now_ms if now_ms is not None else int(time.time() * 1000)
    cutoff = now - int(lookback_days * 24 * 3600 * 1000)
    return int(ts_ms) >= cutoff


def list_files(root: str, suffix: str) -> list[str]:
    if not os.path.isdir(root):
        return []
    paths: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith(suffix):
                paths.append(os.path.join(dirpath, filename))
    return sorted(paths)
```

- [ ] **Step 5: Run helper tests**

Run:

```bash
python3 -m unittest tests.test_aiforai_harvesters.HarvesterBaseTests -v
```

Expected: PASS all `HarvesterBaseTests`.

- [ ] **Step 6: Commit**

Run:

```bash
git add skillopt_sleep/aiforai/harvesters tests/test_aiforai_harvesters.py
git commit -m "feat: add AIForAI harvester utilities"
```

---

### Task 3: Codex Harvester

**Files:**
- Create: `skillopt_sleep/aiforai/harvesters/codex.py`
- Modify: `skillopt_sleep/aiforai/harvesters/__init__.py`
- Modify: `tests/test_aiforai_harvesters.py`

- [ ] **Step 1: Add Codex fixture test**

Append to `tests/test_aiforai_harvesters.py`:

```python
import sqlite3

from skillopt_sleep.aiforai.config import AiforaiConfig
from skillopt_sleep.aiforai.harvesters.codex import CodexHarvester


class CodexHarvesterTests(unittest.TestCase):
    def test_codex_harvester_reads_threads_and_rollout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            codex_home = root / ".codex"
            session_path = codex_home / "sessions/2026/06/09/rollout.jsonl"
            session_path.parent.mkdir(parents=True)
            session_path.write_text(
                "\n".join(
                    [
                        json.dumps({"type": "user_message", "message": "Use ai-model-rd-protocol to plan training"}),
                        json.dumps({"type": "agent_message", "message": "Need training contract"}),
                        json.dumps({"type": "function_call", "name": "exec_command", "arguments": "{}"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            db_path = codex_home / "state_5.sqlite"
            codex_home.mkdir(exist_ok=True)
            con = sqlite3.connect(db_path)
            con.execute(
                "CREATE TABLE threads (id TEXT PRIMARY KEY, rollout_path TEXT NOT NULL, "
                "created_at_ms INTEGER, updated_at_ms INTEGER, cwd TEXT NOT NULL, "
                "title TEXT NOT NULL, git_branch TEXT, model TEXT, reasoning_effort TEXT, "
                "agent_role TEXT)"
            )
            con.execute(
                "INSERT INTO threads VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "thr1",
                    str(session_path),
                    1_800_000_000_000,
                    1_800_000_001_000,
                    "/repo",
                    "train",
                    "main",
                    "gpt-5.5",
                    "xhigh",
                    "worker",
                ),
            )
            con.commit()
            con.close()
            cfg = AiforaiConfig(
                target_skill_repo=str(root / "AIForAI"),
                codex_home=str(codex_home),
                lookback_days=30,
            )

            sessions = CodexHarvester(now_ms=1_800_000_010_000).harvest(cfg)

            self.assertEqual(len(sessions), 1)
            self.assertEqual(sessions[0].source_agent, "codex")
            self.assertEqual(sessions[0].session_id, "thr1")
            self.assertEqual(sessions[0].cwd, "/repo")
            self.assertEqual(sessions[0].tools_used, ["exec_command"])
            self.assertEqual(sessions[0].skill_mentions, ["ai-model-rd-protocol"])
```

- [ ] **Step 2: Run Codex test to confirm failure**

Run:

```bash
python3 -m unittest tests.test_aiforai_harvesters.CodexHarvesterTests -v
```

Expected: FAIL because `skillopt_sleep.aiforai.harvesters.codex` does not exist.

- [ ] **Step 3: Implement Codex harvester**

Create `skillopt_sleep/aiforai/harvesters/codex.py`:

```python
"""Codex trajectory harvester for AIForAI SkillOpt-Sleep."""

from __future__ import annotations

import os
import sqlite3
from typing import Any

from skillopt_sleep.aiforai.config import AiforaiConfig
from skillopt_sleep.aiforai.harvesters.base import (
    Harvester,
    detect_feedback,
    detect_skill_mentions,
    flatten_text,
    iter_jsonl,
    redact_text,
    within_lookback,
)
from skillopt_sleep.aiforai.types import AiforaiSessionDigest


class CodexHarvester(Harvester):
    source_agent = "codex"

    def __init__(self, *, now_ms: int | None = None) -> None:
        self.now_ms = now_ms

    def harvest(self, cfg: AiforaiConfig) -> list[AiforaiSessionDigest]:
        db_path = os.path.join(cfg.codex_home, "state_5.sqlite")
        if not os.path.exists(db_path):
            return []
        rows = self._thread_rows(db_path)
        sessions: list[AiforaiSessionDigest] = []
        for row in rows:
            updated_at = row.get("updated_at_ms") or row.get("created_at_ms")
            if not within_lookback(updated_at, lookback_days=cfg.lookback_days, now_ms=self.now_ms):
                continue
            sessions.append(self._digest_thread(row))
        return sessions

    def _thread_rows(self, db_path: str) -> list[dict[str, Any]]:
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row
        try:
            rows = con.execute(
                "SELECT id, rollout_path, created_at_ms, updated_at_ms, cwd, title, "
                "git_branch, model, reasoning_effort, agent_role FROM threads "
                "ORDER BY updated_at_ms DESC"
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            con.close()

    def _digest_thread(self, row: dict[str, Any]) -> AiforaiSessionDigest:
        rollout_path = str(row.get("rollout_path") or "")
        user_prompts: list[str] = []
        assistant_finals: list[str] = []
        tools: list[str] = []
        files: list[str] = []
        feedback: list[str] = []
        mentions: list[str] = []
        warnings: list[str] = []
        event_count = 0

        for rec in iter_jsonl(rollout_path):
            event_count += 1
            text = self._record_text(rec)
            if text:
                text = redact_text(text).strip()
                mentions.extend(detect_skill_mentions(text))
            rtype = str(rec.get("type") or rec.get("payload", {}).get("type") or "")
            if "user" in rtype:
                if text:
                    user_prompts.append(text)
                    feedback.extend(detect_feedback(text))
            elif "agent" in rtype or "assistant" in rtype or "message" in rtype:
                if text:
                    assistant_finals.append(text)
            elif "function_call" in rtype or rec.get("name"):
                name = str(rec.get("name") or rec.get("payload", {}).get("name") or "")
                if name:
                    tools.append(name)
                args = str(rec.get("arguments") or rec.get("payload", {}).get("arguments") or "")
                if args:
                    mentions.extend(detect_skill_mentions(args))
                    files.extend(self._file_like_tokens(args))
            else:
                if text:
                    mentions.extend(detect_skill_mentions(text))

        if rollout_path and not os.path.exists(rollout_path):
            warnings.append(f"missing rollout_path: {rollout_path}")

        return AiforaiSessionDigest(
            source_agent="codex",
            session_id=str(row.get("id") or os.path.basename(rollout_path)),
            raw_path=rollout_path,
            cwd=str(row.get("cwd") or ""),
            git_branch=str(row.get("git_branch") or ""),
            started_at=str(row.get("created_at_ms") or ""),
            ended_at=str(row.get("updated_at_ms") or ""),
            user_prompts=user_prompts,
            assistant_finals=assistant_finals[-5:],
            tools_used=list(dict.fromkeys(tools)),
            files_touched=list(dict.fromkeys(files))[:40],
            feedback_signals=list(dict.fromkeys(feedback)),
            skill_mentions=list(dict.fromkeys(mentions)),
            event_count=event_count,
            parse_warnings=warnings,
        )

    def _record_text(self, rec: dict[str, Any]) -> str:
        payload = rec.get("payload")
        if isinstance(payload, dict):
            for key in ("message", "text", "content"):
                text = flatten_text(payload.get(key))
                if text:
                    return text
        for key in ("message", "text", "content"):
            text = flatten_text(rec.get(key))
            if text:
                return text
        return ""

    def _file_like_tokens(self, text: str) -> list[str]:
        out: list[str] = []
        for token in text.replace('"', " ").replace("'", " ").split():
            if "/" in token and len(token) < 240:
                out.append(token.strip(",;()[]{}"))
        return out
```

- [ ] **Step 4: Export CodexHarvester**

Update `skillopt_sleep/aiforai/harvesters/__init__.py`:

```python
"""Harvester implementations for AIForAI SkillOpt-Sleep."""

from __future__ import annotations

from skillopt_sleep.aiforai.harvesters.base import Harvester
from skillopt_sleep.aiforai.harvesters.codex import CodexHarvester

__all__ = ["Harvester", "CodexHarvester"]
```

- [ ] **Step 5: Run harvester tests**

Run:

```bash
python3 -m unittest tests.test_aiforai_harvesters -v
```

Expected: PASS all existing harvester tests.

- [ ] **Step 6: Commit**

Run:

```bash
git add skillopt_sleep/aiforai/harvesters tests/test_aiforai_harvesters.py
git commit -m "feat: harvest Codex AIForAI sessions"
```

---

### Task 4: Claude And CodeWhale Harvesters

**Files:**
- Create: `skillopt_sleep/aiforai/harvesters/claude.py`
- Create: `skillopt_sleep/aiforai/harvesters/codewhale.py`
- Modify: `skillopt_sleep/aiforai/harvesters/__init__.py`
- Modify: `tests/test_aiforai_harvesters.py`

- [ ] **Step 1: Add Claude and CodeWhale tests**

Append to `tests/test_aiforai_harvesters.py`:

```python
from skillopt_sleep.aiforai.harvesters.claude import ClaudeHarvester
from skillopt_sleep.aiforai.harvesters.codewhale import CodeWhaleHarvester


class ClaudeHarvesterTests(unittest.TestCase):
    def test_claude_harvester_wraps_project_transcripts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            claude_home = root / ".claude"
            transcript = claude_home / "projects/proj/session1.jsonl"
            transcript.parent.mkdir(parents=True)
            transcript.write_text(
                "\n".join(
                    [
                        json.dumps({
                            "type": "user",
                            "timestamp": "2026-06-09T00:00:00Z",
                            "cwd": "/repo",
                            "gitBranch": "main",
                            "message": {"role": "user", "content": "Use ai-model-rd-protocol"},
                        }),
                        json.dumps({
                            "type": "assistant",
                            "timestamp": "2026-06-09T00:01:00Z",
                            "message": {"role": "assistant", "content": [{"type": "text", "text": "final"}]},
                        }),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            cfg = AiforaiConfig(target_skill_repo=str(root / "AIForAI"), claude_home=str(claude_home))

            sessions = ClaudeHarvester().harvest(cfg)

            self.assertEqual(len(sessions), 1)
            self.assertEqual(sessions[0].source_agent, "claude")
            self.assertEqual(sessions[0].cwd, "/repo")
            self.assertEqual(sessions[0].assistant_finals, ["final"])


class CodeWhaleHarvesterTests(unittest.TestCase):
    def test_codewhale_harvester_reads_runtime_thread_and_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cw_home = root / ".codewhale"
            ds_home = root / ".deepseek"
            runtime = cw_home / "tasks/runtime"
            (runtime / "threads").mkdir(parents=True)
            (runtime / "events").mkdir(parents=True)
            (runtime / "threads/thr1.json").write_text(
                json.dumps({"id": "thr1", "cwd": "/repo", "created_at": "2026-06-09T00:00:00Z"}),
                encoding="utf-8",
            )
            (runtime / "events/thr1.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"role": "user", "content": "请用 ai-model-rd-protocol 做数据获取计划"}),
                        json.dumps({"role": "assistant", "content": "需要 Data Acquisition Hygiene Gate"}),
                        json.dumps({"tool": "mcp_k8s-management_run_task"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            cfg = AiforaiConfig(
                target_skill_repo=str(root / "AIForAI"),
                codewhale_home=str(cw_home),
                deepseek_home=str(ds_home),
            )

            sessions = CodeWhaleHarvester().harvest(cfg)

            self.assertEqual(len(sessions), 1)
            self.assertEqual(sessions[0].source_agent, "codewhale")
            self.assertEqual(sessions[0].session_id, "thr1")
            self.assertIn("mcp_k8s-management_run_task", sessions[0].tools_used)
            self.assertEqual(sessions[0].skill_mentions, ["ai-model-rd-protocol"])
```

- [ ] **Step 2: Run new tests to confirm failure**

Run:

```bash
python3 -m unittest tests.test_aiforai_harvesters.ClaudeHarvesterTests tests.test_aiforai_harvesters.CodeWhaleHarvesterTests -v
```

Expected: FAIL because the harvester modules do not exist.

- [ ] **Step 3: Implement Claude harvester**

Create `skillopt_sleep/aiforai/harvesters/claude.py`:

```python
"""Claude Code trajectory harvester for AIForAI SkillOpt-Sleep."""

from __future__ import annotations

import os

from skillopt_sleep.aiforai.config import AiforaiConfig
from skillopt_sleep.aiforai.harvesters.base import Harvester
from skillopt_sleep.aiforai.types import AiforaiSessionDigest
from skillopt_sleep.harvest import harvest as harvest_claude


class ClaudeHarvester(Harvester):
    source_agent = "claude"

    def harvest(self, cfg: AiforaiConfig) -> list[AiforaiSessionDigest]:
        transcripts_dir = os.path.join(cfg.claude_home, "projects")
        base = harvest_claude(
            transcripts_dir,
            scope="all",
            invoked_project="",
            since_iso=None,
            limit=0,
        )
        sessions: list[AiforaiSessionDigest] = []
        for digest in base:
            sessions.append(
                AiforaiSessionDigest(
                    source_agent="claude",
                    session_id=digest.session_id,
                    raw_path=digest.raw_path,
                    cwd=digest.project,
                    git_branch=digest.git_branch,
                    started_at=digest.started_at,
                    ended_at=digest.ended_at,
                    user_prompts=digest.user_prompts,
                    assistant_finals=digest.assistant_finals,
                    tools_used=digest.tools_used,
                    files_touched=digest.files_touched,
                    feedback_signals=digest.feedback_signals,
                    skill_mentions=[],
                    event_count=digest.n_user_turns + digest.n_assistant_turns,
                    parse_warnings=[],
                )
            )
        return sessions
```

- [ ] **Step 4: Implement CodeWhale harvester**

Create `skillopt_sleep/aiforai/harvesters/codewhale.py`:

```python
"""CodeWhale / DeepSeek TUI harvester for AIForAI SkillOpt-Sleep."""

from __future__ import annotations

import os

from skillopt_sleep.aiforai.config import AiforaiConfig
from skillopt_sleep.aiforai.harvesters.base import (
    Harvester,
    detect_feedback,
    detect_skill_mentions,
    flatten_text,
    iter_jsonl,
    list_files,
    read_json,
    redact_text,
)
from skillopt_sleep.aiforai.types import AiforaiSessionDigest


class CodeWhaleHarvester(Harvester):
    source_agent = "codewhale"

    def harvest(self, cfg: AiforaiConfig) -> list[AiforaiSessionDigest]:
        sessions = self._harvest_runtime(cfg)
        sessions.extend(self._harvest_deepseek_sessions(cfg, seen={s.session_id for s in sessions}))
        return sessions

    def _harvest_runtime(self, cfg: AiforaiConfig) -> list[AiforaiSessionDigest]:
        runtime = os.path.join(cfg.codewhale_home, "tasks", "runtime")
        thread_dir = os.path.join(runtime, "threads")
        event_dir = os.path.join(runtime, "events")
        sessions: list[AiforaiSessionDigest] = []
        for thread_path in list_files(thread_dir, ".json"):
            thread = read_json(thread_path)
            session_id = str(thread.get("id") or os.path.splitext(os.path.basename(thread_path))[0])
            event_path = os.path.join(event_dir, f"{session_id}.jsonl")
            sessions.append(self._digest_records(session_id, thread_path, event_path, thread))
        return sessions

    def _harvest_deepseek_sessions(self, cfg: AiforaiConfig, *, seen: set[str]) -> list[AiforaiSessionDigest]:
        session_dir = os.path.join(cfg.deepseek_home, "sessions")
        sessions: list[AiforaiSessionDigest] = []
        for path in list_files(session_dir, ".json"):
            data = read_json(path)
            session_id = str(data.get("id") or os.path.splitext(os.path.basename(path))[0])
            if session_id in seen:
                continue
            user_prompts: list[str] = []
            assistant_finals: list[str] = []
            tools: list[str] = []
            feedback: list[str] = []
            mentions: list[str] = []
            records = data.get("messages") or data.get("turns") or data.get("events") or []
            if not isinstance(records, list):
                records = []
            for rec in records:
                if not isinstance(rec, dict):
                    continue
                text = redact_text(flatten_text(rec.get("content") or rec.get("message") or rec.get("text"))).strip()
                mentions.extend(detect_skill_mentions(text))
                role = str(rec.get("role") or rec.get("type") or "")
                if "user" in role:
                    user_prompts.append(text)
                    feedback.extend(detect_feedback(text))
                elif "assistant" in role or "agent" in role:
                    assistant_finals.append(text)
                if rec.get("tool"):
                    tools.append(str(rec["tool"]))
            sessions.append(
                AiforaiSessionDigest(
                    source_agent="codewhale",
                    session_id=session_id,
                    raw_path=path,
                    cwd=str(data.get("cwd") or data.get("project") or ""),
                    started_at=str(data.get("created_at") or data.get("started_at") or ""),
                    ended_at=str(data.get("updated_at") or data.get("ended_at") or ""),
                    user_prompts=[p for p in user_prompts if p],
                    assistant_finals=[p for p in assistant_finals if p][-5:],
                    tools_used=list(dict.fromkeys(tools)),
                    feedback_signals=list(dict.fromkeys(feedback)),
                    skill_mentions=list(dict.fromkeys(mentions)),
                    event_count=len(records),
                )
            )
        return sessions

    def _digest_records(
        self,
        session_id: str,
        thread_path: str,
        event_path: str,
        thread: dict,
    ) -> AiforaiSessionDigest:
        user_prompts: list[str] = []
        assistant_finals: list[str] = []
        tools: list[str] = []
        feedback: list[str] = []
        mentions: list[str] = []
        warnings: list[str] = []
        count = 0
        for rec in iter_jsonl(event_path):
            count += 1
            text = redact_text(flatten_text(rec.get("content") or rec.get("message") or rec.get("text"))).strip()
            if text:
                mentions.extend(detect_skill_mentions(text))
            role = str(rec.get("role") or rec.get("type") or "")
            if "user" in role:
                user_prompts.append(text)
                feedback.extend(detect_feedback(text))
            elif "assistant" in role or "agent" in role:
                assistant_finals.append(text)
            tool = rec.get("tool") or rec.get("tool_name") or rec.get("name")
            if tool:
                tools.append(str(tool))
        if event_path and not os.path.exists(event_path):
            warnings.append(f"missing events: {event_path}")
        return AiforaiSessionDigest(
            source_agent="codewhale",
            session_id=session_id,
            raw_path=thread_path,
            cwd=str(thread.get("cwd") or thread.get("project") or ""),
            git_branch=str(thread.get("git_branch") or ""),
            started_at=str(thread.get("created_at") or thread.get("started_at") or ""),
            ended_at=str(thread.get("updated_at") or thread.get("ended_at") or ""),
            user_prompts=[p for p in user_prompts if p],
            assistant_finals=[p for p in assistant_finals if p][-5:],
            tools_used=list(dict.fromkeys(tools)),
            feedback_signals=list(dict.fromkeys(feedback)),
            skill_mentions=list(dict.fromkeys(mentions)),
            event_count=count,
            parse_warnings=warnings,
        )
```

- [ ] **Step 5: Export new harvesters**

Update `skillopt_sleep/aiforai/harvesters/__init__.py`:

```python
"""Harvester implementations for AIForAI SkillOpt-Sleep."""

from __future__ import annotations

from skillopt_sleep.aiforai.harvesters.base import Harvester
from skillopt_sleep.aiforai.harvesters.claude import ClaudeHarvester
from skillopt_sleep.aiforai.harvesters.codex import CodexHarvester
from skillopt_sleep.aiforai.harvesters.codewhale import CodeWhaleHarvester

__all__ = ["Harvester", "ClaudeHarvester", "CodexHarvester", "CodeWhaleHarvester"]
```

- [ ] **Step 6: Run harvester tests**

Run:

```bash
python3 -m unittest tests.test_aiforai_harvesters -v
```

Expected: PASS all harvester tests.

- [ ] **Step 7: Commit**

Run:

```bash
git add skillopt_sleep/aiforai/harvesters tests/test_aiforai_harvesters.py
git commit -m "feat: harvest Claude and CodeWhale AIForAI sessions"
```

---

### Task 5: AIForAI Miner And Source-Stratified Split

**Files:**
- Create: `skillopt_sleep/aiforai/mine.py`
- Create: `tests/test_aiforai_mine.py`

- [ ] **Step 1: Write failing miner tests**

Create `tests/test_aiforai_mine.py`:

```python
from __future__ import annotations

import unittest

from skillopt_sleep.aiforai.mine import mine_tasks, split_tasks
from skillopt_sleep.aiforai.types import AiforaiSessionDigest, AiforaiTaskRecord


def _session(source: str, prompt: str, final: str = "") -> AiforaiSessionDigest:
    return AiforaiSessionDigest(
        source_agent=source,  # type: ignore[arg-type]
        session_id=f"{source}-1",
        raw_path=f"/tmp/{source}.jsonl",
        cwd="/repo",
        user_prompts=[prompt],
        assistant_finals=[final],
        skill_mentions=["ai-model-rd-protocol"],
    )


class AiforaiMineTests(unittest.TestCase):
    def test_mines_training_contract_task(self) -> None:
        tasks, uncheckable = mine_tasks([
            _session("codex", "Start a training run without a contract", "Need a training contract")
        ])

        self.assertEqual(uncheckable, [])
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].task_family, "training_contract")
        self.assertEqual(tasks[0].judge["checks"][0]["op"], "contains")

    def test_mines_data_acquisition_task(self) -> None:
        tasks, _ = mine_tasks([
            _session("claude", "Download the full dataset locally before training")
        ])

        self.assertEqual(tasks[0].task_family, "data_acquisition")

    def test_unrelated_prompt_is_uncheckable(self) -> None:
        tasks, uncheckable = mine_tasks([
            _session("codewhale", "Write a poem about rain")
        ])

        self.assertEqual(tasks, [])
        self.assertEqual(len(uncheckable), 1)

    def test_split_tasks_keeps_sources_represented(self) -> None:
        tasks = [
            AiforaiTaskRecord(
                id=f"{source}-{idx}",
                source_agent=source,
                source_sessions=[f"{source}-{idx}"],
                project="/repo",
                intent="prepare training contract",
                context_excerpt="",
                task_family="training_contract",
                outcome="fail",
                judge={"kind": "rule", "checks": [{"op": "contains", "arg": "training contract"}]},
            )
            for source in ("codex", "claude", "codewhale")
            for idx in range(6)
        ]

        split = split_tasks(tasks, val_fraction=0.25, test_fraction=0.0, seed=7)
        val_sources = {task.source_agent for task in split if task.split == "val"}

        self.assertEqual(val_sources, {"codex", "claude", "codewhale"})
        self.assertTrue(any(task.split == "train" for task in split))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run miner tests to confirm failure**

Run:

```bash
python3 -m unittest tests.test_aiforai_mine -v
```

Expected: FAIL because `skillopt_sleep.aiforai.mine` does not exist.

- [ ] **Step 3: Implement miner**

Create `skillopt_sleep/aiforai/mine.py`:

```python
"""Mine checkable AIForAI tasks from normalized sessions."""

from __future__ import annotations

import hashlib
from collections import defaultdict

from skillopt_sleep.aiforai.types import AiforaiSessionDigest, AiforaiTaskRecord


FAMILY_RULES: list[tuple[str, tuple[str, ...], list[dict]]] = [
    (
        "training_contract",
        ("training contract", "resume training", "start a training", "训练", "train"),
        [
            {"op": "contains", "arg": "training contract"},
            {"op": "contains", "arg": "evaluation contract"},
            {"op": "contains", "arg": "stop criteria"},
            {"op": "contains", "arg": "artifact paths"},
        ],
    ),
    (
        "data_acquisition",
        ("download", "dataset", "data acquisition", "数据", "下载"),
        [
            {"op": "contains", "arg": "scope"},
            {"op": "contains", "arg": "shared storage"},
            {"op": "contains", "arg": "do not download full datasets locally"},
        ],
    ),
    (
        "cluster_preflight",
        ("cluster", "volcano", "k8s", "kubectl", "muxi", "ascend"),
        [
            {"op": "contains", "arg": "image"},
            {"op": "contains", "arg": "dependencies"},
            {"op": "contains", "arg": "data access"},
            {"op": "contains", "arg": "artifact"},
        ],
    ),
    (
        "dirty_worktree_gate",
        ("dirty", "git status", "worktree", "uncommitted"),
        [
            {"op": "contains", "arg": "git status"},
            {"op": "contains", "arg": "dirty"},
            {"op": "contains", "arg": "formal"},
        ],
    ),
    (
        "claim_integrity",
        ("done", "complete", "完成", "ready", "status"),
        [
            {"op": "contains", "arg": "Delivered artifact"},
            {"op": "contains", "arg": "Verified evidence"},
            {"op": "contains", "arg": "Unverified boundary"},
            {"op": "contains", "arg": "Next deliverable"},
        ],
    ),
    (
        "rag_agent_diagnosis",
        ("rag", "retrieval", "agent failure", "tool trajectory", "trajectory"),
        [
            {"op": "contains", "arg": "retrieval"},
            {"op": "contains", "arg": "tool"},
            {"op": "contains", "arg": "trajectory"},
        ],
    ),
]


def mine_tasks(
    sessions: list[AiforaiSessionDigest],
    *,
    max_tasks_per_source: int = 40,
) -> tuple[list[AiforaiTaskRecord], list[dict]]:
    tasks: list[AiforaiTaskRecord] = []
    uncheckable: list[dict] = []
    counts: dict[str, int] = defaultdict(int)
    for session in sessions:
        if counts[session.source_agent] >= max_tasks_per_source:
            continue
        text = "\n".join(session.user_prompts + session.assistant_finals)
        family, checks = classify_family(text)
        if not family:
            uncheckable.append({
                "source_agent": session.source_agent,
                "session_id": session.session_id,
                "reason": "no_checkable_aiforai_family",
                "preview": text[:300],
            })
            continue
        intent = session.user_prompts[0] if session.user_prompts else text[:200]
        task = AiforaiTaskRecord(
            id=_task_id(session.source_agent, family, intent),
            source_agent=session.source_agent,
            source_sessions=[session.session_id],
            project=session.cwd,
            intent=intent[:800],
            context_excerpt=text[:1200],
            task_family=family,
            outcome=_outcome(session.feedback_signals),
            judge={"kind": "rule", "checks": checks},
        )
        tasks.append(task)
        counts[session.source_agent] += 1
    return dedup_tasks(tasks), uncheckable


def classify_family(text: str) -> tuple[str, list[dict]]:
    low = (text or "").lower()
    for family, needles, checks in FAMILY_RULES:
        if any(needle.lower() in low for needle in needles):
            return family, list(checks)
    return "", []


def split_tasks(
    tasks: list[AiforaiTaskRecord],
    *,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> list[AiforaiTaskRecord]:
    groups: dict[tuple[str, str], list[AiforaiTaskRecord]] = defaultdict(list)
    for task in tasks:
        groups[(task.source_agent, task.task_family)].append(task)
    for (_source, _family), group in groups.items():
        ordered = sorted(group, key=lambda task: _stable_hash(f"{seed}:{task.id}"))
        n = len(ordered)
        n_val = 1 if n >= 2 else 0
        n_val = max(n_val, int(round(n * val_fraction)))
        n_test = int(round(n * test_fraction))
        for idx, task in enumerate(ordered):
            if idx < n_val:
                task.split = "val"
            elif idx < n_val + n_test:
                task.split = "test"
            else:
                task.split = "train"
        if n >= 2 and not any(task.split == "train" for task in group):
            ordered[-1].split = "train"
    return tasks


def dedup_tasks(tasks: list[AiforaiTaskRecord]) -> list[AiforaiTaskRecord]:
    by_id: dict[str, AiforaiTaskRecord] = {}
    for task in tasks:
        existing = by_id.get(task.id)
        if existing is None:
            by_id[task.id] = task
            continue
        existing.source_sessions = list(dict.fromkeys(existing.source_sessions + task.source_sessions))
    return list(by_id.values())


def _task_id(source: str, family: str, intent: str) -> str:
    digest = hashlib.sha256(f"{source}:{family}:{intent.lower()}".encode("utf-8")).hexdigest()[:12]
    return f"aiforai_{digest}"


def _stable_hash(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)


def _outcome(signals: list[str]) -> str:
    if any(signal.startswith("neg:") for signal in signals):
        return "fail"
    if any(signal.startswith("pos:") for signal in signals):
        return "success"
    return "unknown"
```

- [ ] **Step 4: Run miner tests**

Run:

```bash
python3 -m unittest tests.test_aiforai_mine -v
```

Expected: PASS all miner tests.

- [ ] **Step 5: Commit**

Run:

```bash
git add skillopt_sleep/aiforai/mine.py tests/test_aiforai_mine.py
git commit -m "feat: mine AIForAI checkable tasks"
```

---

### Task 6: Audit Orchestration And Report

**Files:**
- Create: `skillopt_sleep/aiforai/report.py`
- Create: `skillopt_sleep/aiforai/run.py`
- Modify: `skillopt_sleep/aiforai/cli.py`
- Create: `tests/test_aiforai_run.py`

- [ ] **Step 1: Write failing audit orchestration tests**

Create `tests/test_aiforai_run.py`:

```python
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from skillopt_sleep.aiforai.config import AiforaiConfig
from skillopt_sleep.aiforai.run import run_audit
from skillopt_sleep.aiforai.types import AiforaiSessionDigest


class StaticHarvester:
    def __init__(self, source: str, prompt: str) -> None:
        self.source_agent = source
        self.prompt = prompt

    def harvest(self, cfg: AiforaiConfig):
        return [
            AiforaiSessionDigest(
                source_agent=self.source_agent,  # type: ignore[arg-type]
                session_id=f"{self.source_agent}-1",
                raw_path="/tmp/raw",
                cwd="/repo",
                user_prompts=[self.prompt],
                assistant_finals=["final"],
                skill_mentions=["ai-model-rd-protocol"],
            )
        ]


class AiforaiAuditRunTests(unittest.TestCase):
    def test_run_audit_writes_report_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "AIForAI"
            repo.mkdir()
            cfg = AiforaiConfig(target_skill_repo=str(repo))
            result = run_audit(
                cfg,
                harvesters=[
                    StaticHarvester("codex", "start a training run"),
                    StaticHarvester("claude", "download full dataset locally"),
                    StaticHarvester("codewhale", "write a poem"),
                ],
            )

            self.assertEqual(result.mode, "audit")
            self.assertEqual(result.sessions_by_source["codex"], 1)
            self.assertEqual(result.checkable_tasks, 2)
            self.assertEqual(result.uncheckable_candidates, 1)
            report_path = Path(result.staging_dir) / "audit_report.md"
            manifest_path = Path(result.staging_dir) / "task_manifest.jsonl"
            uncheckable_path = Path(result.staging_dir) / "uncheckable_candidates.jsonl"
            self.assertTrue(report_path.exists())
            self.assertTrue(manifest_path.exists())
            self.assertTrue(uncheckable_path.exists())
            first = json.loads(manifest_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertIn(first["source_agent"], {"codex", "claude"})


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run audit test to confirm failure**

Run:

```bash
python3 -m unittest tests.test_aiforai_run -v
```

Expected: FAIL because `skillopt_sleep.aiforai.run` does not exist.

- [ ] **Step 3: Implement report writers**

Create `skillopt_sleep/aiforai/report.py`:

```python
"""Report writers for AIForAI SkillOpt-Sleep."""

from __future__ import annotations

import json
import os
import time
from collections import Counter

from skillopt_sleep.aiforai.types import AiforaiRunResult, AiforaiSessionDigest, AiforaiTaskRecord


def make_staging_dir(target_skill_repo: str, *, prefix: str) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    path = os.path.join(target_skill_repo, ".skillopt-sleep", "staging", f"{stamp}-{prefix}")
    os.makedirs(path, exist_ok=True)
    return path


def write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_audit_report(
    out_dir: str,
    *,
    sessions: list[AiforaiSessionDigest],
    tasks: list[AiforaiTaskRecord],
    uncheckable: list[dict],
    result: AiforaiRunResult,
) -> None:
    sessions_by_source = Counter(session.source_agent for session in sessions)
    tasks_by_source = Counter(task.source_agent for task in tasks)
    tasks_by_family = Counter(task.task_family for task in tasks)
    lines = [
        "# AIForAI SkillOpt-Sleep Audit Report",
        "",
        "## Source Coverage",
    ]
    for source in ("codex", "claude", "codewhale"):
        lines.append(f"- {source}: sessions={sessions_by_source.get(source, 0)} tasks={tasks_by_source.get(source, 0)}")
    lines.extend(["", "## Task Families"])
    for family, count in sorted(tasks_by_family.items()):
        lines.append(f"- {family}: {count}")
    lines.extend([
        "",
        "## Checkability",
        f"- checkable tasks: {len(tasks)}",
        f"- uncheckable candidates: {len(uncheckable)}",
        "",
        "## Boundary",
        "- This audit is read-only.",
        "- No live AIForAI skill files were modified.",
    ])
    with open(os.path.join(out_dir, "audit_report.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as handle:
        json.dump(result.to_dict(), handle, ensure_ascii=False, indent=2)
    write_jsonl(os.path.join(out_dir, "sessions.jsonl"), [session.to_dict() for session in sessions])
    write_jsonl(os.path.join(out_dir, "task_manifest.jsonl"), [task.to_dict() for task in tasks])
    write_jsonl(os.path.join(out_dir, "uncheckable_candidates.jsonl"), uncheckable)
```

- [ ] **Step 4: Implement audit orchestration**

Create `skillopt_sleep/aiforai/run.py`:

```python
"""Run orchestration for AIForAI SkillOpt-Sleep."""

from __future__ import annotations

from collections import Counter

from skillopt_sleep.aiforai.config import AiforaiConfig
from skillopt_sleep.aiforai.harvesters.claude import ClaudeHarvester
from skillopt_sleep.aiforai.harvesters.codex import CodexHarvester
from skillopt_sleep.aiforai.harvesters.codewhale import CodeWhaleHarvester
from skillopt_sleep.aiforai.mine import mine_tasks, split_tasks
from skillopt_sleep.aiforai.report import make_staging_dir, write_audit_report
from skillopt_sleep.aiforai.types import AiforaiRunResult


def default_harvesters(sources: tuple[str, ...]):
    mapping = {
        "codex": CodexHarvester(),
        "claude": ClaudeHarvester(),
        "codewhale": CodeWhaleHarvester(),
    }
    return [mapping[source] for source in sources if source in mapping]


def run_audit(cfg: AiforaiConfig, *, harvesters=None) -> AiforaiRunResult:
    selected = list(harvesters) if harvesters is not None else default_harvesters(cfg.sources)
    sessions = []
    notes: list[str] = []
    for harvester in selected:
        try:
            sessions.extend(harvester.harvest(cfg))
        except Exception as exc:  # noqa: BLE001
            notes.append(f"{getattr(harvester, 'source_agent', 'unknown')} harvest failed: {exc}")
    tasks, uncheckable = mine_tasks(sessions, max_tasks_per_source=cfg.max_tasks_per_source)
    tasks = split_tasks(tasks, val_fraction=cfg.val_fraction, test_fraction=cfg.test_fraction, seed=cfg.seed)
    out_dir = make_staging_dir(cfg.target_skill_repo, prefix="audit")
    sessions_by_source = dict(Counter(session.source_agent for session in sessions))
    tasks_by_source = dict(Counter(task.source_agent for task in tasks))
    result = AiforaiRunResult(
        mode="audit",
        staging_dir=out_dir,
        sessions_by_source=sessions_by_source,
        tasks_by_source=tasks_by_source,
        checkable_tasks=len(tasks),
        uncheckable_candidates=len(uncheckable),
        notes=notes,
    )
    write_audit_report(out_dir, sessions=sessions, tasks=tasks, uncheckable=uncheckable, result=result)
    return result
```

- [ ] **Step 5: Implement CLI audit command**

Replace `skillopt_sleep/aiforai/cli.py` with:

```python
"""Command-line entrypoint for AIForAI SkillOpt-Sleep."""

from __future__ import annotations

import argparse
import json

from skillopt_sleep.aiforai.config import AiforaiConfig
from skillopt_sleep.aiforai.run import run_audit


def _sources(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in text.split(",") if part.strip())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="skillopt_sleep.aiforai")
    sub = parser.add_subparsers(dest="cmd", required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--target-skill-repo", required=True)
    audit.add_argument("--sources", default="codex,claude,codewhale")
    audit.add_argument("--lookback-days", type=int, default=30)
    audit.add_argument("--max-tasks-per-source", type=int, default=40)
    audit.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.cmd == "audit":
        cfg = AiforaiConfig(
            target_skill_repo=args.target_skill_repo,
            sources=_sources(args.sources),
            lookback_days=args.lookback_days,
            max_tasks_per_source=args.max_tasks_per_source,
        )
        result = run_audit(cfg)
        if args.json:
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(f"[aiforai] audit staged: {result.staging_dir}")
            print(f"[aiforai] sessions: {result.sessions_by_source}")
            print(f"[aiforai] checkable tasks: {result.checkable_tasks}")
        return 0
    return 2
```

- [ ] **Step 6: Run audit tests**

Run:

```bash
python3 -m unittest tests.test_aiforai_run -v
```

Expected: PASS `AiforaiAuditRunTests`.

- [ ] **Step 7: Run CLI smoke against fixture command path**

Run:

```bash
python3 -m skillopt_sleep.aiforai audit --target-skill-repo /tmp --sources codex,claude,codewhale --lookback-days 1 --json
```

Expected: exit 0 and JSON with `"mode": "audit"` and `"staging_dir"`.

- [ ] **Step 8: Commit**

Run:

```bash
git add skillopt_sleep/aiforai tests/test_aiforai_run.py
git commit -m "feat: add AIForAI audit command"
```

---

### Task 7: Protected Skill Adapter And Validators

**Files:**
- Create: `skillopt_sleep/aiforai/skill_adapter.py`
- Create: `tests/test_aiforai_skill_adapter.py`

- [ ] **Step 1: Write failing skill adapter tests**

Create `tests/test_aiforai_skill_adapter.py`:

```python
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from skillopt_sleep.aiforai.skill_adapter import (
    LEARNED_END,
    LEARNED_START,
    apply_learned_rules,
    current_learned_rules,
    run_aiforai_validators,
)


class AiforaiSkillAdapterTests(unittest.TestCase):
    def test_apply_learned_rules_preserves_handwritten_content(self) -> None:
        base = "---\nname: ai-model-rd-protocol\n---\n\n# Skill\n\nKeep this doctrine.\n"

        updated = apply_learned_rules(base, ["Rule A", "Rule A", "Rule B"])

        self.assertIn("Keep this doctrine.", updated)
        self.assertIn(LEARNED_START, updated)
        self.assertIn(LEARNED_END, updated)
        self.assertEqual(current_learned_rules(updated), ["Rule A", "Rule B"])

    def test_apply_learned_rules_replaces_only_protected_block(self) -> None:
        doc = apply_learned_rules("# Skill\n\nManual line.\n", ["Old"])

        updated = apply_learned_rules(doc, ["New"])

        self.assertIn("Manual line.", updated)
        self.assertNotIn("- Old", updated)
        self.assertIn("- New", updated)

    def test_run_validators_captures_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            result = run_aiforai_validators(str(repo))

            self.assertFalse(result["ok"])
            self.assertIn("quick_validate", result["commands"][0]["name"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run skill adapter tests to confirm failure**

Run:

```bash
python3 -m unittest tests.test_aiforai_skill_adapter -v
```

Expected: FAIL because `skillopt_sleep.aiforai.skill_adapter` does not exist.

- [ ] **Step 3: Implement skill adapter**

Create `skillopt_sleep/aiforai/skill_adapter.py`:

```python
"""AIForAI skill document and validator helpers."""

from __future__ import annotations

import os
import subprocess
from typing import Dict, List


LEARNED_START = "<!-- SKILLOPT-AIFORAI:LEARNED START -->"
LEARNED_END = "<!-- SKILLOPT-AIFORAI:LEARNED END -->"
BANNER = (
    "_This block is maintained by AIForAI SkillOpt-Sleep. It is staged and "
    "validated before adoption. Hand-authored content outside this block is "
    "never changed by the optimizer._"
)


def read_skill(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def write_skill(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


def current_learned_rules(doc: str) -> list[str]:
    inner = _extract_learned(doc)
    rules: list[str] = []
    for line in inner.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            rules.append(stripped[2:].strip())
    return rules


def apply_learned_rules(doc: str, rules: list[str]) -> str:
    base = _strip_learned(doc)
    deduped = []
    seen = set()
    for rule in rules:
        clean = rule.strip().lstrip("- ").strip()
        key = " ".join(clean.lower().split())
        if clean and key not in seen:
            seen.add(key)
            deduped.append(clean)
    body = "\n".join(f"- {rule}" for rule in deduped)
    block = (
        f"\n\n{LEARNED_START}\n"
        "## Learned AIForAI Rules\n\n"
        f"{BANNER}\n\n"
        f"{body}\n"
        f"{LEARNED_END}\n"
    )
    return base.rstrip() + block


def run_aiforai_validators(repo: str, *, timeout: int = 120) -> Dict:
    commands = [
        ("quick_validate", ["python3", "scripts/quick_validate.py", "ai-model-rd-protocol"]),
        ("unittest", ["python3", "-m", "unittest", "discover", "-s", "tests", "-v"]),
    ]
    results = []
    ok = True
    for name, cmd in commands:
        try:
            proc = subprocess.run(
                cmd,
                cwd=repo,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            passed = proc.returncode == 0
            output = (proc.stdout or "") + (proc.stderr or "")
        except Exception as exc:  # noqa: BLE001
            passed = False
            output = str(exc)
        ok = ok and passed
        results.append({"name": name, "cmd": cmd, "ok": passed, "output": output[-4000:]})
    return {"ok": ok, "commands": results}


def _extract_learned(doc: str) -> str:
    start = doc.find(LEARNED_START)
    end = doc.find(LEARNED_END)
    if start == -1 or end == -1 or end < start:
        return ""
    return doc[start + len(LEARNED_START):end].strip()


def _strip_learned(doc: str) -> str:
    text = doc
    while True:
        start = text.find(LEARNED_START)
        if start == -1:
            break
        end = text.find(LEARNED_END, start)
        if end == -1:
            text = text[:start]
            break
        text = text[:start] + text[end + len(LEARNED_END):]
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.rstrip()
```

- [ ] **Step 4: Run skill adapter tests**

Run:

```bash
python3 -m unittest tests.test_aiforai_skill_adapter -v
```

Expected: PASS all skill adapter tests.

- [ ] **Step 5: Commit**

Run:

```bash
git add skillopt_sleep/aiforai/skill_adapter.py tests/test_aiforai_skill_adapter.py
git commit -m "feat: manage AIForAI learned skill block"
```

---

### Task 8: Curated Regression Suite And Mock Replay Gate

**Files:**
- Create: `skillopt_sleep/aiforai/regression_suite.py`
- Create: `skillopt_sleep/aiforai/replay.py`
- Modify: `tests/test_aiforai_run.py`

- [ ] **Step 1: Add replay/gate tests**

Append to `tests/test_aiforai_run.py`:

```python
from skillopt_sleep.aiforai.regression_suite import curated_regression_tasks
from skillopt_sleep.aiforai.mine import mine_tasks
from skillopt_sleep.aiforai.replay import evaluate_tasks, gate_candidate, propose_mock_rules


class AiforaiReplayGateTests(unittest.TestCase):
    def test_curated_regression_suite_has_required_families(self) -> None:
        tasks = curated_regression_tasks()
        families = {task.task_family for task in tasks}

        self.assertIn("training_contract", families)
        self.assertIn("data_acquisition", families)
        self.assertIn("claim_integrity", families)

    def test_mock_replay_improves_after_rule_added(self) -> None:
        tasks, _ = mine_tasks([
            AiforaiSessionDigest(
                source_agent="codex",
                session_id="s1",
                raw_path="/tmp/raw",
                cwd="/repo",
                user_prompts=["start a training run"],
            )
        ])
        baseline = evaluate_tasks(tasks, skill="")
        rules = propose_mock_rules(tasks, skill="")
        candidate = "\n".join(rules)
        candidate_score = evaluate_tasks(tasks, skill=candidate)

        self.assertLess(baseline.aggregate_hard, candidate_score.aggregate_hard)
        self.assertTrue(gate_candidate(baseline, candidate_score).accepted)
```

- [ ] **Step 2: Run replay tests to confirm failure**

Run:

```bash
python3 -m unittest tests.test_aiforai_run.AiforaiReplayGateTests -v
```

Expected: FAIL because regression/replay modules do not exist.

- [ ] **Step 3: Implement curated suite**

Create `skillopt_sleep/aiforai/regression_suite.py`:

```python
"""Curated AIForAI regression tasks."""

from __future__ import annotations

from skillopt_sleep.aiforai.types import AiforaiTaskRecord


def curated_regression_tasks() -> list[AiforaiTaskRecord]:
    cases = [
        ("training_contract", "Start a full model training run now.", ["training contract", "evaluation contract", "stop criteria", "artifact paths"]),
        ("data_acquisition", "Download this full dataset locally before training.", ["scope", "shared storage", "do not download full datasets locally"]),
        ("dirty_worktree_gate", "Run a formal controlled experiment in this dirty repo.", ["git status", "dirty", "formal"]),
        ("claim_integrity", "Is this system complete?", ["Delivered artifact", "Verified evidence", "Unverified boundary", "Next deliverable"]),
        ("rag_agent_diagnosis", "The RAG answer is bad; fix generation.", ["retrieval", "tool", "trajectory"]),
        ("cluster_preflight", "Submit the cluster job immediately.", ["image", "dependencies", "data access", "artifact"]),
    ]
    tasks: list[AiforaiTaskRecord] = []
    for idx, (family, intent, required) in enumerate(cases):
        tasks.append(
            AiforaiTaskRecord(
                id=f"curated_{family}_{idx}",
                source_agent="curated",
                source_sessions=[],
                project="AIForAI",
                intent=intent,
                context_excerpt=intent,
                task_family=family,
                outcome="unknown",
                split="val",
                origin="curated",
                judge={"kind": "rule", "checks": [{"op": "contains", "arg": item} for item in required]},
            )
        )
    return tasks
```

- [ ] **Step 4: Implement mock replay and gate**

Create `skillopt_sleep/aiforai/replay.py`:

```python
"""Mock replay and gate utilities for AIForAI SkillOpt-Sleep."""

from __future__ import annotations

from dataclasses import dataclass, field

from skillopt_sleep.aiforai.types import AiforaiReplayResult, AiforaiTaskRecord


@dataclass(slots=True)
class AiforaiScoreSummary:
    aggregate_hard: float
    aggregate_soft: float
    by_source: dict[str, float] = field(default_factory=dict)
    by_family: dict[str, float] = field(default_factory=dict)
    results: list[AiforaiReplayResult] = field(default_factory=list)


@dataclass(slots=True)
class AiforaiGateDecision:
    accepted: bool
    action: str
    reason: str


def evaluate_tasks(tasks: list[AiforaiTaskRecord], *, skill: str) -> AiforaiScoreSummary:
    results: list[AiforaiReplayResult] = []
    for task in tasks:
        hard, soft, missing = _score_task(task, skill)
        results.append(
            AiforaiReplayResult(
                task_id=task.id,
                source_agent=task.source_agent,
                task_family=task.task_family,
                hard=hard,
                soft=soft,
                response=skill,
                fail_reason="missing: " + ", ".join(missing) if missing else "",
            )
        )
    return _summarize(results)


def propose_mock_rules(tasks: list[AiforaiTaskRecord], *, skill: str) -> list[str]:
    rules: list[str] = []
    lower_skill = skill.lower()
    for task in tasks:
        required = [str(check.get("arg")) for check in task.judge.get("checks", [])]
        missing = [item for item in required if item.lower() not in lower_skill]
        if missing:
            rule = f"For {task.task_family} tasks, explicitly include: {', '.join(missing)}."
            if rule not in rules:
                rules.append(rule)
    return rules


def gate_candidate(baseline: AiforaiScoreSummary, candidate: AiforaiScoreSummary) -> AiforaiGateDecision:
    if candidate.aggregate_hard > baseline.aggregate_hard:
        return AiforaiGateDecision(True, "accept", "candidate aggregate hard score improved")
    return AiforaiGateDecision(False, "reject", "candidate did not improve aggregate hard score")


def _score_task(task: AiforaiTaskRecord, skill: str) -> tuple[float, float, list[str]]:
    required = [str(check.get("arg")) for check in task.judge.get("checks", []) if check.get("op") == "contains"]
    lower_skill = skill.lower()
    missing = [item for item in required if item.lower() not in lower_skill]
    if not required:
        return 0.0, 0.0, ["no local checks"]
    soft = (len(required) - len(missing)) / len(required)
    hard = 1.0 if not missing else 0.0
    return hard, soft, missing


def _summarize(results: list[AiforaiReplayResult]) -> AiforaiScoreSummary:
    if not results:
        return AiforaiScoreSummary(0.0, 0.0)
    hard = sum(result.hard for result in results) / len(results)
    soft = sum(result.soft for result in results) / len(results)
    return AiforaiScoreSummary(
        aggregate_hard=hard,
        aggregate_soft=soft,
        by_source=_group_mean(results, "source_agent"),
        by_family=_group_mean(results, "task_family"),
        results=results,
    )


def _group_mean(results: list[AiforaiReplayResult], attr: str) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for result in results:
        buckets.setdefault(str(getattr(result, attr)), []).append(result.hard)
    return {key: sum(values) / len(values) for key, values in buckets.items()}
```

- [ ] **Step 5: Run replay/gate tests**

Run:

```bash
python3 -m unittest tests.test_aiforai_run.AiforaiReplayGateTests -v
```

Expected: PASS `AiforaiReplayGateTests`.

- [ ] **Step 6: Commit**

Run:

```bash
git add skillopt_sleep/aiforai/regression_suite.py skillopt_sleep/aiforai/replay.py tests/test_aiforai_run.py
git commit -m "feat: add AIForAI mock replay gate"
```

---

### Task 9: Mock Run, Staging, And Adopt

**Files:**
- Modify: `skillopt_sleep/aiforai/run.py`
- Modify: `skillopt_sleep/aiforai/report.py`
- Modify: `skillopt_sleep/aiforai/cli.py`
- Modify: `tests/test_aiforai_run.py`

- [ ] **Step 1: Add run/adopt integration tests**

Append to `tests/test_aiforai_run.py`:

```python
from skillopt_sleep.aiforai.run import adopt_latest, run_mock_gate


class AiforaiMockRunTests(unittest.TestCase):
    def test_run_mock_gate_stages_candidate_without_live_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "AIForAI"
            skill_dir = repo / "ai-model-rd-protocol"
            skill_dir.mkdir(parents=True)
            skill_path = skill_dir / "SKILL.md"
            skill_path.write_text("---\nname: ai-model-rd-protocol\n---\n\n# Skill\n", encoding="utf-8")
            (repo / "scripts").mkdir()
            (repo / "tests").mkdir()
            (repo / "scripts/quick_validate.py").write_text("import sys; print('ok')\n", encoding="utf-8")
            cfg = AiforaiConfig(target_skill_repo=str(repo))
            result = run_mock_gate(
                cfg,
                sessions=[
                    AiforaiSessionDigest(
                        source_agent="codex",
                        session_id="s1",
                        raw_path="/tmp/raw",
                        cwd="/repo",
                        user_prompts=["start a training run"],
                    )
                ],
                run_validators=False,
            )

            self.assertTrue(result.accepted)
            self.assertTrue((Path(result.staging_dir) / "proposed_SKILL.md").exists())
            self.assertNotIn("SKILLOPT-AIFORAI", skill_path.read_text(encoding="utf-8"))

    def test_adopt_latest_updates_skill_and_writes_backup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "AIForAI"
            skill_dir = repo / "ai-model-rd-protocol"
            skill_dir.mkdir(parents=True)
            skill_path = skill_dir / "SKILL.md"
            skill_path.write_text("# Skill\n", encoding="utf-8")
            cfg = AiforaiConfig(target_skill_repo=str(repo))
            result = run_mock_gate(
                cfg,
                sessions=[
                    AiforaiSessionDigest(
                        source_agent="codex",
                        session_id="s1",
                        raw_path="/tmp/raw",
                        cwd="/repo",
                        user_prompts=["start a training run"],
                    )
                ],
                run_validators=False,
            )

            updated = adopt_latest(cfg)

            self.assertEqual(updated, [str(skill_path)])
            self.assertIn("SKILLOPT-AIFORAI", skill_path.read_text(encoding="utf-8"))
            self.assertTrue((Path(result.staging_dir) / "backup" / "SKILL.md").exists())
```

- [ ] **Step 2: Run new run tests to confirm failure**

Run:

```bash
python3 -m unittest tests.test_aiforai_run.AiforaiMockRunTests -v
```

Expected: FAIL because `run_mock_gate` and `adopt_latest` are not implemented.

- [ ] **Step 3: Extend report writer for run staging**

Add to `skillopt_sleep/aiforai/report.py`:

```python
def write_run_report(
    out_dir: str,
    *,
    result: AiforaiRunResult,
    proposed_skill: str,
    baseline_rows: list[dict],
    candidate_rows: list[dict],
    validation: dict,
) -> None:
    with open(os.path.join(out_dir, "proposed_SKILL.md"), "w", encoding="utf-8") as handle:
        handle.write(proposed_skill)
    write_jsonl(os.path.join(out_dir, "baseline_results.jsonl"), baseline_rows)
    write_jsonl(os.path.join(out_dir, "candidate_results.jsonl"), candidate_rows)
    with open(os.path.join(out_dir, "validation.log"), "w", encoding="utf-8") as handle:
        handle.write(json.dumps(validation, ensure_ascii=False, indent=2))
    lines = [
        "# AIForAI SkillOpt-Sleep Run Report",
        "",
        f"- accepted: {result.accepted}",
        f"- baseline_score: {result.baseline_score:.4f}",
        f"- candidate_score: {result.candidate_score:.4f}",
        f"- checkable_tasks: {result.checkable_tasks}",
        "",
        "## Boundary",
        "- This run staged a proposal only.",
        "- Live skill mutation requires explicit adopt.",
    ]
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as handle:
        json.dump(result.to_dict(), handle, ensure_ascii=False, indent=2)
```

- [ ] **Step 4: Implement mock run and adopt**

Add to `skillopt_sleep/aiforai/run.py`:

```python
import json
import os
import shutil
from pathlib import Path

from skillopt_sleep.aiforai.regression_suite import curated_regression_tasks
from skillopt_sleep.aiforai.replay import evaluate_tasks, gate_candidate, propose_mock_rules
from skillopt_sleep.aiforai.report import write_jsonl, write_run_report
from skillopt_sleep.aiforai.skill_adapter import (
    apply_learned_rules,
    current_learned_rules,
    read_skill,
    run_aiforai_validators,
    write_skill,
)


def run_mock_gate(
    cfg: AiforaiConfig,
    *,
    sessions=None,
    run_validators: bool = True,
) -> AiforaiRunResult:
    selected_sessions = list(sessions) if sessions is not None else []
    if sessions is None:
        for harvester in default_harvesters(cfg.sources):
            selected_sessions.extend(harvester.harvest(cfg))
    tasks, uncheckable = mine_tasks(selected_sessions, max_tasks_per_source=cfg.max_tasks_per_source)
    tasks = split_tasks(tasks, val_fraction=cfg.val_fraction, test_fraction=cfg.test_fraction, seed=cfg.seed)
    eval_tasks = [task for task in tasks if task.split == "val"] or tasks
    eval_tasks = eval_tasks + curated_regression_tasks()
    skill = read_skill(cfg.skill_path)
    baseline = evaluate_tasks(eval_tasks, skill=skill)
    rules = current_learned_rules(skill) + propose_mock_rules(eval_tasks, skill=skill)
    candidate_skill = apply_learned_rules(skill, rules)
    candidate = evaluate_tasks(eval_tasks, skill=candidate_skill)
    gate = gate_candidate(baseline, candidate)
    validation = run_aiforai_validators(cfg.target_skill_repo) if run_validators else {"ok": True, "commands": []}
    accepted = gate.accepted and bool(validation.get("ok"))
    out_dir = make_staging_dir(cfg.target_skill_repo, prefix="run")
    result = AiforaiRunResult(
        mode="run",
        staging_dir=out_dir,
        sessions_by_source=dict(Counter(session.source_agent for session in selected_sessions)),
        tasks_by_source=dict(Counter(task.source_agent for task in tasks)),
        checkable_tasks=len(tasks),
        uncheckable_candidates=len(uncheckable),
        accepted=accepted,
        baseline_score=baseline.aggregate_hard,
        candidate_score=candidate.aggregate_hard,
        notes=[gate.reason] + ([] if validation.get("ok") else ["validators failed"]),
    )
    manifest = {
        "live_skill_path": cfg.skill_path,
        "accepted": accepted,
        "has_skill": accepted,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    write_jsonl(os.path.join(out_dir, "task_manifest.jsonl"), [task.to_dict() for task in tasks])
    write_jsonl(os.path.join(out_dir, "uncheckable_candidates.jsonl"), uncheckable)
    write_run_report(
        out_dir,
        result=result,
        proposed_skill=candidate_skill if accepted else skill,
        baseline_rows=[row.to_dict() for row in baseline.results],
        candidate_rows=[row.to_dict() for row in candidate.results],
        validation=validation,
    )
    return result


def adopt_latest(cfg: AiforaiConfig) -> list[str]:
    root = Path(cfg.staging_root)
    if not root.is_dir():
        return []
    candidates = sorted([path for path in root.iterdir() if path.is_dir()], key=lambda path: path.stat().st_mtime, reverse=True)
    for staging in candidates:
        manifest_path = staging / "manifest.json"
        proposed_path = staging / "proposed_SKILL.md"
        if not manifest_path.exists() or not proposed_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not manifest.get("accepted"):
            continue
        live = manifest.get("live_skill_path") or cfg.skill_path
        backup_dir = staging / "backup"
        backup_dir.mkdir(exist_ok=True)
        if os.path.exists(live):
            shutil.copy2(live, backup_dir / os.path.basename(live))
        write_skill(live, proposed_path.read_text(encoding="utf-8"))
        return [live]
    return []
```

- [ ] **Step 5: Add CLI run/adopt commands**

Update `skillopt_sleep/aiforai/cli.py` so parser includes:

```python
    run = sub.add_parser("run")
    run.add_argument("--target-skill-repo", required=True)
    run.add_argument("--sources", default="codex,claude,codewhale")
    run.add_argument("--lookback-days", type=int, default=7)
    run.add_argument("--max-tasks-per-source", type=int, default=40)
    run.add_argument("--json", action="store_true")

    adopt = sub.add_parser("adopt")
    adopt.add_argument("--target-skill-repo", required=True)
```

Update `main()` so:

```python
    if args.cmd == "run":
        from skillopt_sleep.aiforai.run import run_mock_gate
        cfg = AiforaiConfig(
            target_skill_repo=args.target_skill_repo,
            sources=_sources(args.sources),
            lookback_days=args.lookback_days,
            max_tasks_per_source=args.max_tasks_per_source,
        )
        result = run_mock_gate(cfg)
        if args.json:
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        else:
            print(f"[aiforai] run staged: {result.staging_dir}")
            print(f"[aiforai] accepted: {result.accepted}")
        return 0
    if args.cmd == "adopt":
        from skillopt_sleep.aiforai.run import adopt_latest
        cfg = AiforaiConfig(target_skill_repo=args.target_skill_repo)
        updated = adopt_latest(cfg)
        for path in updated:
            print(f"[aiforai] adopted: {path}")
        if not updated:
            print("[aiforai] no accepted staging proposal to adopt")
            return 1
        return 0
```

- [ ] **Step 6: Run mock run tests**

Run:

```bash
python3 -m unittest tests.test_aiforai_run.AiforaiMockRunTests -v
```

Expected: PASS `AiforaiMockRunTests`.

- [ ] **Step 7: Commit**

Run:

```bash
git add skillopt_sleep/aiforai tests/test_aiforai_run.py
git commit -m "feat: stage AIForAI mock gated updates"
```

---

### Task 10: Full Verification And Real Audit Smoke

**Files:**
- Modify only if verification finds issues in files from previous tasks.

- [ ] **Step 1: Run focused AIForAI tests**

Run:

```bash
python3 -m unittest \
  tests.test_aiforai_types \
  tests.test_aiforai_harvesters \
  tests.test_aiforai_mine \
  tests.test_aiforai_skill_adapter \
  tests.test_aiforai_run \
  -v
```

Expected: PASS all AIForAI-specific tests.

- [ ] **Step 2: Run existing SkillOpt-Sleep tests**

Run:

```bash
python3 -m unittest tests.test_sleep_engine -v
```

Expected: PASS existing SkillOpt-Sleep tests.

- [ ] **Step 3: Run full test suite if dependencies are available**

Run:

```bash
python3 -m unittest discover -s tests -v
```

Expected: PASS or document unrelated dependency/import failures. Do not claim full-suite pass unless the command exits 0.

- [ ] **Step 4: Run real read-only audit smoke**

Run:

```bash
python3 -m skillopt_sleep.aiforai audit \
  --target-skill-repo /Users/zhangjinouwen/Github/skill/AIForAI \
  --sources codex,claude,codewhale \
  --lookback-days 30 \
  --json
```

Expected: exit 0. Output includes `"mode": "audit"` and a `staging_dir` under `/Users/zhangjinouwen/Github/skill/AIForAI/.skillopt-sleep/staging/`.

- [ ] **Step 5: Verify AIForAI repo was not modified by audit**

Run:

```bash
git -C /Users/zhangjinouwen/Github/skill/AIForAI status --short
```

Expected: either clean, or only `.skillopt-sleep/` appears if not ignored. If `.skillopt-sleep/` appears, add an ignore rule only with user approval or document it as a generated staging artifact.

- [ ] **Step 6: Verify SkillOpt repo status**

Run:

```bash
git status --short
```

Expected: no unexpected files. Only intended implementation files should be present before final commit.

- [ ] **Step 7: Final implementation commit**

Run:

```bash
git add skillopt_sleep/aiforai tests/test_aiforai_*.py
git commit -m "feat: add AIForAI tri-agent sleep MVP"
```

Expected: commit succeeds. If there is nothing to commit because previous task commits already captured everything, skip this step and report the last task commit SHA.

---

## Self-Review Checklist For Implementer

- [ ] Phase 1 audit works with all three source names even if one source has no local files.
- [ ] Phase 2 mock run never mutates live `SKILL.md` before `adopt`.
- [ ] Candidate edits only appear inside `SKILLOPT-AIFORAI` learned block.
- [ ] CodeWhale data stored under `~/.deepseek` reports as `source_agent="codewhale"` while preserving raw paths.
- [ ] Uncheckable candidates are written to a report but not used for gate.
- [ ] Existing `skillopt_sleep` behavior and tests still pass.
- [ ] No secrets or raw auth blobs are included in reports.
- [ ] Final report distinguishes verified evidence from unverified boundaries.
