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
