"""Configuration for AIForAI SkillOpt-Sleep."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

DATACLASS_KWARGS = {"frozen": True, "slots": True} if sys.version_info >= (3, 10) else {"frozen": True}


@dataclass(**DATACLASS_KWARGS)
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
