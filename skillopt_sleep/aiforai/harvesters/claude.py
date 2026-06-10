"""Claude trajectory harvester for AIForAI SkillOpt-Sleep."""

from __future__ import annotations

import os

from skillopt_sleep.aiforai.config import AiforaiConfig
from skillopt_sleep.aiforai.harvesters.base import Harvester, detect_skill_mentions, redact_text
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
            def normalize_texts(values: list[str]) -> list[str]:
                normalized: list[str] = []
                for value in values:
                    text = redact_text(value).strip()
                    if text:
                        normalized.append(text)
                return normalized

            user_prompts = normalize_texts(digest.user_prompts)
            assistant_finals = normalize_texts(digest.assistant_finals)
            skill_mentions: list[str] = []
            for text in user_prompts + assistant_finals:
                skill_mentions.extend(detect_skill_mentions(text))
            sessions.append(
                AiforaiSessionDigest(
                    source_agent="claude",
                    session_id=digest.session_id,
                    raw_path=digest.raw_path,
                    cwd=digest.project,
                    git_branch=digest.git_branch,
                    started_at=digest.started_at,
                    ended_at=digest.ended_at,
                    user_prompts=user_prompts,
                    assistant_finals=assistant_finals,
                    tools_used=digest.tools_used,
                    files_touched=digest.files_touched,
                    feedback_signals=digest.feedback_signals,
                    skill_mentions=list(dict.fromkeys(skill_mentions)),
                    event_count=digest.n_user_turns + digest.n_assistant_turns,
                    parse_warnings=[],
                )
            )
        return sessions
