"""CodeWhale / DeepSeek TUI harvester for AIForAI SkillOpt-Sleep."""

from __future__ import annotations

import os
from typing import Any

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
                text = self._record_text(rec)
                if text:
                    text = redact_text(text).strip()
                    mentions.extend(detect_skill_mentions(text))
                role = str(rec.get("role") or rec.get("type") or "")
                if "user" in role:
                    if text:
                        user_prompts.append(text)
                        feedback.extend(detect_feedback(text))
                elif ("assistant" in role or "agent" in role) and text:
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
                    user_prompts=user_prompts,
                    assistant_finals=assistant_finals[-5:],
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
        thread: dict[str, Any],
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
            text = self._record_text(rec)
            if text:
                text = redact_text(text).strip()
                mentions.extend(detect_skill_mentions(text))
            role = str(rec.get("role") or rec.get("type") or "")
            if "user" in role:
                if text:
                    user_prompts.append(text)
                    feedback.extend(detect_feedback(text))
            elif ("assistant" in role or "agent" in role) and text:
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
            user_prompts=user_prompts,
            assistant_finals=assistant_finals[-5:],
            tools_used=list(dict.fromkeys(tools)),
            feedback_signals=list(dict.fromkeys(feedback)),
            skill_mentions=list(dict.fromkeys(mentions)),
            event_count=count,
            parse_warnings=warnings,
        )

    def _record_text(self, rec: dict[str, Any]) -> str:
        return flatten_text(rec.get("content") or rec.get("message") or rec.get("text"))
