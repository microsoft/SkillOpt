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
            sessions.append(self._digest_thread(row, codex_home=cfg.codex_home))
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

    def _digest_thread(self, row: dict[str, Any], *, codex_home: str = "") -> AiforaiSessionDigest:
        rollout_path = str(row.get("rollout_path") or "")
        user_prompts: list[str] = []
        assistant_finals: list[str] = []
        tools: list[str] = []
        files: list[str] = []
        feedback: list[str] = []
        mentions: list[str] = []
        warnings: list[str] = []
        event_count = 0

        def build_digest() -> AiforaiSessionDigest:
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

        if rollout_path and codex_home and not self._is_within_codex_home(rollout_path, codex_home):
            warnings.append(f"rollout_path outside codex_home: {rollout_path}")
            return build_digest()

        for rec in iter_jsonl(rollout_path):
            event_count += 1
            text = self._record_text(rec)
            if text:
                text = redact_text(text).strip()
                mentions.extend(detect_skill_mentions(text))
            rtype = self._record_type(rec)
            role = self._record_role(rec)
            if rtype == "message":
                if role == "user":
                    if text:
                        user_prompts.append(text)
                        feedback.extend(detect_feedback(text))
                elif role == "assistant" and text:
                    assistant_finals.append(text)
            elif "user" in rtype:
                if text:
                    user_prompts.append(text)
                    feedback.extend(detect_feedback(text))
            elif "agent" in rtype or "assistant" in rtype:
                if text:
                    assistant_finals.append(text)
            elif "function_call" in rtype or self._record_value(rec, "name"):
                name = str(self._record_value(rec, "name") or "")
                if name:
                    tools.append(name)
                args = str(self._record_value(rec, "arguments") or "")
                if args:
                    mentions.extend(detect_skill_mentions(args))
                    files.extend(self._file_like_tokens(args))

        if rollout_path and not os.path.exists(rollout_path):
            warnings.append(f"missing rollout_path: {rollout_path}")

        return build_digest()

    def _record_text(self, rec: dict[str, Any]) -> str:
        for source in (self._payload(rec), rec):
            for key in ("message", "text", "content"):
                text = flatten_text(source.get(key))
                if text:
                    return text
        return ""

    def _record_type(self, rec: dict[str, Any]) -> str:
        return str(self._record_value(rec, "type") or "")

    def _record_role(self, rec: dict[str, Any]) -> str:
        return str(self._record_value(rec, "role") or "").lower()

    def _record_value(self, rec: dict[str, Any], key: str) -> Any:
        payload = self._payload(rec)
        if key in payload:
            return payload.get(key)
        return rec.get(key)

    def _payload(self, rec: dict[str, Any]) -> dict[str, Any]:
        payload = rec.get("payload")
        return payload if isinstance(payload, dict) else {}

    def _is_within_codex_home(self, path: str, codex_home: str) -> bool:
        try:
            return os.path.commonpath((os.path.realpath(path), os.path.realpath(codex_home))) == os.path.realpath(
                codex_home
            )
        except ValueError:
            return False

    def _file_like_tokens(self, text: str) -> list[str]:
        out: list[str] = []
        for token in text.replace('"', " ").replace("'", " ").split():
            if "/" in token and len(token) < 240:
                out.append(token.strip(",;()[]{}"))
        return out
