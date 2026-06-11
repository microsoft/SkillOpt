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


SECRET_KEY_VALUE_PATTERN = re.compile(
    r"""(?ix)
    (?P<prefix>
        (?P<key_quote>["']?)
        (?P<key>OPENAI_API_KEY|ANTHROPIC_API_KEY|AZURE_OPENAI_API_KEY|token|api_key)
        (?P=key_quote)
        \s*[:=]\s*
    )
    (?P<value>
        "(?:[^"\\]|\\.)*"
        |
        '(?:[^'\\]|\\.)*'
        |
        [^\s,}\]]+
    )
    """
)
SECRET_LITERAL_PATTERN = re.compile(r"sk-[A-Za-z0-9_-]{12,}")

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
            part = flatten_text(item)
            if part:
                parts.append(part)
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        if "text" in value:
            text = flatten_text(value["text"])
            if text:
                return text
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
    def replace_secret_value(match: re.Match[str]) -> str:
        value = match.group("value")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            return f'{match.group("prefix")}{value[0]}<redacted>{value[-1]}'
        return f'{match.group("prefix")}<redacted>'

    redacted = SECRET_KEY_VALUE_PATTERN.sub(replace_secret_value, text or "")
    return SECRET_LITERAL_PATTERN.sub("<redacted-secret>", redacted)


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
