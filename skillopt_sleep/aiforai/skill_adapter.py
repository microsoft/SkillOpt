"""AIForAI skill document and validator helpers."""

from __future__ import annotations

import os
import subprocess
from typing import Any


LEARNED_START = "<!-- SKILLOPT-AIFORAI:LEARNED START -->"
LEARNED_END = "<!-- SKILLOPT-AIFORAI:LEARNED END -->"
BANNER = (
    "_This block is maintained by AIForAI SkillOpt-Sleep. It is staged and "
    "validated before adoption. Handwritten content outside this block is "
    "never changed._"
)

_TITLE = "## Learned AIForAI Rules"
_OUTPUT_TAIL = 4000


def read_skill(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def write_skill(path: str, content: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


def current_learned_rules(doc: str) -> list[str]:
    rules: list[str] = []
    seen: set[str] = set()

    for line in _extract_learned(doc).splitlines():
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        rule = _clean_rule(stripped)
        key = _rule_key(rule)
        if rule and key not in seen:
            seen.add(key)
            rules.append(rule)

    return rules


def apply_learned_rules(doc: str, rules: list[str]) -> str:
    base = _strip_learned(doc)
    deduped: list[str] = []
    seen: set[str] = set()

    for rule in rules:
        clean = _clean_rule(rule)
        key = _rule_key(clean)
        if clean and key not in seen:
            seen.add(key)
            deduped.append(clean)

    body = "\n".join(f"- {rule}" for rule in deduped)
    body_section = f"\n\n{body}" if body else ""
    block = (
        f"\n\n{LEARNED_START}\n"
        f"{_TITLE}\n\n"
        f"{BANNER}\n"
        f"{body_section}\n"
        f"{LEARNED_END}\n"
    )
    return (base.rstrip() + block).lstrip("\n")


def run_aiforai_validators(repo: str, *, timeout: int = 120) -> dict[str, Any]:
    command_specs = [
        (
            "quick_validate",
            ["python3", "scripts/quick_validate.py", "ai-model-rd-protocol"],
        ),
        (
            "unittest_discover",
            ["python3", "-m", "unittest", "discover", "-s", "tests", "-v"],
        ),
    ]

    commands: list[dict[str, Any]] = []
    overall_ok = True

    for name, cmd in command_specs:
        try:
            completed = subprocess.run(
                cmd,
                cwd=repo,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            output = (completed.stdout or "") + (completed.stderr or "")
            ok = completed.returncode == 0
            returncode = completed.returncode
        except Exception as exc:  # noqa: BLE001
            output = str(exc)
            ok = False
            returncode = None

        overall_ok = overall_ok and ok
        commands.append(
            {
                "name": name,
                "cmd": cmd,
                "ok": ok,
                "returncode": returncode,
                "output": output[-_OUTPUT_TAIL:],
            }
        )

    return {"ok": overall_ok, "commands": commands}


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


def _clean_rule(rule: str) -> str:
    return rule.strip().lstrip("- ").strip()


def _rule_key(rule: str) -> str:
    return " ".join(rule.lower().split())
