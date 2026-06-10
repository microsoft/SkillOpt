"""AIForAI skill document and validator helpers."""

from __future__ import annotations

import os
import subprocess
import sys
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
    deduped: list[str] = []
    seen: set[str] = set()

    for rule in rules:
        clean = _clean_rule(rule)
        key = _rule_key(clean)
        if clean and key not in seen:
            seen.add(key)
            deduped.append(clean)

    block = _render_learned_block(deduped)
    block_span = _find_learned_block(doc)
    if block_span is None:
        return _append_learned_block(doc, block)
    start, end = block_span
    return doc[:start] + block + doc[end:]


def run_aiforai_validators(repo: str, *, timeout: int = 120) -> dict[str, Any]:
    command_specs = [
        (
            "quick_validate",
            [sys.executable, "scripts/quick_validate.py", "ai-model-rd-protocol"],
        ),
        (
            "unittest_discover",
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        ),
    ]

    commands: list[dict[str, Any]] = []
    overall_ok = True

    for name, cmd in command_specs:
        stdout = ""
        stderr = ""
        returncode = None
        ok = False
        status = "failed"
        failure_type: str | None = None
        try:
            completed = subprocess.run(
                cmd,
                cwd=repo,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            stdout = _coerce_output(completed.stdout)
            stderr = _coerce_output(completed.stderr)
            ok = completed.returncode == 0
            returncode = completed.returncode
            status = "passed" if ok else "failed"
            if not ok:
                failure_type = "exit_code"
        except subprocess.TimeoutExpired as exc:
            stdout = _coerce_output(getattr(exc, "stdout", None) or getattr(exc, "output", None))
            stderr = _coerce_output(getattr(exc, "stderr", None))
            status = "failed"
            failure_type = "timeout"
        except OSError as exc:
            stderr = str(exc)
            status = "failed"
            failure_type = "os_error"
        except Exception as exc:  # noqa: BLE001
            stderr = str(exc)
            status = "failed"
            failure_type = "exception"

        output = _combine_output(stdout, stderr)

        overall_ok = overall_ok and ok
        commands.append(
            {
                "name": name,
                "cmd": cmd,
                "ok": ok,
                "status": status,
                "failure_type": failure_type,
                "returncode": returncode,
                "stdout": stdout[-_OUTPUT_TAIL:],
                "stderr": stderr[-_OUTPUT_TAIL:],
                "output": output[-_OUTPUT_TAIL:],
            }
        )

    return {"ok": overall_ok, "commands": commands}


def _extract_learned(doc: str) -> str:
    block_span = _find_learned_block(doc)
    if block_span is None:
        return ""
    start, end = block_span
    return doc[start + len(LEARNED_START):end - len(LEARNED_END)].strip()


def _strip_learned(doc: str) -> str:
    block_span = _find_learned_block(doc)
    if block_span is None:
        return doc.rstrip()
    start, end = block_span
    text = doc[:start] + doc[end:]
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.rstrip()


def _clean_rule(rule: str) -> str:
    clean = rule.strip()
    if clean.startswith("- "):
        clean = clean[2:]
    return clean.strip()


def _rule_key(rule: str) -> str:
    return " ".join(rule.lower().split())


def _find_learned_block(doc: str) -> tuple[int, int] | None:
    start = doc.find(LEARNED_START)
    if start == -1:
        return None
    after_start = start + len(LEARNED_START)
    end = doc.find(LEARNED_END, after_start)
    nested_start = doc.find(LEARNED_START, after_start)
    if end == -1 or (nested_start != -1 and nested_start < end):
        return None
    return start, end + len(LEARNED_END)


def _render_learned_block(rules: list[str]) -> str:
    body = "\n".join(f"- {rule}" for rule in rules)
    body_section = f"\n\n{body}" if body else ""
    return (
        f"{LEARNED_START}\n"
        f"{_TITLE}\n\n"
        f"{BANNER}\n"
        f"{body_section}\n"
        f"{LEARNED_END}"
    )


def _append_learned_block(doc: str, block: str) -> str:
    base = doc.rstrip("\n")
    if not base:
        return f"{block}\n"
    return f"{base}\n\n{block}\n"


def _coerce_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _combine_output(stdout: str, stderr: str) -> str:
    return stdout + stderr
