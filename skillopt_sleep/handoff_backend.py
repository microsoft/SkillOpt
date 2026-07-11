"""SkillOpt-Sleep — handoff backend (session-executed model calls).

Runs the sleep cycle WITHOUT spawning any model subprocess or API call.
Every intelligent operation (attempt / judge / reflect) is turned into a
prompt file that an interactive agent session answers between engine runs:

    run 1: the engine executes the deterministic stages; every model call
           it needs is recorded as a pending prompt; the run stops and
           writes PROMPTS.md + pending.json into the handoff directory.
    you:   answer each prompt (each in a FRESH context, so the session's
           own history cannot contaminate the held-out gate) and write the
           raw answer text to answers/<id>.md.
    run 2: the engine re-runs; answered prompts resolve from answers/, the
           cycle advances to the next model-dependent stage, and either
           finishes or writes the next PROMPTS.md batch.

Resume needs no serialized engine state: harvest -> mine -> replay is
deterministic, so re-running regenerates identical prompts and the answers
directory acts as a persistent, cross-run call cache. A prompt that embeds
a still-unanswered response (detected via the pending sentinel) aborts the
run immediately so placeholder text never propagates into scores, edits,
or staging. A typical night converges in 3-6 rounds: baseline attempts ->
reflect -> candidate re-scoring per accepted edit.

Limitations (v1): `dream_rollouts > 1` yields no contrastive spread (the
same prompt maps to the same answer file), and tool-loop tasks fall back
to the base single-shot 'TOOL_CALL: <name>' marker convention.
"""
from __future__ import annotations

import json
import os
import threading
from typing import Dict

from skillopt_sleep.backend import CliBackend, skill_hash

PENDING_SENTINEL_PREFIX = "[[SKILLOPT-SLEEP-PENDING:"
PENDING_SENTINEL_SUFFIX = "]]"

# reflect() appends this when a reply fails to parse; with a placeholder
# reply the retry is a dependent call, not a genuinely new question.
_REFLECT_RETRY_MARKER = "your previous reply was not valid JSON"

PROMPTS_FILENAME = "PROMPTS.md"
PENDING_FILENAME = "pending.json"


class PendingCalls(RuntimeError):
    """The cycle cannot advance until pending prompts are answered."""

    def __init__(self, pending: Dict[str, Dict[str, object]]):
        self.pending = dict(pending)
        super().__init__(
            f"{len(self.pending)} model call(s) awaiting handoff answers"
        )


class HandoffBackend(CliBackend):
    """Backend that outsources every model call to prompt/answer files.

    ``_call`` resolves a prompt from ``answers/<sha256[:16]>.md`` when the
    answer exists; otherwise it records the prompt as pending and returns a
    sentinel placeholder so independent calls in the same phase can still
    be collected into one batch. Any call whose prompt was BUILT FROM a
    placeholder raises :class:`PendingCalls` — that call depends on answers
    the user has not provided yet, so continuing would only mint garbage.
    """

    name = "handoff"

    def __init__(self, model: str = "", handoff_dir: str = "") -> None:
        super().__init__(model=model, timeout=0)
        self.handoff_dir = os.path.abspath(
            handoff_dir or os.path.join(os.getcwd(), ".skillopt-sleep-handoff")
        )
        self.answers_dir = os.path.join(self.handoff_dir, "answers")
        os.makedirs(self.answers_dir, exist_ok=True)
        # key -> {"prompt": str, "max_tokens": int}, insertion-ordered
        self.pending: Dict[str, Dict[str, object]] = {}
        self._lock = threading.Lock()

    # ── prompt/answer plumbing ────────────────────────────────────────────
    def answer_path(self, key: str) -> str:
        return os.path.join(self.answers_dir, f"{key}.md")

    def _call(self, prompt: str, *, max_tokens: int = 1024) -> str:
        if PENDING_SENTINEL_PREFIX in prompt:
            # Built from a still-pending response — dependent call.
            raise PendingCalls(self.pending)
        if _REFLECT_RETRY_MARKER in prompt and self.pending:
            # Retry of a reflect whose first reply is the placeholder.
            raise PendingCalls(self.pending)
        key = skill_hash(prompt)
        path = self.answer_path(key)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return f.read().strip()
        with self._lock:
            self.pending[key] = {"prompt": prompt, "max_tokens": max_tokens}
        return f"{PENDING_SENTINEL_PREFIX}{key}{PENDING_SENTINEL_SUFFIX}"

    # ── handoff file emission ─────────────────────────────────────────────
    def flush_pending(self) -> str:
        """Write PROMPTS.md (human/agent-readable) + pending.json (machine).

        Prompts can themselves contain markdown fences, so PROMPTS.md
        delimits each prompt with BEGIN/END marker lines instead of fences.
        Returns the PROMPTS.md path.
        """
        from skillopt_sleep.staging import redact_secrets

        os.makedirs(self.handoff_dir, exist_ok=True)
        with self._lock:
            items = list(self.pending.items())
        payload = {
            "format": "skillopt_sleep.handoff.v1",
            "answers_dir": self.answers_dir,
            "pending": [
                {
                    "id": key,
                    "answer_file": self.answer_path(key),
                    "max_tokens": item["max_tokens"],
                    "prompt": redact_secrets(str(item["prompt"])),
                }
                for key, item in items
            ],
        }
        with open(os.path.join(self.handoff_dir, PENDING_FILENAME), "w",
                  encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")

        lines = [
            "# SkillOpt-Sleep — pending model calls (handoff)",
            "",
            f"{len(items)} prompt(s) below need answers before the sleep "
            "cycle can continue.",
            "",
            "For EACH prompt:",
            "",
            "1. Answer it in a FRESH context (e.g. a subagent with no",
            "   conversation history). Do NOT let the current session's",
            "   context, the other prompts in this file, or the optimization",
            "   run itself leak into the answer — that contaminates the",
            "   held-out validation gate.",
            "2. Write ONLY the raw answer text (no commentary, no code",
            "   fences) to the prompt's answer file.",
            "",
            "When every answer file exists, re-run the same engine command",
            "(`python -m skillopt_sleep run --backend handoff ...`); it",
            "resumes automatically from the answers directory.",
            "",
        ]
        for i, (key, item) in enumerate(items, start=1):
            lines += [
                "---",
                "",
                f"## Prompt {i} of {len(items)}",
                "",
                f"- id: `{key}`",
                f"- answer file: `answers/{key}.md`",
                f"- suggested max tokens: {item['max_tokens']}",
                "",
                f"----- BEGIN PROMPT {key} -----",
                redact_secrets(str(item["prompt"])),
                f"----- END PROMPT {key} -----",
                "",
            ]
        prompts_path = os.path.join(self.handoff_dir, PROMPTS_FILENAME)
        with open(prompts_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return prompts_path
