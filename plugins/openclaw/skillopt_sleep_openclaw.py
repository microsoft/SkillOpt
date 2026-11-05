"""OpenClaw backend for SkillOpt-Sleep.

Adapts the skillopt_sleep Backend protocol to our DeepSeek + Ollama stack:
  - attempt/judge/reflect  ->  DeepSeek V4 Pro (or Flash for cost)
  - embeddings              ->  Ollama nomic-embed-text (already configured)

This backend NEVER mutates live state. It only returns text + EditRecord
proposals that the gate stages for human review.
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Tuple

from skillopt_sleep.backend import Backend, _optimizer_feedback, exact_score
from skillopt_sleep.types import EditRecord, ReplayResult, TaskRecord

# ── DeepSeek + Ollama OpenAI-compatible API client (curl-based, no extra deps) ──


def _chat(messages: List[Dict[str, str]], *, model: str, temperature: float = 0.2, max_tokens: int = 1500) -> str:
    """Call DeepSeek V4 Pro via curl + jq. No extra Python deps needed."""
    import json as _json
    import urllib.request

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        # try loading from .env
        env_path = os.path.expanduser("~/.openclaw/.env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    if line.startswith("DEEPSEEK_API_KEY="):
                        api_key = line.split("=", 1)[1].strip()
                        break

    base = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=_json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = _json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[BACKEND_ERROR] {type(e).__name__}: {str(e)[:200]}"


def _embed(text: str) -> List[float]:
    """Call Ollama for embeddings. Uses the configured nomic-embed-text model."""
    import json as _json
    import urllib.request

    try:
        req = urllib.request.Request(
            "http://127.0.0.1:11434/api/embeddings",
            data=_json.dumps({"model": "nomic-embed-text:latest", "prompt": text[:2000]}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = _json.loads(resp.read().decode("utf-8"))
            return data.get("embedding", [])
    except Exception:
        return []


# ── Backend implementation ────────────────────────────────────────────────────


class OpenClawDeepSeekBackend(Backend):
    """Use DeepSeek V4 Pro for attempt/judge/reflect, Ollama for embeddings.

    - "model" passed to constructor = optimizer model (default: deepseek-v4-pro)
    - "judge_model" = judge model (default: deepseek-v4-pro for quality)
    - "cheap_model" = budget-fallback (deepseek-v4-flash)
    """

    name = "openclaw-deepseek"

    def __init__(
        self,
        model: str = "deepseek-v4-pro",
        judge_model: str = "deepseek-v4-pro",
        cheap_model: str = "deepseek-v4-flash",
    ):
        self._model = model
        self._judge_model = judge_model
        self._cheap_model = cheap_model
        self._tokens = 0  # rough estimate

    def tokens_used(self) -> int:
        return self._tokens

    # ── 1. attempt: produce a response given the task + skill + memory ──
    def attempt(self, task: TaskRecord, skill: str, memory: str) -> str:
        sys = (
            "You are an OpenClaw agent (Kobe ecosystem). Use the skill and memory below to complete the task. "
            "If the task asks for a structured output, follow the rubric exactly. "
            "Be concise. No preamble, no explanation unless the task asks for it."
        )
        usr = f"""## SKILL
{skill or '(no skill yet)'}

## MEMORY
{memory or '(no memory yet)'}

## TASK
{task.intent}

## CONTEXT (if any)
{task.context_excerpt or '(none)'}

## RESPONSE
"""
        out = _chat(
            [{"role": "system", "content": sys}, {"role": "user", "content": usr}],
            model=self._model,
            temperature=0.2,
        )
        self._tokens += len(usr) // 4 + 200
        return out

    # ── 2. judge: score the response ──
    def judge(self, task: TaskRecord, response: str) -> Tuple[float, float, str]:
        # Hard score: exact-match against task.reference (if available)
        hard = exact_score(task.reference or "", response)

        # Soft score: LLM judge against rubric (reference if reference_kind=='rubric')
        rubric_text = task.reference if task.reference_kind == "rubric" else ""
        if rubric_text:
            judge_prompt = f"""You are a strict grader. Score the response 0.0-1.0 against the rubric.

## TASK
{task.intent}

## REFERENCE
{task.reference or '(none)'}

## RUBRIC
{rubric_text}

## RESPONSE
{response[:3000]}

## INSTRUCTIONS
Return ONLY a single float 0.0-1.0 on one line. No explanation. No markdown.
"""
            try:
                j_out = _chat(
                    [{"role": "user", "content": judge_prompt}],
                    model=self._judge_model,
                    temperature=0.0,
                    max_tokens=20,
                ).strip()
                soft = float(re.search(r"[\d.]+", j_out.splitlines()[0]).group())
                soft = max(0.0, min(1.0, soft))
            except Exception:
                soft = hard
            self._tokens += 600
        else:
            soft = hard

        rationale = f"hard={hard:.2f} soft={soft:.2f}"
        return hard, soft, rationale

    # ── 3. reflect: produce bounded EditRecord proposals ──
    def reflect(
        self,
        failures: List[Tuple[TaskRecord, ReplayResult]],
        successes: List[Tuple[TaskRecord, ReplayResult]],
        skill: str,
        memory: str,
        *,
        edit_budget: int,
        evolve_skill: bool,
        evolve_memory: bool,
    ) -> List[EditRecord]:
        # Compact digest of failures + successes
        fail_digest = "\n".join(
            f"- TASK: {t.intent[:200]}\n  RESPONSE: {r.response[:300]}\n"
            f"  ACTIONABLE FEEDBACK: {_optimizer_feedback(t, r)[:240]}\n"
            f"  REFERENCE: {t.reference[:200]}"
            for t, r in failures[:5]
        ) or "(none)"
        succ_digest = "\n".join(
            f"- TASK: {t.intent[:150]} -> OK" for t, _r in successes[:3]
        ) or "(none)"

        rubric_text = ""
        if failures:
            reference_answers = "\n".join(
                f"Q: {t.intent[:120]}\nA: {t.reference}"
                for t, _ in failures[:3]
                if t.reference
            )
            rubric_text = f"\n\n## REFERENCE ANSWERS\n{reference_answers}"

        sys = (
            "You are SkillOpt-Sleep's bounded-edit optimizer. Your job is to propose 1-4 MINIMAL text edits to a skill or memory document "
            "that, if applied, would help future agents do better on the failed tasks. "
            "NEVER propose adding new sections wholesale. NEVER delete entire sections. "
            "Edit primitives: ADD (append a step/rule at end), DELETE (remove a specific line by exact match), REPLACE (swap a specific line for another by exact match). "
            "Use only the actionable semantic feedback; never quote or reconstruct private verifier syntax such as regexes or check source. "
            "If you cannot identify a clear, minimal improvement, return an empty list."
        )
        usr = f"""## CURRENT SKILL
{skill or '(empty)'}

## CURRENT MEMORY
{memory or '(empty)'}

## FAILED TASKS
{fail_digest}

## SUCCESSFUL TASKS
{succ_digest}
{rubric_text}

## CONSTRAINTS
- max {edit_budget} edits total
- edits go to {"skill + memory" if (evolve_skill and evolve_memory) else ("skill" if evolve_skill else "memory")}
- if evolve_skill=False, target="memory" only; if evolve_memory=False, target="skill" only
- target must be "skill" or "memory"

## OUTPUT FORMAT (JSON, no markdown)
{{"edits": [{{"op": "ADD"|"DELETE"|"REPLACE", "target": "skill"|"memory", "content": "the text to add or replace with", "old_text": "for REPLACE/DELETE, the exact line to find", "rationale": "one short sentence why"}}]}}
"""
        out = _chat(
            [{"role": "system", "content": sys}, {"role": "user", "content": usr}],
            model=self._model,
            temperature=0.4,
            max_tokens=2000,
        )
        self._tokens += len(usr) // 3 + 1500

        # parse
        try:
            # strip markdown fences if any
            cleaned = out.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```[a-z]*\n?", "", cleaned)
                cleaned = re.sub(r"\n?```$", "", cleaned)
            data = json.loads(cleaned)
            edits: List[EditRecord] = []
            for e in data.get("edits", [])[:edit_budget]:
                if e.get("op") not in ("ADD", "DELETE", "REPLACE"):
                    continue
                target = e.get("target", "skill")
                if target not in ("skill", "memory"):
                    continue
                if not evolve_skill and target == "skill":
                    continue
                if not evolve_memory and target == "memory":
                    continue
                edits.append(EditRecord(
                    op=e["op"],
                    target=target,
                    content=e.get("content", ""),
                    old_text=e.get("old_text", ""),
                    rationale=e.get("rationale", ""),
                ))
            return edits
        except Exception as e:
            # log + return empty list (no edit is better than a bad edit)
            return []
