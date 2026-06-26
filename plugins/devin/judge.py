#!/usr/bin/env python3
"""Reference judge for SkillOpt-Sleep — score a candidate reply against a rubric.

Tasks harvested without a hard test/build signal get ``verifier: "judge"`` and a
``rubric`` (see ``_build_rubric`` in harvest_devin.py).  This module is the
scorer the validation gate calls for those tasks: given the rubric and a
candidate reply produced during replay, it returns a score in ``[0, 1]``.  The
gate accepts a skill edit only if the *new* skill scores strictly higher on the
held-out tasks.

It is self-contained on purpose — in a full deployment the SkillOpt engine owns
replay+scoring, but having a runnable reference here lets you sanity-check the
judge path without the engine.

Backends (select via ``SKILLOPT_JUDGE``):
  * ``heuristic`` (default) — keyword-coverage, offline, no API key, deterministic.
  * ``claude``              — LLM judge via the Anthropic API (needs ANTHROPIC_API_KEY).

Usage:
    python judge.py --rubric rubric.json --reply reply.txt
    echo "<reply>" | python judge.py --rubric-inline '["Addresses OrderService", ...]'
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import List

_STOPWORDS = {"addresses", "resolves", "implements", "without", "introducing",
              "behavior", "request", "response", "concrete", "actionable", "not",
              "the", "and", "that", "with", "stated", "reported", "actually",
              "preserves", "improving", "structure", "requested", "satisfies"}

# Cheap, fast model is the right default for a judge.
_JUDGE_MODEL = os.environ.get("SKILLOPT_JUDGE_MODEL", "claude-haiku-4-5-20251001")


def _content_words(text: str) -> List[str]:
    return [w for w in re.findall(r"[A-Za-z][A-Za-z0-9_.\-]{3,}", text.lower())
            if w not in _STOPWORDS]


def heuristic_score(reply: str, rubric: List[str]) -> float:
    """Fraction of rubric criteria whose key content words appear in the reply.

    Crude but deterministic: each criterion is 'met' if at least one of its
    content words shows up in the candidate reply. Good enough to smoke-test the
    gate wiring; swap in the claude backend for real judging.
    """
    if not rubric:
        return 0.0
    low = reply.lower()
    met = 0
    for criterion in rubric:
        words = _content_words(criterion)
        if not words:                       # nothing to check → treat as met
            met += 1
            continue
        if any(w in low for w in words):
            met += 1
    return round(met / len(rubric), 3)


def claude_score(reply: str, rubric: List[str]) -> float:
    """LLM judge via the Anthropic API. Returns a 0..1 score.

    Stdlib-only (urllib) so this file stays dependency-free. Falls back to the
    heuristic if the key is missing or the call fails, so the gate never hard-errors.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("[judge] ANTHROPIC_API_KEY unset — using heuristic", file=sys.stderr)
        return heuristic_score(reply, rubric)
    import urllib.request

    rubric_block = "\n".join(f"- {c}" for c in rubric)
    prompt = (
        "You are scoring an AI agent's reply against a rubric. For each criterion, "
        "decide if the reply satisfies it. Respond with ONLY a number between 0 and "
        "1 — the fraction of criteria satisfied.\n\n"
        f"Rubric:\n{rubric_block}\n\nReply:\n{reply}\n\nScore:"
    )
    body = json.dumps({
        "model": _JUDGE_MODEL,
        "max_tokens": 8,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=body,
        headers={"content-type": "application/json", "x-api-key": api_key,
                 "anthropic-version": "2023-06-01"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.load(resp)
        text = "".join(b.get("text", "") for b in data.get("content", []))
        m = re.search(r"[01](?:\.\d+)?", text)
        return max(0.0, min(1.0, float(m.group(0)))) if m else heuristic_score(reply, rubric)
    except Exception as exc:                 # network/auth/parse — degrade gracefully
        print(f"[judge] claude backend failed ({exc}) — using heuristic", file=sys.stderr)
        return heuristic_score(reply, rubric)


def score(reply: str, rubric: List[str]) -> float:
    backend = os.environ.get("SKILLOPT_JUDGE", "heuristic")
    return claude_score(reply, rubric) if backend == "claude" else heuristic_score(reply, rubric)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Score a reply against a rubric (0..1)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--rubric", help="Path to a JSON file containing a list of criteria")
    g.add_argument("--rubric-inline", help="Inline JSON list of criteria")
    p.add_argument("--reply", help="Path to the reply text (default: stdin)")
    args = p.parse_args(argv)

    rubric = (json.load(open(args.rubric, encoding="utf-8")) if args.rubric
              else json.loads(args.rubric_inline))
    reply = (open(args.reply, encoding="utf-8").read() if args.reply
             else sys.stdin.read())
    print(score(reply, rubric))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
