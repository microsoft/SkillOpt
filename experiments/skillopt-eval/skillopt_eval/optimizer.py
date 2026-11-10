"""Candidate proposal for the optimization loop.

The loop treats the optimizer as a black box that emits *bounded edits* to
the current best skill text. Acceptance is decided solely by the held-out
validation gate in experiment.py - never here - so any proposer (human-
authored files, the SkillOpt engine, an LLM backend) plugs in.

Two proposers ship:

* ``FileOptimizer`` - iterates pre-authored candidate SKILL.md files from the
  config (``candidates:``). This is the path a real run uses: a maintainer or
  the SkillOpt optimizer prepares bounded edits offline and the harness gates
  them. Provenance is the file path + content hash.
* ``RuleOptimizer`` - deterministic offline proposer used by the mock runner
  and tests. Appends directive paragraphs from a fixed library; bounded by
  the learning_rate budget.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Protocol

from .harness import skill_hash


@dataclass
class CandidateEdit:
    text: str
    description: str
    source: str            # provenance: file path or rule name
    content_hash: str


class Proposer(Protocol):
    def propose(self, current_text: str, step: int,
                rejected_hashes: List[str]) -> Optional[CandidateEdit]:
        ...


class FileOptimizer:
    """Propose each pre-authored candidate file in order."""

    def __init__(self, paths: List[str]):
        self.paths = [Path(p) for p in paths]
        for p in self.paths:
            if not p.is_file():
                raise FileNotFoundError(f"candidate file not found: {p}")
            if p.is_symlink():
                raise ValueError(f"candidate file must not be a symlink: {p}")

    def propose(self, current_text: str, step: int,
                rejected_hashes: List[str]) -> Optional[CandidateEdit]:
        if step >= len(self.paths):
            return None
        path = self.paths[step]
        text = path.read_text(encoding="utf-8")
        return CandidateEdit(
            text=text,
            description=f"candidate file {path.name}",
            source=str(path),
            content_hash=skill_hash(text),
        )


_DIRECTIVE_LIBRARY = [
    "## Verification\n\nAlways verify your work: re-run the tests after every "
    "edit and confirm they pass before claiming completion.",
    "## Reproduction\n\nFirst reproduce the failure: run the test and observe "
    "the failing result to find the root cause before fixing anything.",
    "## Honesty\n\nReport the test status honestly, including failures and "
    "partial passes; never claim done without evidence.",
    "## Boundaries\n\nFix the source, not the test. Do not modify tests to "
    "make them pass; treat them as the authority.",
    "## Persistence\n\nIf a result looks flaky, retry and re-verify rather "
    "than trusting a stale signal.",
]


class RuleOptimizer:
    """Deterministic bounded-edit proposer for offline runs/tests.

    Each step appends the next directive paragraph from a fixed library,
    respecting a textual learning-rate budget: an edit is refused when the
    appended text would exceed ``learning_rate`` of the current skill length.
    Step order is fixed so runs reproduce exactly.
    """

    def __init__(self, learning_rate: float = 1.0,
                 library: Optional[List[str]] = None):
        self.learning_rate = learning_rate
        self.library = library or _DIRECTIVE_LIBRARY

    def propose(self, current_text: str, step: int,
                rejected_hashes: List[str]) -> Optional[CandidateEdit]:
        if step >= len(self.library):
            return None
        directive = self.library[step]
        if len(current_text) > 0 and len(directive) > len(current_text) * self.learning_rate:
            return None
        text = current_text.rstrip() + "\n\n" + directive + "\n"
        return CandidateEdit(
            text=text,
            description=f"append directive #{step + 1}",
            source=f"rule:{step}",
            content_hash=skill_hash(text),
        )


def build_proposer(candidates: List[str], learning_rate: float,
                   runner_name: str) -> Proposer:
    if candidates:
        return FileOptimizer(candidates)
    return RuleOptimizer(learning_rate=learning_rate)
