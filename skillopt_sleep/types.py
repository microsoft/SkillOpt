"""SkillOpt-Sleep — core data types.

These dataclasses are the interfaces between the sleep-cycle stages
(harvest -> mine -> replay -> consolidate -> stage). They are intentionally
plain (no slots, no heavy deps) so the package imports cleanly on any
Python 3.8+ interpreter and the deterministic experiment runs with zero
external dependencies.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

# ── Stage 1: harvest ──────────────────────────────────────────────────────────

@dataclass
class SessionDigest:
    """A normalized summary of one local agent session transcript.

    Produced by source-specific harvesters from Claude Code transcripts, Codex
    Desktop archived sessions, or Cursor Agent transcripts.
    """

    session_id: str
    project: str
    git_branch: str = ""
    started_at: str = ""
    ended_at: str = ""
    user_prompts: List[str] = field(default_factory=list)
    assistant_finals: List[str] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    # Skill targets invoked through the Claude ``Skill`` tool, in first-seen
    # order. Optional and default-empty: harvesters that do not observe skill
    # invocations, and digests persisted before this field existed, leave it [].
    skills_used: List[str] = field(default_factory=list)
    files_touched: List[str] = field(default_factory=list)
    feedback_signals: List[str] = field(default_factory=list)  # "still broken", "perfect", ...
    n_user_turns: int = 0
    n_assistant_turns: int = 0
    raw_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Stage 2: mine ─────────────────────────────────────────────────────────────

@dataclass
class TaskRecord:
    """A self-contained recurring task mined from one or more sessions.

    This is the *training unit* of the sleep cycle — the analogue of a
    SkillOpt benchmark item.
    """

    id: str
    project: str
    intent: str                       # what the user wanted (the "question")
    context_excerpt: str = ""         # minimal context needed to attempt it
    # Optional system framing for the rollout. When set (e.g. real benchmarks
    # carrying the research repo's exact rollout_system), the backend uses THIS
    # verbatim instead of its generic instruction wrapper — this keeps scoring
    # faithful to the source task and avoids re-deriving framing the benchmark
    # already bakes in.
    system: str = ""
    attempted_solution: str = ""      # what the agent produced before
    outcome: str = "unknown"          # success | fail | mixed | unknown
    reference_kind: str = "none"      # exact | rubric | rule | none
    reference: str = ""               # exact answer, or rubric text
    judge: Dict[str, Any] = field(default_factory=dict)  # gbrain-style rule judge
    tags: List[str] = field(default_factory=list)
    source_sessions: List[str] = field(default_factory=list)
    # split ∈ {train, val, test}.  val + test come ONLY from real mined tasks and
    # never overlap (val gates updates, test is the final held-out measure). train
    # may be dream-augmented (see origin).  Legacy values replay->train,
    # holdout->val are normalized on load.
    split: str = "train"
    # origin ∈ {real, dream}.  'real' = mined from the user's actual sessions;
    # 'dream' = synthetic/augmented for the training pool. Dream tasks are NEVER
    # allowed into val/test, which is the anti-overfitting guarantee.
    origin: str = "real"
    derived_from: str = ""            # for dream tasks: the real task id it varies
    # Which skill this task exercised, when the source session invoked exactly
    # one. Empty means unknown or ambiguous, which keeps the task in the
    # existing managed-skill catch-all path. Kept last for positional compatibility.
    skill_hint: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TaskRecord":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


# ── Stage 3: replay ───────────────────────────────────────────────────────────

@dataclass
class ReplayResult:
    """Outcome of re-running one TaskRecord offline under a given skill+memory."""

    id: str
    hard: float = 0.0                 # 0/1 exact, or continuous reward
    soft: float = 0.0                 # partial credit / judge score 0..1
    response: str = ""
    fail_reason: str = ""
    task_type: str = "task"
    judge_rationale: str = ""
    tools_called: List[str] = field(default_factory=list)
    tokens: int = 0                   # approx tokens this rollout cost (for token objective)
    latency_ms: float = 0.0           # wall-clock for this rollout (for latency objective)
    # Semantic, task-level feedback that may be shown to the skill optimizer.
    # ``fail_reason`` and ``judge_rationale`` remain the verbatim verifier
    # evidence for audit/debugging; they may contain regexes or other grader
    # implementation details and must not be used as learning context.
    optimizer_feedback: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── Stage 4/5: consolidation report ───────────────────────────────────────────

@dataclass
class EditRecord:
    """One bounded edit proposed/applied to skill or memory."""

    target: str                       # "skill" | "memory"
    op: str                           # add | delete | replace
    content: str = ""
    anchor: str = ""                  # for replace/delete: text being changed
    rationale: str = ""


@dataclass
class SkillGroupReport:
    """One skill group's own gate evidence for the night.

    Every field describes that group alone: its baseline, its candidate score,
    its decision, and why. Groups never share rows, so a weak group cannot
    borrow a strong group's evidence or block its acceptance.
    """

    skill_name: str
    status: str = ""                   # consolidated | skipped | failed
    accepted: bool = False
    gate_action: str = ""
    baseline_score: float = 0.0
    candidate_score: float = 0.0
    n_tasks: int = 0
    n_applied_edits: int = 0
    n_rejected_edits: int = 0
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SleepReport:
    """Everything one night produced — written to staging for review."""

    night: int
    project: str
    started_at: str = ""
    ended_at: str = ""
    n_sessions: int = 0
    n_tasks: int = 0
    n_replayed: int = 0
    baseline_score: float = 0.0
    candidate_score: float = 0.0
    accepted: bool = False
    gate_action: str = ""
    # True when the gate's validation slice was not disjoint from the tasks the
    # optimizer saw, so its comparison could not detect overfitting.
    holdout_leaked: bool = False
    no_edits_reason: str = ""
    edits: List[EditRecord] = field(default_factory=list)
    rejected_edits: List[EditRecord] = field(default_factory=list)
    # Proposed edits that changed nothing (anchor absent, duplicate/empty add,
    # unknown op). They were never scored by the gate, so they belong in neither
    # list above — without them a night can report no edits while the optimizer
    # actually produced several.
    unmatched_edits: List[EditRecord] = field(default_factory=list)
    # Per-skill gate rows for a multi-skill night, in first-seen order. Empty on
    # a single-managed-skill night, so the fields above remain the whole story
    # for consumers that predate skill groups.
    skill_groups: List[SkillGroupReport] = field(default_factory=list)
    tokens_used: int = 0
    notes: List[str] = field(default_factory=list)
    # Keep new optional fields last to preserve positional construction.
    gate_no_regression: bool = False
    # Per-candidate held-out comparisons, including rejected intermediate
    # skill/memory trials and the final replay.
    gate_trials: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d
