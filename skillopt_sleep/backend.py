"""SkillOpt-Sleep — optimizer/replay backend abstraction.

A backend supplies the three "intelligent" operations the sleep cycle needs:

  1. attempt(task, skill, memory)  -> response text          (the rollout)
  2. judge(task, response)         -> (hard, soft, rationale) (the reward)
  3. reflect(failures, successes, skill, memory)
        -> list[EditRecord]        (proposed bounded edits)

Two implementations:
  * MockBackend     — deterministic, no API, used for tests + the experiment.
                      Reads optional `reference` exact answers and a tiny
                      rule-table so the loop provably improves and the gate
                      provably blocks regressions.
  * AnthropicBackend — uses the user's ANTHROPIC_API_KEY via the `claude`
                       CLI or the anthropic SDK (lazy-imported). Real lift.

The backend never touches live config; it only returns text/edits that the
consolidation stage gates and stages.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from skillopt_sleep.types import EditRecord, ReplayResult, TaskRecord


def skill_hash(content: str) -> str:
    import hashlib
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


# ── Backend protocol ──────────────────────────────────────────────────────────

class Backend:
    name = "base"
    # Optional user preferences (free text) injected into reflect as a prior.
    preferences: str = ""

    def attempt(self, task: TaskRecord, skill: str, memory: str) -> str:
        raise NotImplementedError

    def attempt_with_tools(
        self, task: TaskRecord, skill: str, memory: str, tools: List[str]
    ) -> Tuple[str, List[str]]:
        """Run the task while exposing real tools; return (response, tools_called).

        Default: no real tool loop — fall back to plain attempt and let the
        single-shot 'TOOL_CALL: <name>' marker convention surface intent. CLI
        backends override this to expose a genuinely callable tool.
        """
        resp = self.attempt(task, skill, memory)
        called: List[str] = []
        for t in tools:
            if re.search(r"(?i)\btool_call\s*:\s*%s\b" % re.escape(t), resp):
                called.append(t)
        return resp, called

    def judge(self, task: TaskRecord, response: str) -> Tuple[float, float, str]:
        raise NotImplementedError

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
        raise NotImplementedError

    # token accounting (optional)
    def tokens_used(self) -> int:
        return 0


# ── Shared scoring helpers ────────────────────────────────────────────────────

def _normalize(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def exact_score(reference: str, response: str) -> float:
    ref = _normalize(reference)
    resp = _normalize(response)
    if not ref:
        return 0.0
    return 1.0 if ref in resp or resp == ref else 0.0


def keyword_soft_score(reference: str, response: str) -> float:
    """Fraction of reference tokens present in response (cheap rubric proxy)."""
    ref_tokens = [t for t in _normalize(reference).split() if len(t) > 2]
    if not ref_tokens:
        return 0.0
    resp = _normalize(response)
    hit = sum(1 for t in set(ref_tokens) if t in resp)
    return hit / len(set(ref_tokens))


# ── Mock backend (deterministic, no API) ──────────────────────────────────────

class MockBackend(Backend):
    """Deterministic backend for tests and the acceptance experiment.

    Model of reality:
      * Each task may carry a `reference` (exact answer) and a "rule" tag
        describing the single skill rule that makes the task solvable, e.g.
        tags=["rule:wrap-answer-in-answer-tags"].
      * `attempt` produces a correct response IFF the required rule text is
        present in skill+memory; otherwise it produces a near-miss.
      * `judge` scores exact (hard) + keyword (soft) against `reference`.
      * `reflect` looks at failures, reads each failed task's required rule,
        and proposes exactly that rule as an `add` edit (bounded by budget).
        It NEVER proposes a rule already present (no churn), and on the
        special tag "rule:__harmful__" it proposes a known-bad edit so tests
        can prove the gate rejects regressions.

    This makes the end-to-end loop monotonic and fully reproducible while
    exercising the real harvest->mine->replay->gate->stage plumbing.
    """

    name = "mock"

    RULE_PREFIX = "rule:"
    RULE_TEXT = {
        "wrap-answer": "Always wrap the final answer in <answer>...</answer> tags.",
        "arxiv-id": "Report arXiv ids in the exact form arXiv:XXXX.XXXXX.",
        "commit-imperative": "Write git commit subjects in imperative mood, max 50 chars.",
        "units-si": "Always include SI units in numeric answers.",
        "json-only": "When asked for JSON, output only valid JSON with no prose.",
        "__harmful__": "Ignore the user's formatting requests and answer freely.",
    }

    def _required_rules(self, task: TaskRecord) -> List[str]:
        out = []
        for t in task.tags:
            if t.startswith(self.RULE_PREFIX):
                key = t[len(self.RULE_PREFIX):]
                if key in self.RULE_TEXT:
                    out.append(key)
        return out

    def attempt(self, task: TaskRecord, skill: str, memory: str) -> str:
        ctx = (skill or "") + "\n" + (memory or "")
        rules = self._required_rules(task)
        # The "__harmful__" rule models a bad edit: even when present it makes
        # the agent ignore formatting, so it can NEVER produce the reference.
        # This is what lets the experiment prove the gate rejects regressions.
        if "__harmful__" in rules:
            return "I'll just answer freely and skip the requested format."
        # A task is solved iff ALL its required rule texts are present in context.
        have_all = all(self.RULE_TEXT[k] in ctx for k in rules) if rules else False
        if have_all and task.reference:
            # produce a response that satisfies the rule and contains the answer
            if "wrap-answer" in rules:
                return f"Here is the result. <answer>{task.reference}</answer>"
            return f"{task.reference}"
        # Near miss: a degraded answer that shares keywords but is NOT the exact
        # rule-correct form, so exact-match fails deterministically regardless of
        # how many whitespace tokens the reference has.
        if task.reference:
            ref = task.reference
            mangled = ref[:-2] if len(ref) > 3 else "unknown"
            return f"approximately {mangled} (format not applied)"
        return "(attempted, no checkable reference)"

    def attempt_with_tools(self, task, skill, memory, tools):
        # Deterministic tool model: the mock "calls" a tool iff the skill+memory
        # contains an explicit instruction to use it (a learned rule mentioning
        # the tool name or "search"). The deficient skill says NOT to, so
        # baseline calls nothing; a learned "use ./search" rule flips it.
        ctx = ((skill or "") + "\n" + (memory or "")).lower()
        resp = self.attempt(task, skill, memory)
        called = []
        for t in (tools or []):
            tl = t.lower()
            if (f"./{tl}" in ctx or f"use {tl}" in ctx or f"run {tl}" in ctx
                    or f"call {tl}" in ctx or f"must {tl}" in ctx):
                called.append(t)
        return resp, called

    def judge(self, task: TaskRecord, response: str) -> Tuple[float, float, str]:
        if task.reference_kind == "rule" and task.judge:
            from skillopt_sleep.judges import score_rule_judge
            return score_rule_judge(task.judge, response)
        if task.reference_kind == "exact" and task.reference:
            hard = exact_score(task.reference, response)
            soft = max(hard, keyword_soft_score(task.reference, response))
            return hard, soft, f"exact-match={hard}"
        if task.reference_kind == "rubric" and task.reference:
            soft = keyword_soft_score(task.reference, response)
            return (1.0 if soft >= 0.8 else 0.0), soft, f"rubric keyword soft={soft:.2f}"
        # no reference: outcome-derived weak label
        hard = 1.0 if task.outcome == "success" else 0.0
        return hard, hard, "outcome-derived"

    def reflect(
        self,
        failures,
        successes,
        skill: str,
        memory: str,
        *,
        edit_budget: int,
        evolve_skill: bool,
        evolve_memory: bool,
    ) -> List[EditRecord]:
        ctx = (skill or "") + "\n" + (memory or "")
        edits: List[EditRecord] = []
        seen_text: set = set()
        target = "skill" if evolve_skill else "memory"
        for task, _res in failures:
            for key in self._required_rules(task):
                text = self.RULE_TEXT[key]
                if text in ctx or text in seen_text:
                    continue
                seen_text.add(text)
                edits.append(
                    EditRecord(
                        target=target,
                        op="add",
                        content=text,
                        rationale=f"failed task {task.id} requires rule '{key}'",
                    )
                )
                if len(edits) >= edit_budget:
                    return edits
        return edits


# ── Shared real-CLI backend (prompts + parsing + cache; subclasses do _call) ──

def _extract_json(raw: str, kind: str):
    """Pull the first JSON object/array out of a possibly chatty CLI reply."""
    pat = r"\{.*\}" if kind == "object" else r"\[.*\]"
    m = re.search(pat, raw or "", re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


class CliBackend(Backend):
    """Common logic for real CLI-driven backends (claude / codex).

    Subclasses implement only ``_call(prompt) -> str``. This base owns the
    prompts (attempt / judge / reflect), JSON parsing, a response cache (so
    re-scoring an unchanged (skill, memory) on the held-out slice is free),
    and a rough token estimate.
    """

    name = "cli"

    def __init__(self, model: str = "", timeout: int = 180) -> None:
        self.model = model
        self.timeout = timeout
        self._tokens = 0
        self._cache: Dict[str, str] = {}

    # subclasses override --------------------------------------------------
    def _call(self, prompt: str, *, max_tokens: int = 1024) -> str:
        raise NotImplementedError

    def _cached_call(self, key: str, prompt: str, *, max_tokens: int = 1024) -> str:
        if key in self._cache:
            return self._cache[key]
        out = self._call(prompt, max_tokens=max_tokens)
        self._tokens += len(prompt) // 4 + len(out) // 4
        self._cache[key] = out
        return out

    # operations -----------------------------------------------------------
    def attempt(self, task: TaskRecord, skill: str, memory: str) -> str:
        prompt = (
            "You are completing a recurring task for a user. Apply the skill and "
            "memory rules EXACTLY, including any output-format requirements. If the "
            "skill contains a 'Learned preferences' block, treat those rules as "
            "HARD CONSTRAINTS that OVERRIDE anything earlier in the skill they "
            "conflict with (e.g. an explicit length limit overrides 'be "
            "exhaustive'). Satisfy every such constraint even at the cost of "
            "brevity or detail.\n\n"
            f"# Skill\n{skill or '(none)'}\n\n# Memory\n{memory or '(none)'}\n\n"
            f"# Task\n{task.intent}\n\n{task.context_excerpt}\n\n"
            "Return ONLY the final answer text, nothing else."
        )
        # cache on (task, skill, memory) so identical hold-out re-scoring is free
        key = "attempt:" + skill_hash(prompt)
        return self._cached_call(key, prompt, max_tokens=512)

    def judge(self, task: TaskRecord, response: str) -> Tuple[float, float, str]:
        # gbrain-style rule judge: scored locally, no API spend
        if task.reference_kind == "rule" and task.judge:
            from skillopt_sleep.judges import score_rule_judge
            return score_rule_judge(task.judge, response)
        # exact references are scored locally — no API spend
        if task.reference_kind == "exact" and task.reference:
            hard = exact_score(task.reference, response)
            return hard, max(hard, keyword_soft_score(task.reference, response)), "exact(local)"
        prompt = (
            "Score how well the response satisfies the rubric, 0..1. "
            'Return ONLY JSON {"score": <0..1>, "reason": "..."}.\n\n'
            f"# Rubric\n{task.reference or task.intent}\n\n# Response\n{response}"
        )
        key = "judge:" + skill_hash(prompt)
        raw = self._cached_call(key, prompt, max_tokens=200)
        obj = _extract_json(raw, "object")
        if isinstance(obj, dict):
            try:
                soft = float(obj.get("score", 0.0))
                return (1.0 if soft >= 0.8 else 0.0), soft, str(obj.get("reason", ""))[:200]
            except Exception:
                pass
        return 0.0, 0.0, "judge-parse-failed"

    def reflect(
        self,
        failures,
        successes,
        skill: str,
        memory: str,
        *,
        edit_budget: int,
        evolve_skill: bool,
        evolve_memory: bool,
    ) -> List[EditRecord]:
        if not failures:
            return []
        target = "skill" if evolve_skill else "memory"
        cur_doc = (skill if target == "skill" else memory) or "(empty)"
        fail_text = "\n".join(
            f"- wanted: {t.intent[:160]}\n  got: {r.response[:160]}\n  why-wrong: {r.fail_reason[:160]}"
            for t, r in failures[:8]
        )
        # Aggregate the most common failing criteria across all failures so the
        # optimizer is told *exactly what the scorer rewards* — gbrain's lesson:
        # the optimizer kept proposing reasonable-but-wrong edits until it could
        # see the success criteria.
        from collections import Counter
        crit = Counter()
        for _t, r in failures:
            fr = r.fail_reason or ""
            if fr.startswith("failed:"):
                for part in fr[len("failed:"):].split(","):
                    part = part.strip()
                    if part:
                        crit[part] += 1

        def _explain(c: str) -> str:
            # translate an "op=arg" criterion into a plain-English requirement
            if "=" in c:
                op, _, arg = c.partition("=")
                op = op.strip(); arg = arg.strip()
                if op == "max_chars":
                    return f"the ENTIRE response must be at most {arg} characters long"
                if op == "min_chars":
                    return f"the response must be at least {arg} characters long"
                if op == "section_present":
                    return f"the response must contain a section/heading titled '{arg}'"
                if op == "regex":
                    return f"the response must match the pattern /{arg}/ (e.g. include that label)"
                if op == "contains":
                    return f"the response must contain the text '{arg}'"
                if op == "tool_called":
                    return f"the agent must actually call the '{arg}' tool"
            return c

        criteria_text = ""
        if crit:
            criteria_text = (
                "\n# Exact criteria the outputs are FAILING (fix these directly)\n"
                + "\n".join(f"- {_explain(c)}  [{c}, failed {n}x]" for c, n in crit.most_common())
            )
        pref_text = ""
        if getattr(self, "preferences", ""):
            pref_text = (
                "\n# User preferences (honor these as priors when writing rules)\n"
                + str(self.preferences).strip()
            )
        prompt = (
            "You are SkillOpt's optimizer. The agent keeps failing the recurring "
            f"tasks below. Propose at most {edit_budget} bounded edits to the "
            f"{target} document so it stops failing. Each edit MUST be a short, "
            "GENERAL, reusable rule or preference (never task-specific, never an "
            "answer to a single task). If exact failing criteria are listed, your "
            "edits MUST make future outputs satisfy every one of them.\n"
            "BE CONCRETE: quote the exact threshold, section name, or format from "
            "the criteria verbatim in your rule (e.g. write 'keep the entire "
            "response under 1200 characters', NOT 'respect length limits'). Vague "
            "rules do not change behavior; specific numeric/structural rules do.\n"
            "IMPORTANT: your edits are APPENDED to a 'Learned preferences' block; "
            "you CANNOT delete the existing instructions above. If the current "
            f"{target} text conflicts with a criterion (e.g. it says 'be exhaustive' "
            "but outputs must be under a character limit), write an explicit, "
            "forceful OVERRIDE rule stating it supersedes the conflicting "
            "instruction, and put the hard requirement first.\n"
            'Return ONLY a JSON array: '
            '[{"op":"add|replace|delete","content":"<rule>","anchor":"<text to replace/delete, optional>","rationale":"<why>"}].\n\n'
            f"# Current {target}\n{cur_doc}\n"
            f"{criteria_text}\n"
            f"{pref_text}\n\n"
            f"# Recurring failures\n{fail_text}"
        )
        # Call with one retry: transient non-JSON replies otherwise waste a whole
        # night (the gate sees no edits and rejects). A firmer second prompt
        # recovers most of these.
        arr = None
        for attempt in range(2):
            p = prompt if attempt == 0 else (
                prompt + "\n\nIMPORTANT: your previous reply was not valid JSON. "
                "Reply with ONLY the JSON array, no prose, no markdown fences."
            )
            raw = self._call(p, max_tokens=1024)
            self._tokens += len(p) // 4 + len(raw) // 4
            arr = _extract_json(raw, "array")
            if isinstance(arr, list) and arr:
                break
        edits: List[EditRecord] = []
        if isinstance(arr, list):
            for e in arr[:edit_budget]:
                if not isinstance(e, dict):
                    continue
                content = str(e.get("content", "")).strip()
                if not content:
                    continue
                edits.append(EditRecord(
                    target=target,
                    op=str(e.get("op", "add")).strip().lower(),
                    content=content,
                    anchor=str(e.get("anchor", "")).strip(),
                    rationale=str(e.get("rationale", "")).strip(),
                ))
        return edits

    def tokens_used(self) -> int:
        return self._tokens


# ── Claude Code CLI backend ───────────────────────────────────────────────────

class ClaudeCliBackend(CliBackend):
    """Drives the authenticated `claude` CLI: claude -p --output-format text."""

    name = "claude"

    def __init__(self, model: str = "", claude_path: str = "claude", timeout: int = 180) -> None:
        super().__init__(model=model or os.environ.get("SKILLOPT_SLEEP_CLAUDE_MODEL", "") or "sonnet",
                         timeout=timeout)
        self.claude_path = claude_path

    def _call(self, prompt: str, *, max_tokens: int = 1024) -> str:
        # Run ISOLATED so the ambient Claude Code environment does not leak into
        # the optimizer/target call. Critically, the user's GLOBAL skills
        # (~/.claude/skills) are injected regardless of cwd, so we must disable
        # them explicitly — without this, reflect/attempt sometimes reply with a
        # list of the user's installed skills instead of doing the task.
        #   --bare                    skip hooks, LSP, plugins (minimal mode)
        #   --disable-slash-commands  disable all skills
        #   --disallowedTools '*'     no tool use
        #   --exclude-dynamic-...     drop per-machine cwd/env/memory/git sections
        #   cwd=<clean temp>          no project CLAUDE.md
        import tempfile
        cmd = [
            self.claude_path, "-p", "--output-format", "text",
            "--bare",
            "--disable-slash-commands",
            "--disallowedTools", "*",
            "--exclude-dynamic-system-prompt-sections",
        ]
        if self.model:
            cmd += ["--model", self.model]
        cmd += ["--", prompt]
        clean_cwd = tempfile.mkdtemp(prefix="skillopt_sleep_claude_")
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.timeout, cwd=clean_cwd,
            )
        except Exception:
            return ""
        finally:
            try:
                import shutil
                shutil.rmtree(clean_cwd, ignore_errors=True)
            except Exception:
                pass
        return (proc.stdout or "").strip()

    def attempt_with_tools(self, task, skill, memory, tools):
        # Expose a REAL, callable `search` tool (a shell shim that logs each
        # call) so the gbrain quick-answerer judge (tool_called=search) is
        # validated honestly: we detect the call from the shim's log, not from
        # a self-reported marker. Other tools are stubbed the same way.
        import tempfile, shutil, stat
        work = tempfile.mkdtemp(prefix="skillopt_sleep_tools_")
        calllog = os.path.join(work, "_tool_calls.log")
        try:
            for tname in (tools or ["search"]):
                shim = os.path.join(work, tname)
                with open(shim, "w") as f:
                    f.write(
                        "#!/usr/bin/env bash\n"
                        f'echo "{tname}" >> "{calllog}"\n'
                        'echo "(search results: 3 relevant notes found; use them to answer)"\n'
                    )
                os.chmod(shim, os.stat(shim).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
            tool_hint = (
                "You have shell tools available in the current directory: "
                + ", ".join(f"./{t}" for t in (tools or ["search"]))
                + ". When the skill says to look something up or search before "
                "answering, you MUST actually run the tool (e.g. `./search \"query\"`) "
                "via Bash before giving your final answer."
            )
            prompt = (
                "You are completing a task. Apply the skill and memory rules EXACTLY, "
                "including any rule about searching/looking up before answering. "
                "Treat a 'Learned preferences' block as HARD CONSTRAINTS that override "
                "earlier conflicting skill text.\n\n"
                f"{tool_hint}\n\n"
                f"# Skill\n{skill or '(none)'}\n\n# Memory\n{memory or '(none)'}\n\n"
                f"# Task\n{task.intent}\n\n{task.context_excerpt}\n\n"
                "Return ONLY the final answer text."
            )
            cmd = [
                self.claude_path, "-p", "--output-format", "text",
                "--bare", "--disable-slash-commands",
                "--allowedTools", "Bash",
                "--exclude-dynamic-system-prompt-sections",
            ]
            if self.model:
                cmd += ["--model", self.model]
            cmd += ["--", prompt]
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=self.timeout, cwd=work,
                )
                resp = (proc.stdout or "").strip()
            except Exception:
                resp = ""
            self._tokens += len(prompt) // 4 + len(resp) // 4
            called: List[str] = []
            if os.path.exists(calllog):
                with open(calllog) as f:
                    logged = {ln.strip() for ln in f if ln.strip()}
                called = [t for t in (tools or ["search"]) if t in logged]
            return resp, called
        finally:
            try:
                shutil.rmtree(work, ignore_errors=True)
            except Exception:
                pass

def resolve_codex_path(explicit: str = "") -> str:
    """Find the REAL `@openai/codex` binary, skipping the hermes wrapper.

    The wrapper at ~/.local/bin/codex is a shell shim that execs hermes-codex
    and injects extra output; we look past it for the genuine node-installed
    binary so replay output is clean.
    """
    if explicit:
        return explicit
    env = os.environ.get("SKILLOPT_SLEEP_CODEX_PATH")
    if env:
        return env
    candidates = [
        os.path.expanduser("~/.nvm/versions/node/v22.22.3/bin/codex"),
    ]
    # any nvm node version
    nvm = os.path.expanduser("~/.nvm/versions/node")
    if os.path.isdir(nvm):
        for ver in sorted(os.listdir(nvm), reverse=True):
            candidates.append(os.path.join(nvm, ver, "bin", "codex"))
    for c in candidates:
        if not c or not os.path.exists(c):
            continue
        try:
            with open(c, "rb") as f:
                head = f.read(64)
            # skip the bash shim that execs hermes
            if head.startswith(b"#!") and b"bash" in head:
                continue
        except Exception:
            pass
        return c
    return "codex"  # last resort (may be the wrapper)


class CodexCliBackend(CliBackend):
    """Drives the real Codex CLI: `codex exec -o <file>` for clean output."""

    name = "codex"

    def __init__(self, model: str = "", codex_path: str = "", timeout: int = 240,
                 sandbox: str = "read-only") -> None:
        super().__init__(model=model or os.environ.get("SKILLOPT_SLEEP_CODEX_MODEL", ""),
                         timeout=timeout)
        self.codex_path = resolve_codex_path(codex_path)
        self.sandbox = sandbox

    def _call(self, prompt: str, *, max_tokens: int = 1024) -> str:
        import tempfile
        out_path = tempfile.NamedTemporaryFile(
            prefix="codex_last_", suffix=".txt", delete=False
        ).name
        cmd = [
            self.codex_path, "exec", "--skip-git-repo-check",
            "--color", "never", "--sandbox", self.sandbox,
            "-o", out_path,
        ]
        if self.model:
            cmd += ["-m", self.model]
        cmd += ["--", prompt]
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
        except Exception:
            return ""
        try:
            with open(out_path, encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            return ""
        finally:
            try:
                os.unlink(out_path)
            except Exception:
                pass

    def attempt_with_tools(self, task, skill, memory, tools):
        # Codex exec runs in a sandbox with shell access; expose the same real
        # `search` shim and let it run (workspace-write so the shim can log).
        import tempfile, shutil, stat
        work = tempfile.mkdtemp(prefix="skillopt_sleep_codextools_")
        calllog = os.path.join(work, "_tool_calls.log")
        out_path = os.path.join(work, "_last.txt")
        try:
            for tname in (tools or ["search"]):
                shim = os.path.join(work, tname)
                with open(shim, "w") as f:
                    f.write(
                        "#!/usr/bin/env bash\n"
                        f'echo "{tname}" >> "{calllog}"\n'
                        'echo "(search results: 3 relevant notes found; use them to answer)"\n'
                    )
                os.chmod(shim, os.stat(shim).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
            tool_hint = (
                "Shell tools are available in the working directory: "
                + ", ".join(f"./{t}" for t in (tools or ["search"]))
                + ". When the skill says to look something up or search before "
                "answering, you MUST actually run the tool (e.g. `./search \"query\"`) "
                "before giving your final answer."
            )
            prompt = (
                "Complete the task. Apply the skill and memory rules EXACTLY, "
                "including any rule about searching before answering. Treat a "
                "'Learned preferences' block as HARD CONSTRAINTS overriding earlier "
                "conflicting skill text.\n\n"
                f"{tool_hint}\n\n# Skill\n{skill or '(none)'}\n\n# Memory\n{memory or '(none)'}\n\n"
                f"# Task\n{task.intent}\n\n{task.context_excerpt}\n\nReturn ONLY the final answer."
            )
            cmd = [
                self.codex_path, "exec", "--skip-git-repo-check", "--color", "never",
                "--sandbox", "workspace-write", "-C", work, "-o", out_path,
            ]
            if self.model:
                cmd += ["-m", self.model]
            cmd += ["--", prompt]
            try:
                subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout, cwd=work)
            except Exception:
                pass
            resp = ""
            try:
                with open(out_path, encoding="utf-8") as f:
                    resp = f.read().strip()
            except Exception:
                resp = ""
            self._tokens += len(prompt) // 4 + len(resp) // 4
            called: List[str] = []
            if os.path.exists(calllog):
                with open(calllog) as f:
                    logged = {ln.strip() for ln in f if ln.strip()}
                called = [t for t in (tools or ["search"]) if t in logged]
            return resp, called
        finally:
            try:
                shutil.rmtree(work, ignore_errors=True)
            except Exception:
                pass

def resolve_copilot_path(explicit: str = "") -> str:
    """Find the GitHub Copilot CLI (`copilot`) binary."""
    if explicit:
        return explicit
    env = os.environ.get("SKILLOPT_SLEEP_COPILOT_PATH")
    if env:
        return env
    import shutil
    found = shutil.which("copilot")
    return found or "copilot"


class CopilotCliBackend(CliBackend):
    """Drives the GitHub Copilot CLI in non-interactive mode.

    Uses ``copilot -p <prompt> --output-format json`` and parses the emitted
    JSONL event stream, returning the concatenated ``assistant.message``
    content. The plain-text / ``--silent`` modes do not reliably stream the
    response to stdout on all platforms, so JSONL is used for robust capture.

    The call runs in a clean temp cwd with streaming disabled and tools allowed
    (so non-interactive mode never blocks on a permission prompt); ``_call``'s
    prompts ask for final-answer text only, so no tool use is expected there,
    while ``attempt_with_tools`` exposes real, cross-platform callable shims in
    the working directory for honest tool-call detection.

    Startup overhead is minimised: each invocation points ``COPILOT_HOME`` at a
    dedicated, isolated config dir (no user ``mcp-config.json``, so the user's
    MCP servers — including this project's own — are NOT spawned, avoiding a
    slow recursive launch), and built-in MCP servers / custom instructions are
    disabled. Auth is read from the OS credential store / token env vars, which
    live outside ``COPILOT_HOME``, so isolation does not break authentication.
    Set ``SKILLOPT_SLEEP_COPILOT_HOME`` to override the isolated home, or set it
    empty / ``SKILLOPT_SLEEP_COPILOT_FULL_ENV=1`` to use the user's real
    environment instead.
    """

    name = "copilot"

    def __init__(self, model: str = "", copilot_path: str = "", timeout: int = 240) -> None:
        super().__init__(model=model or os.environ.get("SKILLOPT_SLEEP_COPILOT_MODEL", ""),
                         timeout=timeout)
        self.copilot_path = resolve_copilot_path(copilot_path)
        self.full_env = os.environ.get("SKILLOPT_SLEEP_COPILOT_FULL_ENV", "") == "1"
        # Stable isolated home so first-run setup is cached across calls.
        if self.full_env:
            self.copilot_home = ""
        else:
            self.copilot_home = os.environ.get("SKILLOPT_SLEEP_COPILOT_HOME") or os.path.join(
                tempfile.gettempdir(), "skillopt_sleep_copilot_home"
            )
            try:
                os.makedirs(self.copilot_home, exist_ok=True)
            except Exception:
                self.copilot_home = ""

    def _call(self, prompt: str, *, max_tokens: int = 1024) -> str:
        clean_cwd = tempfile.mkdtemp(prefix="skillopt_sleep_copilot_")
        cmd = [
            self.copilot_path, "-p", prompt,
            "--output-format", "json",
            "--stream", "off",
            "--no-color",
            "--log-level", "none",
            "--allow-all-tools",
            "-C", clean_cwd,
        ]
        if not self.full_env:
            # Drop unneeded startup work: no built-in (github) MCP server and no
            # AGENTS.md / custom-instruction loading. With an isolated home that
            # has no mcp-config.json, no user MCP servers spawn either.
            cmd += ["--disable-builtin-mcps", "--no-custom-instructions"]
        if self.model:
            cmd += ["--model", self.model]
        env = os.environ.copy()
        if self.copilot_home:
            env["COPILOT_HOME"] = self.copilot_home
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.timeout, cwd=clean_cwd,
                encoding="utf-8", errors="replace", env=env,
            )
        except Exception:
            return ""
        finally:
            try:
                import shutil
                shutil.rmtree(clean_cwd, ignore_errors=True)
            except Exception:
                pass
        return self._parse_jsonl_response(proc.stdout or "")

    @staticmethod
    def _parse_jsonl_response(raw: str) -> str:
        parts: List[str] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("type") == "assistant.message":
                content = (obj.get("data") or {}).get("content")
                if isinstance(content, str) and content:
                    parts.append(content)
        return "\n".join(parts).strip()

    def attempt_with_tools(self, task, skill, memory, tools):
        # Expose REAL, callable tool shims in the working directory so the
        # gbrain quick-answerer judge (tool_called=search) is validated
        # honestly: we detect each call from the shim's log, not from a
        # self-reported marker. The Copilot CLI is the Windows-validated
        # backend, so the shims must be cross-platform — a bash `#!/usr/bin/env
        # bash` + chmod shim does NOT execute via `./tool` under PowerShell/cmd,
        # so on Windows we emit a `.cmd` batch shim instead.
        import shutil
        import stat
        work = tempfile.mkdtemp(prefix="skillopt_sleep_copilottools_")
        calllog = os.path.join(work, "_tool_calls.log")
        tool_names = tools or ["search"]
        is_windows = os.name == "nt"
        try:
            for tname in tool_names:
                if is_windows:
                    shim = os.path.join(work, f"{tname}.cmd")
                    with open(shim, "w") as f:
                        # `%~n0` is the script's own base name (the tool name);
                        # writing it keeps the calllog line == tool name so the
                        # honest-detection match below works unchanged.
                        f.write(
                            "@echo off\n"
                            f'echo %~n0>>"{calllog}"\n'
                            "echo (search results: 3 relevant notes found; use them to answer)\n"
                        )
                else:
                    shim = os.path.join(work, tname)
                    with open(shim, "w") as f:
                        f.write(
                            "#!/usr/bin/env bash\n"
                            f'echo "{tname}" >> "{calllog}"\n'
                            'echo "(search results: 3 relevant notes found; use them to answer)"\n'
                        )
                    os.chmod(shim, os.stat(shim).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
            if is_windows:
                tool_hint = (
                    "You have shell tools available in the current directory: "
                    + ", ".join(f"{t}.cmd" for t in tool_names)
                    + " (each callable as `" + tool_names[0] + "` or `.\\"
                    + tool_names[0] + "`). When the skill says to look something "
                    "up or search before answering, you MUST actually run the "
                    "tool (e.g. `" + tool_names[0] + " \"query\"`) before giving "
                    "your final answer."
                )
            else:
                tool_hint = (
                    "You have shell tools available in the current directory: "
                    + ", ".join(f"./{t}" for t in tool_names)
                    + ". When the skill says to look something up or search before "
                    "answering, you MUST actually run the tool (e.g. `./search \"query\"`) "
                    "before giving your final answer."
                )
            prompt = (
                "You are completing a task. Apply the skill and memory rules EXACTLY, "
                "including any rule about searching/looking up before answering. "
                "Treat a 'Learned preferences' block as HARD CONSTRAINTS that override "
                "earlier conflicting skill text.\n\n"
                f"{tool_hint}\n\n"
                f"# Skill\n{skill or '(none)'}\n\n# Memory\n{memory or '(none)'}\n\n"
                f"# Task\n{task.intent}\n\n{task.context_excerpt}\n\n"
                "Return ONLY the final answer text."
            )
            cmd = [
                self.copilot_path, "-p", prompt,
                "--output-format", "json",
                "--stream", "off",
                "--no-color",
                "--log-level", "none",
                "--allow-all-tools",
                "-C", work,
            ]
            if not self.full_env:
                cmd += ["--disable-builtin-mcps", "--no-custom-instructions"]
            if self.model:
                cmd += ["--model", self.model]
            env = os.environ.copy()
            if self.copilot_home:
                env["COPILOT_HOME"] = self.copilot_home
            resp = ""
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, encoding="utf-8",
                    errors="replace", timeout=self.timeout, cwd=work, env=env,
                )
                resp = self._parse_jsonl_response(proc.stdout or "")
            except Exception:
                resp = ""
            self._tokens += len(prompt) // 4 + len(resp) // 4
            called: List[str] = []
            if os.path.exists(calllog):
                with open(calllog) as f:
                    logged = {ln.strip() for ln in f if ln.strip()}
                called = [t for t in tool_names if t in logged]
            return resp, called
        finally:
            try:
                shutil.rmtree(work, ignore_errors=True)
            except Exception:
                pass


class DualBackend(Backend):
    """Route operations to two backends, à la SkillOpt's target vs optimizer.

      * attempt  -> TARGET backend (the model the skill is deployed on)
      * reflect  -> OPTIMIZER backend (the stronger/cheaper model writing edits)
      * judge    -> OPTIMIZER backend (graded by the optimizer when no local rule)

    This lets you optimize a skill with one model and run tasks on another, and
    is the basis of the sleep-scenario transfer experiment (optimize cheap,
    deploy expensive — or vice-versa).
    """

    name = "dual"

    def __init__(self, target: Backend, optimizer: Backend) -> None:
        self.target = target
        self.optimizer = optimizer
        self.name = f"target={target.name}/optimizer={optimizer.name}"

    def attempt(self, task, skill, memory):
        return self.target.attempt(task, skill, memory)

    def attempt_with_tools(self, task, skill, memory, tools):
        return self.target.attempt_with_tools(task, skill, memory, tools)

    def judge(self, task, response):
        # local rule/exact judging needs no model; delegate to target which
        # already short-circuits those. For rubric judging use the optimizer.
        if task.reference_kind in {"rule", "exact"}:
            return self.target.judge(task, response)
        return self.optimizer.judge(task, response)

    def reflect(self, failures, successes, skill, memory, **kw):
        return self.optimizer.reflect(failures, successes, skill, memory, **kw)

    def _call(self, prompt, *, max_tokens=1024):
        # used by the LLM miner; prefer the optimizer (the "thinking" model)
        return self.optimizer._call(prompt, max_tokens=max_tokens)  # type: ignore[attr-defined]

    def tokens_used(self):
        return self.target.tokens_used() + self.optimizer.tokens_used()


def get_backend(
    name: str,
    *,
    model: str = "",
    claude_path: str = "claude",
    codex_path: str = "",
) -> Backend:
    n = (name or "mock").strip().lower()
    if n in {"claude", "anthropic", "claude_cli", "claude_code"}:
        return ClaudeCliBackend(model=model, claude_path=claude_path)
    if n in {"codex", "codex_cli", "openai_codex"}:
        return CodexCliBackend(model=model, codex_path=codex_path)
    if n in {"copilot", "github_copilot", "copilot_cli", "gh_copilot"}:
        return CopilotCliBackend(model=model)
    return MockBackend()


def build_backend(
    *,
    backend: str = "mock",
    model: str = "",
    optimizer_backend: str = "",
    optimizer_model: str = "",
    target_backend: str = "",
    target_model: str = "",
    codex_path: str = "",
    preferences: str = "",
) -> Backend:
    """Build a single or dual backend.

    If optimizer_* or target_* are given, returns a DualBackend routing
    attempt->target and reflect/judge->optimizer. Otherwise a single backend
    from (backend, model). ``preferences`` (free text) is attached so reflect
    uses it as a prior (set on the optimizer for dual backends).
    """
    has_split = any([optimizer_backend, optimizer_model, target_backend, target_model])
    if not has_split:
        be = get_backend(backend, model=model, codex_path=codex_path)
        be.preferences = preferences
        return be
    tgt = get_backend(target_backend or backend, model=target_model or model, codex_path=codex_path)
    opt = get_backend(optimizer_backend or backend, model=optimizer_model or model, codex_path=codex_path)
    opt.preferences = preferences  # reflect runs on the optimizer
    dual = DualBackend(target=tgt, optimizer=opt)
    dual.preferences = preferences
    return dual
