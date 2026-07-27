"""mem0-backed persistent memory for SkillOpt.

Stores skill iterations and reflection outcomes, and reads a small amount of
relevant history back into the Reflect stage so the memory participates in
training rather than only recording it.

Safety contract
---------------
1. **Nothing is sent unless the run opted in.** The client is only constructed
   from a :class:`~skillopt.memory.settings.Mem0Settings` whose ``usable`` is
   true, which requires ``mem0_enabled`` to have been set explicitly.
2. **Everything outbound is redacted.** Every payload is built through
   :meth:`SkillMemory._payload`, which applies
   :func:`~skillopt.memory.redaction.redact_for_upload` and then truncates.
   There is no code path that reaches ``self._client`` with unredacted text.
3. **Every call is time-bounded.** Network work runs through :meth:`_bounded`,
   so a slow or unreachable service costs at most ``timeout_seconds`` per
   training step instead of blocking it.
4. **Failures never break training.** Every public method returns a benign
   value on error; the trainer hooks additionally swallow anything unexpected.

Usage::

    from skillopt.memory import SkillMemory
    from skillopt.memory.settings import resolve_settings

    settings = resolve_settings(cfg)
    memory = SkillMemory.from_settings(settings)   # None unless opted in
"""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as _FuturesTimeout
from typing import Any

from skillopt.memory.redaction import redact_for_upload
from skillopt.memory.settings import Mem0Settings

try:
    from mem0 import MemoryClient
    _MEM0_AVAILABLE = True
except ImportError:
    _MEM0_AVAILABLE = False
    MemoryClient = None  # type: ignore[assignment,misc]


def mem0_available() -> bool:
    """Whether the optional ``mem0ai`` dependency is importable."""
    return _MEM0_AVAILABLE


class SkillMemory:
    """Persistent memory backend for SkillOpt using mem0.

    Prefer :meth:`from_settings`; the constructor stays explicit so tests can
    inject a fake client without touching the network.
    """

    def __init__(self, settings: Mem0Settings, client: Any | None = None) -> None:
        if client is None:
            if not _MEM0_AVAILABLE:
                raise ImportError("mem0ai is not installed. Run: pip install 'skillopt[mem0]'")
            if not settings.usable:
                raise ValueError("SkillMemory requires mem0_enabled=true and an API key")
            client = MemoryClient(api_key=settings.api_key)

        self.settings = settings
        self.namespace = settings.namespace
        self._client = client
        # One worker: calls are already serialised by the training loop, and a
        # single thread keeps the timeout semantics easy to reason about.
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mem0")

    # ── Construction ──────────────────────────────────────────────────────

    @classmethod
    def from_settings(cls, settings: Mem0Settings, client: Any | None = None) -> "SkillMemory | None":
        """Return a backend, or ``None`` when the run has not opted in.

        Returning ``None`` rather than raising keeps callers free of
        conditionals: every hook already treats ``None`` as "do nothing".
        """
        if not settings.usable and client is None:
            return None
        if client is None and not _MEM0_AVAILABLE:
            return None
        return cls(settings, client=client)

    # ── Internal helpers ──────────────────────────────────────────────────

    def _bounded(self, fn, *args, **kwargs) -> Any:
        """Run *fn* with a hard wall-clock bound; ``None`` on timeout or error.

        The worker thread may keep running after a timeout — Python cannot
        cancel a blocked socket read — but the *training step* is released on
        schedule, which is the property that matters here.
        """
        try:
            future = self._pool.submit(fn, *args, **kwargs)
            return future.result(timeout=self.settings.timeout_seconds)
        except _FuturesTimeout:
            print(f"  [mem0] call exceeded {self.settings.timeout_seconds}s — continuing without it")
            return None
        except Exception as exc:  # noqa: BLE001 - memory must never break training
            print(f"  [mem0] call failed ({type(exc).__name__}: {exc}) — continuing without it")
            return None

    def _payload(self, text: str) -> str:
        """The only route by which text becomes outbound content.

        Redacts first, truncates second, so the cap is measured on exactly the
        string that would be transmitted.
        """
        redacted = redact_for_upload(text, self.settings.project_root)
        if not isinstance(redacted, str):
            redacted = str(redacted)
        limit = max(0, int(self.settings.max_chars))
        if len(redacted) > limit:
            return redacted[:limit] + "\n[...truncated]"
        return redacted

    def _add(self, content: str, metadata: dict | None = None) -> Any:
        messages = [{"role": "user", "content": self._payload(content)}]
        kwargs: dict[str, Any] = {"user_id": self.namespace}
        if metadata:
            kwargs["metadata"] = redact_for_upload(metadata, self.settings.project_root)
        return self._bounded(self._client.add, messages, **kwargs)

    def _search(self, query: str, limit: int) -> list[dict]:
        results = self._bounded(
            self._client.search,
            self._payload(query),
            user_id=self.namespace,
            limit=limit,
        )
        if isinstance(results, list):
            return results
        if isinstance(results, dict):
            found = results.get("results", [])
            return found if isinstance(found, list) else []
        return []

    @staticmethod
    def _short_hash(text: str) -> str:
        return hashlib.sha1(text.encode()).hexdigest()[:8]

    @staticmethod
    def _valid_patches(patches: Any) -> list[dict]:
        """Keep only genuine dict patches.

        The reflection stage may yield ``None`` for a minibatch that produced
        nothing, and a malformed backend reply can yield a bare string. Both
        are filtered here rather than raising inside a ``try`` that swallows
        the error, so a malformed patch costs one skipped record instead of
        the whole memory write.
        """
        if not isinstance(patches, (list, tuple)):
            return []
        return [p for p in patches if isinstance(p, dict)]

    # ── Public API ────────────────────────────────────────────────────────

    def store_skill_iteration(
        self,
        epoch: int,
        step: int,
        skill_text: str,
        score: float,
        metadata: dict | None = None,
    ) -> Any:
        """Store the skill version produced at *epoch*/*step*."""
        skill_hash = self._short_hash(skill_text or "")
        base_meta: dict[str, Any] = {
            "event_type": "skill_iteration",
            "epoch": epoch,
            "step": step,
            "score": round(float(score), 6),
            "skill_hash": skill_hash,
            "skill_length": len(skill_text or ""),
        }
        if metadata:
            base_meta.update(metadata)

        content = (
            f"[SkillOpt] Epoch {epoch} Step {step} — skill_hash={skill_hash} "
            f"score={score:.4f}\n\n"
            f"=== SKILL TEXT ===\n{skill_text or ''}"
        )
        return self._add(content, metadata=base_meta)

    def store_reflection(
        self,
        epoch: int,
        step: int,
        patches: Any,
        scores: dict | None = None,
    ) -> Any:
        """Store a summary of the patches produced by one Reflect stage."""
        valid = self._valid_patches(patches)
        n_patches = len(valid)
        scores_str = json.dumps(scores or {}, ensure_ascii=False)
        try:
            patch_summary = json.dumps(
                [{k: v for k, v in p.items() if k != "skill_text"} for p in valid[:10]],
                ensure_ascii=False,
                default=str,
            )
        except (TypeError, ValueError):
            patch_summary = "[unserialisable patch summary]"

        base_meta: dict[str, Any] = {
            "event_type": "reflection",
            "epoch": epoch,
            "step": step,
            "n_patches": n_patches,
        }
        if scores:
            base_meta.update({f"score_{k}": v for k, v in scores.items()})

        content = (
            f"[SkillOpt] Reflection Epoch {epoch} Step {step} — "
            f"{n_patches} patch(es) generated. Scores: {scores_str}\n\n"
            f"Patch summary:\n{patch_summary}"
        )
        return self._add(content, metadata=base_meta)

    def retrieve_relevant_context(self, query: str, limit: int | None = None) -> list[dict]:
        """Return past memories relevant to *query* (empty on any failure)."""
        if not self.settings.retrieval_enabled:
            return []
        return self._search(query, limit=limit or self.settings.retrieval_limit)

    def format_retrieved_context(self, memories: list[dict], max_chars: int = 2000) -> str:
        """Render retrieved memories as text for inclusion in a prompt.

        Returns ``""`` when there is nothing useful, so callers can append
        unconditionally without producing an empty heading.
        """
        lines: list[str] = []
        for rec in memories or []:
            if not isinstance(rec, dict):
                continue
            text = rec.get("memory") or rec.get("text") or ""
            if not isinstance(text, str) or not text.strip():
                continue
            lines.append(f"- {text.strip()}")
        if not lines:
            return ""

        body = "\n".join(lines)
        if len(body) > max_chars:
            body = body[:max_chars] + "\n[...truncated]"
        return "## Relevant Memory From Previous Runs\n" + body

    def close(self) -> None:
        """Release the worker thread. Safe to call more than once."""
        self._pool.shutdown(wait=False)

    def __repr__(self) -> str:
        return f"<SkillMemory namespace={self.namespace!r} client={self._client.__class__.__name__}>"
