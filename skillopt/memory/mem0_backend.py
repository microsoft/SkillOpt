"""mem0-backed persistent memory for SkillOpt.

Stores skill iterations, reflection results, and experiment outcomes in mem0
so that the ReflACT trainer can retrieve relevant historical context across
runs and identify the best skill versions discovered so far.

Usage::

    from skillopt.memory import SkillMemory

    m = SkillMemory(api_key="m0-...")
    m.store_skill_iteration(epoch=1, step=3, skill_text="...", score=0.82)
    ctx = m.retrieve_relevant_context("handling multi-step navigation")
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any

try:
    from mem0 import MemoryClient
    _MEM0_AVAILABLE = True
except ImportError:
    _MEM0_AVAILABLE = False
    MemoryClient = None  # type: ignore[assignment,misc]


class SkillMemory:
    """Persistent memory backend for SkillOpt using mem0.

    Parameters
    ----------
    api_key : str | None
        mem0 API key. Falls back to ``MEM0_API_KEY`` env var, then to
        ``MEM0_API_KEY`` set on the object at construction time.
    user_id : str
        Logical user/project identifier used to namespace memories in mem0.
    """

    def __init__(
        self,
        api_key: str | None = None,
        user_id: str = "skillopt",
    ) -> None:
        if not _MEM0_AVAILABLE:
            raise ImportError(
                "mem0ai is not installed. Run: pip install mem0ai"
            )

        resolved_key = api_key or os.environ.get("MEM0_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "No mem0 API key provided. Pass api_key= or set MEM0_API_KEY env var."
            )

        self.user_id = user_id
        self._client = MemoryClient(api_key=resolved_key)

    # ── Internal helpers ──────────────────────────────────────────────────

    def _add(self, messages: list[dict], metadata: dict | None = None) -> Any:
        """Low-level add wrapper — always tags with user_id."""
        kwargs: dict[str, Any] = {"user_id": self.user_id}
        if metadata:
            kwargs["metadata"] = metadata
        return self._client.add(messages, **kwargs)

    def _search(self, query: str, limit: int = 5) -> list[dict]:
        """Low-level search wrapper — scoped to this user_id."""
        results = self._client.search(query, user_id=self.user_id, limit=limit)
        # mem0 returns a list of memory dicts
        if isinstance(results, list):
            return results
        # Some versions wrap results in a dict
        if isinstance(results, dict):
            return results.get("results", [])
        return []

    @staticmethod
    def _short_hash(text: str) -> str:
        return hashlib.sha1(text.encode()).hexdigest()[:8]

    # ── Public API ────────────────────────────────────────────────────────

    def store_skill_iteration(
        self,
        epoch: int,
        step: int,
        skill_text: str,
        score: float,
        metadata: dict | None = None,
    ) -> Any:
        """Store a skill version produced during training.

        Parameters
        ----------
        epoch : int
            Current training epoch.
        step : int
            Current training step within the epoch.
        skill_text : str
            Full text of the skill document at this point.
        score : float
            Evaluation score (0–1) for this skill version.
        metadata : dict | None
            Optional additional metadata to attach.
        """
        skill_hash = self._short_hash(skill_text)
        base_meta: dict[str, Any] = {
            "event_type": "skill_iteration",
            "epoch": epoch,
            "step": step,
            "score": round(float(score), 6),
            "skill_hash": skill_hash,
            "skill_length": len(skill_text),
        }
        if metadata:
            base_meta.update(metadata)

        content = (
            f"[SkillOpt] Epoch {epoch} Step {step} — skill_hash={skill_hash} "
            f"score={score:.4f}\n\n"
            f"=== SKILL TEXT ===\n{skill_text[:4000]}"  # cap to avoid huge payloads
        )

        messages = [{"role": "user", "content": content}]
        return self._add(messages, metadata=base_meta)

    def store_reflection(
        self,
        epoch: int,
        step: int,
        patches: list[dict],
        scores: dict | None = None,
    ) -> Any:
        """Store the result of a Reflect stage.

        Parameters
        ----------
        epoch : int
            Current training epoch.
        step : int
            Current training step.
        patches : list[dict]
            Raw patches produced by the reflection stage.
        scores : dict | None
            Optional dict of metric → value (e.g. ``{"hard": 0.7, "soft": 0.8}``).
        """
        n_patches = len(patches)
        scores_str = json.dumps(scores or {}, ensure_ascii=False)
        patch_summary = json.dumps(
            [
                {k: v for k, v in p.items() if k != "skill_text"}
                for p in patches[:10]  # only first 10 to keep it concise
            ],
            ensure_ascii=False,
        )

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

        messages = [{"role": "user", "content": content}]
        return self._add(messages, metadata=base_meta)

    def retrieve_relevant_context(
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict]:
        """Retrieve past memories relevant to *query*.

        Parameters
        ----------
        query : str
            Free-text query describing the context you want to retrieve.
        limit : int
            Maximum number of results to return.

        Returns
        -------
        list[dict]
            List of memory objects (each has at least a ``memory`` field).
        """
        return self._search(query, limit=limit)

    def get_best_skill(self, user_id: str | None = None) -> dict | None:
        """Return the memory record for the highest-scored skill iteration.

        Searches mem0 for skill_iteration records and picks the one with
        the highest ``score`` in its metadata.

        Parameters
        ----------
        user_id : str | None
            Override the user_id for this query (defaults to ``self.user_id``).

        Returns
        -------
        dict | None
            The memory record with the highest score, or ``None`` if no skill
            iterations have been stored yet.
        """
        results = self._client.search(
            "skill iteration score evaluation",
            user_id=user_id or self.user_id,
            limit=50,
        )
        if isinstance(results, dict):
            results = results.get("results", [])
        if not results:
            return None

        best: dict | None = None
        best_score = -1.0
        for rec in results:
            meta = rec.get("metadata") or {}
            if meta.get("event_type") != "skill_iteration":
                continue
            try:
                s = float(meta.get("score", -1))
            except (TypeError, ValueError):
                continue
            if s > best_score:
                best_score = s
                best = rec

        return best

    def store_experiment_result(
        self,
        config_name: str,
        final_score: float,
        skill_hash: str,
    ) -> Any:
        """Store the final outcome of an experiment run.

        Parameters
        ----------
        config_name : str
            Human-readable name / path of the config used for this run.
        final_score : float
            Best evaluation score achieved during the run.
        skill_hash : str
            Hash of the best skill version (from :meth:`store_skill_iteration`).
        """
        base_meta: dict[str, Any] = {
            "event_type": "experiment_result",
            "config_name": config_name,
            "final_score": round(float(final_score), 6),
            "skill_hash": skill_hash,
        }

        content = (
            f"[SkillOpt] Experiment complete — config={config_name!r} "
            f"final_score={final_score:.4f} best_skill_hash={skill_hash}"
        )

        messages = [{"role": "user", "content": content}]
        return self._add(messages, metadata=base_meta)

    def __repr__(self) -> str:
        return (
            f"<SkillMemory user_id={self.user_id!r} "
            f"client={self._client.__class__.__name__}>"
        )
