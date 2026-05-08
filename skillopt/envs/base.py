"""ReflACT environment adapter — abstract interface.

To connect ReflACT to a new environment (benchmark, simulator, etc.),
implement a subclass of :class:`EnvAdapter` with environment-specific
rollout and reflection logic.

Example::

    class MyBenchAdapter(EnvAdapter):
        def build_train_env(self, batch_size, seed, **kw):
            return MyEnvManager(split="train", n=batch_size, seed=seed)

        def build_eval_env(self, env_num, split, seed, **kw):
            return MyEnvManager(split=split, n=env_num, seed=seed)

        def rollout(self, env_manager, skill_content, out_dir, **kw):
            # Run episodes, return [{"id": ..., "hard": 0/1, "soft": 0.0-1.0, ...}]
            ...

        def reflect(self, results, skill_content, out_dir, **kw):
            # Analyze trajectories, return list of patch dicts
            ...

        def get_task_types(self):
            return ["task_a", "task_b"]
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import os
import random

from skillopt.datasets.base import BaseDataLoader, BatchSpec
from skillopt.model.codex_harness import extract_codex_trace_prefix, format_codex_trace_steps, parse_codex_raw
from skillopt.prompts import load_prompt


class EnvAdapter(ABC):
    """Abstract adapter for connecting ReflACT to any environment.

    Subclasses must implement all abstract methods. The ReflACT trainer
    calls these methods at the appropriate pipeline stages.
    """

    # ── Lifecycle hooks ────────────────────────────────────────────────────

    def setup(self, cfg: dict) -> None:
        """Called once by the trainer before the training loop begins.

        Override to perform one-time initialization that requires the full
        config (e.g., data loading, split creation).  Default is a no-op.
        """
        self._cfg = dict(cfg)

    def get_dataloader(self) -> BaseDataLoader | None:
        """Return the task dataloader used by this adapter, if any."""
        return None

    def requires_ray(self) -> bool:
        """Return whether this adapter requires Ray runtime initialization."""
        return False

    def deep_reflect(
        self,
        results: list[dict],
        skill_content: str,
        out_dir: str,
        **kwargs,
    ) -> list[dict | None]:
        """Optional deeper diagnostic reflection pass.

        Default behavior is a no-op. Dataset-backed adapters may override this
        to re-query the student on a small representative subset of the current
        batch using minimally-perturbed diagnostic prompts that expose
        intermediate reasoning state.
        """
        return []

    def build_reference_text(self, item: dict) -> str:
        """Return hidden reference material for deep reflection, if any."""
        return str(item.get("reference_text") or "").strip()

    def get_reference_metadata(self, item: dict) -> dict:
        """Return structured metadata about hidden reference material."""
        reference_text = self.build_reference_text(item)
        if not reference_text:
            return {"fields": [], "preview": ""}
        return {
            "fields": ["reference_text"],
            "preview": reference_text[:400],
        }

    def get_codex_deep_probe_prompt(self) -> str | None:
        env_name = getattr(self, "_cfg", {}).get("env_name")
        return load_prompt("deep_probe_codex", env=env_name)

    def attach_codex_probe_context(
        self,
        results: list[dict],
        prediction_dir: str,
    ) -> list[dict]:
        """Attach compact Codex step metadata for codex-aware deep reflection."""
        enriched: list[dict] = []
        for row in results:
            merged = dict(row)
            tid = str(row.get("id"))
            raw_path = os.path.join(prediction_dir, tid, "codex_raw.txt")
            if os.path.exists(raw_path):
                with open(raw_path, encoding="utf-8") as f:
                    raw = f.read()
                parsed = parse_codex_raw(raw)
                merged["codex_probe_trace_steps"] = format_codex_trace_steps(raw)
                merged["codex_probe_step_count"] = len(parsed["steps"])
            enriched.append(merged)
        return enriched

    def resolve_codex_probe_target(
        self,
        *,
        selected_items: list[dict],
        selected_examples: list[dict],
        prediction_dir: str,
        probe: dict,
    ) -> tuple[list[dict], dict[str, str] | None, dict]:
        """Resolve the teacher-selected codex probe target and raw trace prefix."""
        target_id = str(probe.get("probe_target_id", "")).strip()
        selected_id_set = {str(item["id"]) for item in selected_items}
        if target_id not in selected_id_set:
            target_id = str(selected_items[0]["id"])
        target_item = next(item for item in selected_items if str(item["id"]) == target_id)
        target_result = next(
            (row for row in selected_examples if str(row.get("id")) == target_id),
            None,
        )
        max_probe_step = int((target_result or {}).get("codex_probe_step_count", 0))
        default_probe_step = max_probe_step - 1 if max_probe_step > 1 else max_probe_step
        probe_after_step = int(probe.get("probe_after_step", default_probe_step))
        if max_probe_step > 0:
            probe_after_step = max(0, min(probe_after_step, max_probe_step))
        else:
            probe_after_step = 0
        raw_path = os.path.join(prediction_dir, target_id, "codex_raw.txt")
        trace_prefix = ""
        if os.path.exists(raw_path):
            with open(raw_path, encoding="utf-8") as f:
                trace_prefix = extract_codex_trace_prefix(f.read(), after_step=probe_after_step)
        updated_probe = dict(probe)
        updated_probe["probe_target_id"] = target_id
        updated_probe["probe_after_step"] = probe_after_step
        return [target_item], {target_id: trace_prefix}, updated_probe

    def attach_reference_context(
        self,
        results: list[dict],
        items: list[dict] | None,
    ) -> list[dict]:
        """Attach environment-specific hidden reference text to result dicts."""
        if not results or not items:
            return list(results)

        item_by_id = {
            str(item.get("id")): item
            for item in items
            if isinstance(item, dict) and item.get("id") is not None
        }
        enriched: list[dict] = []
        for row in results:
            merged = dict(row)
            item = item_by_id.get(str(row.get("id")))
            if item:
                reference_text = self.build_reference_text(item)
                if reference_text:
                    merged["reference_text"] = reference_text
            enriched.append(merged)
        return enriched

    def select_representative_items(
        self,
        results: list[dict],
        items: list[dict] | None,
        *,
        n_failures: int,
        n_successes: int,
        seed: int | None = None,
    ) -> list[dict]:
        """Select a small diverse subset of current-batch items by outcome."""
        if not items:
            return []

        item_by_id = {
            str(item.get("id")): item
            for item in items
            if isinstance(item, dict) and item.get("id") is not None
        }
        failures = [
            (result, item_by_id[str(result.get("id"))])
            for result in results
            if not result.get("hard") and str(result.get("id")) in item_by_id
        ]
        successes = [
            (result, item_by_id[str(result.get("id"))])
            for result in results
            if result.get("hard") and str(result.get("id")) in item_by_id
        ]

        rng = random.Random(seed)

        def _pick(pool: list[tuple[dict, dict]], quota: int) -> list[dict]:
            if quota <= 0 or not pool:
                return []
            shuffled = list(pool)
            rng.shuffle(shuffled)

            picked_ids: set[str] = set()
            picked: list[dict] = []
            seen_types: set[str] = set()

            for result, item in shuffled:
                task_type = str(result.get("task_type") or item.get("task_type") or item.get("subtype") or "unknown")
                item_id = str(item["id"])
                if task_type in seen_types or item_id in picked_ids:
                    continue
                picked.append(item)
                picked_ids.add(item_id)
                seen_types.add(task_type)
                if len(picked) >= quota:
                    return picked

            for _, item in shuffled:
                item_id = str(item["id"])
                if item_id in picked_ids:
                    continue
                picked.append(item)
                picked_ids.add(item_id)
                if len(picked) >= quota:
                    break
            return picked

        selected = _pick(failures, n_failures)
        selected_ids = {str(item["id"]) for item in selected}
        selected.extend(
            item for item in _pick(successes, n_successes)
            if str(item["id"]) not in selected_ids
        )
        return selected

    def build_env_from_batch(self, batch: BatchSpec, **kwargs):
        """Build an environment manager or item list from a :class:`BatchSpec`.

        Default behavior preserves the legacy adapter API by routing training
        batches through :meth:`build_train_env` and evaluation batches through
        :meth:`build_eval_env`.
        """
        if batch.phase == "train":
            return self.build_train_env(batch_size=batch.batch_size, seed=batch.seed, **kwargs)
        return self.build_eval_env(
            env_num=batch.batch_size,
            split=batch.split,
            seed=batch.seed,
            **kwargs,
        )

    @abstractmethod
    def build_train_env(self, batch_size: int, seed: int, **kwargs):
        """Build a training environment manager.

        Returns
        -------
        object
            An environment manager that can be passed to :meth:`rollout`.
        """

    @abstractmethod
    def build_eval_env(self, env_num: int, split: str, seed: int, **kwargs):
        """Build an evaluation environment manager.

        Parameters
        ----------
        env_num : int
            Number of evaluation environments.
        split : str
            Dataset split (e.g. ``"valid_seen"``, ``"valid_unseen"``).
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        object
            An environment manager that can be passed to :meth:`rollout`.
        """

    @abstractmethod
    def rollout(
        self,
        env_manager,
        skill_content: str,
        out_dir: str,
        **kwargs,
    ) -> list[dict]:
        """Run a batch of episodes using the current skill.

        Returns
        -------
        list[dict]
            Each dict conforms to :class:`~skillopt.types.RolloutResult`:
            must have ``"id"`` (str), ``"hard"`` (0/1), ``"soft"``
            (float 0-1). May include env-specific fields.
        """

    @abstractmethod
    def reflect(
        self,
        results: list[dict],
        skill_content: str,
        out_dir: str,
        **kwargs,
    ) -> list[dict | None]:
        """Analyze rollout results and produce patches.

        Each returned dict conforms to :class:`~skillopt.types.RawPatch`:
        ``"patch"`` (with ``"edits"`` list) + ``"source_type"``
        (``"failure"`` or ``"success"``).

        Returns
        -------
        list[dict | None]
            Raw analyst outputs; ``None`` entries are filtered out.
        """

    @abstractmethod
    def get_task_types(self) -> list[str]:
        """Return the list of task type names for this environment."""

    # ── Prompt configuration (two-level priority) ────────────────────────
    #
    # Priority: env-specific prompt file  >  generic default prompt file.
    #
    # Prompts are loaded from ``.md`` files via ``load_prompt(name, env)``:
    #   1. ``skillopt/envs/<env>/prompts/<name>.md``  (env-specific)
    #   2. ``skillopt/prompts/<name>.md``             (generic fallback)
    #
    # Subclasses can still override ``get_*_prompt()`` for full control.

    @property
    def _env_name(self) -> str:
        """Derive the env directory name from this adapter's module path."""
        # e.g. "skillopt.envs.searchqa.adapter" → "searchqa"
        module = type(self).__module__
        parts = module.split(".")
        if len(parts) >= 3 and parts[-3] == "envs":
            return parts[-2]
        return ""

    def _load_env_prompt(self, name: str) -> str | None:
        """Load a prompt with env-specific override. Returns None if not found."""
        try:
            return load_prompt(name, env=self._env_name)
        except FileNotFoundError:
            return None

    def get_error_minibatch_prompt(self) -> str | None:
        update_mode = getattr(self, "_cfg", {}).get("skill_update_mode", "patch")
        raw_mode = str(update_mode).strip().lower()
        if raw_mode in {"full_rewrite", "full_rewrite_minibatch", "minibatch_full_rewrite", "skill_rewrite_minibatch"}:
            prompt = self._load_env_prompt("analyst_error_full_rewrite")
            if prompt is not None:
                return prompt
        if raw_mode in {"rewrite", "rewrite_from_suggestions", "suggestions", "rewrite_suggestions"}:
            prompt = self._load_env_prompt("analyst_error_rewrite")
            if prompt is not None:
                return prompt
        return self._load_env_prompt("analyst_error")

    def get_success_minibatch_prompt(self) -> str | None:
        update_mode = getattr(self, "_cfg", {}).get("skill_update_mode", "patch")
        raw_mode = str(update_mode).strip().lower()
        if raw_mode in {"full_rewrite", "full_rewrite_minibatch", "minibatch_full_rewrite", "skill_rewrite_minibatch"}:
            prompt = self._load_env_prompt("analyst_success_full_rewrite")
            if prompt is not None:
                return prompt
        if raw_mode in {"rewrite", "rewrite_from_suggestions", "suggestions", "rewrite_suggestions"}:
            prompt = self._load_env_prompt("analyst_success_rewrite")
            if prompt is not None:
                return prompt
        return self._load_env_prompt("analyst_success")

    def get_deep_probe_prompt(self) -> str | None:
        return self._load_env_prompt("deep_probe")

    def get_meta_reflect_prompt(self) -> str | None:
        update_mode = getattr(self, "_cfg", {}).get("skill_update_mode", "patch")
        if str(update_mode).strip().lower() == "rewrite_from_suggestions":
            prompt = self._load_env_prompt("meta_reflect_rewrite")
            if prompt is not None:
                return prompt
        return self._load_env_prompt("meta_reflect")
