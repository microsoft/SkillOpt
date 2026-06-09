"""
Benchmark Environment Template
===============================
Copy this file and implement the TODO sections to add a new benchmark.

The EnvAdapter is responsible for:
  1. Building per-batch environment managers (train and eval splits).
  2. Running rollouts under the current skill document.
  3. Reflecting on those rollouts into raw patch dicts.
  4. Reporting the distinct task types in your data (for stratified
     sampling).

For a fully worked example see ``skillopt/envs/officeqa/``.
"""
from __future__ import annotations

from skillopt.datasets.base import BatchSpec
from skillopt.envs.base import EnvAdapter
from skillopt.envs._template.loader_template import TemplateBenchmarkLoader


class TemplateBenchmarkEnv(EnvAdapter):
    """
    Environment adapter for <Your Benchmark Name>.

    Rename this class. Each abstract method below is required by
    :class:`skillopt.envs.base.EnvAdapter`. The template implementations
    are minimal so this file is importable and instantiable; replace the
    TODOs with real logic.
    """

    def __init__(
        self,
        split_dir: str = "",
        data_path: str = "",
        split_mode: str = "split_dir",
        split_ratio: str = "2:1:7",
        split_seed: int = 42,
        split_output_dir: str = "",
        workers: int = 4,
        analyst_workers: int = 4,
        failure_only: bool = False,
        minibatch_size: int = 8,
        edit_budget: int = 4,
        seed: int = 42,
        limit: int = 0,
        max_completion_tokens: int = 4096,
    ) -> None:
        self.workers = workers
        self.analyst_workers = analyst_workers
        self.failure_only = failure_only
        self.minibatch_size = minibatch_size
        self.edit_budget = edit_budget
        self.max_completion_tokens = int(max_completion_tokens)
        self.dataloader = TemplateBenchmarkLoader(
            split_dir=split_dir,
            data_path=data_path,
            split_mode=split_mode,
            split_ratio=split_ratio,
            split_seed=split_seed,
            split_output_dir=split_output_dir,
            seed=seed,
            limit=limit,
        )

    # ── Lifecycle hooks ────────────────────────────────────────────────

    def setup(self, cfg: dict) -> None:
        super().setup(cfg)
        self.dataloader.setup(cfg)

    def get_dataloader(self):
        return self.dataloader

    # ── Batch → env manager ────────────────────────────────────────────

    def build_env_from_batch(self, batch: BatchSpec, **kwargs):
        # Dataset-backed envs typically just pass items straight through.
        return list(batch.payload or [])

    def build_train_env(self, batch_size: int, seed: int, **kwargs):
        batch = self.dataloader.build_train_batch(
            batch_size=batch_size, seed=seed, **kwargs
        )
        return self.build_env_from_batch(batch, **kwargs)

    def build_eval_env(self, env_num: int, split: str, seed: int, **kwargs):
        batch = self.dataloader.build_eval_batch(
            env_num=env_num, split=split, seed=seed, **kwargs
        )
        return self.build_env_from_batch(batch, **kwargs)

    # ── Rollout: run episodes under current skill ──────────────────────

    def rollout(
        self,
        env_manager,
        skill_content: str,
        out_dir: str,
        **kwargs,
    ) -> list[dict]:
        """
        Run a batch of episodes under the current skill.

        TODO: replace this loop with your real rollout. For each item:
          1. Build the prompt using `skill_content` as the system message.
          2. Call your target model.
          3. Score the prediction.
          4. Return a dict with at minimum: ``id`` (str), ``hard`` (0|1),
             ``soft`` (float in [0, 1]). Add any env-specific extras you
             need for reflect() — they will be preserved on
             ``RolloutResult.extras``.
        """
        items: list[dict] = env_manager
        results: list[dict] = []
        for item in items:
            # ── REPLACE THIS BLOCK WITH YOUR REAL ROLLOUT ──
            results.append(
                {
                    "id": str(item.get("id", "")),
                    "hard": 0,
                    "soft": 0.0,
                    "predicted_answer": "",
                    "question": item.get("question", ""),
                    "fail_reason": "template rollout — not implemented",
                }
            )
        return results

    # ── Reflect (inherited) ─────────────────────────────────────────────
    #
    # ``reflect`` is inherited from ``EnvAdapter``: the default delegates to
    # ``skillopt.gradient.reflect.run_minibatch_reflect`` using your
    # ``analyst_error_*`` / ``analyst_success_*`` prompts. You do NOT need to
    # implement it — override only if your benchmark needs custom reflection.

    # ── Stratification hint ────────────────────────────────────────────

    def get_task_types(self) -> list[str]:
        """Distinct task-type strings used for stratified sampling."""
        seen: list[str] = []
        all_items = (
            self.dataloader.train_items
            + self.dataloader.val_items
            + self.dataloader.test_items
        )
        for item in all_items:
            tt = str(item.get("task_type") or "template")
            if tt not in seen:
                seen.append(tt)
        return seen or ["template"]
