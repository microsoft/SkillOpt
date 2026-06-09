"""LiveMathematicianBench environment adapter for ReflACT."""
from __future__ import annotations

import json

from skillopt.datasets.base import BatchSpec
from skillopt.envs.base import EnvAdapter
from skillopt.envs.livemathematicianbench.dataloader import LiveMathematicianBenchDataLoader
from skillopt.envs.livemathematicianbench.rollout import run_batch
from skillopt.model import get_target_backend


class LiveMathematicianBenchAdapter(EnvAdapter):
    """LiveMathematicianBench adapter."""

    def build_reference_text(self, item: dict) -> str:
        parts: list[str] = []
        theorem = str(item.get("theorem") or "").strip()
        sketch = str(item.get("sketch") or "").strip()
        if theorem:
            parts.append(f"## Reference Theorem\n{theorem}")
        if sketch:
            parts.append(f"## Reference Sketch\n{sketch}")
        return "\n\n".join(parts)

    def get_reference_metadata(self, item: dict) -> dict:
        fields: list[str] = []
        previews: list[str] = []
        theorem = str(item.get("theorem") or "").strip()
        sketch = str(item.get("sketch") or "").strip()
        if theorem:
            fields.append("theorem")
            previews.append(f"[theorem]\n{theorem[:220]}")
        if sketch:
            fields.append("sketch")
            previews.append(f"[sketch]\n{sketch[:220]}")
        return {
            "fields": fields,
            "preview": "\n\n".join(previews)[:500],
        }

    def __init__(
        self,
        split_dir: str = "",
        data_path: str = "",
        split_mode: str = "ratio",
        split_ratio: str = "2:1:7",
        split_seed: int = 42,
        split_output_dir: str = "",
        max_turns: int = 1,
        exec_timeout: int = 600,
        workers: int = 64,
        analyst_workers: int = 16,
        failure_only: bool = False,
        minibatch_size: int = 8,
        edit_budget: int = 4,
        seed: int = 42,
        limit: int = 0,
        shuffle_choices: bool = True,
        use_theorem: bool = False,
        use_sketch: bool = False,
        max_completion_tokens: int = 16384,
    ) -> None:
        self.max_turns = max_turns
        self.exec_timeout = exec_timeout
        self.workers = workers
        self.max_completion_tokens = int(max_completion_tokens)
        self.analyst_workers = analyst_workers
        self.failure_only = failure_only
        self.minibatch_size = minibatch_size
        self.edit_budget = edit_budget
        self.use_theorem = use_theorem
        self.use_sketch = use_sketch
        self.dataloader = LiveMathematicianBenchDataLoader(
            split_dir=split_dir,
            data_path=data_path,
            split_mode=split_mode,
            split_ratio=split_ratio,
            split_seed=split_seed,
            split_output_dir=split_output_dir,
            seed=seed,
            limit=limit,
            shuffle_choices=shuffle_choices,
        )

    def setup(self, cfg: dict) -> None:
        super().setup(cfg)
        self.dataloader.setup(cfg)

    def get_dataloader(self):
        return self.dataloader

    def build_env_from_batch(self, batch: BatchSpec, **kwargs):
        return list(batch.payload or [])

    def build_train_env(self, batch_size: int, seed: int, **kwargs):
        batch = self.dataloader.build_train_batch(batch_size=batch_size, seed=seed, **kwargs)
        return self.build_env_from_batch(batch, **kwargs)

    def build_eval_env(self, env_num: int, split: str, seed: int, **kwargs):
        batch = self.dataloader.build_eval_batch(env_num=env_num, split=split, seed=seed, **kwargs)
        return self.build_env_from_batch(batch, **kwargs)

    def rollout(
        self,
        env_manager,
        skill_content: str,
        out_dir: str,
        **kwargs,
    ) -> list[dict]:
        items: list[dict] = env_manager
        return run_batch(
            items=items,
            out_root=out_dir,
            skill_content=skill_content,
            max_turns=self.max_turns,
            exec_timeout=self.exec_timeout,
            workers=self.workers,
            max_completion_tokens=self.max_completion_tokens,
            use_theorem=self.use_theorem,
            use_sketch=self.use_sketch,
            diagnostic_mode=kwargs.get("diagnostic_mode", False),
            diagnostic_instruction=kwargs.get("diagnostic_instruction", ""),
            diagnostic_trace_context_by_id=kwargs.get("diagnostic_trace_context_by_id"),
            task_timeout=self.exec_timeout,
        )

    def get_task_types(self) -> list[str]:
        return self.dataloader.get_task_types()
