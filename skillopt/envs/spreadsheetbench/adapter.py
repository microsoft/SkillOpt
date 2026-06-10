"""SpreadsheetBench environment adapter for ReflACT.

Connects the ReflACT training loop to SpreadsheetBench by implementing
:class:`~skillopt.envs.base.EnvAdapter`.
"""
from __future__ import annotations

import json
import os

from skillopt.datasets.base import BatchSpec
from skillopt.envs.base import EnvAdapter
from skillopt.envs.spreadsheetbench.dataloader import SpreadsheetBenchDataLoader
from skillopt.envs.spreadsheetbench.rollout import (
    process_one,
    run_spreadsheet_batch,
    run_spreadsheet_batch_codegen,
)
from skillopt.model import get_target_backend, is_target_exec_backend


# Task types used for per-category breakdowns
TASK_TYPES = ["cell_level", "sheet_level"]


class SpreadsheetBenchAdapter(EnvAdapter):
    """SpreadsheetBench environment adapter."""

    def __init__(
        self,
        split_dir: str = "",
        data_path: str = "",
        split_mode: str = "ratio",
        split_ratio: str = "2:1:7",
        split_seed: int = 42,
        split_output_dir: str = "",
        data_root: str = "",
        mode: str = "single",
        max_turns: int = 30,
        exec_timeout: int = 600,
        workers: int = 64,
        analyst_workers: int = 16,
        failure_only: bool = False,
        minibatch_size: int = 8,
        edit_budget: int = 4,
        seed: int = 42,
        max_completion_tokens: int = 16384,
    ) -> None:
        self.data_root = data_root
        self.mode = mode  # "single", "multi", or "react"
        self.max_turns = max_turns
        self.exec_timeout = exec_timeout
        self.workers = workers
        self.max_completion_tokens = int(max_completion_tokens)
        self.analyst_workers = analyst_workers
        self.failure_only = failure_only
        self.minibatch_size = minibatch_size
        self.edit_budget = edit_budget
        self.dataloader = SpreadsheetBenchDataLoader(
            split_dir=split_dir,
            data_path=data_path,
            split_mode=split_mode,
            split_ratio=split_ratio,
            split_seed=split_seed,
            split_output_dir=split_output_dir,
            data_root=data_root,
            seed=seed,
        )

    def setup(self, cfg: dict) -> None:
        super().setup(cfg)
        if is_target_exec_backend() and self.mode != "single":
            raise NotImplementedError(
                "Exec target backends are currently supported only for SpreadsheetBench mode=single."
            )
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
        """Run agent on all items and return results.

        Dispatches based on ``self.mode``:
          - ``"single"`` / ``"multi"``: codegen agent (no tool-call)
          - ``"react"``: ReAct agent with tool-call (legacy)
        """
        items = env_manager  # For static datasets, env_manager is a list of items
        results_path = os.path.join(out_dir, "results.jsonl")
        os.makedirs(out_dir, exist_ok=True)

        # Resume support
        if os.path.exists(results_path):
            existing: list[dict] = []
            with open(results_path) as f:
                for line in f:
                    try:
                        existing.append(json.loads(line))
                    except Exception:
                        pass
            if existing:
                return existing

        if self.mode in ("single", "multi"):
            results = run_spreadsheet_batch_codegen(
                items=items,
                data_root=self.data_root,
                out_root=out_dir,
                skill_content=skill_content,
                mode=self.mode,
                max_turns=self.max_turns,
                max_completion_tokens=self.max_completion_tokens,
                max_api_workers=self.workers,
                task_timeout=self.exec_timeout,
                use_eval_feedback=kwargs.get("use_eval_feedback", False),
                diagnostic_mode=kwargs.get("diagnostic_mode", False),
                diagnostic_instruction=kwargs.get("diagnostic_instruction", ""),
                diagnostic_trace_context_by_id=kwargs.get("diagnostic_trace_context_by_id"),
            )
        else:
            results = run_spreadsheet_batch(
                items=items,
                data_root=self.data_root,
                out_root=out_dir,
                skill_content=skill_content,
                max_turns=self.max_turns,
                max_completion_tokens=self.max_completion_tokens,
                max_api_workers=self.workers,
                task_timeout=max(600, int(self.exec_timeout) + 60),
                diagnostic_mode=kwargs.get("diagnostic_mode", False),
                diagnostic_instruction=kwargs.get("diagnostic_instruction", ""),
                diagnostic_trace_context_by_id=kwargs.get("diagnostic_trace_context_by_id"),
            )

        with open(results_path, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        return results

    def get_task_types(self) -> list[str]:
        return list(TASK_TYPES)
