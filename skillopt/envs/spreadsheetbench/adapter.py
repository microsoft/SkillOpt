"""SpreadsheetBench environment adapter for ReflACT.

Connects the ReflACT training loop to SpreadsheetBench by implementing
:class:`~skillopt.envs.base.EnvAdapter`.
"""
from __future__ import annotations

import json
import os

from skillopt.gradient.deep_probe import generate_deep_probe_instruction
from skillopt.datasets.base import BatchSpec
from skillopt.envs.base import EnvAdapter
from skillopt.envs.spreadsheetbench.dataloader import SpreadsheetBenchDataLoader
from skillopt.envs.spreadsheetbench.rollout import (
    process_one,
    run_spreadsheet_batch,
    run_spreadsheet_batch_codegen,
)
from skillopt.gradient.reflect import run_minibatch_reflect
from skillopt.model import get_student_backend, is_student_exec_backend


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
        use_deep_reflect: bool = False,
        deep_reflect_failures: int = 4,
        deep_reflect_successes: int = 2,
    ) -> None:
        self.data_root = data_root
        self.mode = mode  # "single", "multi", or "react"
        self.max_turns = max_turns
        self.exec_timeout = exec_timeout
        self.workers = workers
        self.analyst_workers = analyst_workers
        self.failure_only = failure_only
        self.minibatch_size = minibatch_size
        self.edit_budget = edit_budget
        self.use_deep_reflect = use_deep_reflect
        self.deep_reflect_failures = deep_reflect_failures
        self.deep_reflect_successes = deep_reflect_successes
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
        if is_student_exec_backend() and self.mode != "single":
            raise NotImplementedError(
                "Exec student backends are currently supported only for SpreadsheetBench mode=single."
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

    def reflect(
        self,
        results: list[dict],
        skill_content: str,
        out_dir: str,
        **kwargs,
    ) -> list[dict | None]:
        """Analyze rollout results and produce patches (minibatch mode)."""
        prediction_dir = kwargs.get("prediction_dir", os.path.join(out_dir, "predictions"))
        patches_dir = kwargs.get("patches_dir", os.path.join(out_dir, "patches"))
        random_seed = kwargs.get("random_seed")
        step_buffer_context = kwargs.get("step_buffer_context", "")
        meta_skill_context = kwargs.get("meta_skill_context", "")

        return run_minibatch_reflect(
            results=results,
            skill_content=skill_content,
            prediction_dir=prediction_dir,
            patches_dir=patches_dir,
            workers=self.analyst_workers,
            failure_only=self.failure_only,
            minibatch_size=self.minibatch_size,
            edit_budget=self.edit_budget,
            random_seed=random_seed,
            error_system=self.get_error_minibatch_prompt(),
            success_system=self.get_success_minibatch_prompt(),
            step_buffer_context=step_buffer_context,
            meta_skill_context=meta_skill_context,
            update_mode=getattr(self, "_cfg", {}).get("skill_update_mode", "patch"),
        )

    def deep_reflect(
        self,
        results: list[dict],
        skill_content: str,
        out_dir: str,
        **kwargs,
    ) -> list[dict | None]:
        if not self.use_deep_reflect:
            return []

        env_manager = kwargs.get("env_manager")
        if not isinstance(env_manager, list):
            return []

        prediction_dir = kwargs.get("prediction_dir", os.path.join(out_dir, "predictions"))
        random_seed = kwargs.get("random_seed")
        step_buffer_context = kwargs.get("step_buffer_context", "")
        meta_skill_context = kwargs.get("meta_skill_context", "")
        codex_backend = get_student_backend() == "codex_exec"
        selected_items = self.select_representative_items(
            results,
            env_manager,
            n_failures=self.deep_reflect_failures,
            n_successes=self.deep_reflect_successes,
            seed=random_seed,
        )
        if not selected_items:
            return []

        selected_ids = {str(item["id"]) for item in selected_items}
        selected_results = [row for row in results if str(row.get("id")) in selected_ids]
        selected_examples = (
            self.attach_codex_probe_context(selected_results, prediction_dir)
            if codex_backend
            else selected_results
        )
        selected_metadata = [
            {
                "id": str(item["id"]),
                "instruction_type": str(item.get("instruction_type") or ""),
                "answer_position": str(item.get("answer_position") or ""),
            }
            for item in selected_items
        ]

        deep_dir = os.path.join(out_dir, "deep_reflect")
        rollout_dir = os.path.join(deep_dir, "rollout")
        patches_dir = os.path.join(deep_dir, "patches")
        os.makedirs(deep_dir, exist_ok=True)
        print(
            f"    [2b/6 DEEP REFLECT setup] selected={len(selected_items)} "
            f"mode={self.mode}"
        )
        probe = generate_deep_probe_instruction(
            skill_content=skill_content,
            items=selected_examples,
            prediction_dir=prediction_dir,
            system_prompt=self.get_codex_deep_probe_prompt() if codex_backend else self.get_deep_probe_prompt(),
            step_buffer_context=step_buffer_context,
            meta_skill_context=meta_skill_context,
            output_requirements=[
                "- The instruction must ask for a short structured diagnostic readout before the student writes code or starts tool use.",
                "- The readout should focus on task family, source/target region, and decisive transformation rule.",
                "- The student must still complete the original spreadsheet task.",
                "- Keep the readout concise and avoid exhaustive cell enumeration.",
                "- The instruction text should be ready to append directly to the student's prompt.",
            ],
        )
        if not probe:
            return []
        diagnostic_trace_context_by_id = None
        if codex_backend:
            selected_items, diagnostic_trace_context_by_id, probe = self.resolve_codex_probe_target(
                selected_items=selected_items,
                selected_examples=selected_examples,
                prediction_dir=prediction_dir,
                probe=probe,
            )

        with open(os.path.join(deep_dir, "probe.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    **probe,
                    "selected_examples": selected_metadata,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        deep_results = self.rollout(
            selected_items,
            skill_content,
            rollout_dir,
            diagnostic_mode=True,
            diagnostic_instruction=probe["probe_instruction"],
            diagnostic_trace_context_by_id=diagnostic_trace_context_by_id,
        )
        return run_minibatch_reflect(
            results=deep_results,
            skill_content=skill_content,
            prediction_dir=os.path.join(rollout_dir, "predictions"),
            patches_dir=patches_dir,
            workers=self.analyst_workers,
            failure_only=self.failure_only,
            minibatch_size=self.minibatch_size,
            edit_budget=self.edit_budget,
            random_seed=random_seed,
            error_system=self.get_error_minibatch_prompt(),
            success_system=self.get_success_minibatch_prompt(),
            step_buffer_context=step_buffer_context,
            meta_skill_context=meta_skill_context,
            update_mode=getattr(self, "_cfg", {}).get("skill_update_mode", "patch"),
        )

    def get_task_types(self) -> list[str]:
        return list(TASK_TYPES)
