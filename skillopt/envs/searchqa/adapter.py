"""SearchQA environment adapter for ReflACT."""
from __future__ import annotations

import json
import os

from skillopt.gradient.deep_probe import generate_deep_probe_instruction
from skillopt.datasets.base import BatchSpec
from skillopt.envs.base import EnvAdapter
from skillopt.envs.searchqa.dataloader import SearchQADataLoader
from skillopt.envs.searchqa.rollout import run_batch
from skillopt.gradient.reflect import run_minibatch_reflect
from skillopt.model import get_student_backend


class SearchQAAdapter(EnvAdapter):
    """SearchQA environment adapter."""

    def __init__(
        self,
        split_dir: str = "",
        data_path: str = "",
        split_mode: str = "ratio",
        split_ratio: str = "2:1:7",
        split_seed: int = 42,
        split_output_dir: str = "",
        max_turns: int = 1,
        exec_timeout: int = 120,
        workers: int = 64,
        analyst_workers: int = 16,
        failure_only: bool = False,
        minibatch_size: int = 8,
        edit_budget: int = 4,
        seed: int = 42,
        limit: int = 0,
        exec_timeout: int = 600,
        use_deep_reflect: bool = False,
        deep_reflect_failures: int = 4,
        deep_reflect_successes: int = 2,
    ) -> None:
        self.max_turns = max_turns
        self.exec_timeout = exec_timeout
        self.workers = workers
        self.analyst_workers = analyst_workers
        self.failure_only = failure_only
        self.minibatch_size = minibatch_size
        self.edit_budget = edit_budget
        self.exec_timeout = exec_timeout
        self.use_deep_reflect = use_deep_reflect
        self.deep_reflect_failures = deep_reflect_failures
        self.deep_reflect_successes = deep_reflect_successes
        self.dataloader = SearchQADataLoader(
            split_dir=split_dir,
            data_path=data_path,
            split_mode=split_mode,
            split_ratio=split_ratio,
            split_seed=split_seed,
            split_output_dir=split_output_dir,
            seed=seed,
            limit=limit,
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
        env_manager,  # actually list[dict] for SearchQA
        skill_content: str,
        out_dir: str,
        **kwargs,
    ) -> list[dict]:
        """Run QA agent on items. Resume-aware."""
        items: list[dict] = env_manager  # type alias for clarity
        return run_batch(
            items=items,
            out_root=out_dir,
            skill_content=skill_content,
            max_turns=self.max_turns,
            exec_timeout=self.exec_timeout,
            workers=self.workers,
            diagnostic_mode=kwargs.get("diagnostic_mode", False),
            diagnostic_instruction=kwargs.get("diagnostic_instruction", ""),
            diagnostic_trace_context_by_id=kwargs.get("diagnostic_trace_context_by_id"),
            task_timeout=self.exec_timeout,
        )

    def reflect(
        self,
        results: list[dict],
        skill_content: str,
        out_dir: str,
        **kwargs,
    ) -> list[dict | None]:
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
                "question_preview": str(item.get("question") or "")[:200],
                "has_context": bool(str(item.get("context") or "").strip()),
                "n_gold_answers": len(item.get("answers") or []),
            }
            for item in selected_items
        ]

        deep_dir = os.path.join(out_dir, "deep_reflect")
        rollout_dir = os.path.join(deep_dir, "rollout")
        patches_dir = os.path.join(deep_dir, "patches")
        os.makedirs(deep_dir, exist_ok=True)
        print(
            f"    [2b/6 DEEP REFLECT setup] selected={len(selected_items)} "
            f"mode=no_reference_probe"
        )
        probe = generate_deep_probe_instruction(
            skill_content=skill_content,
            items=selected_examples,
            prediction_dir=prediction_dir,
            system_prompt=self.get_codex_deep_probe_prompt() if codex_backend else self.get_deep_probe_prompt(),
            step_buffer_context=step_buffer_context,
            meta_skill_context=meta_skill_context,
            output_requirements=[
                "- There is no hidden reference block. Use only the question, provided context, the student's output, and the evaluation result to infer what intermediate state is worth probing.",
                "- The instruction must explicitly request a short <analysis>...</analysis> block before the final <answer>...</answer>.",
                "- The readout should focus on likely evidence span, top candidate and runner-up, decisive clue, or a few short intermediate conclusions.",
                "- Do not ask for exhaustive copying of the context or a full chain-of-thought.",
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
        return ["qa"]
