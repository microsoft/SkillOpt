from __future__ import annotations

import os

from skillopt.datasets.base import BatchSpec
from skillopt.envs.base import EnvAdapter
from skillopt.envs.deep_reflect import run_no_reference_deep_reflect
from skillopt.envs.docvqa.dataloader import DocVQADataLoader
from skillopt.envs.docvqa.rollout import run_batch
from skillopt.gradient.reflect import run_minibatch_reflect


class DocVQAAdapter(EnvAdapter):
    def __init__(
        self,
        split_dir: str = "",
        data_path: str = "",
        split_mode: str = "split_dir",
        split_ratio: str = "2:1:7",
        split_seed: int = 42,
        split_output_dir: str = "",
        max_turns: int = 1,
        exec_timeout: int = 120,
        workers: int = 16,
        analyst_workers: int = 16,
        failure_only: bool = False,
        minibatch_size: int = 8,
        edit_budget: int = 4,
        seed: int = 42,
        limit: int = 0,
        exec_timeout: int = 600,
        image_detail: str = "auto",
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
        self.image_detail = image_detail
        self.use_deep_reflect = use_deep_reflect
        self.deep_reflect_failures = deep_reflect_failures
        self.deep_reflect_successes = deep_reflect_successes
        self.dataloader = DocVQADataLoader(
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

    def rollout(self, env_manager, skill_content: str, out_dir: str, **kwargs) -> list[dict]:
        items: list[dict] = env_manager
        return run_batch(
            items=items,
            out_root=out_dir,
            skill_content=skill_content,
            max_turns=self.max_turns,
            exec_timeout=self.exec_timeout,
            workers=self.workers,
            image_detail=self.image_detail,
            diagnostic_mode=kwargs.get("diagnostic_mode", False),
            diagnostic_instruction=kwargs.get("diagnostic_instruction", ""),
            task_timeout=self.exec_timeout,
        )

    def reflect(self, results: list[dict], skill_content: str, out_dir: str, **kwargs) -> list[dict | None]:
        prediction_dir = kwargs.get("prediction_dir", os.path.join(out_dir, "predictions"))
        patches_dir = kwargs.get("patches_dir", os.path.join(out_dir, "patches"))
        random_seed = kwargs.get("random_seed")
        step_buffer_context = kwargs.get("step_buffer_context", "")
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
            update_mode=getattr(self, "_cfg", {}).get("skill_update_mode", "patch"),
        )

    def deep_reflect(
        self,
        results: list[dict],
        skill_content: str,
        out_dir: str,
        **kwargs,
    ) -> list[dict | None]:
        return run_no_reference_deep_reflect(
            self,
            results,
            skill_content,
            out_dir,
            env_manager=kwargs.get("env_manager"),
            prediction_dir=kwargs.get("prediction_dir"),
            random_seed=kwargs.get("random_seed"),
            step_buffer_context=kwargs.get("step_buffer_context", ""),
            output_requirements=[
                "- There is no hidden reference block. Use only the document image prompt, student output, and evaluation result to infer what intermediate state is worth probing.",
                "- The instruction must explicitly request a short <analysis>...</analysis> block before the final <answer>...</answer>.",
                "- The readout should focus on visual region, field/table/figure label, OCR text read, candidate answer, and answer-format normalization.",
                "- Do not ask for exhaustive transcription or a full chain-of-thought.",
                "- The instruction text should be ready to append directly to the student's prompt.",
            ],
            metadata_builder=lambda item: {
                "id": str(item.get("id")),
                "task_type": str(item.get("task_type") or "docvqa"),
                "question_preview": str(item.get("question") or "")[:200],
                "image_path": item.get("image_path", ""),
                "docId": item.get("docId", ""),
                "page": item.get("ucsf_document_page_no", ""),
            },
        )

    def get_task_types(self) -> list[str]:
        seen: list[str] = []
        for item in self.dataloader.train_items + self.dataloader.val_items + self.dataloader.test_items:
            task_type = str(item.get("task_type") or "docvqa")
            if task_type not in seen:
                seen.append(task_type)
        return seen or ["docvqa"]
