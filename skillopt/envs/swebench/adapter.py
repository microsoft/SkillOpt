from __future__ import annotations

import os

from skillopt.datasets.base import BatchSpec
from skillopt.envs.base import EnvAdapter
from skillopt.envs.swebench.dataloader import SWEBenchDataLoader
from skillopt.envs.swebench.rollout import run_batch
from skillopt.gradient.reflect import run_minibatch_reflect


class SWEBenchAdapter(EnvAdapter):
    def __init__(
        self,
        split_dir: str = "",
        data_path: str = "",
        split_mode: str = "ratio",
        split_ratio: str = "2:1:7",
        split_seed: int = 42,
        split_output_dir: str = "",
        dataset_name: str = "lite",
        hf_split: str = "test",
        workers: int = 8,
        eval_workers: int = 8,
        analyst_workers: int = 16,
        failure_only: bool = False,
        minibatch_size: int = 4,
        edit_budget: int = 4,
        seed: int = 42,
        limit: int = 0,
        step_limit: int = 50,
        cost_limit: float = 3.0,
        timeout_per_instance: int = 600,
        student_model: str = "",
    ) -> None:
        self.dataset_name = dataset_name
        self.hf_split = hf_split
        self.workers = workers
        self.eval_workers = eval_workers
        self.analyst_workers = analyst_workers
        self.failure_only = failure_only
        self.minibatch_size = minibatch_size
        self.edit_budget = edit_budget
        self.step_limit = step_limit
        self.cost_limit = cost_limit
        self.timeout_per_instance = timeout_per_instance
        self.student_model = student_model
        self.dataloader = SWEBenchDataLoader(
            split_dir=split_dir,
            data_path=data_path,
            split_mode=split_mode,
            split_ratio=split_ratio,
            split_seed=split_seed,
            split_output_dir=split_output_dir,
            seed=seed,
            limit=limit,
            dataset_name=dataset_name,
            hf_split=hf_split,
        )

    def setup(self, cfg: dict) -> None:
        super().setup(cfg)
        self.student_model = str(self.student_model or cfg.get("student_model") or "gpt-5.4").strip()
        self.dataset_name = str(self.dataset_name or cfg.get("dataset_name") or "lite").strip()
        self.hf_split = str(self.hf_split or cfg.get("hf_split") or "test").strip()
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
            student_model=self.student_model,
            dataset_name=self.dataset_name,
            hf_split=self.hf_split,
            workers=self.workers,
            eval_workers=self.eval_workers,
            step_limit=self.step_limit,
            cost_limit=self.cost_limit,
            timeout_per_instance=self.timeout_per_instance,
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

    def get_task_types(self) -> list[str]:
        repos = {
            str(item.get("repo") or "").strip()
            for item in (
                self.dataloader.train_items
                + self.dataloader.val_items
                + self.dataloader.test_items
            )
            if str(item.get("repo") or "").strip()
        }
        return sorted(repos) or ["swebench"]
