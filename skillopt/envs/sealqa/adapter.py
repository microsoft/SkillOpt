from __future__ import annotations

import os

from skillopt.datasets.base import BatchSpec
from skillopt.envs.base import EnvAdapter
from skillopt.envs.deep_reflect import run_no_reference_deep_reflect
from skillopt.envs.sealqa.dataloader import SealQADataLoader
from skillopt.envs.sealqa.rollout import run_batch
from skillopt.gradient.reflect import run_minibatch_reflect


class SealQAAdapter(EnvAdapter):
    def __init__(
        self,
        split_dir: str = '',
        workers: int = 4,
        analyst_workers: int = 8,
        failure_only: bool = False,
        minibatch_size: int = 8,
        edit_budget: int = 4,
        seed: int = 42,
        limit: int = 0,
        max_tool_turns: int = 12,
        use_deep_reflect: bool = False,
        deep_reflect_failures: int = 4,
        deep_reflect_successes: int = 2,
    ) -> None:
        self.workers = workers
        self.analyst_workers = analyst_workers
        self.failure_only = failure_only
        self.minibatch_size = minibatch_size
        self.edit_budget = edit_budget
        self.max_tool_turns = max_tool_turns
        self.use_deep_reflect = use_deep_reflect
        self.deep_reflect_failures = deep_reflect_failures
        self.deep_reflect_successes = deep_reflect_successes
        self.dataloader = SealQADataLoader(split_dir=split_dir, seed=seed, limit=limit)

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
            workers=self.workers,
            max_tool_turns=self.max_tool_turns,
            diagnostic_mode=kwargs.get('diagnostic_mode', False),
            diagnostic_instruction=kwargs.get('diagnostic_instruction', ''),
        )

    def reflect(self, results: list[dict], skill_content: str, out_dir: str, **kwargs) -> list[dict | None]:
        prediction_dir = kwargs.get('prediction_dir', os.path.join(out_dir, 'predictions'))
        patches_dir = kwargs.get('patches_dir', os.path.join(out_dir, 'patches'))
        random_seed = kwargs.get('random_seed')
        step_buffer_context = kwargs.get('step_buffer_context', '')
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
            env_manager=kwargs.get('env_manager'),
            prediction_dir=kwargs.get('prediction_dir'),
            random_seed=kwargs.get('random_seed'),
            step_buffer_context=kwargs.get('step_buffer_context', ''),
            output_requirements=[
                "- There is no hidden reference block. Use only the question, provided evidence, URL/fetch trace, student output, and evaluation result to infer what intermediate state is worth probing.",
                "- The instruction must explicitly request a short <analysis>...</analysis> block before the final <answer>...</answer>.",
                "- The readout should focus on effective time frame, conflicting evidence, decisive source, candidate answer, and answer-finalization rule.",
                "- Do not ask for exhaustive web summaries or a full chain-of-thought.",
                "- The instruction text should be ready to append directly to the student's prompt.",
            ],
            metadata_builder=lambda item: {
                "id": str(item.get('id')),
                "task_type": str(item.get('task_type') or item.get('topic') or 'sealqa'),
                "question_preview": str(item.get('question') or '')[:200],
                "freshness": item.get('freshness', ''),
                "question_types": item.get('question_types', ''),
                "topic": item.get('topic', ''),
            },
        )

    def get_task_types(self) -> list[str]:
        seen: list[str] = []
        for item in self.dataloader.train_items + self.dataloader.val_items + self.dataloader.test_items:
            task_type = str(item.get('task_type') or 'sealqa')
            if task_type not in seen:
                seen.append(task_type)
        return seen or ['sealqa']
