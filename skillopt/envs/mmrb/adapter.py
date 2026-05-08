"""MMRB environment adapter for ReflACT."""
from __future__ import annotations

import json
import os

from skillopt.gradient.deep_probe import generate_deep_probe_instruction
from skillopt.datasets.base import BatchSpec
from skillopt.gradient.reflect import run_minibatch_reflect
from skillopt.envs.base import EnvAdapter
from skillopt.envs.mmrb.dataloader import MMRBDataLoader
from skillopt.envs.mmrb.rollout import run_batch
from skillopt.model import get_student_backend


class MMRBAdapter(EnvAdapter):
    """MMRB adapter."""

    def build_reference_text(self, item: dict) -> str:
        reasoning_steps = item.get("reasoning_steps") or []
        if not reasoning_steps:
            return ""

        blocks: list[str] = []
        for path_idx, path in enumerate(reasoning_steps, 1):
            if not isinstance(path, list) or not path:
                continue
            lines = [f"### Reasoning Path {path_idx}"]
            for step in path:
                if not isinstance(step, dict):
                    continue
                step_no = step.get("reasoning step", "?")
                step_type = str(step.get("reasoning type") or "").strip()
                rationale = str(step.get("rationale") or "").strip()
                if rationale:
                    prefix = f"{step_no}. [{step_type}] " if step_type else f"{step_no}. "
                    lines.append(prefix + rationale)
            if len(lines) > 1:
                blocks.append("\n".join(lines))
        if not blocks:
            return ""
        return "## Reference Reasoning Steps\n" + "\n\n".join(blocks[:3])

    def get_reference_metadata(self, item: dict) -> dict:
        reasoning_steps = item.get("reasoning_steps") or []
        path_count = 0
        preview_parts: list[str] = []
        for path in reasoning_steps:
            if not isinstance(path, list) or not path:
                continue
            path_count += 1
            first = path[0] if isinstance(path[0], dict) else {}
            step_type = str(first.get("reasoning type") or "").strip()
            rationale = str(first.get("rationale") or "").strip()
            preview_parts.append(f"[path {path_count}] {step_type}: {rationale[:180]}")
        if not path_count:
            return {"fields": [], "preview": ""}
        return {
            "fields": ["reasoning_steps"],
            "preview": "\n".join(preview_parts)[:500],
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
        workers: int = 16,
        analyst_workers: int = 16,
        failure_only: bool = False,
        minibatch_size: int = 8,
        edit_budget: int = 4,
        seed: int = 42,
        limit: int = 0,
        image_detail: str = "auto",
        use_deep_reflect: bool = False,
        deep_reflect_failures: int = 4,
        deep_reflect_successes: int = 2,
    ) -> None:
        self.max_turns = max_turns
        self.workers = workers
        self.analyst_workers = analyst_workers
        self.failure_only = failure_only
        self.minibatch_size = minibatch_size
        self.edit_budget = edit_budget
        self.image_detail = image_detail
        self.use_deep_reflect = use_deep_reflect
        self.deep_reflect_failures = deep_reflect_failures
        self.deep_reflect_successes = deep_reflect_successes
        self.dataloader = MMRBDataLoader(
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
            workers=self.workers,
            image_detail=self.image_detail,
            diagnostic_mode=kwargs.get("diagnostic_mode", False),
            diagnostic_instruction=kwargs.get("diagnostic_instruction", ""),
            diagnostic_trace_context_by_id=kwargs.get("diagnostic_trace_context_by_id"),
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
        prediction_dir = kwargs.get("prediction_dir", os.path.join(out_dir, "predictions"))
        random_seed = kwargs.get("random_seed")
        step_buffer_context = kwargs.get("step_buffer_context", "")
        meta_skill_context = kwargs.get("meta_skill_context", "")
        codex_backend = get_student_backend() == "codex_exec"
        selected_items = self.select_representative_items(
            results,
            env_manager if isinstance(env_manager, list) else None,
            n_failures=self.deep_reflect_failures,
            n_successes=self.deep_reflect_successes,
            seed=random_seed,
        )
        if not selected_items:
            return []
        selected_ids = {str(item["id"]) for item in selected_items}
        selected_results = [row for row in results if str(row.get("id")) in selected_ids]
        selected_examples = self.attach_reference_context(selected_results, selected_items)
        if codex_backend:
            selected_examples = self.attach_codex_probe_context(selected_examples, prediction_dir)

        reasoning_count = 0
        selected_metadata = []
        for item in selected_items:
            meta = self.get_reference_metadata(item)
            if meta["fields"]:
                reasoning_count += 1
            selected_metadata.append({
                "id": str(item["id"]),
                "task_type": str(item.get("subtask") or item.get("task_type") or "mmrb"),
                "reference_fields": meta["fields"],
                "reference_preview": meta["preview"],
            })

        deep_dir = os.path.join(out_dir, "deep_reflect")
        rollout_dir = os.path.join(deep_dir, "rollout")
        patches_dir = os.path.join(deep_dir, "patches")
        os.makedirs(deep_dir, exist_ok=True)
        print(
            f"    [2b/6 DEEP REFLECT setup] selected={len(selected_items)} "
            f"reference_fields=reasoning_steps({reasoning_count}/{len(selected_items)})"
        )
        probe = generate_deep_probe_instruction(
            skill_content=skill_content,
            items=selected_examples,
            prediction_dir=prediction_dir,
            system_prompt=self.get_codex_deep_probe_prompt() if codex_backend else self.get_deep_probe_prompt(),
            step_buffer_context=step_buffer_context,
            meta_skill_context=meta_skill_context,
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
        probe_record = {
            **probe,
            "reference_summary": {
                "selected_count": len(selected_items),
                "field_counts": {"reasoning_steps": reasoning_count},
            },
            "selected_examples": selected_metadata,
        }
        with open(os.path.join(deep_dir, "probe.json"), "w", encoding="utf-8") as f:
            json.dump(probe_record, f, ensure_ascii=False, indent=2)
        deep_results = run_batch(
            items=selected_items,
            out_root=rollout_dir,
            skill_content=skill_content,
            max_turns=self.max_turns,
            workers=min(self.workers, max(len(selected_items), 1)),
            image_detail=self.image_detail,
            diagnostic_mode=True,
            diagnostic_instruction=probe["probe_instruction"],
            diagnostic_trace_context_by_id=diagnostic_trace_context_by_id,
        )
        deep_results = self.attach_reference_context(deep_results, selected_items)
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
        return self.dataloader.get_task_types()
