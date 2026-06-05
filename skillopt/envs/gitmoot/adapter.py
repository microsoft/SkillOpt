"""Gitmoot environment adapter."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from skillopt.datasets.base import BatchSpec
from skillopt.envs.base import EnvAdapter
from skillopt.envs.gitmoot.dataloader import GitmootDataLoader
from skillopt.envs.gitmoot.package import TemplateDocument, split_template_document
from skillopt.envs.gitmoot.rollout import run_batch
from skillopt.gradient.reflect import run_minibatch_reflect
from skillopt.optimizer.update_modes import (
    get_payload_items,
    is_full_rewrite_minibatch_mode,
)


class GitmootAdapter(EnvAdapter):
    """Adapter that trains Gitmoot agent-template Markdown from review packages."""

    def __init__(
        self,
        training_package: str = "",
        artifact_root: str = "",
        seed: int = 42,
        limit: int = 0,
        analyst_workers: int = 4,
        minibatch_size: int = 4,
        edit_budget: int = 4,
        failure_only: bool = False,
        max_completion_tokens: int = 4096,
        **kwargs,
    ) -> None:
        del kwargs
        self.analyst_workers = int(analyst_workers)
        self.minibatch_size = int(minibatch_size)
        self.edit_budget = int(edit_budget)
        self.failure_only = bool(failure_only)
        self.max_completion_tokens = int(max_completion_tokens)
        self.dataloader = GitmootDataLoader(
            training_package=training_package,
            artifact_root=artifact_root,
            seed=seed,
            limit=limit,
        )

    def setup(self, cfg: dict) -> None:
        super().setup(cfg)
        self.dataloader.setup(cfg)
        _ensure_package_skill_init(cfg, self.dataloader.initial_skill_content)

    def get_dataloader(self) -> GitmootDataLoader:
        return self.dataloader

    def build_env_from_batch(self, batch: BatchSpec, **kwargs):
        del kwargs
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
    ) -> list[dict[str, Any]]:
        del kwargs
        items: list[dict[str, Any]] = list(env_manager or [])
        cfg = getattr(self, "_cfg", {})
        return run_batch(
            items=items,
            skill_content=skill_content,
            out_root=out_dir,
            max_completion_tokens=self.max_completion_tokens,
            target_artifact_retry_budget=max(0, int(cfg.get("target_artifact_retry_budget", 1) or 0)),
        )

    def reflect(
        self,
        results: list[dict],
        skill_content: str,
        out_dir: str,
        **kwargs,
    ) -> list[dict | None]:
        template = split_template_document(skill_content)
        update_mode = getattr(self, "_cfg", {}).get("skill_update_mode", "patch")
        patches = run_minibatch_reflect(
            results=results,
            skill_content=template.body,
            prediction_dir=kwargs.get("prediction_dir", os.path.join(out_dir, "predictions")),
            patches_dir=kwargs.get("patches_dir", os.path.join(out_dir, "patches")),
            workers=self.analyst_workers,
            failure_only=self.failure_only,
            minibatch_size=self.minibatch_size,
            edit_budget=self.edit_budget,
            random_seed=kwargs.get("random_seed"),
            step_buffer_context=kwargs.get("step_buffer_context", ""),
            meta_skill_context=kwargs.get("meta_skill_context", ""),
            update_mode=update_mode,
            error_system=self.get_error_minibatch_prompt(),
            success_system=self.get_success_minibatch_prompt(),
        )
        if is_full_rewrite_minibatch_mode(update_mode):
            _recompose_full_rewrite_candidates(patches, template)
        return patches

    def get_task_types(self) -> list[str]:
        return ["gitmoot-skillopt"]


def _recompose_full_rewrite_candidates(
    patches: list[dict | None],
    template: TemplateDocument,
) -> None:
    if not template.frontmatter.strip():
        return
    for patch in patches:
        if not isinstance(patch, dict):
            continue
        for candidate in get_payload_items(patch.get("patch", {}), "full_rewrite_minibatch"):
            if not isinstance(candidate, dict):
                continue
            new_skill = str(candidate.get("new_skill", "")).strip()
            if not new_skill:
                continue
            candidate["new_skill"] = template.compose(split_template_document(new_skill).body)


def _ensure_package_skill_init(cfg: dict, content: str) -> None:
    if str(cfg.get("skill_init") or "").strip():
        return
    if not content.strip():
        return
    out_root = Path(str(cfg.get("out_root") or "outputs/gitmoot")).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)
    skill_path = out_root / "gitmoot_initial_skill.md"
    skill_path.write_text(content, encoding="utf-8")
    cfg["skill_init"] = str(skill_path)
