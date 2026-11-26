"""AppWorld environment adapter for SkillOpt (react_code protocol, see rollout.py)."""
from __future__ import annotations

import json
import os

from skillopt.datasets.base import BatchSpec
from skillopt.envs.appworld.dataloader import AppWorldDataLoader
from skillopt.envs.appworld.rollout import run_batch
from skillopt.envs.base import EnvAdapter


class AppWorldAdapter(EnvAdapter):
    """SkillOpt adapter for AppWorld.

    Parameters
    ----------
    appworld_root / appworld_venv : str
        AppWorld data root (``data/tasks`` etc.) and the venv whose ``bin/appworld`` serves it (``$APPWORLD_VENV`` if empty).
    workers : int
        Concurrent episodes, one AppWorld server each.
    port_base / port_span : int
        Servers bind free ports in [port_base, port_base + port_span). Give each concurrently
        running SkillOpt process its own range.
    max_steps : int
        Executed tool calls per episode (AppWorld official: 50).
    max_completion_tokens : int
        Generated-token cap per model call.
    episode_token_wall : int
        Generated-token cap per episode; 0 = off.
    fail_reason_style : str
        Failure text given to the analysts: ``checks`` (default) leads with AppWorld's failed
        checks; ``legacy`` reproduces the original text that opened with the stop reason.
    fill_app_descriptions : bool
        Fill ``{{ app_descriptions }}`` in the react_code prompt with AppWorld's app list, as
        AppWorld's own agent does (default). ``False`` leaves the placeholder as literal text,
        which reproduces runs made before this option existed.
    """

    def __init__(
        self,
        split_dir: str = "",
        data_path: str = "",
        split_mode: str = "split_dir",
        split_ratio: str = "2:1:7",
        split_seed: int = 42,
        split_output_dir: str = "",
        seed: int = 42,
        limit: int = 0,
        appworld_root: str = "~/.appworld-roots/skillopt-appworld",
        appworld_venv: str = "",
        workers: int = 16,
        port_base: int = 23100,
        port_span: int = 64,
        max_steps: int = 50,
        max_completion_tokens: int = 16384,
        episode_token_wall: int = 0,
        server_request_timeout: float = 300.0,
        exec_timeout: int = 1800,
        fail_reason_style: str = "checks",
        fill_app_descriptions: bool = True,
        analyst_workers: int = 16,
        failure_only: bool = False,
        minibatch_size: int = 8,
        edit_budget: int = 4,
    ) -> None:
        self.appworld_root = appworld_root
        self.appworld_venv = appworld_venv or os.environ.get("APPWORLD_VENV", "")
        self.workers = max(int(workers or 1), 1)
        self.port_base = int(port_base)
        self.port_span = int(port_span)
        self.max_steps = int(max_steps)
        self.max_completion_tokens = int(max_completion_tokens)
        self.episode_token_wall = int(episode_token_wall or 0)
        self.server_request_timeout = float(server_request_timeout)
        self.exec_timeout = int(exec_timeout) if exec_timeout else None
        if fail_reason_style not in ("checks", "legacy"):
            raise ValueError(f"fail_reason_style must be 'checks' or 'legacy', got {fail_reason_style!r}")
        self.fail_reason_style = fail_reason_style
        if isinstance(fill_app_descriptions, str):
            fill_app_descriptions = fill_app_descriptions.strip().lower() in ("true", "1", "yes")
        self.fill_app_descriptions = bool(fill_app_descriptions)
        self.analyst_workers = analyst_workers
        self.failure_only = failure_only
        self.minibatch_size = minibatch_size
        self.edit_budget = edit_budget
        self.dataloader = AppWorldDataLoader(
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
        items: list[dict] = list(env_manager)
        results_path = os.path.join(out_dir, "results.jsonl")
        # Resume only a finished batch of exactly these tasks.
        if os.path.exists(results_path):
            with open(results_path, encoding="utf-8") as f:
                existing = [json.loads(line) for line in f if line.strip()]
            if sorted(r["id"] for r in existing) == sorted(str(i["id"]) for i in items):
                return existing
        results = run_batch(
            items=items,
            skill_content=skill_content,
            out_root=out_dir,
            appworld_root=self.appworld_root,
            appworld_venv=self.appworld_venv,
            workers=self.workers,
            port_base=self.port_base,
            port_span=self.port_span,
            max_steps=self.max_steps,
            max_completion_tokens=self.max_completion_tokens,
            episode_token_wall=self.episode_token_wall,
            request_timeout=self.server_request_timeout,
            call_timeout=self.exec_timeout,
            fail_reason_style=self.fail_reason_style,
            fill_app_descriptions=self.fill_app_descriptions,
        )
        with open(results_path, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        return results

    def get_task_types(self) -> list[str]:
        return ["appworld"]
