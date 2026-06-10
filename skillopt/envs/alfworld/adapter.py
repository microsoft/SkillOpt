"""ALFWorld environment adapter for ReflACT.

Connects the ReflACT training loop to ALFWorld by implementing
:class:`~skillopt.envs.base.EnvAdapter`.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os

from skillopt.datasets.base import BatchSpec
from skillopt.envs.base import EnvAdapter
from skillopt.envs.alfworld.dataloader import ALFWorldDataLoader
from skillopt.envs.alfworld.rollout import (
    build_alfworld_env,
    run_alfworld_batch,
    TASKS,
)
from skillopt.utils import compute_score


@dataclass(frozen=True)
class ALFWorldBatchRun:
    """Lazy ALFWorld batch description.

    The adapter materializes this in rollout chunks so a large evaluation set
    does not keep every ALFWorld simulator open at once.
    """

    env_num: int
    eval_dataset: str
    seed: int
    is_train: bool
    workers: int
    specific_gamefiles: list[str] | None = None
    result_ids: list[str] | None = None
    items: list[dict] | None = None

    def __iter__(self):
        return iter(self.items or [])

    def __len__(self) -> int:
        return int(self.env_num or 0)


class ALFWorldAdapter(EnvAdapter):
    """ALFWorld environment adapter.

    Parameters
    ----------
    max_steps : int
        Maximum steps per ALFWorld episode (default 50).
    max_api_workers : int
        Maximum concurrent API calls during rollout (default 8).
    analyst_workers : int
        Parallel workers for analyst stage (default 16).
    failure_only : bool
        If True, only run error analyst (skip success analyst).
    minibatch_size : int
        Trajectories per analyst group, M (default 8).
    edit_budget : int
        Maximum edits per minibatch, L (default 4).
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
        train_size: int = 0,
        max_steps: int = 50,
        workers: int = 8,
        max_api_workers: int = 8,
        analyst_workers: int = 16,
        failure_only: bool = False,
        minibatch_size: int = 8,
        edit_budget: int = 4,
        max_completion_tokens: int = 16384,
    ) -> None:
        self.max_steps = max_steps
        self.workers = max(int(workers or 1), 1)
        self.max_api_workers = max_api_workers
        self.max_completion_tokens = int(max_completion_tokens)
        self.analyst_workers = analyst_workers
        self.failure_only = failure_only
        self.minibatch_size = minibatch_size
        self.edit_budget = edit_budget
        self.dataloader = ALFWorldDataLoader(
            split_dir=split_dir,
            data_path=data_path,
            split_mode=split_mode,
            split_ratio=split_ratio,
            split_seed=split_seed,
            split_output_dir=split_output_dir,
            seed=seed,
            limit=limit,
            train_size=train_size,
        )
        self._traj_cache: dict[str, dict | None] = {}

    def setup(self, cfg: dict) -> None:
        super().setup(cfg)
        self.dataloader.setup(cfg)

    def _load_traj_data(self, item: dict) -> dict | None:
        gamefile = str(item.get("gamefile") or "").strip()
        if not gamefile:
            return None
        if gamefile in self._traj_cache:
            return self._traj_cache[gamefile]

        traj_path = os.path.join(os.path.dirname(gamefile), "traj_data.json")
        try:
            with open(traj_path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = None
        self._traj_cache[gamefile] = data
        return data

    @staticmethod
    def _unique_lines(values: list[str], *, limit: int = 0) -> list[str]:
        lines: list[str] = []
        seen: set[str] = set()
        for raw in values:
            line = str(raw or "").strip()
            if not line or line in seen:
                continue
            seen.add(line)
            lines.append(line)
            if limit > 0 and len(lines) >= limit:
                break
        return lines

    @staticmethod
    def _format_high_pddl(high_pddl: list[dict]) -> list[str]:
        steps: list[str] = []
        for idx, step in enumerate(high_pddl or [], start=1):
            discrete = step.get("discrete_action") or {}
            action = str(discrete.get("action") or "").strip()
            args = [str(arg).strip() for arg in (discrete.get("args") or []) if str(arg).strip()]
            if action and args:
                text = f"{action}({', '.join(args)})"
            elif action:
                text = action
            else:
                planner_action = step.get("planner_action") or {}
                text = str(planner_action.get("action") or "").strip()
            if text:
                steps.append(f"{idx}. {text}")
        return steps

    def _build_reference_bundle(self, item: dict) -> dict:
        data = self._load_traj_data(item)
        if not data:
            return {}

        anns = ((data.get("turk_annotations") or {}).get("anns") or [])
        task_descs = self._unique_lines(
            [ann.get("task_desc", "") for ann in anns],
            limit=3,
        )
        high_descs = self._unique_lines(
            [step for ann in anns for step in (ann.get("high_descs") or [])],
            limit=12,
        )
        pddl_params = {
            key: value
            for key, value in (data.get("pddl_params") or {}).items()
            if value not in ("", None, [], {})
        }
        scene = data.get("scene") or {}
        scene_summary = {
            key: scene.get(key)
            for key in ("floor_plan", "scene_num", "dirty_and_empty")
            if scene.get(key) not in ("", None, [], {})
        }
        high_pddl = self._format_high_pddl((data.get("plan") or {}).get("high_pddl") or [])
        task_type = str(data.get("task_type") or item.get("task_type") or "").strip()
        return {
            "task_type": task_type,
            "task_descs": task_descs,
            "high_descs": high_descs,
            "pddl_params": pddl_params,
            "high_pddl": high_pddl,
            "scene_summary": scene_summary,
        }

    def build_reference_text(self, item: dict) -> str:
        bundle = self._build_reference_bundle(item)
        if not bundle:
            return ""

        parts: list[str] = []
        if bundle["task_type"]:
            parts.append(f"## Reference Task Type\n{bundle['task_type']}")
        if bundle["task_descs"]:
            parts.append(
                "## Reference Human Task Descriptions\n"
                + "\n".join(f"- {line}" for line in bundle["task_descs"])
            )
        if bundle["high_descs"]:
            parts.append(
                "## Reference Human High-Level Steps\n"
                + "\n".join(f"{idx}. {line}" for idx, line in enumerate(bundle["high_descs"], start=1))
            )
        if bundle["pddl_params"]:
            parts.append(
                "## Reference PDDL Params\n"
                + "\n".join(f"- {key}: {value}" for key, value in bundle["pddl_params"].items())
            )
        if bundle["high_pddl"]:
            parts.append(
                "## Reference Planner High-Level Plan\n" + "\n".join(bundle["high_pddl"])
            )
        if bundle["scene_summary"]:
            parts.append(
                "## Reference Scene Summary\n"
                + "\n".join(f"- {key}: {value}" for key, value in bundle["scene_summary"].items())
            )
        return "\n\n".join(parts)

    def get_reference_metadata(self, item: dict) -> dict:
        bundle = self._build_reference_bundle(item)
        if not bundle:
            return {"fields": [], "preview": ""}

        fields: list[str] = []
        previews: list[str] = []
        if bundle["task_type"]:
            fields.append("task_type")
            previews.append(f"[task_type] {bundle['task_type']}")
        if bundle["task_descs"]:
            fields.append("task_desc")
            previews.append("[task_desc]\n" + "\n".join(bundle["task_descs"][:2]))
        if bundle["high_descs"]:
            fields.append("high_descs")
            previews.append("[high_descs]\n" + "\n".join(bundle["high_descs"][:3]))
        if bundle["pddl_params"]:
            fields.append("pddl_params")
            previews.append(
                "[pddl_params]\n"
                + "\n".join(
                    f"{key}: {value}" for key, value in list(bundle["pddl_params"].items())[:4]
                )
            )
        if bundle["high_pddl"]:
            fields.append("plan.high_pddl")
            previews.append("[plan.high_pddl]\n" + "\n".join(bundle["high_pddl"][:3]))
        if bundle["scene_summary"]:
            fields.append("scene")
            previews.append(
                "[scene]\n"
                + "\n".join(
                    f"{key}: {value}" for key, value in bundle["scene_summary"].items()
                )
            )
        return {
            "fields": fields,
            "preview": "\n\n".join(previews)[:600],
        }

    @staticmethod
    def _infer_dataset_from_gamefile(gamefile: str) -> tuple[str, bool]:
        path = str(gamefile or "")
        if "/valid_seen/" in path:
            return "eval_in_distribution", False
        if "/valid_unseen/" in path:
            return "eval_out_of_distribution", False
        return "train", True

    def get_dataloader(self):
        return self.dataloader

    def _comparison_items(self, items: list[dict]) -> list[dict]:
        enriched: list[dict] = []
        for item in items:
            row = dict(item)
            bundle = self._build_reference_bundle(row)
            if bundle.get("task_descs"):
                row["task_description"] = bundle["task_descs"][0]
            elif bundle.get("task_type"):
                row["task_description"] = bundle["task_type"]
            enriched.append(row)
        return enriched

    def requires_ray(self) -> bool:
        return False

    def build_env_from_batch(self, batch: BatchSpec, **kwargs):
        gamefiles = list(batch.metadata.get("gamefiles") or [])
        result_ids = list(batch.metadata.get("result_ids") or [])
        items = self._comparison_items(list(batch.payload or []))
        return ALFWorldBatchRun(
            env_num=batch.batch_size,
            eval_dataset=batch.metadata.get("eval_dataset", batch.split),
            seed=batch.seed,
            is_train=batch.metadata.get("is_train", batch.phase == "train"),
            specific_gamefiles=gamefiles or None,
            result_ids=result_ids or None,
            items=items,
            workers=self.workers,
        )

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

        if isinstance(env_manager, ALFWorldBatchRun):
            results = self._run_batch(
                env_manager,
                skill_content=skill_content,
                out_dir=out_dir,
            )
        else:
            results = run_alfworld_batch(
                env_manager=env_manager,
                skill_content=skill_content,
                max_steps=self.max_steps,
                out_root=out_dir,
                max_api_workers=self.max_api_workers,
                max_completion_tokens=self.max_completion_tokens,
                result_ids=getattr(env_manager, "_skillopt_result_ids", None),
            )

        with open(results_path, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        return results

    @staticmethod
    def _close_env(env_manager) -> None:
        close = getattr(env_manager, "close", None)
        if callable(close):
            close()

    def _run_batch(
        self,
        batch: ALFWorldBatchRun,
        skill_content: str,
        out_dir: str,
        *,
        diagnostic_mode: bool = False,
        diagnostic_instruction: str = "",
    ) -> list[dict]:
        total = int(batch.env_num or 0)
        if total <= 0:
            return []

        workers = max(1, min(int(batch.workers or self.workers), total))
        if total > workers:
            print(
                f"    [alfworld rollout] episodes={total} "
                f"env_workers={workers} chunks={(total + workers - 1) // workers}"
            )

        all_results: list[dict] = []
        for start in range(0, total, workers):
            chunk_size = min(workers, total - start)
            chunk_gamefiles = (
                batch.specific_gamefiles[start:start + chunk_size]
                if batch.specific_gamefiles
                else None
            )
            chunk_ids = (
                batch.result_ids[start:start + chunk_size]
                if batch.result_ids
                else [f"env_{idx:03d}" for idx in range(start, start + chunk_size)]
            )
            chunk_env = build_alfworld_env(
                env_num=chunk_size,
                eval_dataset=batch.eval_dataset,
                seed=batch.seed + start,
                is_train=batch.is_train,
                specific_gamefiles=chunk_gamefiles,
            )
            try:
                chunk_results = run_alfworld_batch(
                    env_manager=chunk_env,
                    skill_content=skill_content,
                    max_steps=self.max_steps,
                    out_root=out_dir,
                    max_api_workers=min(self.max_api_workers, chunk_size),
                    max_completion_tokens=self.max_completion_tokens,
                    diagnostic_mode=diagnostic_mode,
                    diagnostic_instruction=diagnostic_instruction,
                    result_ids=chunk_ids,
                )
            finally:
                self._close_env(chunk_env)
            all_results.extend(chunk_results)
        return all_results

    def get_task_types(self) -> list[str]:
        return list(TASKS)
