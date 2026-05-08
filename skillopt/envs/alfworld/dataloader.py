"""ALFWorld task dataloader."""
from __future__ import annotations

from skillopt.datasets.base import BatchSpec, SplitDataLoader


class ALFWorldDataLoader(SplitDataLoader):
    """ALFWorld batch planner.

    In split_dir mode, batches are fixed gamefile items so ablations differ
    only in how the same training set is batched.
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
        **kwargs,
    ) -> None:
        super().__init__(
            split_dir=split_dir,
            data_path=data_path,
            split_mode=split_mode,
            split_ratio=split_ratio,
            split_seed=split_seed,
            split_output_dir=split_output_dir,
            seed=seed,
            limit=limit,
        )
        self.train_size_override = int(train_size or 0)

    @staticmethod
    def _metadata_for_items(items: list[dict], split: str, phase: str) -> dict:
        gamefiles = [str(item.get("gamefile") or "") for item in items]
        if any(not gamefile for gamefile in gamefiles):
            raise ValueError("ALFWorld split items must contain non-empty gamefile paths.")
        eval_dataset = "train"
        is_train = phase == "train"
        first = gamefiles[0] if gamefiles else ""
        if "/valid_seen/" in first:
            eval_dataset = "eval_in_distribution"
            is_train = False
        elif "/valid_unseen/" in first:
            eval_dataset = "eval_out_of_distribution"
            is_train = False
        return {
            "eval_dataset": eval_dataset,
            "is_train": is_train,
            "gamefiles": gamefiles,
            "result_ids": [str(item.get("id") or idx) for idx, item in enumerate(items)],
        }

    def get_train_size(self) -> int:
        if self.train_size_override > 0:
            return self.train_size_override
        return super().get_train_size()

    def build_train_batch(self, batch_size: int, seed: int, **kwargs) -> BatchSpec:
        batch = super().build_train_batch(batch_size=batch_size, seed=seed, **kwargs)
        items = list(batch.payload or [])
        batch.metadata.update(self._metadata_for_items(items, "train", "train"))
        return BatchSpec(
            phase="train",
            split="train",
            seed=seed,
            batch_size=len(items),
            payload=items,
            metadata=batch.metadata,
        )

    def plan_train_epoch(
        self,
        *,
        epoch: int,
        steps_per_epoch: int,
        accumulation: int,
        batch_size: int,
        seed: int,
        **kwargs,
    ) -> list[BatchSpec]:
        batches = super().plan_train_epoch(
            epoch=epoch,
            steps_per_epoch=steps_per_epoch,
            accumulation=accumulation,
            batch_size=batch_size,
            seed=seed,
            **kwargs,
        )
        for batch in batches:
            items = list(batch.payload or [])
            batch.metadata.update(self._metadata_for_items(items, "train", "train"))
        return batches

    def build_eval_batch(
        self,
        env_num: int,
        split: str,
        seed: int,
        **kwargs,
    ) -> BatchSpec:
        batch = super().build_eval_batch(
            env_num=env_num,
            split=split,
            seed=seed,
            **kwargs,
        )
        items = list(batch.payload or [])
        batch.metadata.update(self._metadata_for_items(items, split, "eval"))
        return BatchSpec(
            phase="eval",
            split=split,
            seed=seed,
            batch_size=len(items),
            payload=items,
            metadata=batch.metadata,
        )
