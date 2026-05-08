from __future__ import annotations

import json
import os
import random
from collections import defaultdict

from skillopt.datasets.base import SplitDataLoader, _parse_split_ratio


_DATASET_ALIASES = {
    "lite": "princeton-nlp/SWE-Bench_Lite",
    "verified": "princeton-nlp/SWE-Bench_Verified",
    "full": "princeton-nlp/SWE-Bench",
}


def _normalize_dataset_name(name: str) -> str:
    key = str(name or "").strip()
    return _DATASET_ALIASES.get(key.lower(), key or _DATASET_ALIASES["lite"])


class SWEBenchDataLoader(SplitDataLoader):
    def __init__(
        self,
        split_dir: str = "",
        data_path: str = "",
        split_mode: str = "ratio",
        split_ratio: str = "2:1:7",
        split_seed: int = 42,
        split_output_dir: str = "",
        seed: int = 42,
        limit: int = 0,
        dataset_name: str = "lite",
        hf_split: str = "test",
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
        self.dataset_name = dataset_name
        self.hf_split = hf_split

    def setup(self, cfg: dict) -> None:
        self.dataset_name = str(
            self.dataset_name or cfg.get("dataset_name") or "lite"
        ).strip()
        self.hf_split = str(self.hf_split or cfg.get("hf_split") or "test").strip()
        super().setup(cfg)

    def load_raw_items(self, data_path: str) -> list[dict]:
        dataset_ref = str(data_path or "").strip()
        if dataset_ref and (os.path.exists(dataset_ref) or dataset_ref.endswith(".json") or dataset_ref.endswith(".jsonl")):
            return super().load_raw_items(dataset_ref)

        dataset_name = _normalize_dataset_name(dataset_ref or self.dataset_name)
        from datasets import load_dataset

        ds = load_dataset(dataset_name, split=self.hf_split)
        return [dict(item) for item in ds]

    def _materialize_ratio_split(self, cfg: dict) -> str:
        dataset_ref = os.path.abspath(str(self.data_path or "").strip()) if str(self.data_path or "").strip() and os.path.exists(str(self.data_path or "").strip()) else str(self.data_path or "").strip()
        if not dataset_ref:
            dataset_ref = _normalize_dataset_name(self.dataset_name)

        items = self.load_raw_items(dataset_ref)
        if not isinstance(items, list) or not items:
            raise ValueError(f"No SWE-bench items available from {dataset_ref!r}")

        ratio = _parse_split_ratio(self.split_ratio)
        parts = list(ratio)
        total_parts = sum(parts)
        rng = random.Random(self.split_seed)

        by_repo: dict[str, list[dict]] = defaultdict(list)
        for item in items:
            repo = str(item.get("repo") or "unknown").strip() or "unknown"
            by_repo[repo].append(dict(item))

        train_items: list[dict] = []
        val_items: list[dict] = []
        test_items: list[dict] = []

        for repo in sorted(by_repo):
            group = list(by_repo[repo])
            rng.shuffle(group)
            n = len(group)
            n_train = round(n * parts[0] / total_parts)
            n_val = round(n * parts[1] / total_parts)

            if n >= 3:
                n_train = max(1, n_train)
                n_val = max(1, n_val)
            elif n == 2:
                n_train, n_val = 1, 0
            else:
                n_train, n_val = 0, 0

            while n_train + n_val >= n and n >= 2:
                if n_val > 1:
                    n_val -= 1
                elif n_train > 1:
                    n_train -= 1
                else:
                    break

            train_items.extend(group[:n_train])
            val_items.extend(group[n_train:n_train + n_val])
            test_items.extend(group[n_train + n_val:])

        rng2 = random.Random(self.split_seed + 1)
        rng2.shuffle(train_items)
        rng2.shuffle(val_items)
        rng2.shuffle(test_items)

        split_dir = self._resolve_split_output_dir(cfg)
        os.makedirs(split_dir, exist_ok=True)
        self.write_split_items(os.path.join(split_dir, "train"), train_items)
        self.write_split_items(os.path.join(split_dir, "val"), val_items)
        self.write_split_items(os.path.join(split_dir, "test"), test_items)

        manifest = {
            "source_data_path": dataset_ref,
            "dataset_name": _normalize_dataset_name(self.dataset_name),
            "hf_split": self.hf_split,
            "split_mode": "ratio",
            "split_ratio": self.split_ratio,
            "split_seed": self.split_seed,
            "strategy": "stratified_by_repo",
            "counts": {
                "train": len(train_items),
                "val": len(val_items),
                "test": len(test_items),
            },
        }
        with open(os.path.join(split_dir, "split_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(
            f"  [SWEBenchDataLoader] generated repo-stratified split {self.split_ratio} "
            f"at {split_dir} from {dataset_ref}"
        )
        return split_dir

