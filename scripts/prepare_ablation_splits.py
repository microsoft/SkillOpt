#!/usr/bin/env python3
"""Prepare fixed data splits for ablation experiments."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "searchqa": {
        "raw": PROJECT_ROOT / "data/searchqa_train_2000.json",
        "out": PROJECT_ROOT / "data/ablation_splits/searchqa",
        "filenames": {"train": "train.json", "val": "selection.json", "test": "test.json"},
    },
    "spreadsheetbench": {
        "raw": PROJECT_ROOT / "data/spreadsheetbench_verified_400/dataset.json",
        "out": PROJECT_ROOT / "data/ablation_splits/spreadsheetbench",
        "filenames": {"train": "train.json", "val": "sel.json", "test": "test.json"},
    },
}

SPLITS = ("1shot", "1:1:8", "2:1:7", "4:1:5")


def load_items(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"Expected JSON array in {path}, got {type(data).__name__}")
    return data


def split_counts(total: int, split: str) -> tuple[int, int, int]:
    if split == "1shot":
        if total < 3:
            raise ValueError(f"Need at least 3 items for 1shot split, got {total}")
        return 1, 1, total - 2

    ratio = split
    weights = [int(part) for part in ratio.split(":")]
    if len(weights) != 3 or min(weights) <= 0:
        raise ValueError(f"Invalid ratio: {ratio}")
    denom = sum(weights)
    raw = [total * weight / denom for weight in weights]
    counts = [int(value) for value in raw]
    remaining = total - sum(counts)
    order = sorted(
        range(3),
        key=lambda idx: (raw[idx] - counts[idx], weights[idx]),
        reverse=True,
    )
    for idx in order[:remaining]:
        counts[idx] += 1
    return counts[0], counts[1], counts[2]


def split_tag(split: str) -> str:
    return "1shot" if split == "1shot" else split.replace(":", "-")


def write_json(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def prepare_dataset(name: str, *, seed: int, force: bool) -> None:
    spec = DATASETS[name]
    raw_path = spec["raw"]
    out_root = spec["out"]
    filenames = spec["filenames"]

    items = load_items(raw_path)
    for split in SPLITS:
        ratio_tag = split_tag(split)
        split_dir = out_root / f"{ratio_tag}_seed{seed}"
        manifest_path = split_dir / "split_manifest.json"
        if manifest_path.exists() and not force:
            print(f"skip {name} {split}: {split_dir} exists")
            continue

        shuffled = list(items)
        random.Random(seed).shuffle(shuffled)
        train_n, val_n, test_n = split_counts(len(shuffled), split)
        train_items = shuffled[:train_n]
        val_items = shuffled[train_n: train_n + val_n]
        test_items = shuffled[train_n + val_n: train_n + val_n + test_n]

        write_json(split_dir / "train" / filenames["train"], train_items)
        write_json(split_dir / "val" / filenames["val"], val_items)
        write_json(split_dir / "test" / filenames["test"], test_items)
        write_json(
            manifest_path,
            {
                "dataset": name,
                "source": str(raw_path),
                "split_mode": "precomputed_ratio",
                "split_name": split,
                "split_ratio": split if split != "1shot" else "1 train / 1 val / rest test",
                "split_seed": seed,
                "counts": {
                    "train": len(train_items),
                    "val": len(val_items),
                    "test": len(test_items),
                },
            },
        )
        print(
            f"wrote {name} {split} -> {split_dir} "
            f"(train={len(train_items)}, val={len(val_items)}, test={len(test_items)})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dataset", choices=sorted(DATASETS), action="append")
    args = parser.parse_args()

    for name in args.dataset or sorted(DATASETS):
        prepare_dataset(name, seed=args.seed, force=args.force)


if __name__ == "__main__":
    main()
