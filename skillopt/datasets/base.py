"""Generic task dataloader abstractions for ReflACT.

ReflACT does not train model parameters directly. Instead, it iterates over
task batches, rolls out the current skill, reflects on failures/successes,
and updates the skill document. Because of that, the "dataloader" abstraction
here is closer to a batch sampler / episode planner than a tensor loader.

Class hierarchy::

    BaseDataLoader          # abstract — simulator-backed envs (e.g. ALFWorld)
    └── SplitDataLoader     # abstract — dataset-backed envs with split_dir

SplitDataLoader supports two dataset entry modes:

1. ``split_mode="split_dir"``: consume an existing split directory.
2. ``split_mode="ratio"``: build a deterministic split directory from a raw
   dataset path using an explicit train:val:test ratio.

In either case, the standardised split layout is:

    split_dir/
    ├── train/      # training items
    ├── val/        # validation / selection items (gate)
    └── test/       # held-out test items

Each subdirectory's contents are benchmark-specific.  Subclasses only need
to implement ``load_split_items(split_path)`` to teach the loader how to
read items from one of those directories.
"""
from __future__ import annotations

import glob
import json
import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class BatchSpec:
    """A concrete batch request consumed by the training loop.

    Parameters
    ----------
    phase : str
        ``"train"`` or ``"eval"``.
    split : str
        Dataset split name, typically ``"train"`` or an eval split.
    seed : int
        Random seed used to construct the batch deterministically.
    batch_size : int
        Requested number of items / episodes in this batch.
    payload : object | None
        Environment-specific batch payload. For dataset-backed environments
        this is often a list of sampled items; for simulator-backed
        environments this may be ``None`` and the seed alone can define the
        batch.
    metadata : dict[str, Any]
        Optional structured metadata for logging, resume, or curriculum logic.
    """

    phase: str
    split: str
    seed: int
    batch_size: int
    payload: object | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseDataLoader(ABC):
    """Abstract base class for task batch planning in ReflACT.

    Subclasses are responsible for defining how a train or eval batch is
    sampled. The default implementation here provides deterministic epoch seed
    planning so all loaders share the same reproducibility behavior.
    """

    def setup(self, cfg: dict) -> None:
        """Optional one-time initialization with the full trainer config."""

    def set_out_root(self, out_root: str) -> None:
        """Optional hook for loaders that persist split files or state."""

    def state_dict(self) -> dict[str, Any]:
        """Return serializable loader state for resume support."""
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore loader state from :meth:`state_dict` output."""

    def get_train_size(self) -> int | None:
        """Return the size of the training pool when known."""
        return None

    @staticmethod
    def make_base_seeds(steps_per_epoch: int, accumulation: int, seed: int) -> list[int]:
        """Return the deterministic seed pool used to define train batches."""
        batches_per_epoch = steps_per_epoch * accumulation
        return [seed + i + 1 for i in range(batches_per_epoch)]

    @staticmethod
    def shuffle_epoch_seeds(base_seeds: list[int], epoch: int, seed: int) -> list[int]:
        """Return the per-epoch deterministic shuffle of *base_seeds*."""
        epoch_rng = random.Random(seed + epoch * 1000)
        shuffled = list(base_seeds)
        epoch_rng.shuffle(shuffled)
        return shuffled

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
        """Build the full list of training batches for one epoch."""
        base_seeds = self.make_base_seeds(
            steps_per_epoch=steps_per_epoch,
            accumulation=accumulation,
            seed=seed,
        )
        shuffled_seeds = self.shuffle_epoch_seeds(base_seeds, epoch=epoch, seed=seed)
        return [
            self.build_train_batch(batch_size=batch_size, seed=batch_seed, **kwargs)
            for batch_seed in shuffled_seeds
        ]

    @abstractmethod
    def build_train_batch(self, batch_size: int, seed: int, **kwargs) -> BatchSpec:
        """Construct one training batch specification."""

    @abstractmethod
    def build_eval_batch(
        self,
        env_num: int,
        split: str,
        seed: int,
        **kwargs,
    ) -> BatchSpec:
        """Construct one evaluation batch specification."""


# ── Split-based dataloader for dataset-backed environments ──────────────

# Canonical split names expected under split_dir/
SPLIT_NAMES = ("train", "val", "test")

# Maps legacy / trainer split names → canonical directory names
_SPLIT_ALIAS: dict[str, str] = {
    "train": "train",
    "valid_seen": "val",
    "selection": "val",
    "val": "val",
    "valid_unseen": "test",
    "test": "test",
}


def _load_json_or_jsonl(path: str) -> list[dict]:
    """Load a list of items from a JSON or JSONL file."""
    with open(path, encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return []

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = None

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        nested = data.get("data")
        if isinstance(nested, list):
            return nested
        return list(data.values())

    items: list[dict] = []
    for line in content.splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def _parse_split_ratio(text: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in str(text or "").split(":") if part.strip()]
    if len(parts) != 3:
        raise ValueError(
            f"split_ratio must be in train:val:test form, got {text!r}"
        )
    try:
        train, val, test = (int(part) for part in parts)
    except ValueError as exc:
        raise ValueError(
            f"split_ratio must contain integers, got {text!r}"
        ) from exc
    if min(train, val, test) <= 0:
        raise ValueError(f"split_ratio parts must be positive, got {text!r}")
    return train, val, test


def _compute_split_counts(total: int, ratio: tuple[int, int, int]) -> tuple[int, int, int]:
    weights = list(ratio)
    denom = sum(weights)
    raw = [total * weight / denom for weight in weights]
    counts = [int(value) for value in raw]
    remaining = total - sum(counts)
    order = sorted(
        range(len(raw)),
        key=lambda idx: (raw[idx] - counts[idx], weights[idx]),
        reverse=True,
    )
    for idx in order[:remaining]:
        counts[idx] += 1
    return counts[0], counts[1], counts[2]


class SplitDataLoader(BaseDataLoader):
    """Base class for dataset-backed environments.

    Supported modes:

    - ``split_mode="split_dir"``: load an existing ``train/``, ``val/``,
      ``test/`` directory tree.
    - ``split_mode="ratio"``: load raw items from ``data_path`` and materialize
      a deterministic split directory with the requested ratio.
    """

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
        **kwargs,
    ) -> None:
        self.split_dir = split_dir
        self.data_path = data_path
        self.split_mode = split_mode
        self.split_ratio = split_ratio
        self.split_seed = int(split_seed)
        self.split_output_dir = split_output_dir
        self.seed = seed
        self.limit = limit
        self._splits: dict[str, list[dict]] = {}

    # ── Setup ────────────────────────────────────────────────────────────

    def setup(self, cfg: dict) -> None:
        if not self.split_mode:
            self.split_mode = str(cfg.get("split_mode", "ratio") or "ratio")
        if not self.split_dir:
            self.split_dir = cfg.get("split_dir", "")
        if not self.data_path:
            self.data_path = cfg.get("data_path", "")
        if not self.split_output_dir:
            self.split_output_dir = cfg.get("split_output_dir", "")
        if "split_seed" in cfg and not self.split_seed:
            self.split_seed = int(cfg.get("split_seed", 0) or 0)
        if not self.split_seed:
            self.split_seed = self.seed
        if not self.split_ratio:
            self.split_ratio = str(cfg.get("split_ratio", "2:1:7") or "2:1:7")

        mode = str(self.split_mode or "ratio").strip().lower()
        if mode not in {"ratio", "split_dir"}:
            raise ValueError(
                f"{type(self).__name__} split_mode must be 'ratio' or 'split_dir', "
                f"got {self.split_mode!r}"
            )
        self.split_mode = mode

        if self.split_mode == "ratio":
            self.split_dir = self._materialize_ratio_split(cfg)
        if not self.split_dir:
            raise ValueError(
                f"{type(self).__name__} requires either "
                "`split_mode=ratio` with `data_path`, or `split_mode=split_dir` "
                f"with `split_dir` pointing to {'/'.join(SPLIT_NAMES)}/."
            )
        self._load_all_splits()

    def _resolve_split_output_dir(self, cfg: dict) -> str:
        if self.split_output_dir:
            return os.path.abspath(self.split_output_dir)
        out_root = os.path.abspath(str(cfg.get("out_root") or os.getcwd()))
        env_name = str(cfg.get("env") or type(self).__name__.replace("DataLoader", "").lower())
        ratio_tag = str(self.split_ratio or "2:1:7").replace(":", "-")
        return os.path.join(out_root, "_generated_splits", f"{env_name}_{ratio_tag}_seed{self.split_seed}")

    def load_raw_items(self, data_path: str) -> list[dict]:
        """Load raw items from a dataset path before ratio splitting.

        Subclasses can override when the raw dataset is not a single JSON/JSONL
        file or when directory layouts require custom normalization.
        """
        if os.path.isdir(data_path):
            if any(os.path.isdir(os.path.join(data_path, name)) for name in SPLIT_NAMES):
                raise ValueError(
                    f"{type(self).__name__} got a split directory as data_path. "
                    "Use split_mode=split_dir and pass it as split_dir instead."
                )
            candidates = sorted(glob.glob(os.path.join(data_path, "*.json")))
            candidates += sorted(glob.glob(os.path.join(data_path, "*.jsonl")))
            if len(candidates) != 1:
                raise ValueError(
                    f"{type(self).__name__} expected data_path to be one JSON/JSONL file "
                    f"or a directory containing exactly one such file, got: {data_path}"
                )
            return _load_json_or_jsonl(candidates[0])
        return _load_json_or_jsonl(data_path)

    def write_split_items(self, split_path: str, items: list[dict]) -> None:
        os.makedirs(split_path, exist_ok=True)
        out_path = os.path.join(split_path, "items.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)

    def _materialize_ratio_split(self, cfg: dict) -> str:
        data_path = os.path.abspath(str(self.data_path or "").strip())
        if not data_path:
            raise ValueError(
                f"{type(self).__name__} requires data_path when split_mode=ratio."
            )

        ratio = _parse_split_ratio(self.split_ratio)
        items = self.load_raw_items(data_path)
        if not isinstance(items, list) or not items:
            raise ValueError(f"No raw items available for ratio split from {data_path}")

        shuffled = list(items)
        rng = random.Random(self.split_seed)
        rng.shuffle(shuffled)

        train_n, val_n, test_n = _compute_split_counts(len(shuffled), ratio)
        train_items = shuffled[:train_n]
        val_items = shuffled[train_n: train_n + val_n]
        test_items = shuffled[train_n + val_n: train_n + val_n + test_n]

        split_dir = self._resolve_split_output_dir(cfg)
        manifest = {
            "source_data_path": data_path,
            "split_mode": "ratio",
            "split_ratio": self.split_ratio,
            "split_seed": self.split_seed,
            "counts": {
                "train": len(train_items),
                "val": len(val_items),
                "test": len(test_items),
            },
        }
        os.makedirs(split_dir, exist_ok=True)
        self.write_split_items(os.path.join(split_dir, "train"), train_items)
        self.write_split_items(os.path.join(split_dir, "val"), val_items)
        self.write_split_items(os.path.join(split_dir, "test"), test_items)
        with open(os.path.join(split_dir, "split_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(
            f"  [{type(self).__name__}] generated ratio split {self.split_ratio} "
            f"at {split_dir} from {data_path}"
        )
        return split_dir

    def _load_all_splits(self) -> None:
        for name in SPLIT_NAMES:
            split_path = os.path.join(self.split_dir, name)
            if not os.path.isdir(split_path):
                raise ValueError(
                    f"Missing '{name}/' subdirectory in split_dir: {self.split_dir}"
                )
            items = self.load_split_items(split_path)
            if self.limit:
                items = items[: self.limit]
            self._splits[name] = items

        counts = " ".join(f"{k}={len(v)}" for k, v in self._splits.items())
        print(f"  [{type(self).__name__}] {counts}  (from {self.split_dir})")

    def load_split_items(self, split_path: str) -> list[dict]:
        """Load items from one split directory (e.g. ``split_dir/train/``).

        Default: finds the first ``.json`` file in the directory and loads it
        as a JSON array.  Subclasses can override for custom formats.
        """
        json_files = sorted(glob.glob(os.path.join(split_path, "*.json")))
        if not json_files:
            raise FileNotFoundError(
                f"No .json file found in {split_path}"
            )
        with open(json_files[0], encoding="utf-8") as f:
            items = json.load(f)
        if not isinstance(items, list):
            raise ValueError(
                f"Expected JSON array in {json_files[0]}, got {type(items).__name__}"
            )
        return items

    # ── Accessors ────────────────────────────────────────────────────────

    @property
    def train_items(self) -> list[dict]:
        return self._splits.get("train", [])

    @property
    def val_items(self) -> list[dict]:
        return self._splits.get("val", [])

    @property
    def test_items(self) -> list[dict]:
        return self._splits.get("test", [])

    def get_split_items(self, split: str) -> list[dict]:
        """Resolve a split name (including legacy aliases) to its item list."""
        canonical = _SPLIT_ALIAS.get(split, split)
        return list(self._splits.get(canonical, self.val_items))

    def get_train_size(self) -> int:
        return len(self.train_items)

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
        """Build one full epoch that covers the train split in shuffled order.

        For split-backed datasets, an epoch should correspond to one pass over
        the available training items rather than repeated independent sampling.
        """
        epoch_rng = random.Random(seed + epoch * 1000)
        items = list(self.train_items)
        epoch_rng.shuffle(items)

        total_batches = steps_per_epoch * accumulation
        if total_batches <= 0:
            return []

        batches: list[BatchSpec] = []
        cursor = 0
        for batch_idx in range(total_batches):
            batch_items = items[cursor: cursor + batch_size]
            cursor += len(batch_items)

            # Extremely small datasets can leave trailing empty microbatches
            # when accumulation > 1. Reuse the shuffled prefix in that case so
            # the trainer still receives the expected batch count.
            if not batch_items and items:
                refill_rng = random.Random(seed + epoch * 1000 + batch_idx + 1)
                batch_items = list(items)
                refill_rng.shuffle(batch_items)
                batch_items = batch_items[:batch_size]

            batches.append(
                BatchSpec(
                    phase="train",
                    split="train",
                    seed=seed + epoch * 1000 + batch_idx + 1,
                    batch_size=len(batch_items),
                    payload=batch_items,
                )
            )

        return batches

    # ── Batch construction ───────────────────────────────────────────────

    def build_train_batch(self, batch_size: int, seed: int, **kwargs) -> BatchSpec:
        rng = random.Random(seed)
        items = list(self.train_items)
        rng.shuffle(items)
        items = items[:batch_size]
        return BatchSpec(
            phase="train",
            split="train",
            seed=seed,
            batch_size=len(items),
            payload=items,
        )

    def build_eval_batch(
        self,
        env_num: int,
        split: str,
        seed: int,
        **kwargs,
    ) -> BatchSpec:
        items = self.get_split_items(split)
        if env_num and env_num < len(items):
            items = items[:env_num]
        return BatchSpec(
            phase="eval",
            split=split,
            seed=seed,
            batch_size=len(items),
            payload=items,
        )
