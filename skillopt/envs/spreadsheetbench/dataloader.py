"""SpreadsheetBench task dataloader."""
from __future__ import annotations

from skillopt.datasets.base import SplitDataLoader


class SpreadsheetBenchDataLoader(SplitDataLoader):
    """SpreadsheetBench dataloader.

    Each split directory contains a .json file (JSON array of task items).
    Spreadsheet files referenced by items live under a separate ``data_root``.
    """

    def __init__(
        self,
        split_dir: str = "",
        data_path: str = "",
        split_mode: str = "ratio",
        split_ratio: str = "2:1:7",
        split_seed: int = 42,
        split_output_dir: str = "",
        data_root: str = "",
        seed: int = 42,
        limit: int = 0,
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
        self.data_root = data_root
