from __future__ import annotations

import json
from pathlib import Path

from skillopt.datasets.base import SplitDataLoader


class AppWorldDataLoader(SplitDataLoader):
    """Load AppWorld task items from ``<split_dir>/<split>/items.json`` (see make_split.py)."""

    def load_split_items(self, split_path: str) -> list[dict]:
        path = Path(split_path) / "items.json"
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} is missing; run `python -m skillopt.envs.appworld.make_split` first"
            )
        items = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(items, list) or not items:
            raise ValueError(f"expected a non-empty JSON array in {path}")
        for item in items:
            item.setdefault("task_type", "appworld")
            item["task_description"] = item.get("instruction", "")
        return items
