"""SearchQA task dataloader."""
from __future__ import annotations

import json

from skillopt.datasets.base import SplitDataLoader


# ── Raw data loading utilities (for preprocessing / standalone eval) ─────

def _load_items(path: str) -> list[dict]:
    """Load items from JSON or JSONL file."""
    with open(path) as f:
        content = f.read().strip()
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("data") or list(data.values())
    except json.JSONDecodeError:
        pass

    items = []
    for line in content.splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


# ── Dataloader ───────────────────────────────────────────────────────────

class SearchQADataLoader(SplitDataLoader):
    """SearchQA dataloader.

    Each split directory (train/, val/, test/) contains a .json file —
    a JSON array of question items.
    """

    def load_raw_items(self, data_path: str) -> list[dict]:
        return _load_items(data_path)
