"""
Benchmark Data Loader Template
================================
Copy this file and implement the TODO sections to load your benchmark data.

The DataLoader is responsible for:
1. Loading raw data from disk
2. Splitting into train / validation / test sets
3. Providing DataItem objects to the training loop
"""
from pathlib import Path


class TemplateBenchmarkLoader:
    """
    Data loader for <Your Benchmark Name>.

    Rename this class and implement the methods below.
    """

    def __init__(self, data_dir: str = "data/your_benchmark", **kwargs):
        self.data_dir = Path(data_dir)
        self.items = []
        self.splits = {}

    def setup(self, cfg: dict):
        """
        Initialize the loader with config.

        Called once before training starts.

        Args:
            cfg: Dict with keys like 'split_mode', 'train_ratio', 'val_ratio', etc.
        """
        # Step 1: Load raw data
        self.items = self._load_items()

        # Step 2: Create splits
        split_mode = cfg.get("split_mode", "ratio")
        if split_mode == "ratio":
            self._split_by_ratio(
                train_ratio=cfg.get("train_ratio", 0.7),
                val_ratio=cfg.get("val_ratio", 0.15),
            )
        elif split_mode == "split_dir":
            self._load_predefined_splits(cfg.get("split_dir", self.data_dir))

    def _load_items(self) -> list:
        """
        Load raw data into structured items.

        TODO: Implement data loading. Each item should have at minimum:
        - id: unique identifier
        - input: the task input (question, instruction, etc.)
        - ground_truth: the expected answer
        - metadata: optional dict with extra info

        Example:
            items = []
            for path in self.data_dir.glob("*.json"):
                data = json.loads(path.read_text())
                for entry in data:
                    items.append({
                        "id": entry["id"],
                        "input": entry["question"],
                        "ground_truth": entry["answer"],
                        "metadata": {"source": path.name},
                    })
            return items
        """
        raise NotImplementedError("Implement _load_items() for your benchmark")

    def _split_by_ratio(self, train_ratio: float, val_ratio: float):
        """Split items by ratio."""
        import random
        random.shuffle(self.items)
        n = len(self.items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        self.splits = {
            "train": self.items[:n_train],
            "valid": self.items[n_train:n_train + n_val],
            "test": self.items[n_train + n_val:],
        }

    def _load_predefined_splits(self, split_dir):
        """Load from pre-split directories."""
        # TODO: Implement if your benchmark has pre-defined splits
        raise NotImplementedError

    def get_split_items(self, split: str) -> list:
        """
        Return items for a given split.

        Args:
            split: One of "train", "valid", "test"

        Returns:
            List of data items for the requested split
        """
        if split not in self.splits:
            raise ValueError(f"Unknown split '{split}'. Available: {list(self.splits.keys())}")
        return self.splits[split]
