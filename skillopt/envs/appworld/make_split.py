"""Materialize the SkillOpt AppWorld split from an AppWorld data root.

    python -m skillopt.envs.appworld.make_split \
        --appworld_root ~/.appworld-roots/skillopt-appworld --out data/appworld_split_full

Layout written (one ``items.json`` per split, read by ``AppWorldDataLoader``):

* train: AppWorld ``train`` (90 tasks)
* val:   AppWorld ``dev`` (57 tasks), used for selection/gating only
* test:  ``--test_set full`` (default): the complete ``test_normal`` (168) + ``test_challenge`` (417);
  ``--test_set scen1``: one task per scenario, ``test_normal_scen1`` (56) + ``test_challenge_scen1``
  (139), the lists in ``skillopt/envs/appworld/splits/`` (seed 20261010), a 195-task subset

Each item keeps ``source_split`` so the test score can be reported per AppWorld split. Task
instructions are read from the local AppWorld data and are not shipped with SkillOpt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

LAYOUT = {
    "train": [("train", None)],
    "val": [("dev", None)],
    "test": [
        ("test_normal_scen1", HERE / "splits" / "test_normal_scen1.txt"),
        ("test_challenge_scen1", HERE / "splits" / "test_challenge_scen1.txt"),
    ],
}


def _task_ids(root: Path, name: str, path: Path | None) -> list[str]:
    source = path if path is not None else root / "data" / "datasets" / f"{name}.txt"
    ids = source.read_text().split()
    if len(ids) != len(set(ids)):
        raise SystemExit(f"duplicate task ids in {source}")
    return ids


def build(root: Path, out: Path, test_set: str = "full") -> dict[str, int]:
    layout = dict(LAYOUT)
    if test_set == "full":  # the complete AppWorld test_normal (168) + test_challenge (417)
        layout["test"] = [("test_normal", None), ("test_challenge", None)]
    elif test_set != "scen1":
        raise ValueError(f"test_set must be 'full' or 'scen1', got {test_set!r}")
    counts: dict[str, int] = {}
    seen: dict[str, str] = {}
    for split, sources in layout.items():
        items = []
        for name, path in sources:
            for task_id in _task_ids(root, name, path):
                if task_id in seen:
                    raise SystemExit(f"{task_id} is in both {seen[task_id]} and {split}")
                seen[task_id] = split
                specs = json.loads((root / "data" / "tasks" / task_id / "specs.json").read_text())
                items.append({
                    "id": task_id,
                    "task_id": task_id,
                    "instruction": specs["instruction"],
                    "source_split": name,
                    "scenario_id": task_id.rsplit("_", 1)[0],
                    "task_type": "appworld",
                })
        split_dir = out / split
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "items.json").write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n")
        counts[split] = len(items)
    manifest = {
        "benchmark": "AppWorld",
        "appworld_root": str(root),
        "layout": {split: [name for name, _ in sources] for split, sources in layout.items()},
        "counts": counts,
    }
    (out / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--appworld_root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/appworld_split_full"))
    parser.add_argument("--test_set", choices=["full", "scen1"], default="full")
    args = parser.parse_args()
    print(build(args.appworld_root.expanduser(), args.out, args.test_set))


if __name__ == "__main__":
    main()
