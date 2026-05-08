#!/usr/bin/env python3
"""Download BabyVision from Hugging Face and convert it to local meta_data.jsonl + images/ format."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--dataset", type=str, default="UnipatAI/BabyVision")
    p.add_argument("--split", type=str, default="train")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Please install `datasets` first: pip install datasets pillow") from exc

    out_dir = Path(args.out_dir).resolve()
    images_dir = out_dir / "images"
    meta_path = out_dir / "meta_data.jsonl"
    images_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset, split=args.split)
    with open(meta_path, "w", encoding="utf-8") as outf:
        for idx, row in enumerate(dataset):
            image = row.get("image")
            if image is None:
                continue
            task_id = str(row.get("taskId") or row.get("id") or idx + 1)
            image_name = f"{task_id}.png"
            image_path = images_dir / image_name
            image.save(image_path)

            record = dict(row)
            record["image"] = image_name
            outf.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved BabyVision to {out_dir}")
    print(f"Metadata: {meta_path}")
    print(f"Images:   {images_dir}")


if __name__ == "__main__":
    main()
