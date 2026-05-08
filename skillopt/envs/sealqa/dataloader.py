from __future__ import annotations

import csv
from pathlib import Path

from skillopt.datasets.base import SplitDataLoader


def _normalize_row(row: dict[str, str], index: int) -> dict:
    canary = str(row.get('canary') or '').strip()
    base_id = str(row.get('question_id') or row.get('id') or '').strip()
    if not base_id:
        base_id = f"{canary or 'sealqa'}:{index:04d}"
    return {
        'id': base_id,
        'question': str(row.get('question') or '').strip(),
        'ground_truth': str(row.get('answer') or row.get('ground_truth') or '').strip(),
        'answers': [str(row.get('answer') or row.get('ground_truth') or '').strip()],
        'task_type': str(row.get('topic') or 'sealqa').strip() or 'sealqa',
        'topic': str(row.get('topic') or 'sealqa').strip() or 'sealqa',
        'urls': str(row.get('urls') or '').strip(),
        'search_results': str(row.get('search_results') or '').strip(),
        'freshness': str(row.get('freshness') or '').strip(),
        'question_types': str(row.get('question_types') or '').strip(),
        'canary': canary,
    }


class SealQADataLoader(SplitDataLoader):
    def load_split_items(self, split_path: str) -> list[dict]:
        path = Path(split_path)
        csv_files = sorted(path.glob('*.csv'))
        if not csv_files:
            raise FileNotFoundError(f'No .csv file found in {split_path}')
        with csv_files[0].open(encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            return [_normalize_row(row, idx) for idx, row in enumerate(reader, start=1)]
