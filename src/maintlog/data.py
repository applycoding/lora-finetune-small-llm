"""Read and write the synthetic JSONL dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset

from maintlog.config import resolve_path


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    dataset_path = resolve_path(path)
    rows: list[dict[str, Any]] = []
    with dataset_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_number} in {dataset_path} is not an object.")
            rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    dataset_path = resolve_path(path)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with dataset_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    return dataset_path


def iter_split(path: str | Path, split: str) -> list[dict[str, Any]]:
    rows = [row for row in read_jsonl(path) if row.get("split") == split]
    if not rows:
        raise ValueError(f"No rows with split {split!r} in {path}.")
    return rows


def load_sft_split(path: str | Path, split: str) -> Dataset:
    """Dataset of prompt/completion message lists for TRL SFTTrainer."""
    records = [
        {"prompt": row["prompt"], "completion": row["completion"]}
        for row in iter_split(path, split)
    ]
    return Dataset.from_list(records)
