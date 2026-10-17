"""Generator tests. No model and no network."""

from __future__ import annotations

import json

import pytest

from maintlog.config import DataConfig
from maintlog.data import read_jsonl
from maintlog.schema import DATASET_NOTICE, validate_record
from maintlog.synthetic import generate_rows


def test_rows_are_labeled_synthetic_and_schema_valid() -> None:
    rows = generate_rows(4, 2, 1)
    assert [row["split"] for row in rows] == ["train"] * 4 + ["eval"] * 2
    assert len({row["id"] for row in rows}) == len(rows)
    for row in rows:
        assert row["synthetic"] is True
        assert row["dataset_label"] == "synthetic"
        assert row["notice"] == DATASET_NOTICE
        assert "SYNTHETIC" in row["notice"]
        validate_record(row["record"])
        assert row["prompt"][0]["role"] == "system"
        assert row["prompt"][1] == {"role": "user", "content": row["log"]}
        assert row["completion"][0]["role"] == "assistant"
        assert json.loads(row["completion"][0]["content"]) == row["record"]
        for value in row["record"].values():
            assert str(value) in row["log"]


def test_generation_is_deterministic() -> None:
    assert generate_rows(3, 2, 9) == generate_rows(3, 2, 9)
    assert generate_rows(3, 2, 9) != generate_rows(3, 2, 10)


def test_counts_must_be_positive() -> None:
    with pytest.raises(ValueError):
        generate_rows(0, 1, 1)


def test_committed_file_matches_the_generator() -> None:
    config = DataConfig.from_yaml("configs/data.yaml")
    expected = generate_rows(config.n_train, config.n_eval, config.seed)
    assert read_jsonl(config.output_path) == expected
