"""Dataset loading."""

from __future__ import annotations

import pytest

from maintlog.config import DataConfig
from maintlog.data import iter_split, load_sft_split, write_jsonl
from maintlog.synthetic import generate_rows


def test_sft_split_keeps_prompt_and_completion() -> None:
    config = DataConfig.from_yaml("configs/data.yaml")
    train = load_sft_split(config.output_path, "train")
    evaluation = load_sft_split(config.output_path, "eval")
    assert train.column_names == ["prompt", "completion"]
    assert len(train) == config.n_train
    assert len(evaluation) == config.n_eval
    assert train[0]["prompt"][1]["role"] == "user"
    assert evaluation[0]["completion"][0]["role"] == "assistant"
    train_logs = {row["prompt"][1]["content"] for row in train}
    eval_logs = {row["prompt"][1]["content"] for row in evaluation}
    assert train_logs.isdisjoint(eval_logs)


def test_unknown_split_raises(tmp_path) -> None:
    rows = generate_rows(1, 1, 3)
    path = write_jsonl(tmp_path / "logs.jsonl", rows)
    with pytest.raises(ValueError):
        iter_split(path, "test")
