"""YAML config parsing."""

from __future__ import annotations

import pytest

from maintlog.config import DataConfig, TrainConfig


def test_unknown_key_is_rejected(tmp_path) -> None:
    path = tmp_path / "data.yaml"
    path.write_text(
        "output_path: data/out.jsonl\nn_train: 1\nn_eval: 1\nseed: 1\nextra: true\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unknown config keys"):
        DataConfig.from_yaml(path)


def test_missing_key_is_rejected(tmp_path) -> None:
    path = tmp_path / "data.yaml"
    path.write_text("output_path: data/out.jsonl\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing config keys"):
        DataConfig.from_yaml(path)


def test_override_replaces_a_value(tmp_path) -> None:
    path = tmp_path / "train.yaml"
    path.write_text(
        "\n".join(
            [
                "model_name_or_path: tiny",
                "dataset_path: data.jsonl",
                "output_dir: out",
                "max_steps: 2",
                "learning_rate: 0.0001",
                "per_device_train_batch_size: 1",
                "lora_r: 4",
                "lora_alpha: 8",
                "lora_dropout: 0.0",
                "target_modules: [q_proj]",
                "max_length: 64",
            ]
        ),
        encoding="utf-8",
    )
    config = TrainConfig.from_yaml(path, overrides={"max_steps": 3, "output_dir": str(tmp_path / "run")})
    assert config.max_steps == 3
    assert config.output_dir == str(tmp_path / "run")
    assert config.use_cpu is False
