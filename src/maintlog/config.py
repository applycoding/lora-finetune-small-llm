"""YAML config loading. Unknown keys are rejected so typos fail immediately."""

from __future__ import annotations

from dataclasses import MISSING, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return repo_root() / path


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = resolve_path(path)
    with config_path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return loaded


@dataclass
class DataConfig:
    output_path: str
    n_train: int
    n_eval: int
    seed: int

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: dict[str, Any] | None = None) -> DataConfig:
        return _from_mapping(cls, load_yaml(path), overrides)


@dataclass
class TrainConfig:
    model_name_or_path: str
    dataset_path: str
    output_dir: str
    max_steps: int
    learning_rate: float
    per_device_train_batch_size: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: list[str]
    max_length: int
    gradient_accumulation_steps: int = 1
    num_train_epochs: float = 1.0
    logging_steps: int = 10
    save_strategy: str = "no"
    save_steps: int = 50
    eval_strategy: str = "no"
    seed: int = 42
    use_cpu: bool = False
    eos_token: str | None = None
    chat_template: str | None = None
    train_split: str = "train"

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: dict[str, Any] | None = None) -> TrainConfig:
        return _from_mapping(cls, load_yaml(path), overrides)


@dataclass
class EvalConfig:
    model_name_or_path: str
    dataset_path: str
    output_path: str
    adapter_path: str | None = None
    split: str = "eval"
    max_new_tokens: int = 256
    max_samples: int | None = None
    use_cpu: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: dict[str, Any] | None = None) -> EvalConfig:
        return _from_mapping(cls, load_yaml(path), overrides)


@dataclass
class MergeConfig:
    base_model: str
    adapter_path: str
    output_dir: str
    repo_id: str
    private: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path, overrides: dict[str, Any] | None = None) -> MergeConfig:
        return _from_mapping(cls, load_yaml(path), overrides)


def _from_mapping(cls: type[Any], raw: dict[str, Any], overrides: dict[str, Any] | None) -> Any:
    data = dict(raw)
    if overrides:
        data.update(overrides)
    known = {item.name for item in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"Unknown config keys for {cls.__name__}: {unknown}")
    required_missing = []
    for item in fields(cls):
        if item.name in data:
            continue
        has_default = item.default is not MISSING or item.default_factory is not MISSING
        if not has_default:
            required_missing.append(item.name)
    if required_missing:
        raise ValueError(f"Missing config keys for {cls.__name__}: {required_missing}")
    kwargs = {key: data[key] for key in known if key in data}
    return cls(**kwargs)
