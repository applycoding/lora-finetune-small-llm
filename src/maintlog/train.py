"""LoRA supervised fine-tuning with TRL SFTTrainer."""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
from pathlib import Path
from typing import Any

from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from maintlog.config import TrainConfig, resolve_path
from maintlog.data import load_sft_split
from maintlog.modeling import load_causal_lm, load_tokenizer, parameter_counts

logger = logging.getLogger("maintlog.train")


def train_from_config(
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = TrainConfig.from_yaml(config_path, overrides)
    return run_training(config)


def run_training(config: TrainConfig) -> dict[str, Any]:
    if config.use_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    output_dir = resolve_path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(
        config.model_name_or_path,
        eos_token=config.eos_token,
        chat_template=config.chat_template,
    )
    model = load_causal_lm(config.model_name_or_path, tokenizer)
    train_dataset = load_sft_split(config.dataset_path, config.train_split)

    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(config.target_modules),
    )
    args = build_sft_config(config, output_dir)
    trainer = SFTTrainer(**_trainer_kwargs(model, args, train_dataset, tokenizer, peft_config))
    trainable, total = parameter_counts(trainer.model)
    logger.info("Trainable parameters %s / %s", trainable, total)
    result = trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    summary = {
        "train_loss": float(result.training_loss),
        "global_step": int(result.global_step),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "output_dir": str(output_dir),
        "model_name_or_path": config.model_name_or_path,
    }
    summary_path = output_dir / "train_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logger.info("Saved adapter to %s", output_dir)
    return summary


def _trainer_kwargs(
    model: Any,
    args: SFTConfig,
    train_dataset: Any,
    tokenizer: Any,
    peft_config: LoraConfig,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "peft_config": peft_config,
    }
    parameters = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in parameters:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in parameters:
        kwargs["tokenizer"] = tokenizer
    return kwargs


def build_sft_config(config: TrainConfig, output_dir: Path) -> SFTConfig:
    """Build SFTConfig, dropping keys this TRL version does not accept."""
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "max_steps": config.max_steps,
        "num_train_epochs": config.num_train_epochs,
        "logging_steps": config.logging_steps,
        "save_strategy": config.save_strategy,
        "save_steps": config.save_steps,
        "bf16": False,
        "fp16": False,
        "tf32": False,
        "report_to": "none",
        "max_length": config.max_length,
        "seed": config.seed,
        "data_seed": config.seed,
        "gradient_checkpointing": False,
        "optim": "adamw_torch",
        "dataloader_pin_memory": False,
        "dataloader_num_workers": 0,
        "completion_only_loss": True,
        "packing": False,
        "use_cpu": config.use_cpu,
        "warmup_steps": 0,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "lr_scheduler_type": "constant",
        "logging_first_step": True,
        "disable_tqdm": False,
        "remove_unused_columns": True,
    }
    if config.eos_token:
        kwargs["eos_token"] = config.eos_token

    parameters = inspect.signature(SFTConfig).parameters
    if "eval_strategy" in parameters:
        kwargs["eval_strategy"] = config.eval_strategy
    elif "evaluation_strategy" in parameters:
        kwargs["evaluation_strategy"] = config.eval_strategy

    supported = {key: value for key, value in kwargs.items() if key in parameters}
    dropped = sorted(set(kwargs) - set(supported))
    if dropped:
        logger.info("This TRL build ignores SFTConfig keys: %s", dropped)
    return SFTConfig(**supported)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="LoRA-SFT a small causal LM on synthetic maintenance logs.")
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args(argv)
    summary = train_from_config(args.config)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
