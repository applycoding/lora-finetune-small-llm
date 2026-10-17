"""Generate on the held-out split and score JSON validity plus field accuracy."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel

from maintlog.config import EvalConfig, resolve_path
from maintlog.data import iter_split
from maintlog.metrics import score_predictions
from maintlog.modeling import load_causal_lm, load_tokenizer

logger = logging.getLogger("maintlog.evaluate")


def evaluate_from_config(
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = EvalConfig.from_yaml(config_path, overrides)
    return run_evaluation(config)


def run_evaluation(config: EvalConfig) -> dict[str, Any]:
    rows = iter_split(config.dataset_path, config.split)
    if config.max_samples is not None:
        rows = rows[: config.max_samples]
    tokenizer = load_tokenizer(config.model_name_or_path)
    model = load_causal_lm(config.model_name_or_path, tokenizer)
    if config.adapter_path:
        adapter_path = resolve_path(config.adapter_path)
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")
        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    if config.use_cpu:
        model.to("cpu")

    predictions = [
        generate_completion(model, tokenizer, row["prompt"], config.max_new_tokens) for row in rows
    ]
    golds = [row["record"] for row in rows]
    metrics = score_predictions(golds, predictions)
    report = {
        "status": "measured",
        "disclaimer": (
            "Scores from this run only. The committed results table is a placeholder "
            "and is not updated by this script unless you point output_path at it."
        ),
        "model_name_or_path": config.model_name_or_path,
        "adapter_path": config.adapter_path,
        "split": config.split,
        **metrics,
        "predictions": [
            {"id": row["id"], "prediction": prediction, "gold": row["record"]}
            for row, prediction in zip(rows, predictions, strict=True)
        ],
    }
    output_path = resolve_path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %s", output_path)
    return report


@torch.inference_mode()
def generate_completion(
    model: Any,
    tokenizer: Any,
    prompt_messages: list[dict[str, str]],
    max_new_tokens: int,
) -> str:
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    prompt_ids = prompt_ids.to(model.device)
    generated = model.generate(
        prompt_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    new_tokens = generated[0, prompt_ids.shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Score JSON validity and field accuracy.")
    parser.add_argument("--config", default="configs/eval.yaml")
    args = parser.parse_args(argv)
    report = evaluate_from_config(args.config)
    printable = {key: value for key, value in report.items() if key != "predictions"}
    print(json.dumps(printable, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
