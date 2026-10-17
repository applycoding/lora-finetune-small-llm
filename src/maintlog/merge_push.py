"""Merge a LoRA adapter into the base model and optionally push it to the Hub.

Pushing requires HF_TOKEN. CI does not run this script.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi
from peft import PeftModel

from maintlog.config import MergeConfig, resolve_path
from maintlog.modeling import load_causal_lm, load_tokenizer

_MODEL_CARD = """---
library_name: transformers
base_model: {base_model}
tags:
- lora
- synthetic
- portfolio-demo
---

# Maintenance-log JSON model

Merged LoRA adapter for a portfolio demo. The training data is **synthetic**.
This card does not report benchmark scores. None were measured for the upload.
"""


class MissingTokenError(RuntimeError):
    """HF_TOKEN is absent, so the Hub push cannot run."""


def resolve_hf_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise MissingTokenError(
            "HF_TOKEN is required to push a merged model to the Hugging Face Hub. "
            "This script is not part of CI. Re-run with --merge-only to save locally."
        )
    return token


def merge_adapter(config: MergeConfig) -> Path:
    output_dir = resolve_path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = resolve_path(config.adapter_path)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")

    tokenizer = load_tokenizer(config.base_model)
    model = load_causal_lm(config.base_model, tokenizer)
    peft_model = PeftModel.from_pretrained(model, str(adapter_path))
    merged = peft_model.merge_and_unload()
    if getattr(merged, "generation_config", None) is not None:
        merged.generation_config.pad_token_id = tokenizer.pad_token_id
    merged.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))
    card = _MODEL_CARD.format(base_model=config.base_model)
    (output_dir / "README.md").write_text(card, encoding="utf-8")
    return output_dir


def push_folder(folder: Path, repo_id: str, token: str, private: bool) -> None:
    api = HfApi(token=token)
    api.create_repo(repo_id, exist_ok=True, repo_type="model", private=private)
    api.upload_folder(folder_path=str(folder), repo_id=repo_id, repo_type="model")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Merge a LoRA adapter and push the result to the Hugging Face Hub."
    )
    parser.add_argument("--config", default="configs/merge.yaml")
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Save the merged model locally. Does not read HF_TOKEN and does not push.",
    )
    args = parser.parse_args(argv)
    config = MergeConfig.from_yaml(args.config)
    token: str | None
    if args.merge_only:
        token = None
    else:
        try:
            token = resolve_hf_token()
        except MissingTokenError as exc:
            print(str(exc), file=sys.stderr)
            return 2

    output_dir = merge_adapter(config)
    print(f"Merged model saved to {output_dir}.")
    if token is None:
        print("Push skipped (--merge-only).")
        return 0
    push_folder(output_dir, config.repo_id, token, private=config.private)
    print(f"Pushed merged model to {config.repo_id}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
