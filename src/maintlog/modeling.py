"""Model and tokenizer loading shared by train, eval, merge, and the demo."""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from maintlog.prompts import PLAIN_CHAT_TEMPLATE

_CHAT_TEMPLATES = {"plain": PLAIN_CHAT_TEMPLATE}


def load_tokenizer(
    model_name: str,
    eos_token: str | None = None,
    chat_template: str | None = None,
) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
    if eos_token:
        tokenizer.eos_token = eos_token
    if tokenizer.pad_token is None or tokenizer.pad_token_id is None or tokenizer.pad_token_id < 0:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    if chat_template:
        template = _CHAT_TEMPLATES.get(chat_template)
        if template is None:
            known = ", ".join(sorted(_CHAT_TEMPLATES))
            raise ValueError(f"Unknown chat_template {chat_template!r}. Known values: {known}.")
        tokenizer.chat_template = template
    elif not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = PLAIN_CHAT_TEMPLATE
    return tokenizer


def load_causal_lm(model_name: str, tokenizer: PreTrainedTokenizerBase) -> PreTrainedModel:
    shared = {"trust_remote_code": False}
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32, **shared)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, **shared
        )
    align_pad_token(tokenizer, model)
    model.config.use_cache = False
    return model


def align_pad_token(tokenizer: PreTrainedTokenizerBase, model: PreTrainedModel) -> None:
    pad_id = tokenizer.pad_token_id
    if pad_id is None or pad_id < 0:
        pad_id = tokenizer.eos_token_id
    model.config.pad_token_id = pad_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = pad_id


def parameter_counts(model: PreTrainedModel) -> tuple[int, int]:
    trainable = 0
    total = 0
    for param in model.parameters():
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count
    return trainable, total
