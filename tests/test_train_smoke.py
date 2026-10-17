"""A few LoRA steps on a tiny random Llama. CPU only. Not a quality run."""

from __future__ import annotations

import math

from maintlog.train import train_from_config

TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def test_tiny_random_llama_lora_steps(tmp_path) -> None:
    summary = train_from_config(
        "configs/smoke.yaml",
        overrides={
            "output_dir": str(tmp_path / "adapter"),
            "max_steps": 2,
            "model_name_or_path": TINY_MODEL,
        },
    )
    assert summary["model_name_or_path"] == TINY_MODEL
    assert summary["global_step"] == 2
    assert math.isfinite(summary["train_loss"])
    assert summary["train_loss"] > 0.0
    assert 0 < summary["trainable_parameters"] < summary["total_parameters"]
    adapter_dir = tmp_path / "adapter"
    assert (adapter_dir / "adapter_config.json").is_file()
    weights = list(adapter_dir.glob("adapter_model.*"))
    assert weights, "LoRA weights were not saved"
