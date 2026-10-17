"""Gradio demo: rule-based preview, or a local LoRA adapter when one exists."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Literal, assert_never

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
from peft import PeftModel

from maintlog.config import repo_root, resolve_path
from maintlog.data import read_jsonl
from maintlog.evaluate import generate_completion
from maintlog.fallback import PREVIEW_NOTE, preview_parse
from maintlog.modeling import load_causal_lm, load_tokenizer
from maintlog.prompts import build_prompt

Mode = Literal["preview", "adapter"]
_MODEL_CACHE: dict[tuple[str, str], tuple[Any, Any]] = {}


def build_demo() -> gr.Blocks:
    examples = _example_rows()
    with gr.Blocks(title="Maintenance log to JSON") as demo:
        gr.Markdown(
            """
# Maintenance log to JSON

Portfolio demo. The samples are **synthetic**. The default path is a rule-based
preview of the labeled format, not a fine-tuned model. Load a local LoRA adapter
only after you have trained one.
            """.strip()
        )
        with gr.Row():
            log = gr.Textbox(
                label="Maintenance log",
                lines=8,
                placeholder="Paste a synthetic maintenance log.",
            )
            output = gr.Code(label="JSON", language="json")
        mode = gr.Radio(
            choices=[
                ("Rule-based preview", "preview"),
                ("Local LoRA adapter", "adapter"),
            ],
            value="preview",
            label="Decoder",
        )
        with gr.Accordion("Local adapter", open=False):
            adapter_path = gr.Textbox(
                label="Adapter directory",
                value="outputs/lora-qwen2.5-0.5b",
            )
            base_model = gr.Textbox(
                label="Base model",
                value="Qwen/Qwen2.5-0.5B-Instruct",
            )
        status = gr.Textbox(label="Status", interactive=False)
        run = gr.Button("Extract JSON", variant="primary")
        run.click(
            fn=extract,
            inputs=[log, mode, adapter_path, base_model],
            outputs=[output, status],
        )
        if examples:
            gr.Examples(examples=examples, inputs=[log])
    return demo


def extract(log: str, mode: str, adapter_path: str, base_model: str) -> tuple[str, str]:
    if not log or not log.strip():
        return "", "Paste a maintenance log first."
    try:
        selected = _parse_mode(mode)
    except ValueError as exc:
        return "", str(exc)
    if selected == "preview":
        parsed = preview_parse(log)
        return json.dumps(parsed, indent=2, sort_keys=True), PREVIEW_NOTE
    if selected == "adapter":
        return _extract_with_adapter(log, adapter_path, base_model)
    assert_never(selected)


def _parse_mode(mode: str) -> Mode:
    if mode == "preview":
        return "preview"
    if mode == "adapter":
        return "adapter"
    raise ValueError(f"Unknown decoder {mode!r}.")


def _extract_with_adapter(log: str, adapter_path: str, base_model: str) -> tuple[str, str]:
    path = resolve_path(adapter_path)
    if not path.exists():
        return (
            "",
            f"No adapter at {path}. Train one, or switch to the rule-based preview.",
        )
    if not base_model.strip():
        return "", "Set a base model id or local path before loading an adapter."
    try:
        model, tokenizer = _load_adapter(base_model.strip(), path)
        text = generate_completion(model, tokenizer, build_prompt(log), max_new_tokens=256)
    except (OSError, RuntimeError, ValueError) as exc:
        return "", f"Adapter inference failed: {exc}"
    return text, f"Generated with adapter {path} on base {base_model.strip()}."


def _load_adapter(base_model: str, adapter_path: Path) -> tuple[Any, Any]:
    key = (base_model, str(adapter_path))
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    tokenizer = load_tokenizer(base_model)
    model = load_causal_lm(base_model, tokenizer)
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    model.to("cpu")
    _MODEL_CACHE[key] = (model, tokenizer)
    return model, tokenizer


def _example_rows() -> list[list[str]]:
    dataset = repo_root() / "data" / "synthetic" / "maintenance_logs.jsonl"
    if not dataset.exists():
        return []
    logs = [row["log"] for row in read_jsonl(dataset)[:4]]
    return [[log] for log in logs]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch the maintenance-log Gradio demo.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18765)
    args = parser.parse_args(argv)
    demo = build_demo()
    demo.launch(server_name=args.host, server_port=args.port, show_error=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
