"""Demo dispatch without launching a server or loading weights."""

from __future__ import annotations

import json

from maintlog.demo_app import build_demo, extract
from maintlog.synthetic import generate_rows


def test_preview_extract_returns_json() -> None:
    row = generate_rows(1, 1, 4)[0]
    text, note = extract(row["log"], "preview", "outputs/missing", "Qwen/Qwen2.5-0.5B-Instruct")
    assert json.loads(text) == row["record"]
    assert "not a fine-tuned model" in note


def test_blank_and_unknown_mode() -> None:
    assert extract("  ", "preview", "", "")[0] == ""
    assert "Paste" in extract("  ", "preview", "", "")[1]
    assert "Unknown decoder" in extract("a log", "nope", "", "")[1]


def test_missing_adapter_does_not_download(tmp_path) -> None:
    text, note = extract("a log", "adapter", str(tmp_path / "no-adapter"), "Qwen/Qwen2.5-0.5B-Instruct")
    assert text == ""
    assert "No adapter" in note


def test_demo_builds() -> None:
    demo = build_demo()
    assert demo is not None
