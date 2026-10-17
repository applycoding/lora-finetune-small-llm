"""Prompt text for the maintenance-log extraction task."""

from __future__ import annotations

from typing import Any

from maintlog.schema import REQUIRED_FIELDS, canonical_json

Message = dict[str, str]

SYSTEM_PROMPT = (
    "You convert one maintenance log into one JSON object. "
    f"Use only these keys: {', '.join(REQUIRED_FIELDS)}. "
    "severity is one of low, medium, high, critical. "
    "status is one of open, in_progress, resolved. "
    "downtime_minutes is an integer. "
    "Reply with JSON only, no markdown."
)

# Used for tokenizers that do not ship a usable chat template, including the
# tiny random Llama in the CPU smoke test. The generation prompt is a prefix
# of the full conversation, which is what completion-only loss expects.
PLAIN_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ message['role'] }}: {{ message['content'] }}\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}assistant: {% endif %}"
)


def build_prompt(log: str) -> list[Message]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": log},
    ]


def build_completion(record: dict[str, Any]) -> list[Message]:
    return [{"role": "assistant", "content": canonical_json(record)}]
