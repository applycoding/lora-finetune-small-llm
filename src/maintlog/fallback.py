"""Rule-based preview for the Gradio demo.

This parser is not a model. It only recognizes the labeled synthetic logs.
"""

from __future__ import annotations

import re
from typing import Any

from maintlog.schema import REQUIRED_FIELDS, SEVERITIES, STATUSES

_ASSET_RE = re.compile(r"\b([A-Z]{2,8}-\d{2,6})\b")
_TIMESTAMP_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)\b")
_DOWNTIME_RE = re.compile(
    r"downtime(?:\s+minutes|_minutes)?\s*[:=]?\s*(\d+)",
    re.IGNORECASE,
)
_LABELED_RE = {
    "site": re.compile(r"site\s*[:=]\s*([^.|\n]+)", re.IGNORECASE),
    "technician": re.compile(
        r"technician\s*[:=]\s*([A-Z]\.\s+[A-Z][a-z]+)",
        re.IGNORECASE,
    ),
    "component": re.compile(r"component\s*[:=]\s*([^.|\n]+)", re.IGNORECASE),
    "symptom": re.compile(r"symptom\s*[:=]\s*([^.|\n]+)", re.IGNORECASE),
    "action_taken": re.compile(
        r"action(?:\s+taken|_taken)\s*[:=]\s*([^.|\n]+)",
        re.IGNORECASE,
    ),
}

PREVIEW_NOTE = (
    "Rule-based preview of the labeled synthetic format. "
    "This is not a fine-tuned model prediction and it is not an accuracy claim."
)


def preview_parse(log: str) -> dict[str, Any]:
    """Fill schema fields the labeled preview can see. Unseen fields are null."""
    parsed: dict[str, Any] = {field: None for field in REQUIRED_FIELDS}
    if not log or not log.strip():
        return parsed

    asset = _ASSET_RE.search(log)
    if asset:
        parsed["asset_id"] = asset.group(1)

    timestamp = _TIMESTAMP_RE.search(log)
    if timestamp:
        parsed["reported_at"] = timestamp.group(1)

    downtime = _DOWNTIME_RE.search(log)
    if downtime:
        parsed["downtime_minutes"] = int(downtime.group(1))

    parsed["severity"] = _first_enum(log, SEVERITIES)
    parsed["status"] = _first_enum(log, STATUSES)

    for field, pattern in _LABELED_RE.items():
        match = pattern.search(log)
        if match:
            parsed[field] = match.group(1).strip().rstrip(".")
    return parsed


def _first_enum(text: str, allowed: frozenset[str]) -> str | None:
    ordered = sorted(allowed, key=len, reverse=True)
    pattern = re.compile(r"\b(" + "|".join(ordered) + r")\b", re.IGNORECASE)
    match = pattern.search(text)
    if match is None:
        return None
    return match.group(1).lower()
