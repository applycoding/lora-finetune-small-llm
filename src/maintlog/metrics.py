"""JSON-validity and field-accuracy scoring.

These functions define the metrics. They do not ship measured scores.
"""

from __future__ import annotations

import json
import re
from typing import Any

from maintlog.schema import REQUIRED_FIELDS

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Return the first JSON object found in model text, or None."""
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    if not stripped:
        return None
    candidates = [stripped]
    fenced = _FENCE_RE.search(stripped)
    if fenced:
        candidates.append(fenced.group(1))
    spanned = _first_object_span(stripped)
    if spanned is not None:
        candidates.append(spanned)
    for candidate in candidates:
        parsed = _loads_object(candidate)
        if parsed is not None:
            return parsed
    return None


def score_predictions(golds: list[dict[str, Any]], predictions: list[str]) -> dict[str, Any]:
    """Score gold records against raw model strings.

    json_validity is the fraction of predictions that contain a JSON object.
    field_accuracy is the mean, over examples, of matched required fields
    divided by the number of required fields. An invalid prediction scores 0
    on every field. Enum fields are compared case-insensitively. Other strings
    are stripped and compared exactly. downtime_minutes accepts an int or a
    whole-number float or digit string.
    """
    if len(golds) != len(predictions):
        raise ValueError("golds and predictions must have the same length.")
    count = len(golds)
    if count == 0:
        raise ValueError("Cannot score an empty prediction list.")

    valid_flags: list[bool] = []
    example_scores: list[float] = []
    field_hits = {field: 0 for field in REQUIRED_FIELDS}

    for gold, prediction in zip(golds, predictions, strict=True):
        parsed = extract_json_object(prediction)
        valid = parsed is not None
        valid_flags.append(valid)
        matched = 0
        for field in REQUIRED_FIELDS:
            hit = valid and _values_equal(field, parsed.get(field) if parsed else None, gold[field])
            if hit:
                matched += 1
                field_hits[field] += 1
        example_scores.append(matched / len(REQUIRED_FIELDS))

    valid_count = sum(valid_flags)
    return {
        "n": count,
        "json_valid_count": valid_count,
        "json_validity": valid_count / count,
        "field_accuracy": sum(example_scores) / count,
        "per_field_accuracy": {
            field: field_hits[field] / count for field in REQUIRED_FIELDS
        },
    }


def _loads_object(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict):
        return parsed
    return None


def _first_object_span(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(text[start:], start):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _values_equal(field: str, predicted: Any, gold: Any) -> bool:
    return _normalize(field, predicted) == _normalize(field, gold)


def _normalize(field: str, value: Any) -> Any:
    if field == "downtime_minutes":
        return _normalize_downtime(value)
    if field in {"severity", "status"}:
        if isinstance(value, str):
            return value.strip().lower()
        return None
    if isinstance(value, str):
        return value.strip()
    return None


def _normalize_downtime(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
    return None
