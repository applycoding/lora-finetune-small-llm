"""Target JSON schema for a maintenance-log record."""

from __future__ import annotations

import json
from typing import Any

REQUIRED_FIELDS: tuple[str, ...] = (
    "asset_id",
    "site",
    "reported_at",
    "severity",
    "component",
    "symptom",
    "action_taken",
    "downtime_minutes",
    "status",
    "technician",
)

SEVERITIES: frozenset[str] = frozenset({"low", "medium", "high", "critical"})
STATUSES: frozenset[str] = frozenset({"open", "in_progress", "resolved"})

DATASET_LABEL = "synthetic"
DATASET_NOTICE = (
    "SYNTHETIC — generated maintenance log for a portfolio demo. Not a real work order."
)


class RecordError(ValueError):
    """A record does not match the maintenance-log schema."""


def validate_record(record: dict[str, Any]) -> None:
    """Raise RecordError when a record is missing fields or has illegal values."""
    if not isinstance(record, dict):
        raise RecordError("Record must be a JSON object.")
    missing = [field for field in REQUIRED_FIELDS if field not in record]
    if missing:
        raise RecordError(f"Missing fields: {missing}")
    extra = [key for key in record if key not in REQUIRED_FIELDS]
    if extra:
        raise RecordError(f"Unexpected fields: {extra}")

    asset_id = record["asset_id"]
    if not isinstance(asset_id, str) or not _is_asset_id(asset_id):
        raise RecordError(f"Invalid asset_id: {asset_id!r}")

    for field in ("site", "component", "symptom", "action_taken", "technician"):
        value = record[field]
        if not isinstance(value, str) or not value.strip():
            raise RecordError(f"Invalid {field}: {value!r}")

    reported_at = record["reported_at"]
    if not isinstance(reported_at, str) or not _is_timestamp(reported_at):
        raise RecordError(f"Invalid reported_at: {reported_at!r}")

    if record["severity"] not in SEVERITIES:
        raise RecordError(f"Invalid severity: {record['severity']!r}")
    if record["status"] not in STATUSES:
        raise RecordError(f"Invalid status: {record['status']!r}")

    downtime = record["downtime_minutes"]
    if isinstance(downtime, bool) or not isinstance(downtime, int) or downtime < 0:
        raise RecordError(f"Invalid downtime_minutes: {downtime!r}")


def canonical_json(record: dict[str, Any]) -> str:
    """Serialize a validated record with stable key order and no extra space."""
    validate_record(record)
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _is_asset_id(value: str) -> bool:
    prefix, separator, number = value.partition("-")
    return bool(separator) and prefix.isalpha() and prefix.isupper() and number.isdigit()


def _is_timestamp(value: str) -> bool:
    if len(value) != 20 or not value.endswith("Z"):
        return False
    date, tee, time = value[:-1].partition("T")
    if tee != "T" or len(date) != 10 or len(time) != 8:
        return False
    year, month, day = date.split("-")
    hour, minute, second = time.split(":")
    parts = (year, month, day, hour, minute, second)
    return all(part.isdigit() for part in parts)
