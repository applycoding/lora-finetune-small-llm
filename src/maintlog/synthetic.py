"""Deterministic synthetic maintenance logs.

Every row is labeled synthetic. Nothing here is a real work order.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any

from maintlog.prompts import build_completion, build_prompt
from maintlog.schema import DATASET_LABEL, DATASET_NOTICE, validate_record

_COMPONENTS: tuple[dict[str, tuple[str, ...]], ...] = (
    {
        "name": "bearing",
        "symptoms": (
            "grinding noise and heat at the housing",
            "axial play above the wear limit",
        ),
        "actions": (
            "replaced the part and verified rotation",
            "greased the housing and scheduled a follow-up",
        ),
    },
    {
        "name": "mechanical seal",
        "symptoms": (
            "oil weep at the seal face",
            "steady drip at the gland",
        ),
        "actions": (
            "cleaned the seal face and reinstalled",
            "replaced the seal kit and leak-checked",
        ),
    },
    {
        "name": "drive belt",
        "symptoms": (
            "belt glaze and slippage",
            "squeal on startup",
        ),
        "actions": (
            "aligned the drive and retensioned",
            "fitted a new belt and checked alignment",
        ),
    },
    {
        "name": "contactor",
        "symptoms": (
            "intermittent trip on startup",
            "chatter when the unit starts",
        ),
        "actions": (
            "reset the trip and monitored for one hour",
            "replaced the contactor and confirmed pull-in",
        ),
    },
    {
        "name": "suction filter",
        "symptoms": (
            "filter differential above the limit",
            "starved flow at the inlet",
        ),
        "actions": (
            "swapped the filter element",
            "cleaned the strainer and logged the differential",
        ),
    },
    {
        "name": "impeller",
        "symptoms": (
            "vibration at running speed",
            "rub marks on the shroud",
        ),
        "actions": (
            "dressed the shroud and rebalanced",
            "isolated the unit and tagged it for repair",
        ),
    },
)

_SITES: tuple[str, ...] = (
    "North Plant",
    "Dock 3",
    "Boiler House",
    "Line 2 Compressor Room",
    "Warehouse Cold Store",
    "East Utility Bay",
)

_TECHNICIANS: tuple[str, ...] = (
    "A. Okonkwo",
    "M. Chen",
    "R. Ibarra",
    "S. Novak",
    "L. Rahman",
    "P. Duarte",
)

_PREFIXES: tuple[str, ...] = ("PUMP", "AHU", "CMP", "FAN", "MTR", "VLV")
_SEVERITIES: tuple[str, ...] = ("low", "medium", "high", "critical")
_STATUSES: tuple[str, ...] = ("open", "in_progress", "resolved")

_TEMPLATES: tuple[str, ...] = (
    "Work order. site: {site}. technician: {technician}. asset_id: {asset_id}. "
    "reported_at: {reported_at}. component: {component}. symptom: {symptom}. "
    "severity: {severity}. action_taken: {action_taken}. "
    "downtime_minutes: {downtime_minutes}. status: {status}.",
    "Shift note from {site}. site: {site}. technician: {technician}. "
    "asset_id: {asset_id}. reported_at: {reported_at}. component: {component}. "
    "symptom: {symptom}. severity: {severity}. action_taken: {action_taken}. "
    "downtime_minutes: {downtime_minutes}. status: {status}.",
    "Follow-up on asset {asset_id}. site: {site}. technician: {technician}. "
    "asset_id: {asset_id}. reported_at: {reported_at}. component: {component}. "
    "symptom: {symptom}. severity: {severity}. action_taken: {action_taken}. "
    "downtime_minutes: {downtime_minutes}. status: {status}.",
    "Logged {reported_at}. site: {site}. technician: {technician}. "
    "asset_id: {asset_id}. reported_at: {reported_at}. component: {component}. "
    "symptom: {symptom}. severity: {severity}. action_taken: {action_taken}. "
    "downtime_minutes: {downtime_minutes}. status: {status}.",
)


def generate_rows(n_train: int, n_eval: int, seed: int) -> list[dict[str, Any]]:
    """Build a deterministic train/eval list. Both counts must be positive."""
    if n_train < 1 or n_eval < 1:
        raise ValueError("n_train and n_eval must both be at least 1.")
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    total = n_train + n_eval
    for index in range(total):
        split = "train" if index < n_train else "eval"
        rows.append(_build_row(rng, index, split, seed))
    return rows


def _build_row(rng: random.Random, index: int, split: str, seed: int) -> dict[str, Any]:
    component = rng.choice(_COMPONENTS)
    record = {
        "asset_id": f"{rng.choice(_PREFIXES)}-{1000 + index}",
        "site": rng.choice(_SITES),
        "reported_at": _random_timestamp(rng),
        "severity": rng.choice(_SEVERITIES),
        "component": component["name"],
        "symptom": rng.choice(component["symptoms"]),
        "action_taken": rng.choice(component["actions"]),
        "downtime_minutes": rng.randrange(0, 241, 5),
        "status": rng.choice(_STATUSES),
        "technician": rng.choice(_TECHNICIANS),
    }
    validate_record(record)
    log = rng.choice(_TEMPLATES).format(**record)
    _assert_log_contains_record(log, record)
    row_id = f"syn-{split}-{index:04d}"
    return {
        "id": row_id,
        "split": split,
        "synthetic": True,
        "dataset_label": DATASET_LABEL,
        "notice": DATASET_NOTICE,
        "generator": "maintlog.synthetic",
        "seed": seed,
        "log": log,
        "record": record,
        "prompt": build_prompt(log),
        "completion": build_completion(record),
    }


def _random_timestamp(rng: random.Random) -> str:
    start = datetime(2023, 1, 1)
    moment = start + timedelta(days=rng.randint(0, 699))
    moment = moment.replace(
        hour=rng.randint(0, 23),
        minute=rng.choice((0, 15, 30, 45)),
        second=0,
    )
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _assert_log_contains_record(log: str, record: dict[str, Any]) -> None:
    for key, value in record.items():
        token = str(value)
        if token not in log:
            raise ValueError(f"Synthetic log dropped {key} value {token!r}.")
