"""JSON validity and field accuracy. These check the math, not a trained model."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from maintlog.metrics import extract_json_object, score_predictions
from maintlog.schema import REQUIRED_FIELDS, canonical_json


def _gold() -> dict:
    return {
        "asset_id": "PUMP-1000",
        "site": "North Plant",
        "reported_at": "2024-03-12T08:15:00Z",
        "severity": "high",
        "component": "bearing",
        "symptom": "grinding noise and heat at the housing",
        "action_taken": "replaced the part and verified rotation",
        "downtime_minutes": 45,
        "status": "resolved",
        "technician": "A. Okonkwo",
    }


def test_exact_object_scores_one() -> None:
    gold = _gold()
    scored = score_predictions([gold], [canonical_json(gold)])
    assert scored["n"] == 1
    assert scored["json_validity"] == 1.0
    assert scored["field_accuracy"] == 1.0
    assert scored["per_field_accuracy"]["asset_id"] == 1.0


def test_invalid_json_scores_zero() -> None:
    scored = score_predictions([_gold()], ["not json"])
    assert scored["json_valid_count"] == 0
    assert scored["json_validity"] == 0.0
    assert scored["field_accuracy"] == 0.0
    assert all(value == 0.0 for value in scored["per_field_accuracy"].values())


def test_one_wrong_field_on_two_examples() -> None:
    gold = _gold()
    wrong = dict(gold)
    wrong["asset_id"] = "FAN-1000"
    scored = score_predictions([gold, gold], [canonical_json(gold), canonical_json(wrong)])
    assert scored["json_validity"] == 1.0
    assert scored["per_field_accuracy"]["asset_id"] == 0.5
    assert scored["per_field_accuracy"]["site"] == 1.0
    assert scored["field_accuracy"] == pytest.approx((1.0 + 0.9) / 2)


def test_prose_around_json_and_numeric_downtime() -> None:
    gold = _gold()
    payload = dict(gold)
    payload["severity"] = "High"
    payload["downtime_minutes"] = "45"
    text = "Here is the record:\n" + json.dumps(payload) + "\nDone."
    parsed = extract_json_object(text)
    assert parsed is not None
    assert parsed["severity"] == "High"
    scored = score_predictions([gold], [text])
    assert scored["json_validity"] == 1.0
    assert scored["field_accuracy"] == 1.0


def test_fenced_json_and_non_object() -> None:
    gold = _gold()
    fenced = "```json\n" + canonical_json(gold) + "\n```"
    assert score_predictions([gold], [fenced])["field_accuracy"] == 1.0
    assert extract_json_object("[1, 2, 3]") is None
    assert score_predictions([gold], ["[1, 2, 3]"])["json_validity"] == 0.0


def test_missing_field_is_a_miss() -> None:
    gold = _gold()
    partial = {key: value for key, value in gold.items() if key != "technician"}
    scored = score_predictions([gold], [json.dumps(partial)])
    assert scored["json_validity"] == 1.0
    assert scored["per_field_accuracy"]["technician"] == 0.0
    assert scored["field_accuracy"] == pytest.approx((len(REQUIRED_FIELDS) - 1) / len(REQUIRED_FIELDS))


def test_empty_prediction_list_is_rejected() -> None:
    with pytest.raises(ValueError):
        score_predictions([], [])


def test_results_file_is_an_empty_placeholder() -> None:
    payload = json.loads(Path("results/eval_results.json").read_text(encoding="utf-8"))
    assert payload["status"] == "placeholder"
    assert payload["json_validity"] is None
    assert payload["field_accuracy"] is None
    assert set(payload["per_field_accuracy"]) == set(REQUIRED_FIELDS)
    assert all(value is None for value in payload["per_field_accuracy"].values())
