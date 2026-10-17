"""Rule-based preview. It is not a model score."""

from __future__ import annotations

from maintlog.fallback import PREVIEW_NOTE, preview_parse
from maintlog.schema import REQUIRED_FIELDS
from maintlog.synthetic import generate_rows


def test_preview_recovers_generated_logs() -> None:
    for row in generate_rows(8, 2, 42):
        assert preview_parse(row["log"]) == row["record"]


def test_blank_log_is_all_null() -> None:
    parsed = preview_parse("   ")
    assert set(parsed) == set(REQUIRED_FIELDS)
    assert all(value is None for value in parsed.values())


def test_preview_note_names_what_it_is() -> None:
    assert "Rule-based preview" in PREVIEW_NOTE
    assert "not a fine-tuned model" in PREVIEW_NOTE
