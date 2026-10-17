"""The GitHub workflow stays on the local test command."""

from __future__ import annotations

from pathlib import Path


def test_ci_runs_pytest_and_does_not_push() -> None:
    text = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert "pytest" in text
    assert "merge_push" not in text
    assert "HF_TOKEN" not in text
