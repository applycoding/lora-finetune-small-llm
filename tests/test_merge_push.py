"""Hub push is refused without HF_TOKEN. No network and no weights."""

from __future__ import annotations

import pytest

from maintlog.merge_push import MissingTokenError, main, resolve_hf_token


def test_missing_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(MissingTokenError):
        resolve_hf_token()


def test_blank_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "   ")
    with pytest.raises(MissingTokenError):
        resolve_hf_token()


def test_main_stops_before_loading_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert main(["--config", "configs/merge.yaml"]) == 2
