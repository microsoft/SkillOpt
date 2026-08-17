"""Tests for MiniMax service-region selection in the minimax_chat backend."""

from __future__ import annotations

import importlib
import os
from collections.abc import Iterator
from typing import Any

import pytest

from skillopt.model import minimax_backend

_GLOBAL_BASE_URL = "https://api.minimax.io/v1"
_CN_BASE_URL = "https://api.minimaxi.com/v1"
_ENV_KEYS = ("MINIMAX_REGION", "MINIMAX_BASE_URL")
_GLOBAL_KEYS = ("REGION", "BASE_URL")


@pytest.fixture(autouse=True)
def isolate_minimax_region() -> Iterator[None]:
    env_snapshot = {key: os.environ.get(key) for key in _ENV_KEYS}
    global_snapshot = {key: getattr(minimax_backend, key) for key in _GLOBAL_KEYS}
    try:
        yield
    finally:
        for key, value in global_snapshot.items():
            setattr(minimax_backend, key, value)
        for key, value in env_snapshot.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _reload_with_env(monkeypatch: pytest.MonkeyPatch, **env: str | None) -> Any:
    for key, value in env.items():
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)
    return importlib.reload(minimax_backend)


def test_supported_regions_expose_both_endpoints() -> None:
    assert minimax_backend.REGION_BASE_URLS == {
        "global_en": _GLOBAL_BASE_URL,
        "cn_zh": _CN_BASE_URL,
    }
    assert minimax_backend.DEFAULT_REGION == "global_en"


def test_blank_region_falls_back_to_global() -> None:
    assert minimax_backend.normalize_region(None) == "global_en"
    assert minimax_backend.normalize_region("  ") == "global_en"
    assert minimax_backend.base_url_for_region(None) == _GLOBAL_BASE_URL


def test_region_name_is_normalized() -> None:
    assert minimax_backend.normalize_region("CN_ZH") == "cn_zh"
    assert minimax_backend.normalize_region(" cn-zh ") == "cn_zh"
    assert minimax_backend.base_url_for_region("cn-zh") == _CN_BASE_URL


def test_unsupported_region_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported MiniMax region"):
        minimax_backend.normalize_region("apac")


def test_region_env_selects_base_url_at_import(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _reload_with_env(monkeypatch, MINIMAX_REGION="cn_zh", MINIMAX_BASE_URL=None)
    try:
        assert module.REGION == "cn_zh"
        assert module.BASE_URL == _CN_BASE_URL
    finally:
        monkeypatch.undo()
        importlib.reload(minimax_backend)


def test_explicit_base_url_env_wins_over_region(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _reload_with_env(
        monkeypatch,
        MINIMAX_REGION="cn_zh",
        MINIMAX_BASE_URL="https://proxy.internal/v1",
    )
    try:
        assert module.REGION == "cn_zh"
        assert module.BASE_URL == "https://proxy.internal/v1"
    finally:
        monkeypatch.undo()
        importlib.reload(minimax_backend)


def test_configure_region_switches_base_url() -> None:
    minimax_backend.configure_minimax_chat(region="cn_zh")
    assert minimax_backend.get_region() == "cn_zh"
    assert minimax_backend.get_base_url() == _CN_BASE_URL
    assert minimax_backend._chat_url() == f"{_CN_BASE_URL}/chat/completions"

    minimax_backend.configure_minimax_chat(region="global_en")
    assert minimax_backend.get_region() == "global_en"
    assert minimax_backend.get_base_url() == _GLOBAL_BASE_URL


def test_configure_base_url_overrides_region() -> None:
    minimax_backend.configure_minimax_chat(
        region="cn_zh",
        base_url="https://proxy.internal/v1",
    )
    assert minimax_backend.get_region() == "cn_zh"
    assert minimax_backend.get_base_url() == "https://proxy.internal/v1"


def test_configure_rejects_unsupported_region() -> None:
    before = minimax_backend.get_base_url()
    with pytest.raises(ValueError, match="Unsupported MiniMax region"):
        minimax_backend.configure_minimax_chat(region="apac")
    assert minimax_backend.get_base_url() == before
