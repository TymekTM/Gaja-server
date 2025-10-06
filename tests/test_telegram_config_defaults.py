"""Tests for Telegram configuration defaults."""

from __future__ import annotations

import sys
import types

if "httpx" not in sys.modules:
    sys.modules["httpx"] = types.ModuleType("httpx")

if "loguru" not in sys.modules:
    dummy_loguru = types.ModuleType("loguru")

    class _Logger:
        def __getattr__(self, _name):  # pragma: no cover - simple stub
            def _noop(*_args, **_kwargs):
                return None

            return _noop

    dummy_loguru.logger = _Logger()
    sys.modules["loguru"] = dummy_loguru

import json

from config.config_loader import create_default_config, load_config
from integrations.telegram.service import load_telegram_config


def test_create_default_config_enables_telegram_integration():
    config = create_default_config()
    telegram_config = config["integrations"]["telegram"]

    assert telegram_config["enabled"] is True


def test_create_default_config_contains_default_chat_mapping():
    config = create_default_config()
    telegram_config = config["integrations"]["telegram"]

    assert telegram_config["chat_user_map"]["1092300083"] == "1"


def test_load_telegram_config_defaults_to_enabled():
    telegram_config = load_telegram_config({})

    assert telegram_config.enabled is True


def test_load_telegram_config_respects_disabled_flag():
    telegram_config = load_telegram_config({"enabled": False})

    assert telegram_config.enabled is False


def test_load_telegram_config_provides_default_user_mapping():
    telegram_config = load_telegram_config({})

    assert telegram_config.default_user_id == "1"
    assert telegram_config.resolve_user_id("123") == "1"


def test_load_telegram_config_uses_custom_default_user():
    telegram_config = load_telegram_config({"default_user_id": " 42 "})

    assert telegram_config.default_user_id == "42"
    assert telegram_config.resolve_user_id("999") == "42"


def test_load_telegram_config_respects_chat_user_map():
    base_config = create_default_config()["integrations"]["telegram"]
    telegram_config = load_telegram_config(base_config)

    assert telegram_config.resolve_user_id("1092300083") == "1"


def test_load_config_backfills_default_user_and_mapping(tmp_path, monkeypatch):
    legacy_config = {
        "integrations": {
            "telegram": {
                "default_user_id": "",
                "chat_user_map": {}
            }
        }
    }

    config_path = tmp_path / "server_config.json"
    config_path.write_text(json.dumps(legacy_config), encoding="utf-8")

    monkeypatch.setenv("GAJA_CONFIG_PATH", str(tmp_path))

    try:
        config = load_config("server_config.json")
    finally:
        monkeypatch.delenv("GAJA_CONFIG_PATH", raising=False)

    telegram_config = config["integrations"]["telegram"]

    assert telegram_config["default_user_id"] == "1"
    assert telegram_config["chat_user_map"]["1092300083"] == "1"
