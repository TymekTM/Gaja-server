"""config_loader.py Uproszczony loader konfiguracji dla serwera."""

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_config(config_file: str = "server_config.json") -> dict[str, Any]:
    """Ładuje konfigurację z pliku JSON i nadpisuje wartości zmiennymi środowiskowymi.

    Args:
        config_file: Nazwa pliku konfiguracyjnego

    Returns:
        Dict z konfiguracją
    """
    config_path = Path(config_file)

    # Jeśli plik nie istnieje, utwórz domyślną konfigurację
    if not config_path.exists():
        logger.warning(f"Config file {config_file} not found, creating default")
        default_config = create_default_config()
        save_config(default_config, config_file)
        config = default_config
    else:
        try:
            with open(config_path, encoding="utf-8") as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_file}")
        except Exception as e:
            logger.error(f"Error loading config {config_file}: {e}")
            config = create_default_config()

    # Nadpisz wartości zmiennymi środowiskowymi
    if "GAJA_HOST" in os.environ:
        config.setdefault("server", {})["host"] = os.environ["GAJA_HOST"]
    if "GAJA_PORT" in os.environ:
        config.setdefault("server", {})["port"] = int(os.environ["GAJA_PORT"])

    return config


def save_config(config: dict[str, Any], config_file: str = "server_config.json"):
    """Zapisuje konfigurację do pliku."""
    try:
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        logger.info(f"Saved configuration to {config_file}")
    except Exception as e:
        logger.error(f"Error saving config {config_file}: {e}")


def create_default_config() -> dict[str, Any]:
    """Tworzy domyślną konfigurację serwera."""
    return {
        "server": {"host": "0.0.0.0", "port": 8000, "debug": False},
        "database": {"url": "sqlite:///./server_data.db", "echo": False},
        "ai": {
            "provider": "openai",
            "model": "gpt-4.1-nano",
            "temperature": 0.7,
            "max_tokens": 1000,
        },
        "api_keys": {
            "openai": "YOUR_OPENAI_API_KEY_HERE",
            "anthropic": "YOUR_ANTHROPIC_API_KEY_HERE",
        },
        "plugins": {
            "auto_load": True,
            "default_enabled": ["weather_module", "search_module"],
        },
        "logging": {"level": "INFO", "file": "logs/server_{time:YYYY-MM-DD}.log"},
        "ui_language": "en",
    }


class ConfigLoader:
    """Klasa do zarządzania konfiguracją serwera."""

    def __init__(self, config_file: str = "server_config.json"):
        self.config_file = config_file
        self._config = None
        self.load()

    def load(self) -> dict[str, Any]:
        """Załaduj konfigurację z pliku."""
        self._config = load_config(self.config_file)
        return self._config

    def get_config(self) -> dict[str, Any]:
        """Pobierz aktualną konfigurację."""
        if self._config is None:
            self.load()
        return self._config

    def save_config(self, config: dict[str, Any] = None):
        """Zapisz konfigurację."""
        if config is not None:
            self._config = config
        save_config(self._config, self.config_file)

    def update_config(self, updates: dict[str, Any]):
        """Aktualizuj konfigurację."""
        if self._config is None:
            self.load()
        self._config.update(updates)
        self.save_config()

    def get(self, key: str, default: Any = None):
        """Pobierz wartość z konfiguracji."""
        if self._config is None:
            self.load()
        return self._config.get(key, default)


# Stare zmienne dla kompatybilności
_config = load_config()
STT_MODEL = _config.get("ai", {}).get("stt_model", "base")
MAIN_MODEL = _config.get("ai", {}).get("model", "gpt-4.1-nano")
PROVIDER = _config.get("ai", {}).get("provider", "openai")
DEEP_MODEL = _config.get("ai", {}).get("deep_model", "gpt-4.1-nano")
