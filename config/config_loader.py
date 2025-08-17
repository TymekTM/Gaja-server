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
            # Domyślny model zaktualizowany do gpt-5-nano (wcześniej gpt-4.1-nano)
            # Jeśli chcesz wymusić inny model ustaw zmienną środowiskową GAJA_AI_MODEL
            "model": "gpt-5-nano",
            "temperature": 0.7,
            "max_tokens": 1000,
        },
        "api_keys": {
            "openai": "YOUR_OPENAI_API_KEY_HERE",
            "anthropic": "YOUR_ANTHROPIC_API_KEY_HERE",
        },
        "plugins": {
            "auto_load": True,
            "default_enabled": ["weather_module_refactored", "search_module"],
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
        """Pobierz aktualną konfigurację.

        Zawsze zwraca dict – jeśli _config jest None, ładuje ponownie.
        """
        if self._config is None:
            self.load()
        # Defensive fallback
        if self._config is None:
            self._config = create_default_config()
        return self._config  # type: ignore[return-value]

    def save_config(self, config: dict[str, Any] | None = None):
        """Zapisz konfigurację.

        Args:
            config: Opcjonalnie nowy obiekt konfiguracji
        """
        if config is not None:
            self._config = config
        if self._config is None:
            self._config = create_default_config()
        save_config(self._config, self.config_file)

    def update_config(self, updates: dict[str, Any]):
        """Aktualizuj konfigurację."""
        if self._config is None:
            self.load()
        if self._config is None:  # still None -> create default
            self._config = create_default_config()
        self._config.update(updates)
        self.save_config()

    def get(self, key: str, default: Any = None):
        """Pobierz wartość z konfiguracji."""
        if self._config is None:
            self.load()
        if self._config is None:
            self._config = create_default_config()
        
        # Support dotted notation like 'app.name'
        if '.' in key:
            keys = key.split('.')
            value = self._config
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            return value
        
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Ustaw wartość w konfiguracji."""
        if self._config is None:
            self.load()
        if self._config is None:
            self._config = create_default_config()
        
        # Support dotted notation like 'app.name'
        if '.' in key:
            keys = key.split('.')
            config = self._config
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            config[keys[-1]] = value
        else:
            self._config[key] = value
    
    def has(self, key: str) -> bool:
        """Sprawdź czy klucz istnieje."""
        if self._config is None:
            self.load()
        if self._config is None:
            self._config = create_default_config()
        
        # Support dotted notation like 'app.name'
        if '.' in key:
            keys = key.split('.')
            value = self._config
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return False
            return True
        
        return key in self._config
    
    def delete(self, key: str):
        """Usuń klucz z konfiguracji."""
        if self._config is None:
            self.load()
        if self._config is None:
            self._config = create_default_config()
        
        # Support dotted notation like 'app.name'
        if '.' in key:
            keys = key.split('.')
            config = self._config
            for k in keys[:-1]:
                if isinstance(config, dict) and k in config:
                    config = config[k]
                else:
                    return
            if isinstance(config, dict) and keys[-1] in config:
                del config[keys[-1]]
        else:
            if key in self._config:
                del self._config[key]
    
    def to_dict(self) -> dict[str, Any]:
        """Zwróć konfigurację jako słownik."""
        if self._config is None:
            self.load()
        if self._config is None:
            self._config = create_default_config()
        return self._config.copy()
    
    def merge(self, other_config: dict[str, Any]):
        """Połącz z inną konfiguracją."""
        if self._config is None:
            self.load()
        if self._config is None:
            self._config = create_default_config()
        
        def deep_merge(base: dict, update: dict):
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = value
        
        deep_merge(self._config, other_config)
    
    def load_from_file(self, file_path: str):
        """Załaduj konfigurację z pliku."""
        self.config_file = file_path
        self.load()
    
    def save_to_file(self, file_path: str):
        """Zapisz konfigurację do pliku."""
        if self._config is None:
            self._config = create_default_config()
        save_config(self._config, file_path)
    
    def load_from_env(self, prefix: str = "GAJA_"):
        """Załaduj ustawienia ze zmiennych środowiskowych."""
        if self._config is None:
            self._config = create_default_config()
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower().replace('_', '.')
                # Try to convert to appropriate type
                if value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'
                elif value.isdigit():
                    value = int(value)
                else:
                    # Try to convert to float only if it's a valid float
                    try:
                        # Check if it's a valid float (single decimal point)
                        if '.' in value and value.count('.') == 1:
                            parts = value.split('.')
                            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                                value = float(value)
                    except ValueError:
                        pass  # Keep as string
                
                self.set(config_key, value)
    
    @property
    def config_data(self):
        """Get config data."""
        return self.get_config()
    
    @config_data.setter
    def config_data(self, value):
        """Set config data."""
        self._config = value


# Stare zmienne dla kompatybilności
_config = load_config()
STT_MODEL = _config.get("ai", {}).get("stt_model", "base")
# Pozwól nadpisać model zmienną środowiskową GAJA_AI_MODEL
_env_model = os.environ.get("GAJA_AI_MODEL")
_configured_model = _config.get("ai", {}).get("model", "gpt-5-nano")

# Backward compatibility: jeśli ktoś nadal ma stary wpis w configu "gpt-4.1-nano" zmieniamy na nowy
if _configured_model == "gpt-4.1-nano":
    _configured_model = "gpt-5-nano"

# Opcjonalny droższy model: jeśli ustawiono GAJA_USE_GPT5_MINI=1 i nie ustawiono ręcznie modelu, przełącz na gpt-5-mini
if (
    os.environ.get("GAJA_USE_GPT5_MINI") in {"1", "true", "True"}
    and not _env_model
):
    _configured_model = "gpt-5-mini"

MAIN_MODEL = _env_model or _configured_model
PROVIDER = _config.get("ai", {}).get("provider", "openai")
DEEP_MODEL = _config.get("ai", {}).get("deep_model", MAIN_MODEL)
