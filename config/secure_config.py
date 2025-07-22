#!/usr/bin/env python3
"""
GAJA Assistant - Secure Environment Configuration Manager
Zarządza bezpiecznym ładowaniem konfiguracji z zmiennych środowiskowych.
"""

import os
import sys
from pathlib import Path
from typing import Any

from loguru import logger


class SecureConfigManager:
    """Zarządza bezpiecznym ładowaniem konfiguracji."""

    def __init__(self, config_file: str = "server_config.json"):
        self.config_file = config_file
        self.required_env_vars = {
            "OPENAI_API_KEY": "OpenAI API key for AI services",
            "ANTHROPIC_API_KEY": "Anthropic API key for Claude models",
            "DEEPSEEK_API_KEY": "Deepseek API key for alternative AI models",
            "SECRET_KEY": "Secret key for JWT token signing",
        }
        self.optional_env_vars = {
            "AZURE_SPEECH_KEY": "Azure Speech Services API key",
            "GOOGLE_API_KEY": "Google Services API key",
            "WEATHER_API_KEY": "Weather service API key",
            "NEWS_API_KEY": "News service API key",
        }

    def load_environment_variables(self) -> dict[str, str]:
        """Ładuje zmienne środowiskowe z .env file."""
        env_file = Path(".env")
        if env_file.exists():
            self._load_dotenv(env_file)
            logger.info("Environment variables loaded from .env file")
        else:
            logger.warning(".env file not found, using system environment variables")

        return self._validate_environment()

    def _load_dotenv(self, env_file: Path) -> None:
        """Ładuje zmienne z pliku .env."""
        try:
            with open(env_file, encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()

                    # Pomiń komentarze i puste linie
                    if not line or line.startswith("#"):
                        continue

                    # Parsuj zmienną
                    if "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")

                        # Ustaw zmienną środowiskową
                        os.environ[key] = value
                    else:
                        logger.warning(f"Invalid line {line_num} in .env file: {line}")

        except Exception as e:
            logger.error(f"Error loading .env file: {e}")
            raise

    def _validate_environment(self) -> dict[str, str]:
        """Waliduje czy wszystkie wymagane zmienne są ustawione."""
        config = {}
        missing_vars = []

        # Sprawdź wymagane zmienne
        for var_name, description in self.required_env_vars.items():
            value = os.getenv(var_name)
            if not value or value.startswith("YOUR_") or value == "":
                missing_vars.append(f"{var_name}: {description}")
            else:
                config[var_name] = value

        # Sprawdź opcjonalne zmienne
        for var_name, _description in self.optional_env_vars.items():
            value = os.getenv(var_name)
            if value and not value.startswith("YOUR_") and value != "":
                config[var_name] = value

        # Wygeneruj SECRET_KEY jeśli nie jest ustawiony
        if "SECRET_KEY" not in config:
            import secrets

            secret_key = secrets.token_urlsafe(32)
            os.environ["SECRET_KEY"] = secret_key
            config["SECRET_KEY"] = secret_key
            logger.warning("Generated new SECRET_KEY - consider setting it permanently")

        if missing_vars:
            logger.error(
                "Missing required environment variables:\n"
                + "\n".join(f"  - {var}" for var in missing_vars)
            )
            self._create_env_template()
            raise ValueError(
                f"Missing {len(missing_vars)} required environment variables"
            )

        logger.info(
            f"Environment validation successful: {len(config)} variables loaded"
        )
        return config

    def _create_env_template(self) -> None:
        """Tworzy przykładowy plik .env.template."""
        template_file = Path(".env.template")

        if template_file.exists():
            return

        try:
            with open(template_file, "w", encoding="utf-8") as f:
                f.write("# GAJA Assistant Environment Variables\n")
                f.write("# Copy this file to .env and fill in your actual API keys\n\n")

                f.write("# Required Variables\n")
                for var_name, description in self.required_env_vars.items():
                    f.write(f"# {description}\n")
                    f.write(f"{var_name}=your_{var_name.lower()}_here\n\n")

                f.write("# Optional Variables\n")
                for var_name, description in self.optional_env_vars.items():
                    f.write(f"# {description}\n")
                    f.write(f"# {var_name}=your_{var_name.lower()}_here\n\n")

                f.write("# Production Settings\n")
                f.write("# GAJA_ENVIRONMENT=production\n")
                f.write("# GAJA_DEBUG=false\n")
                f.write("# GAJA_LOG_LEVEL=INFO\n")

            logger.info(f"Created environment template: {template_file}")

        except Exception as e:
            logger.error(f"Failed to create .env template: {e}")

    def mask_sensitive_value(self, key: str, value: str) -> str:
        """Maskuje wrażliwe wartości do logowania."""
        sensitive_patterns = [
            "key",
            "secret",
            "token",
            "password",
            "auth",
            "credential",
        ]

        if any(pattern in key.lower() for pattern in sensitive_patterns):
            if len(value) <= 8:
                return "***MASKED***"
            else:
                return f"{value[:4]}...{value[-4:]}"

        return value

    def get_config_summary(self) -> dict[str, Any]:
        """Zwraca podsumowanie konfiguracji z zamaskowanymi wartościami."""
        config = {}

        for var_name in self.required_env_vars.keys():
            value = os.getenv(var_name, "NOT_SET")
            config[var_name] = self.mask_sensitive_value(var_name, value)

        for var_name in self.optional_env_vars.keys():
            value = os.getenv(var_name)
            if value is not None:
                config[var_name] = self.mask_sensitive_value(var_name, value)

        return config

    def verify_api_keys(self) -> dict[str, bool]:
        """Weryfikuje czy klucze API mają właściwy format."""
        results = {}

        api_key_patterns = {
            "OPENAI_API_KEY": {"prefix": "sk-", "min_length": 20},
            "ANTHROPIC_API_KEY": {"prefix": "sk-ant-", "min_length": 20},
            "DEEPSEEK_API_KEY": {"prefix": "sk-", "min_length": 20},
        }

        for var_name, pattern in api_key_patterns.items():
            value = os.getenv(var_name)
            if value:
                is_valid = (
                    isinstance(pattern["prefix"], str)
                    and value.startswith(pattern["prefix"])
                    and len(value) >= pattern["min_length"]
                    and not value.startswith("YOUR_")
                )
                results[var_name] = is_valid

                if not is_valid:
                    logger.warning(f"API key {var_name} may have invalid format")
            else:
                results[var_name] = False

        return results


# Globalna instancja
secure_config = SecureConfigManager()


def load_secure_environment() -> dict[str, str]:
    """Główna funkcja do ładowania bezpiecznej konfiguracji."""
    return secure_config.load_environment_variables()


def get_api_key(provider: str) -> str | None:
    """Bezpiecznie pobiera klucz API dla danego providera."""
    key_mapping = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "azure": "AZURE_SPEECH_KEY",
        "google": "GOOGLE_API_KEY",
        "weather": "WEATHER_API_KEY",
        "news": "NEWS_API_KEY",
    }

    env_var = key_mapping.get(provider.lower())
    if not env_var:
        logger.warning(f"Unknown API provider: {provider}")
        return None

    api_key = os.getenv(env_var)
    if not api_key or api_key.startswith("YOUR_"):
        logger.warning(f"API key not configured for provider: {provider}")
        return None

    return api_key


if __name__ == "__main__":
    """Testuj konfigurację środowiska."""
    try:
        config = load_secure_environment()
        print("✅ Environment configuration loaded successfully")

        summary = secure_config.get_config_summary()
        print("\nConfiguration summary:")
        for key, value in summary.items():
            print(f"  {key}: {value}")

        verification = secure_config.verify_api_keys()
        print("\nAPI key verification:")
        for key, is_valid in verification.items():
            status = "✅ Valid" if is_valid else "❌ Invalid"
            print(f"  {key}: {status}")

    except Exception as e:
        print(f"❌ Configuration error: {e}")
        sys.exit(1)
