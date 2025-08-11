"""
environment_manager.py - Bezpieczne zarządzanie zmiennymi środowiskowymi i kluczami API
"""

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class EnvironmentManager:
    """Zarządza zmiennymi środowiskowymi i bezpiecznym ładowaniem konfiguracji."""

    def __init__(self, env_file: str = ".env"):
        self.env_file = Path(env_file)
        self._load_env_file()

    def _load_env_file(self):
        """Ładuje zmienne z pliku .env jeśli istnieje."""
        if self.env_file.exists():
            try:
                with open(self.env_file, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            key, value = line.split("=", 1)
                            # Don't override if already set in environment
                            if key not in os.environ:
                                os.environ[key] = value
                logger.info(f"Loaded environment variables from {self.env_file}")
            except Exception as e:
                logger.error(f"Error loading .env file: {e}")

    def get_api_key(self, service: str) -> str | None:
        """Pobiera klucz API dla danej usługi z zmiennych środowiskowych.

        Args:
            service: Nazwa usługi (openai, anthropic, etc.)

        Returns:
            Klucz API lub None jeśli nie znaleziono
        """
        # Standardize service name to uppercase
        env_key = f"{service.upper()}_API_KEY"

        # Special cases for legacy compatibility
        if service.lower() == "openai":
            return os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
        elif service.lower() == "azure":
            return os.getenv("AZURE_SPEECH_KEY") or os.getenv("AZURE_API_KEY")

        return os.getenv(env_key)

    def get_database_url(self) -> str:
        """Pobiera URL bazy danych z zmiennych środowiskowych."""
        return os.getenv("DATABASE_URL", "sqlite:///./gaja_assistant.db")

    def get_server_config(self) -> dict[str, Any]:
        """Pobiera konfigurację serwera z zmiennych środowiskowych."""
        return {
            "host": os.getenv("SERVER_HOST", "localhost"),
            "port": int(os.getenv("SERVER_PORT", "8001")),
            "secret_key": os.getenv("SECRET_KEY"),
            "cors_origins": os.getenv("CORS_ORIGINS", "http://localhost:3000").split(
                ","
            ),
            "max_connections_per_user": int(os.getenv("MAX_CONNECTIONS_PER_USER", "5")),
            "session_timeout_hours": int(os.getenv("SESSION_TIMEOUT_HOURS", "24")),
        }

    def sanitize_config_for_logging(self, config: dict[str, Any]) -> dict[str, Any]:
        """Czyści konfigurację z wrażliwych danych przed logowaniem.

        Args:
            config: Słownik konfiguracji

        Returns:
            Oczyszczony słownik bez wrażliwych danych
        """
        sanitized = {}
        sensitive_keys = {
            "api_key",
            "password",
            "secret",
            "token",
            "auth",
            "credential",
            "openai_api_key",
            "anthropic_api_key",
            "deepseek_api_key",
            "azure_speech_key",
            "together_api_key",
            "groq_api_key",
        }

        for key, value in config.items():
            key_lower = key.lower()

            # Check if key contains sensitive information
            if any(sensitive in key_lower for sensitive in sensitive_keys):
                # Show only first and last 4 characters for API keys
                if isinstance(value, str) and len(value) > 8:
                    sanitized[key] = f"{value[:4]}***{value[-4:]}"
                else:
                    sanitized[key] = "***HIDDEN***"
            elif isinstance(value, dict):
                # Recursively sanitize nested dictionaries
                sanitized[key] = self.sanitize_config_for_logging(value)
            else:
                sanitized[key] = value

        return sanitized

    def validate_required_keys(self, required_services: list[str]) -> dict[str, bool]:
        """Sprawdza czy wymagane klucze API są dostępne.

        Args:
            required_services: Lista wymaganych usług

        Returns:
            Słownik z wynikami walidacji
        """
        validation_results = {}

        for service in required_services:
            api_key = self.get_api_key(service)
            validation_results[service] = (
                api_key is not None and len(api_key.strip()) > 0
            )

            if not validation_results[service]:
                logger.warning(f"Missing or empty API key for {service}")

        return validation_results


# Global instance
env_manager = EnvironmentManager()
