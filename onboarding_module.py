"""Onboarding Module for GAJA Assistant Server Obsługa pierwszego uruchomienia i
konfiguracji wstępnej w architekturze klient-serwer."""

import json
import logging
from pathlib import Path
from typing import Any

import aiohttp
from config_loader import ConfigLoader
from config_manager import DatabaseManager

logger = logging.getLogger(__name__)


class OnboardingModule:
    """Moduł obsługujący pierwszy setup systemu."""

    def __init__(self, config_loader: ConfigLoader, db_manager: DatabaseManager):
        self.config_loader = config_loader
        self.db_manager = db_manager
        self.config_path = Path("server_config.json")
        logger.info("OnboardingModule initialized")

    async def is_first_run(self) -> bool:
        """Sprawdź czy to pierwsze uruchomienie systemu."""
        try:
            config = self.config_loader.get_config()
            return config.get("FIRST_RUN", True)
        except Exception as e:
            logger.error(f"Error checking first run status: {e}")
            return True

    async def get_location_from_ip(self) -> str:
        """Pobierz lokalizację na podstawie IP."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "http://ip-api.com/json/", timeout=5
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("status") == "success":
                            city = data.get("city", "Warsaw")
                            country_code = data.get("countryCode", "PL")
                            return f"{city},{country_code}"
        except Exception as e:
            logger.warning(f"Failed to get location from IP: {e}")

        return "Warsaw,PL"  # Default fallback

    async def get_onboarding_status(self, user_id: str) -> dict[str, Any]:
        """Pobierz status onboarding dla użytkownika."""
        try:
            config = self.config_loader.get_config()
            user_data = await self.db_manager.get_user_data(user_id)

            return {
                "first_run": config.get("FIRST_RUN", True),
                "user_configured": user_data is not None,
                "required_steps": [
                    "user_name",
                    "location",
                    "daily_briefing",
                    "api_keys",
                    "voice_settings",
                ],
                "completed_steps": (
                    user_data.get("onboarding_steps", []) if user_data else []
                ),
            }
        except Exception as e:
            logger.error(f"Error getting onboarding status: {e}")
            return {"error": str(e)}

    async def save_onboarding_step(
        self, user_id: str, step: str, data: dict[str, Any]
    ) -> dict[str, Any]:
        """Zapisz krok onboarding."""
        try:
            # Pobierz obecne dane użytkownika
            user_data = await self.db_manager.get_user_data(user_id) or {}

            # Dodaj krok do ukończonych
            completed_steps = user_data.get("onboarding_steps", [])
            if step not in completed_steps:
                completed_steps.append(step)

            # Aktualizuj dane użytkownika
            user_data["onboarding_steps"] = completed_steps
            user_data.update(data)

            # Zapisz do bazy danych
            await self.db_manager.save_user_data(user_id, user_data)

            # Aktualizuj konfigurację globalną
            await self._update_global_config(step, data)

            logger.info(f"Onboarding step '{step}' completed for user {user_id}")
            return {"success": True, "step": step, "completed_steps": completed_steps}
        except Exception as e:
            logger.error(f"Error saving onboarding step: {e}")
            return {"success": False, "error": str(e)}

    async def _update_global_config(self, step: str, data: dict[str, Any]):
        """Aktualizuj globalną konfigurację na podstawie kroku onboarding."""
        try:
            config = self.config_loader.get_config()

            if step == "user_name":
                config["USER_NAME"] = data.get("name", "User")

            elif step == "location":
                config["DEFAULT_LOCATION"] = data.get("location", "Warsaw,PL")

            elif step == "daily_briefing":
                if "daily_briefing" not in config:
                    config["daily_briefing"] = {}
                config["daily_briefing"].update(data.get("daily_briefing", {}))

            elif step == "api_keys":
                # Zapisz klucze API (ostrożnie z bezpieczeństwem)
                api_keys = data.get("api_keys", {})
                for service, key in api_keys.items():
                    if key and key.strip():
                        config[f"{service.upper()}_API_KEY"] = key.strip()

            elif step == "voice_settings":
                if "tts" not in config:
                    config["tts"] = {}
                config["tts"].update(data.get("voice_settings", {}))

            # Zapisz zaktualizowaną konfigurację
            self.config_loader.save_config(config)

        except Exception as e:
            logger.error(f"Error updating global config: {e}")

    async def complete_onboarding(self, user_id: str) -> dict[str, Any]:
        """Zakończ proces onboarding."""
        try:
            # Oznacz onboarding jako ukończony w konfiguracji
            config = self.config_loader.get_config()
            config["FIRST_RUN"] = False
            self.config_loader.save_config(config)

            # Oznacz w danych użytkownika
            user_data = await self.db_manager.get_user_data(user_id) or {}
            user_data["onboarding_completed"] = True
            user_data["onboarding_completed_at"] = json.dumps(
                {"timestamp": str(self.db_manager.get_current_time())}
            )
            await self.db_manager.save_user_data(user_id, user_data)

            logger.info(f"Onboarding completed for user {user_id}")
            return {
                "success": True,
                "message": "Onboarding completed successfully",
                "redirect_to": "/dashboard",
            }
        except Exception as e:
            logger.error(f"Error completing onboarding: {e}")
            return {"success": False, "error": str(e)}

    async def get_default_config_template(self) -> dict[str, Any]:
        """Pobierz szablon domyślnej konfiguracji dla onboarding."""
        location = await self.get_location_from_ip()

        return {
            "user_name": {
                "type": "text",
                "label": "Jak mam się do Ciebie zwracać?",
                "placeholder": "Twoje imię",
                "required": True,
            },
            "location": {
                "type": "text",
                "label": "Twoja lokalizacja",
                "placeholder": "Miasto, Kraj",
                "default": location,
                "required": True,
            },
            "daily_briefing": {
                "type": "object",
                "label": "Codzienny briefing",
                "fields": {
                    "enabled": {
                        "type": "checkbox",
                        "label": "Włącz codzienny briefing",
                        "default": True,
                    },
                    "time": {
                        "type": "time",
                        "label": "Godzina briefingu",
                        "default": "08:00",
                    },
                    "include_weather": {
                        "type": "checkbox",
                        "label": "Dołącz pogodę",
                        "default": True,
                    },
                    "include_news": {
                        "type": "checkbox",
                        "label": "Dołącz wiadomości",
                        "default": True,
                    },
                },
            },
            "api_keys": {
                "type": "object",
                "label": "Klucze API (opcjonalne)",
                "fields": {
                    "openai": {
                        "type": "password",
                        "label": "OpenAI API Key",
                        "placeholder": "sk-...",
                        "optional": True,
                    },
                    "weather": {
                        "type": "password",
                        "label": "Weather API Key",
                        "placeholder": "Klucz do serwisu pogodowego",
                        "optional": True,
                    },
                },
            },
            "voice_settings": {
                "type": "object",
                "label": "Ustawienia głosu",
                "fields": {
                    "voice_speed": {
                        "type": "range",
                        "label": "Prędkość mowy",
                        "min": 0.5,
                        "max": 2.0,
                        "step": 0.1,
                        "default": 1.0,
                    },
                    "voice_volume": {
                        "type": "range",
                        "label": "Głośność",
                        "min": 0.1,
                        "max": 1.0,
                        "step": 0.1,
                        "default": 0.8,
                    },
                },
            },
        }


# Global functions for function calling system
async def get_onboarding_status(user_id: str = "1") -> dict[str, Any]:
    """Pobierz status procesu onboarding."""
    # Import locally to avoid circular imports
    from server_main import server_app

    if hasattr(server_app, "onboarding_module"):
        return await server_app.onboarding_module.get_onboarding_status(user_id)
    return {"error": "Onboarding module not available"}


async def save_onboarding_step(user_id: str, step: str, data: str) -> dict[str, Any]:
    """Zapisz krok procesu onboarding."""
    # Import locally to avoid circular imports
    from server_main import server_app

    try:
        # Parse data if it's a JSON string
        if isinstance(data, str):
            data = json.loads(data)

        if hasattr(server_app, "onboarding_module"):
            return await server_app.onboarding_module.save_onboarding_step(
                user_id, step, data
            )
        return {"error": "Onboarding module not available"}
    except Exception as e:
        logger.error(f"Error in save_onboarding_step: {e}")
        return {"error": str(e)}


async def complete_onboarding(user_id: str = "1") -> dict[str, Any]:
    """Zakończ proces onboarding."""
    # Import locally to avoid circular imports
    from server_main import server_app

    if hasattr(server_app, "onboarding_module"):
        return await server_app.onboarding_module.complete_onboarding(user_id)
    return {"error": "Onboarding module not available"}


def get_functions():
    """Zwróć listę funkcji dostępnych w module onboarding."""
    return [
        {
            "name": "get_onboarding_status",
            "description": "Pobierz status procesu pierwszego uruchomienia systemu",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "ID użytkownika",
                        "default": "1",
                    }
                },
                "required": [],
            },
        },
        {
            "name": "save_onboarding_step",
            "description": "Zapisz krok procesu konfiguracji wstępnej",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string", "description": "ID użytkownika"},
                    "step": {
                        "type": "string",
                        "description": "Nazwa kroku (user_name, location, daily_briefing, api_keys, voice_settings)",
                    },
                    "data": {
                        "type": "string",
                        "description": "Dane kroku w formacie JSON",
                    },
                },
                "required": ["user_id", "step", "data"],
            },
        },
        {
            "name": "complete_onboarding",
            "description": "Zakończ proces pierwszego uruchomienia systemu",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "ID użytkownika",
                        "default": "1",
                    }
                },
                "required": [],
            },
        },
    ]
