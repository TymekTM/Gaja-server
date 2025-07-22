"""Onboarding Module for Plugin System Moduł onboarding dostępny przez system
pluginów."""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class OnboardingPluginModule:
    """Moduł onboarding jako plugin."""

    def __init__(self):
        logger.info("OnboardingPluginModule initialized")

    async def get_onboarding_status(self, user_id: str = "1") -> dict[str, Any]:
        """Pobierz status onboarding."""
        from server_main import server_app

        if hasattr(server_app, "onboarding_module"):
            return await server_app.onboarding_module.get_onboarding_status(user_id)
        return {"error": "Onboarding module not available"}

    async def save_onboarding_step(
        self, user_id: str, step: str, data: str
    ) -> dict[str, Any]:
        """Zapisz krok onboarding."""
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

    async def complete_onboarding(self, user_id: str = "1") -> dict[str, Any]:
        """Zakończ onboarding."""
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
