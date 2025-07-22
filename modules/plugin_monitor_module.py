"""Plugin Monitor Module for Plugin System Moduł monitorowania pluginów dostępny przez
system pluginów."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class PluginMonitorModule:
    """Moduł monitorowania pluginów jako plugin."""

    def __init__(self):
        logger.info("PluginMonitorModule initialized")

    async def start_plugin_monitoring(self) -> dict[str, Any]:
        """Uruchom monitorowanie pluginów."""
        from server_main import server_app

        try:
            if hasattr(server_app, "plugin_monitor"):
                success = await server_app.plugin_monitor.start_monitoring()
                return {
                    "success": success,
                    "message": (
                        "Plugin monitoring started"
                        if success
                        else "Failed to start plugin monitoring"
                    ),
                }
            return {"success": False, "error": "Plugin monitor not available"}
        except Exception as e:
            logger.error(f"Error starting plugin monitoring: {e}")
            return {"success": False, "error": str(e)}

    async def stop_plugin_monitoring(self) -> dict[str, Any]:
        """Zatrzymaj monitorowanie pluginów."""
        from server_main import server_app

        try:
            if hasattr(server_app, "plugin_monitor"):
                await server_app.plugin_monitor.stop_monitoring()
                return {"success": True, "message": "Plugin monitoring stopped"}
            return {"success": False, "error": "Plugin monitor not available"}
        except Exception as e:
            logger.error(f"Error stopping plugin monitoring: {e}")
            return {"success": False, "error": str(e)}

    async def get_plugin_monitoring_status(self) -> dict[str, Any]:
        """Pobierz status monitorowania pluginów."""
        from server_main import server_app

        try:
            if hasattr(server_app, "plugin_monitor"):
                status = await server_app.plugin_monitor.get_monitoring_status()
                return {"success": True, **status}
            return {"success": False, "error": "Plugin monitor not available"}
        except Exception as e:
            logger.error(f"Error getting plugin monitoring status: {e}")
            return {"success": False, "error": str(e)}

    async def reload_plugin(self, plugin_name: str) -> dict[str, Any]:
        """Ręcznie przeładuj plugin."""
        from server_main import server_app

        try:
            if hasattr(server_app, "plugin_monitor"):
                result = await server_app.plugin_monitor.reload_plugin_manually(
                    plugin_name
                )
                return result
            return {"success": False, "error": "Plugin monitor not available"}
        except Exception as e:
            logger.error(f"Error reloading plugin {plugin_name}: {e}")
            return {"success": False, "error": str(e)}


def get_functions():
    """Zwróć listę funkcji dostępnych w module monitorowania pluginów."""
    return [
        {
            "name": "start_plugin_monitoring",
            "description": "Uruchom automatyczne monitorowanie i przeładowywanie pluginów",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "stop_plugin_monitoring",
            "description": "Zatrzymaj automatyczne monitorowanie pluginów",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "get_plugin_monitoring_status",
            "description": "Pobierz status systemu monitorowania pluginów i statystyki",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "reload_plugin",
            "description": "Ręcznie przeładuj konkretny plugin",
            "parameters": {
                "type": "object",
                "properties": {
                    "plugin_name": {
                        "type": "string",
                        "description": "Nazwa pluginu do przeładowania (bez .py)",
                    }
                },
                "required": ["plugin_name"],
            },
        },
    ]
