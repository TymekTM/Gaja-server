"""Plugin Monitor System for GAJA Assistant Server Automatyczne monitorowanie i
przeładowywanie pluginów."""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

from plugin_manager import plugin_manager
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger(__name__)


class PluginFileHandler(FileSystemEventHandler):
    """Handler dla zdarzeń systemu plików dotyczących pluginów."""

    def __init__(self, plugin_monitor: "PluginMonitor"):
        self.plugin_monitor = plugin_monitor
        self.last_reload_time: dict[str, float] = {}
        self.reload_delay = (
            2.0  # Sekundy opóźnienia dla uniknięcia wielokrotnych przeładowań
        )

    def on_modified(self, event):
        """Obsługa modyfikacji pliku pluginu."""
        if event.is_directory:
            return

        file_path = Path(event.src_path)

        # Sprawdź czy to plik Pythona w katalogu modules
        if file_path.suffix == ".py" and "modules" in str(file_path):
            current_time = time.time()
            last_time = self.last_reload_time.get(str(file_path), 0)

            # Uniknij wielokrotnych przeładowań
            if current_time - last_time > self.reload_delay:
                self.last_reload_time[str(file_path)] = current_time
                self.plugin_monitor.schedule_reload(file_path)

    def on_created(self, event):
        """Obsługa utworzenia nowego pliku pluginu."""
        if event.is_directory:
            return

        file_path = Path(event.src_path)
        if file_path.suffix == ".py" and "modules" in str(file_path):
            self.plugin_monitor.schedule_reload(file_path, is_new=True)

    def on_deleted(self, event):
        """Obsługa usunięcia pliku pluginu."""
        if event.is_directory:
            return

        file_path = Path(event.src_path)
        if file_path.suffix == ".py" and "modules" in str(file_path):
            self.plugin_monitor.schedule_unload(file_path)


class PluginMonitor:
    """System monitorowania i automatycznego przeładowywania pluginów."""

    def __init__(self, modules_path: str = "modules"):
        # Ensure absolute path relative to server directory
        if not Path(modules_path).is_absolute():
            server_dir = Path(__file__).parent  # server directory
            self.modules_path = server_dir / modules_path
        else:
            self.modules_path = Path(modules_path)

        self.observer = None
        self.running = False
        self.reload_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.unload_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.reload_task = None

        # Statistics
        self.stats: dict[str, Any] = {
            "reloads_count": 0,
            "successful_reloads": 0,
            "failed_reloads": 0,
            "last_reload_time": None,
            "monitored_files": set(),
            "active_plugins": set(),
        }

        logger.info(f"PluginMonitor initialized for path: {self.modules_path}")

    async def start_monitoring(self):
        """Rozpocznij monitorowanie plików pluginów."""
        if self.running:
            logger.warning("Plugin monitoring already running")
            return False

        try:
            # Sprawdź czy katalog istnieje
            if not self.modules_path.exists():
                logger.error(f"Modules directory not found: {self.modules_path}")
                return False

            # Inicjalizuj obserwator plików
            self.observer = Observer()
            event_handler = PluginFileHandler(self)

            self.observer.schedule(
                event_handler, str(self.modules_path), recursive=True
            )

            # Uruchom obserwator
            self.observer.start()

            # Uruchom task przeładowywania
            self.reload_task = asyncio.create_task(self._reload_worker())

            self.running = True

            # Skanuj istniejące pliki
            await self._initial_scan()

            logger.info("Plugin monitoring started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start plugin monitoring: {e}")
            return False

    async def stop_monitoring(self):
        """Zatrzymaj monitorowanie pluginów."""
        if not self.running:
            return

        self.running = False

        try:
            # Zatrzymaj obserwator plików
            if self.observer:
                self.observer.stop()
                self.observer.join()
                self.observer = None

            # Zatrzymaj task przeładowywania
            if self.reload_task:
                self.reload_task.cancel()
                try:
                    await self.reload_task
                except asyncio.CancelledError:
                    pass
                self.reload_task = None

            logger.info("Plugin monitoring stopped")

        except Exception as e:
            logger.error(f"Error stopping plugin monitoring: {e}")

    def schedule_reload(self, file_path: Path, is_new: bool = False):
        """Zaplanuj przeładowanie pluginu."""
        try:
            plugin_name = file_path.stem
            if plugin_name.startswith("__"):  # Ignoruj __init__.py i __pycache__
                return

            logger.info(
                f"Scheduling {'new plugin' if is_new else 'reload'} for: {plugin_name}"
            )

            # Dodaj do kolejki przeładowań
            asyncio.create_task(
                self.reload_queue.put(
                    {
                        "action": "reload",
                        "plugin_name": plugin_name,
                        "file_path": str(file_path),
                        "is_new": is_new,
                        "timestamp": time.time(),
                    }
                )
            )

        except Exception as e:
            logger.error(f"Error scheduling reload for {file_path}: {e}")

    def schedule_unload(self, file_path: Path):
        """Zaplanuj odładowanie pluginu."""
        try:
            plugin_name = file_path.stem
            if plugin_name.startswith("__"):
                return

            logger.info(f"Scheduling unload for: {plugin_name}")

            asyncio.create_task(
                self.unload_queue.put(
                    {
                        "action": "unload",
                        "plugin_name": plugin_name,
                        "file_path": str(file_path),
                        "timestamp": time.time(),
                    }
                )
            )

        except Exception as e:
            logger.error(f"Error scheduling unload for {file_path}: {e}")

    async def _reload_worker(self):
        """Worker do przeładowywania pluginów."""
        while self.running:
            try:
                # Sprawdź kolejkę przeładowań
                try:
                    reload_task = await asyncio.wait_for(
                        self.reload_queue.get(), timeout=1.0
                    )
                    await self._handle_reload(reload_task)
                except TimeoutError:
                    pass

                # Sprawdź kolejkę odładowań
                try:
                    unload_task = await asyncio.wait_for(
                        self.unload_queue.get(), timeout=0.1
                    )
                    await self._handle_unload(unload_task)
                except TimeoutError:
                    pass

            except Exception as e:
                logger.error(f"Error in reload worker: {e}")
                await asyncio.sleep(1)

    async def _handle_reload(self, task: dict[str, Any]):
        """Obsłuż przeładowanie pluginu."""
        plugin_name = task["plugin_name"]
        is_new = task.get("is_new", False)

        try:
            self.stats["reloads_count"] = self.stats.get("reloads_count", 0) + 1

            logger.info(
                f"{'Loading new' if is_new else 'Reloading'} plugin: {plugin_name}"
            )

            # Jeśli to istniejący plugin, najpierw go odładuj
            if not is_new and plugin_manager.is_plugin_loaded(plugin_name):
                await self._unload_plugin(plugin_name)

            # Załaduj/przeładuj plugin
            success = await self._load_plugin(plugin_name)

            if success:
                self.stats["successful_reloads"] = (
                    self.stats.get("successful_reloads", 0) + 1
                )
                active_plugins = self.stats.get("active_plugins", set())
                if isinstance(active_plugins, set):
                    active_plugins.add(plugin_name)
                    self.stats["active_plugins"] = active_plugins
                logger.info(
                    f"Successfully {'loaded' if is_new else 'reloaded'} plugin: {plugin_name}"
                )
            else:
                self.stats["failed_reloads"] = self.stats.get("failed_reloads", 0) + 1
                logger.error(
                    f"Failed to {'load' if is_new else 'reload'} plugin: {plugin_name}"
                )

            self.stats["last_reload_time"] = time.time()

        except Exception as e:
            self.stats["failed_reloads"] = self.stats.get("failed_reloads", 0) + 1
            logger.error(
                f"Error {'loading' if is_new else 'reloading'} plugin {plugin_name}: {e}"
            )

    async def _handle_unload(self, task: dict[str, Any]):
        """Obsłuż odładowanie pluginu."""
        plugin_name = task["plugin_name"]

        try:
            logger.info(f"Unloading plugin: {plugin_name}")
            await self._unload_plugin(plugin_name)

            active_plugins = self.stats.get("active_plugins", set())
            if isinstance(active_plugins, set) and plugin_name in active_plugins:
                active_plugins.remove(plugin_name)
                self.stats["active_plugins"] = active_plugins

            logger.info(f"Successfully unloaded plugin: {plugin_name}")

        except Exception as e:
            logger.error(f"Error unloading plugin {plugin_name}: {e}")

    async def _load_plugin(self, plugin_name: str) -> bool:
        """Załaduj plugin."""
        try:
            # Użyj plugin_manager do załadowania
            result = await plugin_manager.load_plugin(plugin_name)
            return result
        except Exception as e:
            logger.error(f"Error loading plugin {plugin_name}: {e}")
            return False

    async def _unload_plugin(self, plugin_name: str) -> bool:
        """Odładuj plugin."""
        try:
            # Skorzystaj z plugin_manager do odładowania
            await plugin_manager.unload_plugin(plugin_name)

            # Usuń z sys.modules jeśli jest
            import sys

            module_key = f"modules.{plugin_name}"
            if module_key in sys.modules:
                del sys.modules[module_key]

            logger.info(f"Unloaded plugin: {plugin_name}")
            return True

        except Exception as e:
            logger.error(f"Error unloading plugin {plugin_name}: {e}")
            return False

    async def _initial_scan(self):
        """Skanuj istniejące pliki pluginów."""
        try:
            for file_path in self.modules_path.glob("*.py"):
                if not file_path.name.startswith("__"):
                    self.stats["monitored_files"].add(str(file_path))

                    # Sprawdź czy plugin jest już załadowany
                    plugin_name = file_path.stem
                    if plugin_manager.is_plugin_loaded(plugin_name):
                        self.stats["active_plugins"].add(plugin_name)

            logger.info(
                f"Initial scan complete. Monitoring {len(self.stats['monitored_files'])} files, "
                f"{len(self.stats['active_plugins'])} active plugins"
            )

        except Exception as e:
            logger.error(f"Error during initial scan: {e}")

    async def get_monitoring_status(self) -> dict[str, Any]:
        """Pobierz status monitorowania."""
        active_plugins = self.stats.get("active_plugins", set())
        monitored_files = self.stats.get("monitored_files", set())

        return {
            "running": self.running,
            "modules_path": str(self.modules_path),
            "stats": {
                **self.stats,
                "active_plugins": list(active_plugins)
                if isinstance(active_plugins, set)
                else [],
                "monitored_files": list(monitored_files)
                if isinstance(monitored_files, set)
                else [],
            },
            "observer_alive": (
                self.observer.is_alive() if self.observer is not None else False
            ),
        }

    async def reload_plugin_manually(self, plugin_name: str) -> dict[str, Any]:
        """Ręcznie przeładuj plugin."""
        try:
            if not self.running:
                return {"success": False, "error": "Plugin monitoring not running"}

            file_path = self.modules_path / f"{plugin_name}.py"
            if not file_path.exists():
                return {
                    "success": False,
                    "error": f"Plugin file not found: {file_path}",
                }

            self.schedule_reload(file_path, is_new=False)

            return {
                "success": True,
                "message": f"Reload scheduled for plugin: {plugin_name}",
            }

        except Exception as e:
            logger.error(f"Error manually reloading plugin {plugin_name}: {e}")
            return {"success": False, "error": str(e)}


# Global instance
plugin_monitor = PluginMonitor()


# Global functions for function calling system
async def start_plugin_monitoring() -> dict[str, Any]:
    """Uruchom monitorowanie pluginów."""
    try:
        success = await plugin_monitor.start_monitoring()
        return {
            "success": success,
            "message": (
                "Plugin monitoring started"
                if success
                else "Failed to start plugin monitoring"
            ),
        }
    except Exception as e:
        logger.error(f"Error starting plugin monitoring: {e}")
        return {"success": False, "error": str(e)}


async def stop_plugin_monitoring() -> dict[str, Any]:
    """Zatrzymaj monitorowanie pluginów."""
    try:
        await plugin_monitor.stop_monitoring()
        return {"success": True, "message": "Plugin monitoring stopped"}
    except Exception as e:
        logger.error(f"Error stopping plugin monitoring: {e}")
        return {"success": False, "error": str(e)}


async def get_plugin_monitoring_status() -> dict[str, Any]:
    """Pobierz status monitorowania pluginów."""
    try:
        status = await plugin_monitor.get_monitoring_status()
        return {"success": True, **status}
    except Exception as e:
        logger.error(f"Error getting plugin monitoring status: {e}")
        return {"success": False, "error": str(e)}


async def reload_plugin(plugin_name: str) -> dict[str, Any]:
    """Ręcznie przeładuj plugin."""
    try:
        result = await plugin_monitor.reload_plugin_manually(plugin_name)
        return result
    except Exception as e:
        logger.error(f"Error reloading plugin {plugin_name}: {e}")
        return {"success": False, "error": str(e)}


def get_functions():
    """Zwróć listę funkcji dostępnych w systemie monitorowania pluginów."""
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
