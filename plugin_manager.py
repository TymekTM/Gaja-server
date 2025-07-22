"""Enhanced Plugin Manager for GAJA Assistant Server Manages dynamic loading/unloading
of plugins per user."""

import importlib
import importlib.util
import inspect
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger
from plugin_protocol import PluginProtocol


@dataclass
class PluginInfo:
    """Information about a plugin."""

    name: str
    module_path: str
    description: str
    version: str
    author: str
    functions: list[str]
    dependencies: list[str]
    enabled: bool = False
    loaded: bool = False
    module: PluginProtocol | None = None


class PluginManager:
    """Enhanced plugin manager with per-user control."""

    def __init__(self, plugins_directory: str = "modules"):
        # Handle both relative and absolute paths
        if not Path(plugins_directory).is_absolute():
            # If we're running from the main directory, plugins are in server/modules
            current_file_dir = Path(__file__).parent
            self.plugins_directory = current_file_dir / plugins_directory
        else:
            self.plugins_directory = Path(plugins_directory)

        self.plugins: dict[str, PluginInfo] = {}
        self.user_plugins: dict[
            str, dict[str, bool]
        ] = {}  # user_id -> {plugin_name: enabled}
        self.function_registry: dict[str, Callable] = {}
        # Create plugins directory if it doesn't exist
        self.plugins_directory.mkdir(exist_ok=True)

    async def discover_plugins(self) -> list[PluginInfo]:
        """Discover all available plugins."""
        discovered = []

        try:
            # Scan modules directory
            for file_path in self.plugins_directory.glob("*.py"):
                if file_path.name.startswith("__"):
                    continue

                plugin_info = await self._analyze_plugin(file_path)
                if plugin_info:
                    discovered.append(plugin_info)
                    self.plugins[plugin_info.name] = plugin_info

            logger.info(f"Discovered {len(discovered)} plugins")
            return discovered

        except Exception as e:
            logger.error(f"Error discovering plugins: {e}")
            return []

    async def _analyze_plugin(self, file_path: Path) -> PluginInfo | None:
        """Analyze a plugin file to extract metadata."""
        try:
            module_name = file_path.stem

            # Try to load module temporarily to analyze
            try:
                # Use proper module name with package structure
                full_module_name = f"modules.{module_name}"

                spec = importlib.util.spec_from_file_location(
                    full_module_name, file_path
                )
                if not spec or not spec.loader:
                    return None

                module = importlib.util.module_from_spec(spec)

                # Set package for relative imports
                module.__package__ = "modules"

                spec.loader.exec_module(module)

                # Extract metadata
                name = getattr(module, "PLUGIN_NAME", module_name)
                description = getattr(module, "PLUGIN_DESCRIPTION", "No description")
                version = getattr(module, "PLUGIN_VERSION", "1.0.0")
                author = getattr(module, "PLUGIN_AUTHOR", "Unknown")
                dependencies = getattr(module, "PLUGIN_DEPENDENCIES", [])

                # Find available functions
                functions = []
                if hasattr(module, "get_functions"):
                    try:
                        available_functions = module.get_functions()
                        if isinstance(available_functions, list):
                            functions = [
                                func.get("name", f"function_{i}")
                                for i, func in enumerate(available_functions)
                            ]
                        else:
                            functions = list(available_functions.keys())
                    except Exception as e:
                        logger.warning(f"Error getting functions from {name}: {e}")

            except Exception as e:
                logger.warning(
                    f"Could not import plugin {module_name} (dependencies missing): {e}"
                )
                # Fallback: Use basic file analysis
                name = module_name
                description = f"Plugin {module_name} (import failed)"
                version = "1.0.0"
                author = "Unknown"
                dependencies = []
                functions = []

                # Try to detect functions by reading file text
                try:
                    with open(file_path, encoding="utf-8") as f:
                        content = f.read()
                        # Look for get_functions definition
                        if "def get_functions(" in content:
                            functions = ["get_functions_detected"]
                except Exception:
                    pass

            # Create plugin info
            plugin_info = PluginInfo(
                name=name,
                module_path=str(file_path),
                description=description,
                version=version,
                author=author,
                functions=functions,
                dependencies=dependencies,
            )

            logger.debug(f"Analyzed plugin: {name} ({len(functions)} functions)")
            return plugin_info

        except Exception as e:
            logger.error(f"Error analyzing plugin {file_path}: {e}")
            return None

    async def load_plugin(self, plugin_name: str) -> bool:
        """Load a plugin into memory."""
        if plugin_name not in self.plugins:
            logger.error(f"Plugin {plugin_name} not found")
            return False

        plugin_info = self.plugins[plugin_name]

        if plugin_info.loaded:
            logger.debug(f"Plugin {plugin_name} already loaded")
            return True

        try:
            # Check dependencies
            for dep in plugin_info.dependencies:
                if not self._check_dependency(dep):
                    logger.error(f"Missing dependency for {plugin_name}: {dep}")
                    return False
            # Load module
            # Add parent directory to sys.path for relative imports
            # Use proper module name with package structure
            full_module_name = f"modules.{plugin_name}"

            spec = importlib.util.spec_from_file_location(
                full_module_name, plugin_info.module_path
            )
            if not spec or not spec.loader:
                return False

            module = importlib.util.module_from_spec(spec)

            # Set package for relative imports
            module.__package__ = "modules"

            spec.loader.exec_module(module)

            # Store module reference
            plugin_info.module = module
            plugin_info.loaded = True

            # Register functions
            if hasattr(module, "get_functions"):
                functions = module.get_functions()
                # Handle list format (current implementation)
                if isinstance(functions, list):
                    for func_info in functions:
                        func_name = func_info.get("name")
                        if func_name:
                            full_name = f"{plugin_name}.{func_name}"
                            # Store function info for later use
                            self.function_registry[full_name] = func_info
                            logger.debug(f"Registered function: {full_name}")
                # Handle dict format (legacy)
                elif isinstance(functions, dict):
                    for func_name, func in functions.items():
                        full_name = f"{plugin_name}.{func_name}"
                        self.function_registry[full_name] = func
                        logger.debug(f"Registered function: {full_name}")

            logger.info(f"Plugin {plugin_name} loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Error loading plugin {plugin_name}: {e}")
            plugin_info.loaded = False
            return False

    async def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin from memory."""
        if plugin_name not in self.plugins:
            return False

        plugin_info = self.plugins[plugin_name]

        if not plugin_info.loaded:
            return True

        try:
            # Unregister functions
            functions_to_remove = [
                func_name
                for func_name in self.function_registry.keys()
                if func_name.startswith(f"{plugin_name}.")
            ]

            for func_name in functions_to_remove:
                del self.function_registry[func_name]
                logger.debug(f"Unregistered function: {func_name}")

            # Clear module reference
            plugin_info.module = None
            plugin_info.loaded = False

            logger.info(f"Plugin {plugin_name} unloaded successfully")
            return True

        except Exception as e:
            logger.error(f"Error unloading plugin {plugin_name}: {e}")
            return False

    def _check_dependency(self, dependency: str) -> bool:
        """Check if a dependency is available."""
        try:
            importlib.import_module(dependency)
            return True
        except ImportError:
            return False

    async def enable_plugin_for_user(self, user_id: str, plugin_name: str) -> bool:
        """Enable a plugin for a specific user."""
        if plugin_name not in self.plugins:
            logger.error(f"Plugin {plugin_name} not found")
            return False

        # Initialize user plugin settings if needed
        if user_id not in self.user_plugins:
            self.user_plugins[user_id] = {}

        # Load plugin if not loaded
        if not self.plugins[plugin_name].loaded:
            if not await self.load_plugin(plugin_name):
                return False

        # Enable for user
        self.user_plugins[user_id][plugin_name] = True
        self.plugins[plugin_name].enabled = True

        logger.info(f"Plugin {plugin_name} enabled for user {user_id}")
        return True

    async def disable_plugin_for_user(self, user_id: str, plugin_name: str) -> bool:
        """Disable a plugin for a specific user."""
        if user_id not in self.user_plugins:
            return True

        # Disable for user
        self.user_plugins[user_id][plugin_name] = False

        # Check if any other users have this plugin enabled
        other_users_enabled = any(
            plugins.get(plugin_name, False)
            for uid, plugins in self.user_plugins.items()
            if uid != user_id
        )

        # If no other users have it enabled, unload it
        if not other_users_enabled and plugin_name in self.plugins:
            await self.unload_plugin(plugin_name)
            self.plugins[plugin_name].enabled = False

        logger.info(f"Plugin {plugin_name} disabled for user {user_id}")
        return True

    def get_user_plugins(self, user_id: str) -> dict[str, bool]:
        """Get plugin status for a user."""
        return self.user_plugins.get(user_id, {})

    def is_plugin_loaded(self, plugin_name: str) -> bool:
        """Check if a plugin is currently loaded."""
        plugin_info = self.plugins.get(plugin_name)
        return plugin_info is not None and plugin_info.loaded

    def get_available_functions(self, user_id: str) -> dict[str, Callable]:
        """Get available functions for a user."""
        user_plugins = self.get_user_plugins(user_id)
        available_functions = {}

        for func_name, func in self.function_registry.items():
            plugin_name = func_name.split(".")[0]
            if user_plugins.get(plugin_name, False):
                available_functions[func_name] = func

        return available_functions

    def get_modules_for_user(self, user_id: str) -> dict[str, Any]:
        """Get loaded modules for a user for function calling."""
        user_plugins = self.get_user_plugins(user_id)
        modules = {}

        for plugin_name, enabled in user_plugins.items():
            if enabled and plugin_name in self.plugins:
                plugin_info = self.plugins[plugin_name]
                if plugin_info.loaded and plugin_info.module:
                    modules[plugin_name] = plugin_info.module

        return modules

    async def execute_function(self, user_id: str, function_name: str, **kwargs) -> Any:
        """Execute a function if available for user."""
        available_functions = self.get_available_functions(user_id)

        if function_name not in available_functions:
            logger.error(f"Function {function_name} not available for user {user_id}")
            return None

        try:
            func = available_functions[function_name]

            # Check if function is async
            if inspect.iscoroutinefunction(func):
                result = await func(**kwargs)
            else:
                result = func(**kwargs)

            logger.debug(f"Executed function {function_name} for user {user_id}")
            return result

        except Exception as e:
            logger.error(f"Error executing function {function_name}: {e}")
            return None

    async def call_plugin_function(
        self,
        user_id: str,
        plugin_name: str,
        function_name: str,
        parameters: dict[str, Any],
    ) -> Any:
        """Call a plugin function using the plugin's execute_function method."""
        try:
            # Check if user has plugin enabled
            user_plugins = self.get_user_plugins(user_id)
            if not user_plugins.get(plugin_name, False):
                logger.error(f"Plugin {plugin_name} not enabled for user {user_id}")
                return {"success": False, "error": f"Plugin {plugin_name} not enabled"}

            # Check if plugin is loaded
            if not self.is_plugin_loaded(plugin_name):
                await self.load_plugin(plugin_name)

            plugin_info = self.plugins.get(plugin_name)
            if not plugin_info or not plugin_info.module:
                logger.error(f"Plugin {plugin_name} not loaded")
                return {"success": False, "error": f"Plugin {plugin_name} not loaded"}

            # Call plugin's execute_function
            if hasattr(plugin_info.module, "execute_function"):
                result = await plugin_info.module.execute_function(
                    function_name, parameters, int(user_id)
                )
                logger.debug(f"Called {plugin_name}.{function_name} for user {user_id}")
                return result
            else:
                logger.error(f"Plugin {plugin_name} has no execute_function method")
                return {
                    "success": False,
                    "error": f"Plugin {plugin_name} has no execute_function method",
                }

        except Exception as e:
            logger.error(f"Error calling {plugin_name}.{function_name}: {e}")
            return {"success": False, "error": str(e)}

    def get_plugin_info(self, plugin_name: str) -> PluginInfo | None:
        """Get information about a plugin."""
        return self.plugins.get(plugin_name)

    def get_all_plugins(self) -> dict[str, PluginInfo]:
        """Get all available plugins."""
        return self.plugins.copy()

    async def reload_plugin(self, plugin_name: str) -> bool:
        """Reload a plugin (useful for development)."""
        if plugin_name not in self.plugins:
            return False

        # Get users who had this plugin enabled
        enabled_users = [
            user_id
            for user_id, plugins in self.user_plugins.items()
            if plugins.get(plugin_name, False)
        ]

        # Unload plugin
        await self.unload_plugin(plugin_name)

        # Re-analyze plugin file
        plugin_info = await self._analyze_plugin(
            Path(self.plugins[plugin_name].module_path)
        )
        if plugin_info:
            self.plugins[plugin_name] = plugin_info

        # Re-enable for users who had it enabled
        for user_id in enabled_users:
            await self.enable_plugin_for_user(user_id, plugin_name)

        logger.info(f"Plugin {plugin_name} reloaded successfully")
        return True


# Global plugin manager instance
plugin_manager = PluginManager("modules")
