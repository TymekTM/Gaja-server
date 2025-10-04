"""function_calling_system.py.

OpenAI Function Calling system for Gaja AI assistant. Converts the existing module
system to OpenAI function calling format.
"""

import logging
import time
import threading
from typing import Any

logger = logging.getLogger(__name__)


_FCS_SINGLETON = None
_FCS_LOCK = threading.Lock()


def get_function_calling_system() -> "FunctionCallingSystem":
    """Return global singleton instance of FunctionCallingSystem.

    Uses double-checked locking for thread-safe lazy initialization. Keeps
    conversion caches (modules -> functions) warm across multiple AI calls.
    """
    global _FCS_SINGLETON
    if _FCS_SINGLETON is None:
        with _FCS_LOCK:
            if _FCS_SINGLETON is None:  # second check inside lock
                inst = FunctionCallingSystem()
                _FCS_SINGLETON = inst
    return _FCS_SINGLETON  # type: ignore[return-value]


class FunctionCallingSystem:
    """Manages conversion of modules to OpenAI function calling format."""

    def __init__(self):
        """Initialize function calling system."""
        self.modules = {}
        # Handlers mapping function name -> handler info dict
        self.function_handlers = {}
        # Cache for instantiated module singletons
        self._cached_module_instances = {}
        # Flag to track if module instances were initialized
        self._instances_initialized = False
        # Cache of converted functions (list of function defs) to avoid repeated imports
        self._functions_cache = None

    async def initialize(self):
        """Asynchroniczna inicjalizacja systemu funkcji."""
        # Tutaj można załadować podstawowe moduły
        logger.info("Function calling system initialized")
        # Clear cache to force refresh of module instances
        self._functions_cache = None
        self._instances_initialized = False
        pass

    async def initialize_module_instances(self):
        """Initialize all cached module instances that have an initialize method."""
        for module_name, instance in self._cached_module_instances.items():
            if hasattr(instance, 'initialize'):
                try:
                    await instance.initialize()
                    logger.debug(f"Initialized {module_name} module instance")
                except Exception as init_err:
                    logger.error(f"Failed to initialize {module_name} instance: {init_err}")
        logger.info("All module instances initialized")

    def register_module(self, module_name: str, module_data: dict[str, Any]) -> None:
        """Register a module with the function calling system.

        Args:
            module_name: Name of the module to register
            module_data: Module configuration containing handler and schema info
        """
        self.modules[module_name] = module_data

        # Register main handler if available
        if "handler" in module_data:
            self.function_handlers[module_name] = module_data["handler"]

        # Register sub-command handlers if available
        if "sub_commands" in module_data:
            for sub_name, sub_data in module_data["sub_commands"].items():
                if "function" in sub_data:
                    handler_name = f"{module_name}_{sub_name}"
                    self.function_handlers[handler_name] = sub_data["function"]

        logger.info(f"Registered module: {module_name}")

    def convert_modules_to_functions(self) -> list[dict[str, Any]]:
        """Convert plugin manager & server modules to OpenAI function definitions."""
        # Return cached conversion if available
        if self._functions_cache is not None:
            return self._functions_cache

        start_time = time.perf_counter()
        functions: list[dict[str, Any]] = []
        seen_names: set[str] = set()

        # ---- Server modules (canonical tool set) ----
        import importlib

        module_specs = [
            ("weather", "weather_module"),
            ("search", "search_module"),
            ("core", "core_module"),
            ("music", "music_module"),
            ("web", "open_web_module"),
            ("notes", "notes_module"),
            ("tasks", "tasks_module"),
        ]

        core_import_names = {imp for _, imp in module_specs}

        for module_name, import_name in module_specs:
            try:
                module = importlib.import_module(f"modules.{import_name}")
            except Exception as e:  # pragma: no cover
                logger.error(f"Failed to import server module {import_name}: {e}")
                continue
            if not hasattr(module, "get_functions"):
                continue
            try:
                module_functions = module.get_functions()
            except Exception as e:  # pragma: no cover
                logger.error(f"get_functions failed for {module_name}: {e}")
                continue
            for func in module_functions:
                try:
                    handler_name = f"{module_name}_{func['name']}"
                    if handler_name in seen_names:
                        # Skip duplicate function name
                        continue
                    openai_func = {
                        "type": "function",
                        "function": {
                            "name": handler_name,
                            "description": func.get("description", "No description"),
                            "parameters": func.get(
                                "parameters",
                                {"type": "object", "properties": {}, "required": []},
                            ),
                        },
                    }
                    functions.append(openai_func)
                    seen_names.add(handler_name)
                    if module_name not in self._cached_module_instances:
                        # Instantiate module class if present
                        instance = None
                        for cls_name in [
                            "WeatherModule",
                            "SearchModule",
                            "CoreModule",
                            "MusicModule",
                            "WebModule",
                            "NotesModule",
                            "TasksModule",
                        ]:
                            if hasattr(module, cls_name):
                                try:
                                    instance = getattr(module, cls_name)()
                                except Exception as inst_err:  # pragma: no cover
                                    logger.error(
                                        f"Failed instantiating {cls_name} for {module_name}: {inst_err}"
                                    )
                                break
                        if instance:
                            self._cached_module_instances[module_name] = instance
                        else:
                            # Skip handler registration if no instance
                            continue
                    module_instance = self._cached_module_instances[module_name]
                    self.function_handlers[handler_name] = {
                        "module": module_instance,
                        "function_name": func["name"],
                        "module_name": module_name,
                        "type": "server_module",
                    }
                except Exception as func_err:  # pragma: no cover
                    logger.error(
                        f"Error processing function {func} in module {module_name}: {func_err}"
                    )

        # ---- Plugin manager functions (only for non-core/extra plugins) ----
        try:
            from core.plugin_manager import plugin_manager

            registry = getattr(plugin_manager, "function_registry", {}) or {}
            for full_func_name, func_info in registry.items():
                parts = full_func_name.split(".")
                if len(parts) != 2:
                    continue
                plugin_name, func_name = parts

                # Skip functions provided by canonical server modules to avoid duplicates
                if plugin_name in core_import_names:
                    continue

                if isinstance(func_info, dict):
                    description = func_info.get(
                        "description", f"Function {full_func_name}"
                    )
                    parameters = func_info.get(
                        "parameters",
                        {"type": "object", "properties": {}, "required": []},
                    )
                else:
                    description = f"Function {full_func_name}"
                    parameters = {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    }

                handler_name = full_func_name.replace(".", "_")
                if handler_name in seen_names:
                    continue

                openai_function = {
                    "type": "function",
                    "function": {
                        "name": handler_name,
                        "description": description,
                        "parameters": parameters,
                    },
                }
                functions.append(openai_function)
                seen_names.add(handler_name)
                self.function_handlers[handler_name] = {
                    "original_name": full_func_name,
                    "plugin_name": plugin_name,
                    "function_name": func_name,
                }
        except Exception:  # pragma: no cover
            logger.debug("Plugin manager not available for function conversion")

        elapsed = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"Converted {len(functions)} functions for OpenAI in {elapsed:.1f} ms"
        )
        # Cache result to avoid repeated import/instantiation overhead
        self._functions_cache = functions
        return functions

    def invalidate_cache(self) -> None:
        """Force cache rebuild so newly loaded plugins become visible."""
        self._functions_cache = None
        logger.debug("Function definitions cache invalidated")

    def _create_main_function(
        self, module_name: str, module_info: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Create main function definition for a module."""
        if "handler" not in module_info:
            return None

        function_name = f"{module_name}_main"
        self.function_handlers[function_name] = module_info["handler"]

        # Enhanced description for better OpenAI understanding
        base_description = module_info.get("description", f"{module_name} module")
        enhanced_description = self._enhance_main_function_description(
            module_name, base_description
        )

        # Build parameters schema
        parameters = {
            "type": "object",
            "properties": {
                "params": {
                    "type": "string",
                    "description": f"Parameters for {module_name} module",
                }
            },
            "required": [],
        }

        # Add sub-command selection if module has sub-commands
        sub_commands = module_info.get("sub_commands", {})
        if sub_commands:
            parameters["properties"]["action"] = {
                "type": "string",
                "description": f"Action to perform. Available: {', '.join(sub_commands.keys())}",
                "enum": list(sub_commands.keys()),
            }

        return {
            "type": "function",
            "function": {
                "name": function_name,
                "description": enhanced_description,
                "parameters": parameters,
            },
        }

    def _create_sub_function(
        self, module_name: str, sub_name: str, sub_info: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Create function definition for a sub-command."""
        function_name = f"{module_name}_{sub_name}"
        self.function_handlers[function_name] = sub_info["function"]

        # Build parameters from sub_info
        parameters = {"type": "object", "properties": {}, "required": []}

        # Extract parameters from sub_info - try different formats
        if "parameters" in sub_info:
            # New format with detailed parameter definitions
            sub_params = sub_info["parameters"]
            if isinstance(sub_params, dict):
                for param_name, param_info in sub_params.items():
                    if isinstance(param_info, dict):
                        parameters["properties"][param_name] = {
                            "type": param_info.get("type", "string"),
                            "description": param_info.get(
                                "description", f"Parameter {param_name}"
                            ),
                        }
                        if param_info.get("required", False):
                            parameters["required"].append(param_name)
        elif "params_desc" in sub_info:
            # Legacy format with description string like '<seconds>' or '<datetime> <note>'
            params_desc = sub_info["params_desc"].strip()
            if params_desc:
                # Parse simple parameter descriptions
                if (
                    params_desc.startswith("<")
                    and params_desc.endswith(">")
                    and params_desc.count("<") == 1
                ):
                    # Single parameter like '<seconds>'
                    param_name = params_desc[1:-1]
                    param_description = self._get_enhanced_param_description(
                        sub_name, param_name
                    )
                    parameters["properties"][param_name] = {
                        "type": "string",
                        "description": param_description,
                    }
                    parameters["required"].append(param_name)
                elif "<" in params_desc:
                    # Multiple parameters like '<datetime> <note>'
                    import re

                    param_matches = re.findall(r"<([^>]+)>", params_desc)
                    for param in param_matches:
                        param_description = self._get_enhanced_param_description(
                            sub_name, param
                        )
                        parameters["properties"][param] = {
                            "type": "string",
                            "description": param_description,
                        }
                        parameters["required"].append(param)
                else:
                    # Generic parameter description
                    parameters["properties"]["params"] = {
                        "type": "string",
                        "description": params_desc
                        or f"Parameters for {sub_name} command",
                    }

        # Fallback to generic params if no specific parameters defined
        if not parameters["properties"]:
            parameters["properties"]["params"] = {
                "type": "string",
                "description": f"Parameters for {sub_name} command",
            }

        # Enhanced description for better OpenAI understanding
        base_description = sub_info.get(
            "description", f"{module_name} {sub_name} command"
        )
        enhanced_description = self._enhance_function_description(
            module_name, sub_name, base_description
        )

        return {
            "type": "function",
            "function": {
                "name": function_name,
                "description": enhanced_description,
                "parameters": parameters,
            },
        }

    def _enhance_main_function_description(
        self, module_name: str, base_description: str
    ) -> str:
        """Enhance main function descriptions for better OpenAI understanding."""

        enhanced_main_descriptions = {
            "core": "Core functionality module for timers, calendar events, reminders, shopping lists, and to-do tasks. Use this when user wants to manage time, schedule, or organize tasks",
            "memory": "Long-term memory management module. Use this when user wants to save, remember, recall, or manage stored information",
            "search": "Internet search module. Use this when user wants to search for information online, find facts, or research topics",
            "music": "Music control module for playing, pausing, skipping tracks on Spotify or using media keys. Use when user wants to control music playback",
        }

        if module_name in enhanced_main_descriptions:
            return enhanced_main_descriptions[module_name]

        return base_description

    def _enhance_function_description(
        self, module_name: str, sub_name: str, base_description: str
    ) -> str:
        """Enhance function descriptions for better OpenAI understanding."""

        # Enhanced descriptions for common functions
        enhanced_descriptions = {
            "core": {
                "set_timer": "Set a countdown timer for a specified duration in seconds. Use this when user wants to set a timer, minutnik, stoper, or countdown",
                "timer": "Set a countdown timer for a specified duration in seconds. Use this when user wants to set a timer, minutnik, stoper, or countdown",
                "view_timers": "View all active timers and their remaining time. Use this when user wants to check active timers",
                "timers": "View all active timers and their remaining time. Use this when user wants to check active timers",
                "add_event": "Add a calendar event with date and description. Use for scheduling appointments, meetings, or events",
                "event": "Add a calendar event with date and description. Use for scheduling appointments, meetings, or events",
                "set_reminder": "Set a reminder for a specific date and time with a note. Use when user wants to be reminded about something",
                "reminder": "Set a reminder for a specific date and time with a note. Use when user wants to be reminded about something",
                "add_item": "Add an item to the shopping list. Use when user wants to add something to buy or purchase",
                "item": "Add an item to the shopping list. Use when user wants to add something to buy or purchase",
                "add_task": "Add a task to the to-do list. Use when user wants to add something to do or complete",
                "task": "Add a task to the to-do list. Use when user wants to add something to do or complete",
            },
            "memory": {
                "add": "Save information to long-term memory. Use when user wants to remember, store, or save something for later",
                "get": "Retrieve information from long-term memory. Use when user asks to recall, remember, or check stored information",
                "show": "Retrieve information from long-term memory. Use when user asks to recall, remember, or check stored information",
                "check": "Retrieve information from long-term memory. Use when user asks to recall, remember, or check stored information",
            },
            "search": {
                "main": "Search the internet for information and provide summarized results. Use when user asks to search, find, or look up information online"
            },
        }

        if (
            module_name in enhanced_descriptions
            and sub_name in enhanced_descriptions[module_name]
        ):
            return enhanced_descriptions[module_name][sub_name]

        # Fallback to original description
        return base_description

    def _get_enhanced_param_description(self, sub_name: str, param_name: str) -> str:
        """Get enhanced parameter descriptions for better OpenAI understanding."""

        param_descriptions = {
            "seconds": "Duration in seconds for the timer (e.g., 60 for 1 minute, 300 for 5 minutes)",
            "datetime": "Date and time in ISO format (e.g., 2025-05-30T14:30:00)",
            "note": "Text note or description for the reminder",
            "desc": "Description or details for the event",
            "item": "Name of the item to add to the shopping list",
            "task": "Description of the task to add to the to-do list",
            "task_number": "Number of the task to complete or remove",
        }

        if param_name in param_descriptions:
            return param_descriptions[param_name]
        return f"The {param_name} parameter for {sub_name} command"

    def _check_missing_parameters(self, function_name: str, parameters: dict[str, Any]) -> dict[str, Any] | None:
        """Check if function has missing required parameters and generate clarification request.
        
        Args:
            function_name: Name of the function to check
            parameters: Parameters provided by the user
            
        Returns:
            Clarification request dict if parameters are missing, None otherwise
        """
        # Check if this is a weather function that needs location
        if function_name.startswith("weather_") and not parameters.get("location"):
            return {
                "type": "clarification_request",
                "action_type": "clarification_request",
                "message": "Potrzebuję informacji o lokalizacji aby sprawdzić pogodę.",
                "clarification_data": {
                    "question": "Dla jakiej lokalizacji chcesz sprawdzić pogodę?",
                    "parameter": "location",
                    "function": function_name,
                    "provided_parameters": parameters
                }
            }

        # Notes module: create_note requires content
        if function_name == "notes_create_note" and not parameters.get("content"):
            return {
                "type": "clarification_request",
                "action_type": "clarification_request",
                "message": "Brakuje treści notatki.",
                "clarification_data": {
                    "question": "Jaka treść notatki mam zapisać?",
                    "parameter": "content",
                    "function": function_name,
                    "provided_parameters": parameters
                }
            }

        # Tasks module: add_task requires title
        if function_name == "tasks_add_task" and not parameters.get("title"):
            return {
                "type": "clarification_request",
                "action_type": "clarification_request",
                "message": "Brakuje tytułu zadania.",
                "clarification_data": {
                    "question": "Jakie zadanie chcesz dodać? Podaj krótki tytuł.",
                    "parameter": "title",
                    "function": function_name,
                    "provided_parameters": parameters
                }
            }
        # Core module: add_event requires title and date
        if function_name == "core_add_event":
            title = (parameters.get("title") or "").strip()
            date = (parameters.get("date") or "").strip()
            if not title or not date:
                missing = []
                if not title:
                    missing.append("tytuł")
                if not date:
                    missing.append("data (YYYY-MM-DD)")
                return {
                    "type": "clarification_request",
                    "action_type": "clarification_request",
                    "message": f"Brakuje pól: {', '.join(missing)} do dodania wydarzenia.",
                    "clarification_data": {
                        "question": "Podaj tytuł oraz datę (YYYY-MM-DD) i ewentualnie godzinę (HH:MM).",
                        "parameter": "event_details",
                        "function": function_name,
                        "provided_parameters": parameters,
                    },
                }

        # Core module: set_timer requires positive duration
        if function_name == "core_set_timer":
            dur = str(parameters.get("duration") or "").strip()
            if not dur or dur in ("0", "0s", "0m", "0h"):
                return {
                    "type": "clarification_request",
                    "action_type": "clarification_request",
                    "message": "Nieprawidłowy czas timera. Podaj dodatni czas (np. 5m, 30s, 1h).",
                    "clarification_data": {
                        "question": "Na jak długo ustawić timer? (np. 15m)",
                        "parameter": "duration",
                        "function": function_name,
                        "provided_parameters": parameters,
                    },
                }

        # Add more function-specific parameter checks here as needed
        # if function_name.startswith("music_") and not parameters.get("song"):
        #     return clarification for missing song parameter...
        
        return None

    async def execute_function(
        self, function_name: str, arguments: dict[str, Any], conversation_history=None
    ):
        """Execute a function call through the plugin manager or server modules."""
        try:
            # Check for missing required parameters first
            clarification = self._check_missing_parameters(function_name, arguments)
            if clarification:
                logger.info(f"Missing required parameters for {function_name}, requesting clarification")
                return clarification
                
            # Get handler info
            handler_info = self.function_handlers.get(function_name)
            if not handler_info:
                return f"Function {function_name} not found"

            # Handle server modules
            if handler_info.get("type") == "server_module":
                module = handler_info["module"]
                func_name = handler_info["function_name"]
                module_name = handler_info["module_name"]

                logger.info(
                    f"Executing server module function: {module_name}.{func_name}"
                )
                
                logger.debug(f"Module instance type: {type(module)}")
                logger.debug(f"Has execute_function: {hasattr(module, 'execute_function')}")

                # Initialize module instances if not already done
                if not self._instances_initialized:
                    logger.info("Initializing module instances for the first time")
                    await self.initialize_module_instances()
                    self._instances_initialized = True
                else:
                    logger.debug("Module instances already initialized")

                if hasattr(module, "execute_function"):
                    # Use the module's execute_function method (async)
                    exec_start = time.perf_counter()
                    result = await module.execute_function(
                        func_name, arguments, user_id=1
                    )
                    exec_elapsed = (time.perf_counter() - exec_start) * 1000
                    logger.debug(
                        f"Executed {module_name}.{func_name} in {exec_elapsed:.1f} ms"
                    )
                    logger.info(f"Server module function result: {result}")
                    return result
                else:
                    # Fallback: call module-level execute_function from modules.<module_name>_module
                    try:
                        import importlib as _importlib

                        mod_obj = _importlib.import_module(f"modules.{module_name}_module")
                        if hasattr(mod_obj, "execute_function"):
                            exec_start = time.perf_counter()
                            result = await mod_obj.execute_function(
                                func_name, arguments, user_id=1
                            )
                            exec_elapsed = (time.perf_counter() - exec_start) * 1000
                            logger.debug(
                                f"Executed module-level {module_name}.{func_name} in {exec_elapsed:.1f} ms"
                            )
                            return result
                    except Exception as fallback_err:  # pragma: no cover
                        logger.debug(
                            f"Fallback module-level execute_function failed for {module_name}: {fallback_err}"
                        )

                    # Final fallback: try to call the method directly on the instance
                    # e.g. SearchModule.search(user_id=..., **arguments)
                    try:
                        method = getattr(module, func_name, None)
                        if callable(method):
                            import inspect as _inspect
                            exec_start = time.perf_counter()
                            if _inspect.iscoroutinefunction(method):
                                data = await method(user_id=1, **arguments)
                            else:
                                data = method(user_id=1, **arguments)
                            exec_elapsed = (time.perf_counter() - exec_start) * 1000
                            logger.debug(
                                f"Executed direct method {module_name}.{func_name} in {exec_elapsed:.1f} ms"
                            )
                            # Normalize return to standard structure
                            if isinstance(data, dict) and (
                                "success" in data or "data" in data or "error" in data
                            ):
                                return data
                            return {"success": True, "data": data}
                    except Exception as direct_err:
                        logger.debug(
                            f"Direct method call failed for {module_name}.{func_name}: {direct_err}"
                        )
                    # Return structured error instead of plain string
                    return {
                        "success": False,
                        "error": f"Module {module_name} does not support execute_function",
                        "module": module_name,
                        "function": func_name,
                    }

            # Handle plugin manager functions (legacy)
            plugin_name = handler_info.get("plugin_name")
            func_name = handler_info.get("function_name")

            if plugin_name and func_name:
                from core.plugin_manager import plugin_manager

                # Get the plugin
                plugin_info = plugin_manager.plugins.get(plugin_name)
                if not plugin_info or not plugin_info.loaded:
                    return f"Plugin {plugin_name} not loaded"

                # Get the function from the plugin module
                if plugin_info.module and hasattr(plugin_info.module, "execute_function"):
                    # Use the plugin's execute_function method (async)
                    result = await plugin_info.module.execute_function(
                        func_name, arguments, user_id=1
                    )
                    return result
                elif hasattr(plugin_info.module, func_name):
                    # Call the function directly
                    func = getattr(plugin_info.module, func_name)
                    if callable(func):
                        # Check if function is async
                        import asyncio

                        if asyncio.iscoroutinefunction(func):
                            result = await func(**arguments)
                        else:
                            result = func(**arguments)
                        return result
                    else:
                        return f"Function {func_name} is not callable"
                else:
                    return f"Function {func_name} not found in plugin {plugin_name}"
            else:
                return f"Invalid handler info for function {function_name}"

        except Exception as e:
            logger.error(f"Error executing function {function_name}: {e}")
            return f"Error executing function: {str(e)}"


def convert_module_system_to_function_calling(
    modules: dict[str, Any],
) -> FunctionCallingSystem:
    """Convert the entire module system to function calling format."""
    system = get_function_calling_system()
    system.modules = modules
    return system
