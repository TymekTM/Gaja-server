"""function_calling_system.py.

OpenAI Function Calling system for Gaja AI assistant. Converts the existing module
system to OpenAI function calling format.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class FunctionCallingSystem:
    """Manages conversion of modules to OpenAI function calling format."""

    def __init__(self):
        """Initialize function calling system."""
        self.modules = {}
        self.function_handlers: dict[str, dict[str, Any]] = {}

    async def initialize(self):
        """Asynchroniczna inicjalizacja systemu funkcji."""
        # Tutaj można załadować podstawowe moduły
        logger.info("Function calling system initialized")
        pass

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
        """Convert plugin manager modules and server modules to OpenAI function calling
        format."""
        functions = []

        # First try plugin manager (legacy support)
        try:
            from plugin_manager import plugin_manager

            if plugin_manager.function_registry:
                # Get functions from plugin manager's function registry
                for (
                    full_func_name,
                    func_info,
                ) in plugin_manager.function_registry.items():
                    try:
                        # Parse plugin name and function name
                        parts = full_func_name.split(".")
                        if len(parts) != 2:
                            logger.warning(
                                f"Skipping function with invalid name format: {full_func_name}"
                            )
                            continue

                        plugin_name, func_name = parts

                        # Create OpenAI function format
                        openai_function = {
                            "type": "function",
                            "function": {
                                "name": full_func_name.replace(
                                    ".", "_"
                                ),  # OpenAI doesn't like dots
                                "description": func_info.get(
                                    "description", f"Function {full_func_name}"
                                ),
                                "parameters": func_info.get(
                                    "parameters",
                                    {
                                        "type": "object",
                                        "properties": {},
                                        "required": [],
                                    },
                                ),
                            },
                        }

                        functions.append(openai_function)
                        logger.debug(
                            f"Converted function: {full_func_name} -> {openai_function['function']['name']}"
                        )

                        # Store handler info for later execution
                        handler_name = full_func_name.replace(".", "_")
                        self.function_handlers[handler_name] = {
                            "original_name": full_func_name,
                            "plugin_name": plugin_name,
                            "function_name": func_name,
                        }

                    except Exception as e:
                        logger.error(f"Error converting function {full_func_name}: {e}")
                        continue
        except Exception as e:
            logger.debug(f"Plugin manager not available or has no functions: {e}")

        # Add server modules directly (main source of functions)
        import sys

        print("DEBUG: Starting server modules section", flush=True)
        sys.stderr.write("DEBUG: Starting server modules section\n")
        sys.stderr.flush()
        try:
            print("DEBUG: Importing server modules...")
            from modules import (  # onboarding_plugin_module,  # Disabled - not needed; plugin_monitor_module,  # Disabled - not needed
                api_module,
                core_module,
                memory_module,
                music_module,
                open_web_module,
                search_module,
                weather_module,
            )

            print("DEBUG: Server modules imported successfully")

            server_modules = [
                ("weather", weather_module),
                ("search", search_module),
                ("core", core_module),
                ("music", music_module),
                ("api", api_module),
                ("web", open_web_module),
                ("memory", memory_module),
                # ("monitor", plugin_monitor_module),  # Disabled - not needed
                # ("onboarding", onboarding_plugin_module),  # Disabled - not needed
            ]

            print(f"DEBUG: Processing {len(server_modules)} server modules")

            for module_name, module in server_modules:
                print(f"DEBUG: Processing module {module_name}")
                try:
                    if hasattr(module, "get_functions"):
                        print(f"DEBUG: Module {module_name} has get_functions")
                        # Call get_functions directly on the module
                        module_functions = module.get_functions()
                        logger.debug(
                            f"Got {len(module_functions)} functions from {module_name}"
                        )
                        print(
                            f"DEBUG: Got {len(module_functions)} functions from {module_name}"
                        )

                        for func in module_functions:
                            openai_func = {
                                "type": "function",
                                "function": {
                                    "name": f"{module_name}_{func['name']}",
                                    "description": func["description"],
                                    "parameters": func["parameters"],
                                },
                            }
                            functions.append(openai_func)

                            # Store handler for execution - create module instance
                            handler_name = f"{module_name}_{func['name']}"

                            # Create module instance based on module type
                            if hasattr(module, "WeatherModule"):
                                module_instance = module.WeatherModule()
                            elif hasattr(module, "SearchModule"):
                                module_instance = module.SearchModule()
                            elif hasattr(module, "CoreModule"):
                                module_instance = module.CoreModule()
                            elif hasattr(module, "MusicModule"):
                                module_instance = module.MusicModule()
                            elif hasattr(module, "APIModule"):
                                module_instance = module.APIModule()
                            elif hasattr(module, "WebModule"):
                                module_instance = module.WebModule()
                            elif hasattr(module, "MemoryModule"):
                                module_instance = module.MemoryModule()
                            elif hasattr(module, "PluginMonitorModule"):
                                module_instance = module.PluginMonitorModule()
                            elif hasattr(module, "OnboardingPluginModule"):
                                module_instance = module.OnboardingPluginModule()
                            else:
                                logger.warning(
                                    f"Could not create instance for {module_name}"
                                )
                                print(
                                    f"DEBUG: Could not create instance for {module_name}"
                                )
                                continue

                            self.function_handlers[handler_name] = {
                                "module": module_instance,
                                "function_name": func["name"],
                                "module_name": module_name,
                                "type": "server_module",
                            }

                            logger.debug(f"Added server function: {handler_name}")
                            print(f"DEBUG: Added server function: {handler_name}")
                    else:
                        logger.warning(
                            f"Module {module_name} does not have get_functions method"
                        )
                        print(
                            f"DEBUG: Module {module_name} does not have get_functions method"
                        )
                except Exception as e:
                    logger.error(f"Error loading functions from {module_name}: {e}")
                    print(f"DEBUG: Error loading functions from {module_name}: {e}")
                    import traceback

                    logger.error(traceback.format_exc())
                    traceback.print_exc()
        except Exception as e:
            logger.error(f"Error loading server modules: {e}")
            print(f"DEBUG: Error loading server modules: {e}")
            import traceback

            traceback.print_exc()

        logger.info(f"Converted {len(functions)} functions for OpenAI")
        return functions

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

    async def execute_function(
        self, function_name: str, arguments: dict[str, Any], conversation_history=None
    ):
        """Execute a function call through the plugin manager or server modules."""
        try:
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

                if hasattr(module, "execute_function"):
                    # Use the module's execute_function method (async)
                    result = await module.execute_function(
                        func_name, arguments, user_id=1
                    )
                    logger.info(f"Server module function result: {result}")
                    return result
                else:
                    return f"Module {module_name} does not support execute_function"

            # Handle plugin manager functions (legacy)
            plugin_name = handler_info.get("plugin_name")
            func_name = handler_info.get("function_name")

            if plugin_name and func_name:
                from plugin_manager import plugin_manager

                # Get the plugin
                plugin_info = plugin_manager.plugins.get(plugin_name)
                if not plugin_info or not plugin_info.loaded:
                    return f"Plugin {plugin_name} not loaded"

                # Get the function from the plugin module
                if hasattr(plugin_info.module, "execute_function"):
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
    system = FunctionCallingSystem()
    system.modules = modules
    return system
