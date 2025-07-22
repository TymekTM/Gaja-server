"""
Refactored core_module for GAJA Assistant - AGENTS.md Compliant

This module provides core functionality including:
- Timers with async polling
- Calendar events
- Reminders
- Task management
- List management

All functions are async/await compatible and use asyncio.to_thread() for non-blocking I/O.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)

# Determine the appropriate directory for persistent application data
APP_NAME = "Asystent_Server"
user_data_dir = ""

# Try to establish a user-specific data directory
try:
    if os.name == "nt":  # Windows
        # Use APPDATA, fallback to user's home directory
        base_dir = os.getenv("APPDATA")
        if not base_dir:
            base_dir = os.path.expanduser("~")
        user_data_dir = os.path.join(base_dir, APP_NAME)
    else:  # macOS, Linux
        user_data_dir = os.path.join(os.path.expanduser("~"), f".{APP_NAME.lower()}")

    # Create the directory if it doesn't exist
    os.makedirs(user_data_dir, exist_ok=True)
    logger.info(f"Application data will be stored in: {user_data_dir}")

except Exception as e:
    logger.error(
        f"Could not create/access user-specific data directory: {e}. Falling back."
    )
    # Fallback: try to create a data directory in the current working directory
    fallback_dir_name = f"{APP_NAME}_data_fallback"
    try:
        # Try to use the directory where the script/executable is located if possible
        if getattr(
            sys, "frozen", False
        ):  # If application is frozen (e.g. PyInstaller bundle)
            app_run_dir = os.path.dirname(sys.executable)
        else:  # If running as a script
            app_run_dir = os.path.dirname(os.path.abspath(__file__))

        user_data_dir = os.path.join(app_run_dir, fallback_dir_name)
        os.makedirs(user_data_dir, exist_ok=True)
        logger.info(f"Using fallback data directory: {user_data_dir}")
    except Exception as fallback_e:
        logger.error(f"Fallback directory creation also failed: {fallback_e}")
        # If even this fails, set user_data_dir to something to prevent crash on join, though writes will fail
        user_data_dir = "."

STORAGE_FILE = os.path.join(user_data_dir, "core_storage.json")
logger.info(f"Using storage file: {STORAGE_FILE}")

# Global timer polling task
_timer_polling_task = None


# Required plugin functions
def get_functions() -> list[dict[str, Any]]:
    """Returns list of available functions in the plugin."""
    return [
        {
            "name": "set_timer",
            "description": "Set a timer with specified duration and label",
            "parameters": {
                "type": "object",
                "properties": {
                    "duration": {
                        "type": "string",
                        "description": "Duration (e.g., '5m', '30s', '1h')",
                    },
                    "label": {
                        "type": "string",
                        "description": "Timer label/description",
                        "default": "timer",
                    },
                },
                "required": ["duration"],
            },
        },
        {
            "name": "view_timers",
            "description": "View active timers",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "add_event",
            "description": "Add a calendar event",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Event title"},
                    "date": {
                        "type": "string",
                        "description": "Event date (YYYY-MM-DD)",
                    },
                    "time": {
                        "type": "string",
                        "description": "Event time (HH:MM)",
                        "default": "12:00",
                    },
                },
                "required": ["title", "date"],
            },
        },
        {
            "name": "view_calendar",
            "description": "View calendar events",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "set_reminder",
            "description": "Set a reminder",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Reminder text"},
                    "time": {
                        "type": "string",
                        "description": "Reminder time (ISO format)",
                    },
                },
                "required": ["text", "time"],
            },
        },
        {
            "name": "view_reminders",
            "description": "View reminders",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "get_reminders_for_today",
            "description": "Get reminders due today",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "add_task",
            "description": "Add a task",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "Task description"},
                    "priority": {
                        "type": "string",
                        "description": "Priority level",
                        "default": "medium",
                    },
                },
                "required": ["task"],
            },
        },
        {
            "name": "view_tasks",
            "description": "View tasks",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "complete_task",
            "description": "Mark a task as completed",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "integer",
                        "description": "Task ID to complete",
                    },
                },
                "required": ["task_id"],
            },
        },
        {
            "name": "remove_task",
            "description": "Remove a task",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "integer", "description": "Task ID to remove"},
                },
                "required": ["task_id"],
            },
        },
        {
            "name": "add_item",
            "description": "Add item to a list",
            "parameters": {
                "type": "object",
                "properties": {
                    "list_name": {"type": "string", "description": "List name"},
                    "item": {"type": "string", "description": "Item to add"},
                },
                "required": ["list_name", "item"],
            },
        },
        {
            "name": "view_list",
            "description": "View items in a list",
            "parameters": {
                "type": "object",
                "properties": {
                    "list_name": {"type": "string", "description": "List name"},
                },
                "required": ["list_name"],
            },
        },
        {
            "name": "remove_item",
            "description": "Remove item from a list",
            "parameters": {
                "type": "object",
                "properties": {
                    "list_name": {"type": "string", "description": "List name"},
                    "item": {"type": "string", "description": "Item to remove"},
                },
                "required": ["list_name", "item"],
            },
        },
        {
            "name": "get_current_time",
            "description": "Get current time",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
        {
            "name": "ask_for_clarification",
            "description": "🔍 CRITICAL: Use this function when user's request is unclear, ambiguous, or missing essential information. DO NOT guess or ask in text - use this function instead! Examples: weather without location, music without specification, timer without duration, reminder without details. This provides better user experience by properly handling unclear requests.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Clear, specific question asking for the missing information. Examples: 'What city would you like the weather for?', 'What song or artist should I play?', 'How long should the timer be?'",
                    },
                    "context": {
                        "type": "string",
                        "description": "What you understood from the user's request so far",
                    },
                },
                "required": ["question"],
            },
        },
    ]


async def execute_function(
    function_name: str, parameters: dict[str, Any], user_id: int
) -> dict[str, Any]:
    """Executes plugin function asynchronously."""
    try:
        if function_name == "set_timer":
            duration = parameters.get("duration", "")
            label = parameters.get("label", "timer")
            return await set_timer({"duration": duration, "label": label})

        elif function_name == "view_timers":
            return await view_timers({})

        elif function_name == "add_event":
            title = parameters.get("title", "")
            date = parameters.get("date", "")
            time = parameters.get("time", "12:00")
            return await add_event({"title": title, "date": date, "time": time})

        elif function_name == "view_calendar":
            return await view_calendar({})

        elif function_name == "set_reminder":
            text = parameters.get("text", "")
            time = parameters.get("time", "")
            return await set_reminder({"text": text, "time": time})

        elif function_name == "view_reminders":
            return await view_reminders({})

        elif function_name == "get_reminders_for_today":
            return await get_reminders_for_today({})

        elif function_name == "add_task":
            task = parameters.get("task", "")
            priority = parameters.get("priority", "medium")
            return await add_task({"task": task, "priority": priority})

        elif function_name == "view_tasks":
            return await view_tasks({})

        elif function_name == "complete_task":
            task_id = parameters.get("task_id", -1)
            return await complete_task({"task_id": task_id})

        elif function_name == "remove_task":
            task_id = parameters.get("task_id", -1)
            return await remove_task({"task_id": task_id})

        elif function_name == "add_item":
            list_name = parameters.get("list_name", "")
            item = parameters.get("item", "")
            return await add_item({"list_name": list_name, "item": item})

        elif function_name == "view_list":
            list_name = parameters.get("list_name", "")
            return await view_list({"list_name": list_name})

        elif function_name == "remove_item":
            list_name = parameters.get("list_name", "")
            item = parameters.get("item", "")
            return await remove_item({"list_name": list_name, "item": item})

        elif function_name == "get_current_time":
            return await get_current_time({})

        elif function_name == "ask_for_clarification":
            question = parameters.get("question", "")
            context = parameters.get("context", "")
            return await ask_for_clarification(
                {"question": question, "context": context}
            )

        else:
            return {
                "success": False,
                "message": f"Unknown function: {function_name}",
                "error": f"Function '{function_name}' not supported",
            }

    except Exception as e:
        logger.error(f"Error in core module function {function_name}: {e}")
        return {
            "success": False,
            "message": f"Error executing {function_name}: {str(e)}",
            "error": str(e),
        }


# Storage functions
async def _init_storage():
    """Initialize storage file with default structure if it doesn't exist."""
    if not os.path.exists(STORAGE_FILE):
        try:

            def _write_init_file():
                with open(STORAGE_FILE, "w") as f:
                    json.dump(
                        {
                            "timers": [],
                            "events": [],
                            "reminders": [],
                            "shopping_list": [],  # Legacy compatibility
                            "tasks": [],
                            "lists": {},
                        },
                        f,
                        indent=2,
                    )

            await asyncio.to_thread(_write_init_file)
            logger.info(f"Initialized new storage file at: {STORAGE_FILE}")
        except Exception as e:
            logger.error(
                f"Failed to create or write initial storage file at {STORAGE_FILE}: {e}"
            )
            raise


async def _load_storage():
    """Load storage data from file asynchronously."""
    try:

        def _read_file():
            with open(STORAGE_FILE) as f:
                return f.read()

        content = await asyncio.to_thread(_read_file)
        return json.loads(content)
    except FileNotFoundError:
        # Initialize storage if file doesn't exist
        await _init_storage()
        return {
            "timers": [],
            "events": [],
            "reminders": [],
            "shopping_list": [],
            "tasks": [],
            "lists": {},
        }
    except Exception as e:
        logger.error(f"Error loading storage: {e}")
        raise


async def _save_storage(data):
    """Save storage data to file asynchronously."""
    try:

        def _write_file():
            with open(STORAGE_FILE, "w") as f:
                json.dump(data, f, indent=2)

        await asyncio.to_thread(_write_file)
    except Exception as e:
        logger.error(f"Error saving storage: {e}")
        raise


# Timer polling loop
async def _timer_polling_loop():
    """Async timer polling loop - checks for expired timers."""
    while True:
        try:
            data = await _load_storage()
            now = datetime.now()
            changed = False
            remaining = []

            for t in data["timers"]:
                target = datetime.fromisoformat(t["target"])
                if target <= now:
                    logger.info(f"Timer finished: {t['label']}")
                    logger.warning(f"⏰ TIMER FINISHED: {t['label']}")
                    changed = True
                else:
                    remaining.append(t)

            if changed:
                data["timers"] = remaining
                await _save_storage(data)

        except Exception as e:
            logger.error(f"Timer polling error: {e}")

        # Use async sleep with longer interval to reduce CPU usage
        await asyncio.sleep(30)  # Check timers every 30 seconds instead of 1 second


def _start_timer_polling_task():
    """Start the async timer polling task."""
    global _timer_polling_task
    if _timer_polling_task is None:
        _timer_polling_task = asyncio.create_task(_timer_polling_loop())


# Core functions
async def set_timer(params) -> dict[str, Any]:
    """Set a timer with specified duration and label."""
    try:
        # Parse timer parameters
        seconds = 0
        label = "timer"

        if isinstance(params, dict):
            duration = params.get("duration", "")
            label = params.get("label", "timer")

            if not duration:
                return {
                    "success": False,
                    "message": "Duration parameter is required",
                    "error": "Missing duration parameter",
                }

            # Parse duration (support "5m", "30s", "1h" format)
            if isinstance(duration, str):
                if duration.endswith("s"):
                    seconds = int(duration[:-1])
                elif duration.endswith("m"):
                    seconds = int(duration[:-1]) * 60
                elif duration.endswith("h"):
                    seconds = int(duration[:-1]) * 3600
                else:
                    seconds = int(duration)
            else:
                seconds = int(duration)

        else:
            return {
                "success": False,
                "message": "Invalid parameters format",
                "error": "Parameters must be a dictionary",
            }

        if seconds <= 0:
            return {
                "success": False,
                "message": "Duration must be positive",
                "error": "Invalid duration value",
            }

        # Create timer
        target = datetime.now() + timedelta(seconds=seconds)
        data = await _load_storage()

        timer_entry = {
            "label": str(label),
            "target": target.isoformat(),
            "sound": "beep",
            "created": datetime.now().isoformat(),
        }

        data["timers"].append(timer_entry)
        await _save_storage(data)

        logger.info(f"Set timer: {label} for {seconds} seconds")
        return {
            "success": True,
            "message": f'Timer "{label}" set for {seconds} seconds.',
            "timer": timer_entry,
        }

    except ValueError as e:
        return {
            "success": False,
            "message": f"Invalid duration format: {e}",
            "error": str(e),
        }
    except Exception as e:
        logger.error(f"Error setting timer: {e}")
        return {
            "success": False,
            "message": f"Error setting timer: {e}",
            "error": str(e),
        }


async def view_timers(params) -> dict[str, Any]:
    """View active timers."""
    try:
        data = await _load_storage()
        now = datetime.now()
        active = []

        for t in data["timers"]:
            target = datetime.fromisoformat(t["target"])
            if target > now:
                remaining = target - now
                remaining_seconds = int(remaining.total_seconds())
                active.append(
                    {
                        "label": t["label"],
                        "remaining_seconds": remaining_seconds,
                        "remaining_formatted": str(remaining).split(".")[0],
                    }
                )

        if not active:
            return {"success": True, "message": "No active timers.", "timers": []}

        result_message = "Active timers:\n"
        for i, timer in enumerate(active, 1):
            result_message += (
                f"{i}. {timer['label']}: {timer['remaining_formatted']} remaining\n"
            )

        return {"success": True, "message": result_message.strip(), "timers": active}

    except Exception as e:
        logger.error(f"Error viewing timers: {e}")
        return {
            "success": False,
            "message": f"Error viewing timers: {e}",
            "error": str(e),
        }


async def add_event(params) -> dict[str, Any]:
    """Add a calendar event."""
    try:
        if not isinstance(params, dict):
            return {
                "success": False,
                "message": "Parameters must be a dictionary",
                "error": "Invalid parameter format",
            }

        title = params.get("title", "").strip()
        date = params.get("date", "").strip()
        time = params.get("time", "12:00").strip()

        if not title or not date:
            return {
                "success": False,
                "message": "Title and date parameters are required",
                "error": "Missing required parameters",
            }

        # Combine date and time
        when_str = f"{date}T{time}"
        when = datetime.fromisoformat(when_str)

        data = await _load_storage()
        event_entry = {
            "title": title,
            "time": when.isoformat(),
            "desc": title,  # Legacy compatibility
            "created": datetime.now().isoformat(),
        }

        data["events"].append(event_entry)
        await _save_storage(data)

        logger.info(f"Added event: {title} at {when_str}")
        return {
            "success": True,
            "message": f'Event "{title}" added for {when.strftime("%Y-%m-%d %H:%M")}',
            "event": event_entry,
        }

    except ValueError as e:
        return {
            "success": False,
            "message": f"Invalid date/time format: {e}",
            "error": str(e),
        }
    except Exception as e:
        logger.error(f"Error adding event: {e}")
        return {
            "success": False,
            "message": f"Error adding event: {e}",
            "error": str(e),
        }


async def view_calendar(params) -> dict[str, Any]:
    """View calendar events."""
    try:
        data = await _load_storage()
        events = data.get("events", [])

        if not events:
            return {"success": True, "message": "No calendar events.", "events": []}

        # Sort events by time
        sorted_events = sorted(events, key=lambda x: x["time"])

        result_message = "Calendar events:\n"
        for i, event in enumerate(sorted_events, 1):
            event_time = datetime.fromisoformat(event["time"])
            title = event.get("title") or event.get("desc", "Untitled")
            result_message += (
                f"{i}. {event_time.strftime('%Y-%m-%d %H:%M')} - {title}\n"
            )

        return {
            "success": True,
            "message": result_message.strip(),
            "events": sorted_events,
        }

    except Exception as e:
        logger.error(f"Error viewing calendar: {e}")
        return {
            "success": False,
            "message": f"Error viewing calendar: {e}",
            "error": str(e),
        }


async def set_reminder(params) -> dict[str, Any]:
    """Set a reminder."""
    try:
        if not isinstance(params, dict):
            return {
                "success": False,
                "message": "Parameters must be a dictionary",
                "error": "Invalid parameter format",
            }

        text = params.get("text", "").strip()
        time_str = params.get("time", "").strip()

        if not text or not time_str:
            return {
                "success": False,
                "message": "Text and time parameters are required",
                "error": "Missing required parameters",
            }

        # Parse reminder time
        reminder_time = datetime.fromisoformat(time_str)

        data = await _load_storage()
        reminder_entry = {
            "text": text,
            "time": reminder_time.isoformat(),
            "created": datetime.now().isoformat(),
        }

        data["reminders"].append(reminder_entry)
        await _save_storage(data)

        logger.info(f"Set reminder: {text} at {time_str}")
        return {
            "success": True,
            "message": f'Reminder "{text}" set for {reminder_time.strftime("%Y-%m-%d %H:%M")}',
            "reminder": reminder_entry,
        }

    except ValueError as e:
        return {
            "success": False,
            "message": f"Invalid time format: {e}",
            "error": str(e),
        }
    except Exception as e:
        logger.error(f"Error setting reminder: {e}")
        return {
            "success": False,
            "message": f"Error setting reminder: {e}",
            "error": str(e),
        }


async def view_reminders(params) -> dict[str, Any]:
    """View reminders."""
    try:
        data = await _load_storage()
        reminders = data.get("reminders", [])

        if not reminders:
            return {"success": True, "message": "No reminders.", "reminders": []}

        # Sort reminders by time
        sorted_reminders = sorted(reminders, key=lambda x: x["time"])

        result_message = "Reminders:\n"
        for i, reminder in enumerate(sorted_reminders, 1):
            reminder_time = datetime.fromisoformat(reminder["time"])
            result_message += f"{i}. {reminder_time.strftime('%Y-%m-%d %H:%M')} - {reminder['text']}\n"

        return {
            "success": True,
            "message": result_message.strip(),
            "reminders": sorted_reminders,
        }

    except Exception as e:
        logger.error(f"Error viewing reminders: {e}")
        return {
            "success": False,
            "message": f"Error viewing reminders: {e}",
            "error": str(e),
        }


async def get_reminders_for_today(params) -> dict[str, Any]:
    """Get reminders due today."""
    try:
        data = await _load_storage()
        reminders = data.get("reminders", [])
        today = datetime.now().date()

        today_reminders = []
        for reminder in reminders:
            reminder_time = datetime.fromisoformat(reminder["time"])
            if reminder_time.date() == today:
                today_reminders.append(reminder)

        if not today_reminders:
            return {
                "success": True,
                "message": "No reminders for today.",
                "reminders": [],
            }

        # Sort by time
        sorted_reminders = sorted(today_reminders, key=lambda x: x["time"])

        result_message = "Today's reminders:\n"
        for i, reminder in enumerate(sorted_reminders, 1):
            reminder_time = datetime.fromisoformat(reminder["time"])
            result_message += (
                f"{i}. {reminder_time.strftime('%H:%M')} - {reminder['text']}\n"
            )

        return {
            "success": True,
            "message": result_message.strip(),
            "reminders": sorted_reminders,
        }

    except Exception as e:
        logger.error(f"Error getting today's reminders: {e}")
        return {
            "success": False,
            "message": f"Error getting today's reminders: {e}",
            "error": str(e),
        }


async def add_task(params) -> dict[str, Any]:
    """Add a task."""
    try:
        if not isinstance(params, dict):
            return {
                "success": False,
                "message": "Parameters must be a dictionary",
                "error": "Invalid parameter format",
            }

        task_text = params.get("task", "").strip()
        priority = params.get("priority", "medium").strip().lower()

        if not task_text:
            return {
                "success": False,
                "message": "Task parameter is required",
                "error": "Missing task parameter",
            }

        # Validate priority
        valid_priorities = ["low", "medium", "high", "urgent"]
        if priority not in valid_priorities:
            priority = "medium"

        data = await _load_storage()
        task_entry = {
            "task": task_text,
            "priority": priority,
            "completed": False,
            "created": datetime.now().isoformat(),
        }

        data["tasks"].append(task_entry)
        await _save_storage(data)

        logger.info(f"Added task: {task_text} (priority: {priority})")
        return {
            "success": True,
            "message": f'Task "{task_text}" added with {priority} priority.',
            "task": task_entry,
        }

    except Exception as e:
        logger.error(f"Error adding task: {e}")
        return {"success": False, "message": f"Error adding task: {e}", "error": str(e)}


async def view_tasks(params) -> dict[str, Any]:
    """View tasks."""
    try:
        data = await _load_storage()
        tasks = data.get("tasks", [])

        if not tasks:
            return {"success": True, "message": "No tasks.", "tasks": []}

        result_message = "Tasks:\n"
        for i, task in enumerate(tasks):
            status = "✓" if task.get("completed", False) else "○"
            priority = task.get("priority", "medium").upper()
            result_message += f"{i}. {status} [{priority}] {task['task']}\n"

        return {"success": True, "message": result_message.strip(), "tasks": tasks}

    except Exception as e:
        logger.error(f"Error viewing tasks: {e}")
        return {
            "success": False,
            "message": f"Error viewing tasks: {e}",
            "error": str(e),
        }


async def complete_task(params) -> dict[str, Any]:
    """Mark a task as completed."""
    try:
        if not isinstance(params, dict):
            return {
                "success": False,
                "message": "Parameters must be a dictionary",
                "error": "Invalid parameter format",
            }

        task_id = params.get("task_id", -1)

        if task_id < 0:
            return {
                "success": False,
                "message": "Task ID parameter is required",
                "error": "Missing or invalid task_id parameter",
            }

        data = await _load_storage()
        tasks = data.get("tasks", [])

        if task_id >= len(tasks):
            return {
                "success": False,
                "message": f"Task ID {task_id} not found",
                "error": "Invalid task ID",
            }

        tasks[task_id]["completed"] = True
        await _save_storage(data)

        task_text = tasks[task_id]["task"]
        logger.info(f"Completed task: {task_text}")
        return {
            "success": True,
            "message": f'Task "{task_text}" marked as completed.',
            "task": tasks[task_id],
        }

    except Exception as e:
        logger.error(f"Error completing task: {e}")
        return {
            "success": False,
            "message": f"Error completing task: {e}",
            "error": str(e),
        }


async def remove_task(params) -> dict[str, Any]:
    """Remove a task."""
    try:
        if not isinstance(params, dict):
            return {
                "success": False,
                "message": "Parameters must be a dictionary",
                "error": "Invalid parameter format",
            }

        task_id = params.get("task_id", -1)

        if task_id < 0:
            return {
                "success": False,
                "message": "Task ID parameter is required",
                "error": "Missing or invalid task_id parameter",
            }

        data = await _load_storage()
        tasks = data.get("tasks", [])

        if task_id >= len(tasks):
            return {
                "success": False,
                "message": f"Task ID {task_id} not found",
                "error": "Invalid task ID",
            }

        removed_task = tasks.pop(task_id)
        await _save_storage(data)

        task_text = removed_task["task"]
        logger.info(f"Removed task: {task_text}")
        return {
            "success": True,
            "message": f'Task "{task_text}" removed.',
            "removed_task": removed_task,
        }

    except Exception as e:
        logger.error(f"Error removing task: {e}")
        return {
            "success": False,
            "message": f"Error removing task: {e}",
            "error": str(e),
        }


async def add_item(params) -> dict[str, Any]:
    """Add item to a list."""
    try:
        if not isinstance(params, dict):
            return {
                "success": False,
                "message": "Parameters must be a dictionary",
                "error": "Invalid parameter format",
            }

        list_name = params.get("list_name", "").strip()
        item = params.get("item", "").strip()

        if not list_name or not item:
            return {
                "success": False,
                "message": "List name and item parameters are required",
                "error": "Missing required parameters",
            }

        data = await _load_storage()
        if "lists" not in data:
            data["lists"] = {}

        if list_name not in data["lists"]:
            data["lists"][list_name] = []

        data["lists"][list_name].append(item)
        await _save_storage(data)

        logger.info(f"Added item '{item}' to list '{list_name}'")
        return {
            "success": True,
            "message": f'Item "{item}" added to list "{list_name}".',
            "list_name": list_name,
            "item": item,
        }

    except Exception as e:
        logger.error(f"Error adding item to list: {e}")
        return {
            "success": False,
            "message": f"Error adding item to list: {e}",
            "error": str(e),
        }


async def view_list(params) -> dict[str, Any]:
    """View items in a list."""
    try:
        if not isinstance(params, dict):
            return {
                "success": False,
                "message": "Parameters must be a dictionary",
                "error": "Invalid parameter format",
            }

        list_name = params.get("list_name", "").strip()

        if not list_name:
            return {
                "success": False,
                "message": "List name parameter is required",
                "error": "Missing list_name parameter",
            }

        data = await _load_storage()
        lists = data.get("lists", {})

        if list_name not in lists:
            return {
                "success": True,
                "message": f'List "{list_name}" is empty or does not exist.',
                "items": [],
            }

        items = lists[list_name]

        if not items:
            return {
                "success": True,
                "message": f'List "{list_name}" is empty.',
                "items": [],
            }

        result_message = f'List "{list_name}":\n'
        for i, item in enumerate(items, 1):
            result_message += f"{i}. {item}\n"

        return {
            "success": True,
            "message": result_message.strip(),
            "list_name": list_name,
            "items": items,
        }

    except Exception as e:
        logger.error(f"Error viewing list: {e}")
        return {
            "success": False,
            "message": f"Error viewing list: {e}",
            "error": str(e),
        }


async def remove_item(params) -> dict[str, Any]:
    """Remove item from a list."""
    try:
        if not isinstance(params, dict):
            return {
                "success": False,
                "message": "Parameters must be a dictionary",
                "error": "Invalid parameter format",
            }

        list_name = params.get("list_name", "").strip()
        item = params.get("item", "").strip()

        if not list_name or not item:
            return {
                "success": False,
                "message": "List name and item parameters are required",
                "error": "Missing required parameters",
            }

        data = await _load_storage()
        lists = data.get("lists", {})

        if list_name not in lists:
            return {
                "success": False,
                "message": f'List "{list_name}" does not exist.',
                "error": "List not found",
            }

        if item not in lists[list_name]:
            return {
                "success": False,
                "message": f'Item "{item}" not found in list "{list_name}".',
                "error": "Item not found",
            }

        lists[list_name].remove(item)
        await _save_storage(data)

        logger.info(f"Removed item '{item}' from list '{list_name}'")
        return {
            "success": True,
            "message": f'Item "{item}" removed from list "{list_name}".',
            "list_name": list_name,
            "removed_item": item,
        }

    except Exception as e:
        logger.error(f"Error removing item from list: {e}")
        return {
            "success": False,
            "message": f"Error removing item from list: {e}",
            "error": str(e),
        }


async def get_current_time(params) -> dict[str, Any]:
    """Get current time."""
    try:
        now = datetime.now()
        formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")

        return {
            "success": True,
            "message": f"Current time: {formatted_time}",
            "current_time": now.isoformat(),
            "formatted_time": formatted_time,
        }

    except Exception as e:
        logger.error(f"Error getting current time: {e}")
        return {
            "success": False,
            "message": f"Error getting current time: {e}",
            "error": str(e),
        }


async def ask_for_clarification(params) -> dict[str, Any]:
    """Ask user for clarification when AI doesn't understand something."""
    try:
        if not isinstance(params, dict):
            return {
                "success": False,
                "message": "Parameters must be a dictionary",
                "error": "Invalid parameter format",
            }

        question = params.get("question", "").strip()
        context = params.get("context", "").strip()

        if not question:
            return {
                "success": False,
                "message": "Question parameter is required",
                "error": "Missing question parameter",
            }

        # Log the clarification request
        logger.info(f"AI requesting clarification: {question}")

        # Create clarification message that will be sent via WebSocket to client
        clarification_data = {
            "type": "clarification_request",
            "question": question,
            "context": context,
            "timestamp": datetime.now().isoformat(),
            "actions": {
                "wait_for_tts_completion": True,  # Wait for TTS to finish instead of stopping it
                "start_recording_after_tts": True,  # Start recording only after TTS completes
                "show_clarification_ui": True,
            },
        }

        # This will be picked up by the WebSocket handler and sent to client
        # The actual WebSocket sending will be handled by the calling code
        return {
            "success": True,
            "message": f"Clarification requested: {question}",
            "clarification_data": clarification_data,
            "question": question,
            "context": context,
            "requires_user_response": True,
            "action_type": "clarification_request",
        }

    except Exception as e:
        logger.error(f"Error requesting clarification: {e}")
        return {
            "success": False,
            "message": f"Error requesting clarification: {e}",
            "error": str(e),
        }


# Legacy handler functions for backward compatibility
async def handler(
    params: str = "", conversation_history: list[Any] | None = None
) -> str:
    """Legacy handler for backward compatibility."""
    # Simple command parsing for legacy support
    if not params:
        return "Specify a command: set_timer, view_timers, add_event, etc."

    parts = str(params).strip().split()
    if not parts:
        return "Specify a command: set_timer, view_timers, add_event, etc."

    command = parts[0].lower()

    try:
        if command == "set_timer" and len(parts) >= 2:
            duration = parts[1]
            label = " ".join(parts[2:]) if len(parts) > 2 else "timer"
            result = await execute_function(
                "set_timer", {"duration": duration, "label": label}, user_id=0
            )
            return result["message"]

        elif command == "view_timers":
            result = await execute_function("view_timers", {}, user_id=0)
            return result["message"]

        else:
            return f"Unknown command: {command}"

    except Exception as e:
        logger.error(f"Error in legacy handler: {e}")
        return f"Error: {e}"


def register():
    """Register the core module for backward compatibility."""
    return {
        "command": "core",
        "aliases": ["core", "timer", "calendar", "reminder", "task", "list"],
        "description": "Core functionality: timers, events, reminders, tasks, lists",
        "handler": handler,
        "sub_commands": {
            "timer": {
                "description": "Timer management",
                "parameters": {
                    "duration": {
                        "type": "string",
                        "description": "Duration (e.g., '5m', '30s')",
                        "required": True,
                    }
                },
            }
        },
    }


# Initialize storage and start timer polling when module is imported
async def _initialize_module():
    """Initialize the module asynchronously."""
    try:
        await _init_storage()
        _start_timer_polling_task()
        logger.info("Core module initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing core module: {e}")


# Note: In a real async application, this should be called by the main event loop
# For now, we'll start the timer polling when needed
def start_core_module():
    """Start the core module (to be called by the main application)."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is already running, create task
            asyncio.create_task(_initialize_module())
        else:
            # If no loop is running, run initialization
            loop.run_until_complete(_initialize_module())
    except Exception as e:
        logger.error(f"Error starting core module: {e}")


# Start the module when imported (will be handled by main application)
try:
    _start_timer_polling_task()
except Exception as e:
    logger.warning(
        f"Could not start timer polling immediately: {e}. Will retry when event loop is available."
    )


class CoreModule:
    """Core module wrapper class for function calling system."""

    def __init__(self):
        """Initialize the core module."""
        logger.info("CoreModule initialized")
        start_core_module()

    def get_functions(self):
        """Return list of available functions."""
        return [
            {
                "name": "set_timer",
                "description": "Set a timer for a specified duration",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "duration_minutes": {
                            "type": "number",
                            "description": "Duration in minutes",
                        },
                        "label": {
                            "type": "string",
                            "description": "Optional label for the timer",
                        },
                    },
                    "required": ["duration_minutes"],
                },
            },
            {
                "name": "get_active_timers",
                "description": "Get all active timers",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "add_calendar_event",
                "description": "Add an event to the calendar",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Event title"},
                        "date": {
                            "type": "string",
                            "description": "Event date (YYYY-MM-DD)",
                        },
                        "time": {"type": "string", "description": "Event time (HH:MM)"},
                        "description": {
                            "type": "string",
                            "description": "Optional event description",
                        },
                    },
                    "required": ["title", "date", "time"],
                },
            },
            {
                "name": "set_reminder",
                "description": "Set a reminder for a specific date and time",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Reminder message",
                        },
                        "date": {
                            "type": "string",
                            "description": "Reminder date (YYYY-MM-DD)",
                        },
                        "time": {
                            "type": "string",
                            "description": "Reminder time (HH:MM)",
                        },
                    },
                    "required": ["message", "date", "time"],
                },
            },
            {
                "name": "add_task",
                "description": "Add a task to the todo list",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {"type": "string", "description": "Task description"},
                        "priority": {
                            "type": "string",
                            "description": "Task priority (low, medium, high)",
                        },
                        "due_date": {
                            "type": "string",
                            "description": "Optional due date (YYYY-MM-DD)",
                        },
                    },
                    "required": ["task"],
                },
            },
            {
                "name": "add_shopping_item",
                "description": "Add an item to the shopping list",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "item": {
                            "type": "string",
                            "description": "Item to add to shopping list",
                        },
                        "quantity": {
                            "type": "string",
                            "description": "Optional quantity",
                        },
                    },
                    "required": ["item"],
                },
            },
            {
                "name": "get_current_time",
                "description": "Get the current date and time",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "ask_for_clarification",
                "description": "🔍 CRITICAL: Use this function when user's request is unclear, ambiguous, or missing essential information. DO NOT guess or ask in text - use this function instead! The system will wait for TTS to complete, then start recording user's response. Examples: weather without location, music without specification, timer without duration, reminder without details. This provides better user experience by properly handling unclear requests.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "Clear, specific question asking for the missing information. Examples: 'What city would you like the weather for?', 'What song or artist should I play?', 'How long should the timer be?'",
                        },
                        "context": {
                            "type": "string",
                            "description": "What you understood from the user's request so far",
                        },
                    },
                    "required": ["question"],
                },
            },
        ]

    async def execute_function(
        self, function_name: str, parameters: dict[str, Any], user_id: int = 0
    ) -> dict[str, Any]:
        """Execute a function through the CoreModule."""
        try:
            # Map function names to the module-level execute_function
            return await execute_function(function_name, parameters, user_id)
        except Exception as e:
            logger.error(f"Error executing CoreModule function {function_name}: {e}")
            return {
                "success": False,
                "message": f"Error executing function: {e}",
                "error": str(e),
            }
