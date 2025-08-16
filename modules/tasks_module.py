"""Tasks Module

Advanced task management with priorities (1 high - 5 low) and due dates.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from config.config_manager import get_database_manager

logger = logging.getLogger(__name__)

PRIORITY_MAP = {
    "urgent": 1,
    "high": 1,
    "medium": 3,
    "normal": 3,
    "low": 5,
}


class TasksModule:
    def __init__(self):
        self.db = get_database_manager()
        logger.info("TasksModule initialized")

    def get_functions(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "add_task",
                "description": "Add a task with optional priority and due date",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "description": {"type": "string"},
                        "priority": {"type": "string", "description": "urgent|high|medium|low or numeric 1-5"},
                        "due_at": {"type": "string", "description": "ISO datetime (YYYY-MM-DDTHH:MM)"},
                    },
                    "required": ["title"],
                },
            },
            {
                "name": "list_tasks",
                "description": "List tasks with ordering by status, priority, due date",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "enum": ["open", "in_progress", "done", "cancelled"]},
                        "include_done": {"type": "boolean", "default": True},
                        "limit": {"type": "integer", "default": 100},
                    },
                    "required": [],
                },
            },
            {
                "name": "update_task_status",
                "description": "Update status of a task",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "integer"},
                        "status": {"type": "string", "enum": ["open", "in_progress", "done", "cancelled"]},
                    },
                    "required": ["task_id", "status"],
                },
            },
            {
                "name": "update_task",
                "description": "Update task fields (title, description, priority, due_at)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "integer"},
                        "title": {"type": "string"},
                        "description": {"type": "string"},
                        "priority": {"type": "string"},
                        "due_at": {"type": "string"},
                    },
                    "required": ["task_id"],
                },
            },
            {
                "name": "delete_task",
                "description": "Delete a task",
                "parameters": {
                    "type": "object",
                    "properties": {"task_id": {"type": "integer"}},
                    "required": ["task_id"],
                },
            },
            {
                "name": "get_overdue_tasks",
                "description": "List overdue tasks (not done and past due date)",
                "parameters": {"type": "object", "properties": {}},
            },
        ]

    def _priority_to_int(self, value: str | int | None) -> int:
        if value is None:
            return 3
        if isinstance(value, int):
            return min(5, max(1, value))
        v = value.strip().lower()
        if v in PRIORITY_MAP:
            return PRIORITY_MAP[v]
        try:
            return min(5, max(1, int(v)))
        except ValueError:
            return 3

    def _parse_due(self, due_str: str | None) -> datetime | None:
        if not due_str:
            return None
        try:
            # Accept both date and datetime
            if len(due_str) == 10:
                return datetime.fromisoformat(due_str + "T00:00:00")
            return datetime.fromisoformat(due_str)
        except Exception:
            return None

    async def execute_function(self, function_name: str, parameters: dict[str, Any], user_id: int) -> dict[str, Any]:
        try:
            if function_name == "add_task":
                title = (parameters.get("title") or "").strip()
                if not title:
                    return {"success": False, "message": "title required", "error": "missing_title"}
                description = parameters.get("description") or ""
                priority_raw = parameters.get("priority")
                prio = self._priority_to_int(priority_raw)
                due_at = self._parse_due(parameters.get("due_at"))
                task_id = self.db.add_task_db(user_id, title=title, description=description, priority=prio, due_at=due_at)
                return {"success": True, "message": f"Task created (id={task_id})", "task_id": task_id, "priority": prio}

            if function_name == "list_tasks":
                status = parameters.get("status")
                include_done = bool(parameters.get("include_done", True))
                limit = int(parameters.get("limit", 100))
                tasks = self.db.list_tasks(user_id, status=status, include_done=include_done, limit=limit)
                if not tasks:
                    return {"success": True, "message": "No tasks", "tasks": []}
                lines = ["Tasks:"]
                for t in tasks:
                    lines.append(f"#{t['id']} [P{t['priority']}] {t['title']} ({t['status']})" + (f" due {t['due_at']}" if t['due_at'] else ""))
                return {"success": True, "message": "\n".join(lines), "tasks": tasks}

            if function_name == "update_task_status":
                task_id = int(parameters.get("task_id", 0))
                status = parameters.get("status")
                if not task_id or not status:
                    return {"success": False, "message": "task_id and status required", "error": "missing_params"}
                ok = self.db.update_task_status(user_id, task_id, status)
                return {"success": ok, "message": ("Status updated" if ok else "Not found"), "task_id": task_id, "status": status}

            if function_name == "update_task":
                task_id = int(parameters.get("task_id", 0))
                if not task_id:
                    return {"success": False, "message": "task_id required", "error": "missing_task_id"}
                title = parameters.get("title")
                description = parameters.get("description")
                priority_raw = parameters.get("priority")
                priority_val = self._priority_to_int(priority_raw) if priority_raw is not None else None
                due_at = self._parse_due(parameters.get("due_at")) if parameters.get("due_at") else None
                ok = self.db.update_task(user_id, task_id, title=title, description=description, priority=priority_val, due_at=due_at)
                return {"success": ok, "message": ("Task updated" if ok else "Not found"), "task_id": task_id}

            if function_name == "delete_task":
                task_id = int(parameters.get("task_id", 0))
                if not task_id:
                    return {"success": False, "message": "task_id required", "error": "missing_task_id"}
                ok = self.db.delete_task(user_id, task_id)
                return {"success": ok, "message": ("Task deleted" if ok else "Not found"), "task_id": task_id}

            if function_name == "get_overdue_tasks":
                tasks = self.db.get_overdue_tasks(user_id)
                return {"success": True, "message": f"{len(tasks)} overdue tasks" if tasks else "No overdue tasks", "tasks": tasks}

            return {"success": False, "message": f"Unknown function {function_name}", "error": "unknown_function"}
        except Exception as e:
            logger.error(f"TasksModule error in {function_name}: {e}")
            return {"success": False, "message": f"Error: {e}", "error": str(e)}


def get_functions():  # dynamic loader compatibility
    return TasksModule().get_functions()
