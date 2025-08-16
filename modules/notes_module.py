"""Notes Module

Provides CRUD operations over user notes with tag filtering and search.
"""
from __future__ import annotations

import logging
from typing import Any

from config.config_manager import get_database_manager

logger = logging.getLogger(__name__)


class NotesModule:
    def __init__(self):
        self.db = get_database_manager()
        logger.info("NotesModule initialized")

    def get_functions(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "create_note",
                "description": "Create a note with optional title and tags",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Note title"},
                        "content": {"type": "string", "description": "Note content"},
                        "tags": {"type": "array", "items": {"type": "string"}, "description": "List of tags"},
                    },
                    "required": ["content"],
                },
            },
            {
                "name": "list_notes",
                "description": "List notes (filter by tag or search term)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tag": {"type": "string"},
                        "search": {"type": "string"},
                        "limit": {"type": "integer", "default": 50},
                    },
                    "required": [],
                },
            },
            {
                "name": "update_note",
                "description": "Update note fields (title/content/tags)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "note_id": {"type": "integer"},
                        "title": {"type": "string"},
                        "content": {"type": "string"},
                        "tags": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["note_id"],
                },
            },
            {
                "name": "delete_note",
                "description": "Delete a note permanently",
                "parameters": {
                    "type": "object",
                    "properties": {"note_id": {"type": "integer"}},
                    "required": ["note_id"],
                },
            },
        ]

    async def execute_function(self, function_name: str, parameters: dict[str, Any], user_id: int) -> dict[str, Any]:
        try:
            if function_name == "create_note":
                content = (parameters.get("content") or "").strip()
                if not content:
                    return {"success": False, "message": "content required", "error": "missing_content"}
                title = parameters.get("title")
                tags = parameters.get("tags") if isinstance(parameters.get("tags"), list) else None
                note_id = self.db.add_note(user_id, title=title, content=content, tags=tags)
                return {"success": True, "message": f"Note created (id={note_id})", "note_id": note_id}

            if function_name == "list_notes":
                tag = parameters.get("tag")
                search = parameters.get("search")
                limit = int(parameters.get("limit", 50))
                notes = self.db.list_notes(user_id, tag=tag, search=search, limit=limit)
                if not notes:
                    return {"success": True, "message": "No notes found", "notes": []}
                lines = ["Notes:"]
                for n in notes:
                    tags_repr = ("[" + ",".join(n["tags"]) + "]") if n["tags"] else ""
                    title_part = n["title"] or n["content"][:30]
                    lines.append(f"#{n['id']} {title_part} {tags_repr}")
                return {"success": True, "message": "\n".join(lines), "notes": notes}

            if function_name == "update_note":
                note_id = int(parameters.get("note_id", 0))
                if not note_id:
                    return {"success": False, "message": "note_id required", "error": "missing_note_id"}
                title = parameters.get("title")
                content = parameters.get("content")
                tags = parameters.get("tags") if isinstance(parameters.get("tags"), list) else None
                ok = self.db.update_note(user_id, note_id, title=title, content=content, tags=tags)
                return {"success": ok, "message": ("Note updated" if ok else "Not found"), "note_id": note_id}

            if function_name == "delete_note":
                note_id = int(parameters.get("note_id", 0))
                if not note_id:
                    return {"success": False, "message": "note_id required", "error": "missing_note_id"}
                ok = self.db.delete_note(user_id, note_id)
                return {"success": ok, "message": ("Note deleted" if ok else "Not found"), "note_id": note_id}

            return {"success": False, "message": f"Unknown function {function_name}", "error": "unknown_function"}
        except Exception as e:
            logger.error(f"NotesModule error in {function_name}: {e}")
            return {"success": False, "message": f"Error: {e}", "error": str(e)}


def get_functions():  # dynamic loader compatibility
    return NotesModule().get_functions()
