"""Memory module for GAJA Assistant Simple database-backed memory system for the
server."""

import logging
from typing import Any

from config_manager import DatabaseManager

logger = logging.getLogger(__name__)


class MemoryModule:
    """Simple memory module using database backend."""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        logger.info("MemoryModule initialized")

    async def store_memory(
        self, user_id: str, key: str, value: Any, category: str = "general"
    ) -> bool:
        """Store a memory item for a user."""
        try:
            # Simple implementation - could be expanded
            await self.db_manager.save_interaction(
                user_id, f"MEMORY_STORE:{key}", str(value)
            )
            logger.info(f"Stored memory for user {user_id}: {key}")
            return True
        except Exception as e:
            logger.error(f"Error storing memory for user {user_id}: {e}")
            return False

    async def get_memory(self, user_id: str, key: str) -> str | None:
        """Get a memory item for a user."""
        try:
            # Simple implementation - get from history
            history = await self.db_manager.get_user_history(user_id)
            for item in history:
                if item["role"] == "user" and item["content"].startswith(
                    f"MEMORY_STORE:{key}"
                ):
                    return item["content"].replace(f"MEMORY_STORE:{key}", "").strip()
            return None
        except Exception as e:
            logger.error(f"Error getting memory for user {user_id}: {e}")
            return None

    async def search_memories(self, user_id: str, query: str) -> list[dict]:
        """Search memories for a user."""
        try:
            history = await self.db_manager.get_user_history(user_id)
            results = []
            for item in history:
                if query.lower() in item["content"].lower():
                    results.append(item)
            return results
        except Exception as e:
            logger.error(f"Error searching memories for user {user_id}: {e}")
            return []


# Plugin functions for the function calling system
async def store_memory(
    user_id: str, key: str, value: str, category: str = "general"
) -> dict[str, Any]:
    """Store a memory item."""
    try:
        from server_main import server_app

        from . import server_main

        success = await server_app.memory_module.store_memory(
            user_id, key, value, category
        )
        return {
            "success": success,
            "message": f"Stored memory: {key}" if success else "Failed to store memory",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


async def get_memory(user_id: str, key: str) -> dict[str, Any]:
    """Get a memory item."""
    try:
        from server_main import server_app

        from . import server_main

        value = await server_app.memory_module.get_memory(user_id, key)
        return {
            "success": value is not None,
            "value": value,
            "message": f"Retrieved memory: {key}" if value else "Memory not found",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


async def search_memories(user_id: str, query: str) -> dict[str, Any]:
    """Search memories."""
    try:
        from server_main import server_app

        from . import server_main

        results = await server_app.memory_module.search_memories(user_id, query)
        return {"success": True, "results": results, "count": len(results)}
    except Exception as e:
        return {"success": False, "error": str(e)}


# Plugin metadata
PLUGIN_FUNCTIONS = {
    "store_memory": {
        "function": store_memory,
        "description": "Store a memory item for the user",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Memory key/identifier"},
                "value": {"type": "string", "description": "Memory value to store"},
                "category": {
                    "type": "string",
                    "description": "Memory category",
                    "default": "general",
                },
            },
            "required": ["key", "value"],
        },
    },
    "get_memory": {
        "function": get_memory,
        "description": "Retrieve a memory item by key",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Memory key to retrieve"}
            },
            "required": ["key"],
        },
    },
    "search_memories": {
        "function": search_memories,
        "description": "Search through user memories",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Search query"}},
            "required": ["query"],
        },
    },
}
