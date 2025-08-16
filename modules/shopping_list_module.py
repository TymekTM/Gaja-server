"""Shopping List Module

Provides structured shopping list management backed by DatabaseManager.
Exposes function-calling compatible interface for adding, listing, updating
status and clearing shopping list items. Supports multiple named lists.
"""
from __future__ import annotations

import logging
from typing import Any

from config.config_manager import get_database_manager

logger = logging.getLogger(__name__)


class ShoppingListModule:
    """Structured shopping list operations."""

    def __init__(self):
        self.db = get_database_manager()
        logger.info("ShoppingListModule initialized")

    # Function calling schema
    def get_functions(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "add_shopping_item",
                "description": "Add an item to a shopping list (creates list if missing)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "item": {"type": "string", "description": "Item name"},
                        "quantity": {"type": "string", "description": "Quantity e.g. '2' or '500g'", "default": "1"},
                        "list_name": {"type": "string", "description": "List name", "default": "shopping"},
                    },
                    "required": ["item"],
                },
            },
            {
                "name": "view_shopping_list",
                "description": "View items in a shopping list (optionally filter status)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "list_name": {"type": "string", "description": "List name", "default": "shopping"},
                        "status": {"type": "string", "description": "Filter by status: pending|bought", "enum": ["pending", "bought"],},
                    },
                    "required": [],
                },
            },
            {
                "name": "update_shopping_item_status",
                "description": "Update status of a shopping list item (pending|bought|removed)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "item_id": {"type": "integer", "description": "ID of the item"},
                        "status": {"type": "string", "enum": ["pending", "bought", "removed"],},
                    },
                    "required": ["item_id", "status"],
                },
            },
            {
                "name": "remove_shopping_item",
                "description": "Remove an item (soft by default, set hard=true for permanent)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "item_id": {"type": "integer", "description": "ID of the item"},
                        "hard": {"type": "boolean", "description": "Permanent delete"},
                    },
                    "required": ["item_id"],
                },
            },
            {
                "name": "clear_shopping_list",
                "description": "Clear (soft remove) items in a list.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "list_name": {"type": "string", "default": "shopping"},
                        "include_bought": {"type": "boolean", "description": "Also clear bought items", "default": True},
                    },
                    "required": [],
                },
            },
        ]

    async def execute_function(self, function_name: str, parameters: dict[str, Any], user_id: int) -> dict[str, Any]:
        try:
            if function_name == "add_shopping_item":
                item = parameters.get("item", "").strip()
                if not item:
                    return {"success": False, "message": "Item is required", "error": "missing_item"}
                quantity = parameters.get("quantity") or "1"
                list_name = parameters.get("list_name") or "shopping"
                item_id = self.db.add_shopping_item(user_id, item=item, quantity=quantity, list_name=list_name)
                return {"success": True, "message": f'Item "{item}" added to {list_name}', "item_id": item_id}

            if function_name == "view_shopping_list":
                list_name = parameters.get("list_name") or "shopping"
                status = parameters.get("status")
                items = self.db.list_shopping_items(user_id, list_name=list_name, status=status)
                if not items:
                    return {"success": True, "message": f'List "{list_name}" empty.', "items": []}
                lines = [f'Shopping list "{list_name}":']
                for it in items:
                    lines.append(f"#{it['id']} {it['item']} x{it['quantity']} [{it['status']}]")
                return {"success": True, "message": "\n".join(lines), "items": items}

            if function_name == "update_shopping_item_status":
                item_id = int(parameters.get("item_id", 0))
                status = parameters.get("status")
                if not item_id or not status:
                    return {"success": False, "message": "item_id and status required", "error": "missing_params"}
                ok = self.db.update_shopping_item_status(user_id, item_id, status)
                return {"success": ok, "message": ("Status updated" if ok else "Item not found"), "item_id": item_id, "status": status}

            if function_name == "remove_shopping_item":
                item_id = int(parameters.get("item_id", 0))
                hard = bool(parameters.get("hard", False))
                if not item_id:
                    return {"success": False, "message": "item_id required", "error": "missing_item_id"}
                ok = self.db.remove_shopping_item(user_id, item_id, hard=hard)
                return {"success": ok, "message": ("Item removed" if ok else "Item not found"), "item_id": item_id, "hard": hard}

            if function_name == "clear_shopping_list":
                list_name = parameters.get("list_name") or "shopping"
                include_bought = bool(parameters.get("include_bought", True))
                cleared = self.db.clear_shopping_list(user_id, list_name, include_bought=include_bought)
                return {"success": True, "message": f"Cleared {cleared} items from {list_name}", "cleared": cleared}

            return {"success": False, "message": f"Unknown function {function_name}", "error": "unknown_function"}
        except Exception as e:
            logger.error(f"Error executing shopping list function {function_name}: {e}")
            return {"success": False, "message": f"Error: {e}", "error": str(e)}


def get_functions() -> list[dict[str, Any]]:  # compatibility for dynamic loader
    return ShoppingListModule().get_functions()
