import os
import tempfile
import pytest
import asyncio

from config.config_manager import DatabaseManager
from modules.shopping_list_module import ShoppingListModule


@pytest.fixture()
def temp_db(monkeypatch):
    """Provide isolated temporary database with a test user (ID captured)."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    dbm = DatabaseManager(path)
    user_id = dbm.create_user("testuser")  # ensure FK target exists
    assert user_id > 0
    yield dbm
    dbm.close()
    os.remove(path)


@pytest.fixture()
def module(temp_db, monkeypatch):
    # patch get_database_manager used inside module
    from config import config_manager as cm

    cm._db_manager = temp_db  # type: ignore
    return ShoppingListModule()


def test_add_and_view_basic(module):
    # Basic smoke test: add single item and view
    milk_res = async_run(module.execute_function("add_shopping_item", {"item": "Milk"}, 1))
    assert milk_res["success"]
    view = async_run(module.execute_function("view_shopping_list", {}, 1))
    assert any(it["item"].lower() == "milk" for it in view["items"])


def async_run(coro):
    """Run coroutine in a fresh event loop (compatible with sync pytest)."""
    return asyncio.run(coro)


def test_add_and_view(module):
    # Add items
    async_run(module.execute_function("add_shopping_item", {"item": "Milk", "quantity": "2"}, 1))
    async_run(module.execute_function("add_shopping_item", {"item": "Bread"}, 1))
    async_run(module.execute_function("add_shopping_item", {"item": "Eggs", "list_name": "weekend"}, 1))

    # View default list
    res_default = async_run(module.execute_function("view_shopping_list", {"list_name": "shopping"}, 1))
    assert res_default["success"]
    assert any(it["item"].lower() == "milk" for it in res_default["items"])

    # View weekend list
    res_weekend = async_run(module.execute_function("view_shopping_list", {"list_name": "weekend"}, 1))
    assert res_weekend["success"]
    assert len(res_weekend["items"]) == 1
    assert res_weekend["items"][0]["item"].lower() == "eggs"


def test_status_and_clearing(module):
    # Add two items
    milk = async_run(module.execute_function("add_shopping_item", {"item": "Milk"}, 1))
    bread = async_run(module.execute_function("add_shopping_item", {"item": "Bread"}, 1))
    milk_id = milk["item_id"]
    bread_id = bread["item_id"]

    # Update status
    upd = async_run(module.execute_function("update_shopping_item_status", {"item_id": milk_id, "status": "bought"}, 1))
    assert upd["success"]

    # Clear only pending (should remove Bread but keep Milk if include_bought=False)
    cleared = async_run(module.execute_function("clear_shopping_list", {"include_bought": False}, 1))
    assert cleared["success"]

    # View list: milk should remain with status bought
    res = async_run(module.execute_function("view_shopping_list", {}, 1))
    assert any(it["id"] == milk_id and it["status"] == "bought" for it in res["items"])

    # Now clear including bought
    cleared2 = async_run(module.execute_function("clear_shopping_list", {"include_bought": True}, 1))
    assert cleared2["success"]
    res2 = async_run(module.execute_function("view_shopping_list", {}, 1))
    assert res2["items"] == []
