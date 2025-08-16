import os
import tempfile
import pytest
import asyncio

from config.config_manager import DatabaseManager
from modules.notes_module import NotesModule


def async_run(coro):
    return asyncio.run(coro)


@pytest.fixture()
def temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    dbm = DatabaseManager(path)
    user_id = dbm.create_user("testuser")
    assert user_id > 0
    yield dbm
    dbm.close()
    os.remove(path)


@pytest.fixture()
def module(temp_db, monkeypatch):
    from config import config_manager as cm
    cm._db_manager = temp_db  # type: ignore
    return NotesModule()


def test_create_and_list(module):
    n1 = async_run(module.execute_function("create_note", {"content": "Buy milk", "tags": ["shopping", "urgent"]}, 1))
    assert n1["success"]
    n2 = async_run(module.execute_function("create_note", {"title": "Ideas", "content": "Build AI assistant", "tags": ["project"]}, 1))
    assert n2["success"]

    lst_all = async_run(module.execute_function("list_notes", {}, 1))
    assert lst_all["success"] and len(lst_all["notes"]) == 2

    lst_tag = async_run(module.execute_function("list_notes", {"tag": "shopping"}, 1))
    assert lst_tag["success"] and len(lst_tag["notes"]) == 1

    search = async_run(module.execute_function("list_notes", {"search": "milk"}, 1))
    assert search["success"] and len(search["notes"]) == 1


def test_update_and_delete(module):
    note = async_run(module.execute_function("create_note", {"content": "Test note"}, 1))
    note_id = note["note_id"]
    upd = async_run(module.execute_function("update_note", {"note_id": note_id, "title": "Updated", "tags": ["x"]}, 1))
    assert upd["success"]
    lst = async_run(module.execute_function("list_notes", {"search": "Updated"}, 1))
    assert lst["notes"]
    dele = async_run(module.execute_function("delete_note", {"note_id": note_id}, 1))
    assert dele["success"]
    lst2 = async_run(module.execute_function("list_notes", {}, 1))
    # Depending on order, might have only previous notes -> ensure note removed
    assert all(n["id"] != note_id for n in lst2["notes"]) or lst2["notes"] == []
