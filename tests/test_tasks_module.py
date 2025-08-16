import os
import tempfile
import pytest
import asyncio
from datetime import datetime, timedelta, timezone

from config.config_manager import DatabaseManager
from modules.tasks_module import TasksModule


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
    return TasksModule()


def test_add_and_list(module):
    t1 = async_run(module.execute_function("add_task", {"title": "Write tests", "priority": "high"}, 1))
    assert t1["success"] and t1["priority"] == 1
    t2 = async_run(module.execute_function("add_task", {"title": "Walk dog", "priority": "low"}, 1))
    assert t2["success"]
    lst = async_run(module.execute_function("list_tasks", {}, 1))
    assert lst["success"] and len(lst["tasks"]) == 2


def test_status_update_and_overdue(module):
    # Use timezone-aware UTC time to avoid naive datetime warnings
    due_past = (datetime.now(timezone.utc) - timedelta(hours=1)).replace(tzinfo=None).strftime("%Y-%m-%dT%H:%M:%S")
    past = async_run(module.execute_function("add_task", {"title": "Past", "due_at": due_past}, 1))
    # Overdue should show up
    overdue = async_run(module.execute_function("get_overdue_tasks", {}, 1))
    assert any(t["id"] == past["task_id"] for t in overdue["tasks"])

    # Update status to done and ensure it disappears from overdue
    upd = async_run(module.execute_function("update_task_status", {"task_id": past["task_id"], "status": "done"}, 1))
    assert upd["success"]
    overdue2 = async_run(module.execute_function("get_overdue_tasks", {}, 1))
    assert all(t["id"] != past["task_id"] for t in overdue2["tasks"]) or overdue2["tasks"] == []


def test_update_and_delete(module):
    t = async_run(module.execute_function("add_task", {"title": "Temp"}, 1))
    tid = t["task_id"]
    upd = async_run(module.execute_function("update_task", {"task_id": tid, "priority": 1}, 1))
    assert upd["success"]
    delres = async_run(module.execute_function("delete_task", {"task_id": tid}, 1))
    assert delres["success"]
