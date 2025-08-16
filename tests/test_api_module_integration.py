"""Integration test for `api_module` make_api_request.

Directly invokes execute_function (no AI indirection) to avoid prompt ambiguity
while still exercising database logging (api_usage) and HTTP layer.

Test Mode Considerations:
 - Uses https://example.com (stable minimal response, no side effects).
 - Ensures user with id=2 exists (creates if missing) to satisfy FOREIGN KEY.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

SERVER_DIR = Path(__file__).resolve().parent.parent
DB_PATH = SERVER_DIR / "server_data.db"
MODULES_DIR = SERVER_DIR / "modules"


def _ensure_user(user_id: int, username: str = "testuser2") -> None:
    if not DB_PATH.exists():  # if database not yet created tests likely not running full server
        pytest.skip("Brak bazy danych – uruchom wcześniej serwer aby zainicjalizować DB")
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        # Check existence
        cur = conn.execute("SELECT id FROM users WHERE id=?", (user_id,))
        row = cur.fetchone()
        if row:
            return
        # If id=2 not present, either create explicit id or fallback
        try:
            conn.execute(
                "INSERT INTO users (id, username, email, password_hash, is_active) VALUES (?, ?, ?, '', 1)",
                (user_id, username, f"{username}@example.com"),
            )
        except sqlite3.IntegrityError:
            conn.execute(
                "INSERT OR IGNORE INTO users (username, email, password_hash, is_active) VALUES (?, ?, '', 1)",
                (username, f"{username}@example.com"),
            )
        conn.commit()
    finally:
        conn.close()


def _load_api_module():
    import importlib.util

    path = MODULES_DIR / "api_module.py"
    spec = importlib.util.spec_from_file_location("modules.api_module", path)
    if not spec or not spec.loader:
        raise RuntimeError("Nie udało się załadować api_module")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


@pytest.mark.integration
def test_make_api_request_success():
    user_id = 2
    _ensure_user(user_id)
    mod = _load_api_module()
    assert hasattr(mod, "execute_function") and hasattr(mod, "get_functions")

    params = {
        "method": "GET",
        "url": "https://example.com",
        "headers": {},
        "params": {},
    }
    import asyncio

    result = asyncio.run(mod.execute_function("make_api_request", params, user_id))  # type: ignore
    assert isinstance(result, dict) and result.get("success"), result
    data = result.get("data") or {}
    assert data.get("status") == 200, data
    assert "headers" in data, data

    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.execute(
            "SELECT COUNT(*) FROM api_usage WHERE user_id=? AND endpoint LIKE ?", (user_id, "%example.com%")
        )
        count = cur.fetchone()[0]
        assert count >= 1, "Brak wpisu w api_usage"
    finally:
        conn.close()
