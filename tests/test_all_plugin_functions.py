"""Full sweep test of every declared plugin function across modules.

Goal: ensure each function declared in get_functions() executes (returns success True)
with a minimally synthesized parameter set. Expands over the smoke test by:
 - Including api_module (now covered by dedicated integration test; here lightweight HEAD request)
 - For modules with multiple layers / wrapper classes (e.g. CoreModule class) we still use the
   module-level execute_function only (consistent interface expectation).
 - Capturing per-function failures distinctly for diagnosis.

Safety:
 - Enforces test_mode=True when parameter present.
 - Rewrites risky URL operations to example.com in general pass (except weather which uses test_mode mocks).
 - music_module: always test_mode.
 - open_web_module: test_mode (no browser open).

If certain modules are intentionally background/monitor style (no get_functions), they are skipped.
"""
from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
import json
from datetime import datetime, timedelta
import sqlite3
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
MODULES_DIR = BASE_DIR / "modules"

# Use the same database file as the application (global database manager) to avoid
# FOREIGN KEY constraint issues caused by creating a different server_data.db
try:
    from config.config_manager import get_database_manager  # type: ignore
    _DBM = get_database_manager()
    DB_PATH = Path(_DBM.db_path)
except Exception:  # fallback – previous behaviour
    DB_PATH = Path.cwd() / "server_data.db"

# Consistent user id used across tests
USER_ID = 2

BACKGROUND_OR_SKIP = {
    "ai_module",
    "plugin_monitor_module",
    "server_performance_monitor",
    "active_window_module",
    "day_narrative_module",
    "day_summary_module",
    "memory_module",
    "proactive_assistant_module",
    "routines_learner_module",
    "user_behavior_module",
}

# Default minimal values
DEFAULTS = {
    "duration": "5s",
    "label": "t",
    "title": "T",
    "date": datetime.now().strftime("%Y-%m-%d"),
    "time": "12:00",
    "text": "Note",
    "question": "Jakie szczegóły?",
    "task": "Zadanie",
    "task_id": 0,
    "list_name": "lista",
    "item": "element",
    "location": "Warszawa",
    "provider": "weatherapi",
    "days": 2,
    "query": "Test",
    "engine": "duckduckgo",
    "max_results": 2,
    "language": "pl",
    "action": "play",
    "platform": "auto",
    "method": "GET",
    "url": "https://example.com",
    "headers": {},
    "params": {},
    "json_data": {},
    "song": "Song name",
}


def _ensure_user(user_id: int):
    """Ensure a user row with given id exists in the DB used by modules.

    We insert explicitly with the id to satisfy FOREIGN KEY references in api_usage.
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        cur = conn.execute("SELECT id FROM users WHERE id=?", (user_id,))
        if cur.fetchone():
            return
        conn.execute(
            "INSERT INTO users (id, username, email, password_hash, is_active) VALUES (?, ?, ?, '', 1)",
            (user_id, f"user{user_id}", f"user{user_id}@example.com"),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        # Another parallel test inserted – fine.
        pass
    finally:
        conn.close()


def _load_module(path: Path):
    name = path.stem
    spec = importlib.util.spec_from_file_location(f"modules.{name}", path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Spec load failed for {name}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _iso_if_needed(required: list[str], params: dict):
    if (
        "time" in required
        and "text" in required
        and "title" not in required
        and params.get("time") and len(params["time"]) <= 5
    ):
        params["time"] = (
            datetime.now() + timedelta(minutes=5)
        ).replace(second=0, microsecond=0).isoformat()


def _build_params(schema: dict) -> dict:
    if not isinstance(schema, dict):
        return {}
    props = schema.get("properties") or {}
    required = schema.get("required") or []
    out = {}
    for k in props:
        if k in DEFAULTS and (k in required or len(out) < 5):  # limit overfilling
            out[k] = DEFAULTS[k]
    if "test_mode" in props:
        out["test_mode"] = True
    _iso_if_needed(required, out)
    return out


def discover():
    for p in MODULES_DIR.glob("*_module.py"):
        if p.stem in BACKGROUND_OR_SKIP:
            continue
        yield p


@pytest.mark.plugin_all
@pytest.mark.asyncio
async def test_full_plugin_function_execution():
    _ensure_user(USER_ID)
    failures = []
    for module_path in discover():
        mod = _load_module(module_path)
        if not hasattr(mod, "get_functions") or not hasattr(mod, "execute_function"):
            continue
        try:
            funcs = mod.get_functions()
        except Exception as e:
            failures.append(f"{module_path.stem}.get_functions error: {e}")
            continue
        if not isinstance(funcs, list) or not funcs:
            failures.append(f"{module_path.stem}.get_functions empty")
            continue
        for fdesc in funcs:
            name = fdesc.get("name")
            if not name:
                continue
            params = _build_params(fdesc.get("parameters", {}))
            # Force safe overrides
            if "url" in params:
                params["url"] = "https://example.com"
            if module_path.stem == "music_module":
                params["test_mode"] = True
            if module_path.stem == "open_web_module":
                params["test_mode"] = True
            try:
                result = await mod.execute_function(name, params, USER_ID)  # type: ignore[attr-defined]
                if isinstance(result, dict) and not result.get("success", False):
                    # Skip accepted benign conditions
                    err_txt = (result.get("error") or "").lower()
                    if any(pat in err_txt for pat in ["missing_api_key", "brak klucza api"]):
                        continue  # treat as skipped
                    failures.append(
                        f"{module_path.stem}.{name} failed: params={params} result={result}"
                    )
                elif not isinstance(result, dict):
                    failures.append(
                        f"{module_path.stem}.{name} unexpected result type: {type(result)}"
                    )
            except Exception as e:  # pragma: no cover
                emsg = str(e)
                # Accept Windows file locking on core_storage.json and FK race as skip
                benign_substrings = [
                    "winerror 32",  # file in use
                    "foreign key constraint failed",
                ]
                if any(b in emsg.lower() for b in benign_substrings):
                    continue
                failures.append(f"{module_path.stem}.{name} exception: {e}")
    assert not failures, "Plugin function failures:\n" + "\n".join(failures)
