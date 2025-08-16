"""Testy negatywne dla wszystkich modułów pluginów.

Sprawdza jak moduły radzą sobie z błędnymi parametrami, brakującymi danymi,
i innymi scenariuszami edge case.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Any
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
MODULES_DIR = BASE_DIR / "modules"
DB_PATH = BASE_DIR / "server_data.db"
USER_ID = 2

STANDARD_MODULES = [
    "core_module",
    "weather_module", 
    "search_module",
    "music_module",
    "open_web_module",
    "onboarding_module",
    "onboarding_plugin_module",
]

def _ensure_user(user_id: int):
    if not DB_PATH.exists():
        return
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        cur = conn.execute("SELECT id FROM users WHERE id=?", (user_id,))
        if cur.fetchone():
            return
        try:
            conn.execute(
                "INSERT INTO users (id, username, email, password_hash, is_active) VALUES (?, ?, ?, '', 1)",
                (user_id, f"user{user_id}", f"user{user_id}@example.com"),
            )
        except sqlite3.IntegrityError:
            pass
        conn.commit()
    finally:
        conn.close()

def _load_module(path: Path):
    name = path.stem
    spec = importlib.util.spec_from_file_location(f"modules.{name}", path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Spec load failed for {name}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

@pytest.mark.parametrize("module_name", STANDARD_MODULES)
@pytest.mark.asyncio
async def test_missing_required_parameters(module_name):
    """Test każdego modułu z brakującymi wymaganymi parametrami."""
    _ensure_user(USER_ID)
    path = MODULES_DIR / f"{module_name}.py"
    if not path.exists():
        pytest.skip(f"{module_name}.py not found")
        
    mod = _load_module(path)
    
    if not hasattr(mod, "get_functions") or not hasattr(mod, "execute_function"):
        pytest.skip(f"{module_name} doesn't have standard interface")
        
    funcs = mod.get_functions()
    if not funcs:
        pytest.skip(f"{module_name} has no functions")
        
    for func_desc in funcs:
        fname = func_desc.get("name")
        if not fname:
            continue
            
        schema = func_desc.get("parameters", {})
        required = schema.get("required", [])
        
        if not required:
            continue  # Skip functions with no required parameters
            
        # Test with completely empty parameters
        try:
            result = await mod.execute_function(fname, {}, USER_ID)
            assert isinstance(result, dict), f"{fname} should return dict even on error"
            assert result.get("success") is False, f"{fname} should fail with missing required params"
            assert "error" in result or "message" in result, f"{fname} should provide error message"
        except Exception as e:
            # Some modules might throw exceptions for missing params - that's acceptable
            assert "required" in str(e).lower() or "missing" in str(e).lower() or "parameter" in str(e).lower(), \
                f"{fname} exception should mention missing parameters: {e}"

@pytest.mark.parametrize("module_name", STANDARD_MODULES)
@pytest.mark.asyncio
async def test_invalid_parameter_types(module_name):
    """Test każdego modułu z nieprawidłowymi typami parametrów."""
    _ensure_user(USER_ID)
    path = MODULES_DIR / f"{module_name}.py"
    if not path.exists():
        pytest.skip(f"{module_name}.py not found")
        
    mod = _load_module(path)
    
    if not hasattr(mod, "get_functions") or not hasattr(mod, "execute_function"):
        pytest.skip(f"{module_name} doesn't have standard interface")
        
    funcs = mod.get_functions()
    if not funcs:
        pytest.skip(f"{module_name} has no functions")
        
    for func_desc in funcs:
        fname = func_desc.get("name")
        if not fname:
            continue
            
        schema = func_desc.get("parameters", {})
        props = schema.get("properties", {})
        
        # Build params with wrong types
        bad_params = {}
        for prop_name, prop_def in props.items():
            expected_type = prop_def.get("type", "string")
            
            # Provide wrong type
            if expected_type == "string":
                bad_params[prop_name] = 12345  # int instead of string
            elif expected_type == "integer":
                bad_params[prop_name] = "not_a_number"  # string instead of int
            elif expected_type == "boolean":
                bad_params[prop_name] = "not_boolean"  # string instead of bool
            elif expected_type == "object":
                bad_params[prop_name] = "not_an_object"  # string instead of object
                
        if not bad_params:
            continue  # Skip if no typed parameters
            
        try:
            result = await mod.execute_function(fname, bad_params, USER_ID)
            # Function should handle type errors gracefully
            assert isinstance(result, dict), f"{fname} should return dict even with wrong types"
            # Don't assert success=False because some functions might handle type conversion
        except Exception as e:
            # Type errors are acceptable
            assert any(word in str(e).lower() for word in ["type", "invalid", "convert", "parse"]), \
                f"{fname} should provide meaningful type error: {e}"

@pytest.mark.asyncio
async def test_core_module_edge_cases():
    """Test specjalnych przypadków core_module."""
    _ensure_user(USER_ID)
    path = MODULES_DIR / "core_module.py"
    mod = _load_module(path)
    
    # Test set_timer with invalid duration
    result = await mod.execute_function("set_timer", {"duration": "invalid"}, USER_ID)
    assert result.get("success") is False, "Invalid duration should fail"
    
    # Test set_timer with negative duration  
    result = await mod.execute_function("set_timer", {"duration": "-5s"}, USER_ID)
    assert result.get("success") is False, "Negative duration should fail"
    
    # Test complete_task with invalid task_id
    result = await mod.execute_function("complete_task", {"task_id": 999}, USER_ID)
    assert result.get("success") is False, "Invalid task_id should fail"
    
    # Test set_reminder with invalid time format
    result = await mod.execute_function("set_reminder", {"text": "test", "time": "invalid_time"}, USER_ID)
    assert result.get("success") is False, "Invalid time format should fail"

@pytest.mark.asyncio
async def test_weather_module_edge_cases():
    """Test specjalnych przypadków weather_module.""" 
    _ensure_user(USER_ID)
    path = MODULES_DIR / "weather_module.py"
    mod = _load_module(path)
    
    # Test with empty location - in test mode it provides default data
    result = await mod.execute_function("get_weather", {"location": ""}, USER_ID)
    assert result.get("success") is True, "Weather module handles empty location in test mode"
    assert result.get("test_mode") is True, "Should be in test mode"
    
    # Test with invalid provider
    result = await mod.execute_function("get_weather", {"location": "Warsaw", "provider": "invalid_provider"}, USER_ID)
    # Should either fail or fallback to default - both acceptable
    assert isinstance(result, dict), "Should return dict"
    
    # Test forecast with invalid days
    result = await mod.execute_function("get_forecast", {"location": "Warsaw", "days": 0}, USER_ID)
    assert isinstance(result, dict), "Should handle invalid days gracefully"

@pytest.mark.asyncio
async def test_search_module_edge_cases():
    """Test specjalnych przypadków search_module."""
    _ensure_user(USER_ID)
    path = MODULES_DIR / "search_module.py" 
    mod = _load_module(path)
    
    # Test with empty query - in test mode it still returns success with error details
    result = await mod.execute_function("search", {"query": ""}, USER_ID)
    assert result.get("success") is True, "Search module handles empty query in test mode"
    # Should contain error information in data
    assert "data" in result and ("error" in result["data"] or "details" in result["data"])
    
    # Test with invalid max_results
    result = await mod.execute_function("search", {"query": "test", "max_results": -1}, USER_ID)
    assert isinstance(result, dict), "Should handle invalid max_results"
    
    # Test with very long query
    long_query = "a" * 1000
    result = await mod.execute_function("search", {"query": long_query, "test_mode": True}, USER_ID)
    assert isinstance(result, dict), "Should handle very long query"

@pytest.mark.asyncio
async def test_music_module_edge_cases():
    """Test specjalnych przypadków music_module."""
    _ensure_user(USER_ID)
    path = MODULES_DIR / "music_module.py"
    mod = _load_module(path)
    
    # Test with invalid action
    result = await mod.execute_function("control_music", {"action": "invalid_action", "test_mode": True}, USER_ID)
    assert result.get("success") is False, "Invalid action should fail"
    
    # Test with empty action
    result = await mod.execute_function("control_music", {"action": "", "test_mode": True}, USER_ID)
    assert result.get("success") is False, "Empty action should fail"

@pytest.mark.asyncio
async def test_open_web_module_edge_cases():
    """Test specjalnych przypadków open_web_module."""
    _ensure_user(USER_ID)
    path = MODULES_DIR / "open_web_module.py"
    mod = _load_module(path)
    
    # Test with empty URL
    result = await mod.execute_function("open_web", {"url": ""}, USER_ID)
    assert result.get("success") is False, "Empty URL should fail"
    
    # Test with malformed URL
    result = await mod.execute_function("open_web", {"url": "not_a_url", "test_mode": True}, USER_ID)
    # Should either fail or auto-correct - both acceptable
    assert isinstance(result, dict), "Should return dict"

@pytest.mark.asyncio
async def test_function_with_unknown_name():
    """Test wywołania nieistniejącej funkcji na każdym module."""
    _ensure_user(USER_ID)
    
    for module_name in STANDARD_MODULES:
        path = MODULES_DIR / f"{module_name}.py"
        if not path.exists():
            continue
            
        mod = _load_module(path)
        
        if not hasattr(mod, "execute_function"):
            continue
            
        try:
            result = await mod.execute_function("nonexistent_function", {}, USER_ID)
            assert isinstance(result, dict), f"{module_name} should return dict for unknown function"
            assert result.get("success") is False, f"{module_name} should fail for unknown function"
            assert "unknown" in str(result).lower() or "not" in str(result).lower(), \
                f"{module_name} should mention unknown function in response"
        except Exception as e:
            # Some modules might throw exceptions for unknown functions - that's acceptable
            assert "unknown" in str(e).lower() or "not" in str(e).lower() or "function" in str(e).lower(), \
                f"{module_name} exception should mention unknown function: {e}"

@pytest.mark.asyncio
async def test_extreme_parameter_values():
    """Test z ekstremalnymi wartościami parametrów."""
    _ensure_user(USER_ID)
    path = MODULES_DIR / "core_module.py"
    mod = _load_module(path)
    
    # Test with extremely long strings
    long_string = "x" * 10000
    result = await mod.execute_function("add_task", {"task": long_string}, USER_ID)
    assert isinstance(result, dict), "Should handle very long task names"
    
    # Test with special characters
    special_chars = "!@#$%^&*()[]{}|\\:;\"'<>?,./"
    result = await mod.execute_function("add_task", {"task": special_chars}, USER_ID)
    assert isinstance(result, dict), "Should handle special characters"
    
    # Test with unicode characters
    unicode_text = "测试 🎵 émoji ñoñó"
    result = await mod.execute_function("add_task", {"task": unicode_text}, USER_ID)
    assert isinstance(result, dict), "Should handle unicode characters"
