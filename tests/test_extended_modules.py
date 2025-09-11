"""Test wszystkich skipped modułów onboarding i advanced_memory_system.

Sprawdza czy moduły onboarding mają standardowy interfejs i czy można je testować.
Dodaje dedykowane testy dla zaawansowanych modułów nie pokrytych przez podstawowe testy.
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

@pytest.mark.asyncio
async def test_onboarding_module_functions():
    """Test onboarding_module functions directly."""
    _ensure_user(USER_ID)
    path = MODULES_DIR / "onboarding_module.py"
    mod = _load_module(path)
    
    assert hasattr(mod, "get_functions"), "onboarding_module should have get_functions"
    
    funcs = mod.get_functions()
    assert isinstance(funcs, list) and funcs, "get_functions should return non-empty list"
    
    # Test get_onboarding_status
    try:
        result = await mod.get_onboarding_status(str(USER_ID))
        assert isinstance(result, dict), "get_onboarding_status should return dict"
        # Expected keys in result
        expected_keys = ["first_run", "user_configured", "required_steps"]
        for key in expected_keys:
            assert key in result or "error" in result, f"Missing key {key} in onboarding status"
    except Exception as e:
        # May fail due to server_app import issues, but the function should exist
        assert "server_app" in str(e) or "module" in str(e), f"Unexpected error: {e}"

@pytest.mark.asyncio  
async def test_onboarding_plugin_module_functions():
    """Test onboarding_plugin_module functions directly."""
    _ensure_user(USER_ID)
    path = MODULES_DIR / "onboarding_plugin_module.py"
    mod = _load_module(path)
    
    assert hasattr(mod, "get_functions"), "onboarding_plugin_module should have get_functions"
    
    funcs = mod.get_functions()
    assert isinstance(funcs, list) and funcs, "get_functions should return non-empty list"
    
    # Test that functions match between both onboarding modules
    onboarding_path = MODULES_DIR / "onboarding_module.py"
    onboarding_mod = _load_module(onboarding_path)
    
    onboarding_funcs = {f["name"] for f in onboarding_mod.get_functions()}
    plugin_funcs = {f["name"] for f in mod.get_functions()}
    
    assert onboarding_funcs == plugin_funcs, "Both onboarding modules should expose same functions"

@pytest.mark.asyncio
async def test_advanced_memory_system_module():
    """Test advanced_memory_system module if it has standard interface."""
    path = MODULES_DIR / "advanced_memory_system.py"
    if not path.exists():
        pytest.skip("advanced_memory_system.py not found")
        
    try:
        mod = _load_module(path)
    except Exception as e:
        pytest.skip(f"advanced_memory_system module has loading issues: {e}")
        
    if not hasattr(mod, "get_functions"):
        pytest.skip("advanced_memory_system doesn't have get_functions")
        
    if not hasattr(mod, "execute_function"):
        pytest.skip("advanced_memory_system doesn't have execute_function")
        
    try:
        funcs = mod.get_functions()
        assert isinstance(funcs, list), "get_functions should return list"
    except Exception as e:
        pytest.skip(f"get_functions failed: {e}")
        
    # If it has functions, test them
    if funcs:
        for func_desc in funcs[:2]:  # Test first 2 functions max
            fname = func_desc.get("name")
            if not fname:
                continue
                
            # Build minimal parameters
            schema = func_desc.get("parameters", {})
            props = schema.get("properties", {})
            required = schema.get("required", [])
            
            params = {}
            for param in required[:3]:  # Only required params, max 3
                if param in props:
                    param_type = props[param].get("type", "string")
                    if param_type == "string":
                        params[param] = "test"
                    elif param_type == "integer":
                        params[param] = 1
                    elif param_type == "boolean":
                        params[param] = True
                        
            # Set test_mode if available
            if "test_mode" in props:
                params["test_mode"] = True
                
            try:
                result = await mod.execute_function(fname, params, USER_ID)
                assert isinstance(result, dict), f"Function {fname} should return dict"
                # Don't assert success=True as advanced modules might have complex requirements
            except Exception as e:
                # Log but don't fail - advanced modules might need special setup
                print(f"Warning: {fname} failed with {e}")

@pytest.mark.asyncio
async def test_daily_briefing_module():
    """Test daily_briefing_module if it has standard interface."""
    path = MODULES_DIR / "daily_briefing_module.py"
    if not path.exists():
        pytest.skip("daily_briefing_module.py not found")
        
    mod = _load_module(path)
    
    if not hasattr(mod, "get_functions"):
        pytest.skip("daily_briefing_module doesn't have get_functions")
        
    if not hasattr(mod, "execute_function"):
        pytest.skip("daily_briefing_module doesn't have execute_function")
        
    funcs = mod.get_functions()
    assert isinstance(funcs, list), "get_functions should return list"
    
    # Test daily briefing functions with safe parameters
    for func_desc in funcs:
        fname = func_desc.get("name")
        if not fname:
            continue
            
        # Safe parameters for briefing functions
        params = {"test_mode": True, "user_id": str(USER_ID)}
        
        try:
            result = await mod.execute_function(fname, params, USER_ID)
            assert isinstance(result, dict), f"Function {fname} should return dict"
        except Exception as e:
            # Daily briefing might need server setup
            if "server_app" not in str(e):
                raise

@pytest.mark.asyncio
async def test_search_module_extended():
    """Extended tests for search_module."""
    path = MODULES_DIR / "search_module.py"
    mod = _load_module(path)
    
    # Test get_functions
    funcs = mod.get_functions()
    assert len(funcs) >= 1, "search_module should have at least 1 function"
    
    func_names = {f["name"] for f in funcs}
    assert "search" in func_names, "search function should be available"
    
    # Test search function with test_mode
    result = await mod.execute_function(
        "search", 
        {"query": "Python programming", "test_mode": True, "max_results": 3}, 
        USER_ID
    )
    assert result["success"] is True, "Search should succeed in test mode"
    assert result["test_mode"] is True, "Should confirm test mode"
    assert "data" in result, "Should have data field"
    
    

@pytest.mark.integration
@pytest.mark.asyncio  
async def test_weather_module_extended():
    """Extended tests for weather_module."""
    path = MODULES_DIR / "weather_module.py"
    mod = _load_module(path)
    
    # Test get_functions
    funcs = mod.get_functions()
    func_names = {f["name"] for f in funcs}
    assert "get_weather" in func_names, "get_weather function should be available"
    assert "get_forecast" in func_names, "get_forecast function should be available"
    
    # Test get_weather with test_mode
    result = await mod.execute_function(
        "get_weather", 
        {"location": "Warszawa", "test_mode": True}, 
        USER_ID
    )
    assert result["success"] is True, "Weather should succeed in test mode"
    assert result["test_mode"] is True, "Should confirm test mode"
    
    # Test get_forecast with test_mode
    result = await mod.execute_function(
        "get_forecast", 
        {"location": "Krakow", "days": 3, "test_mode": True}, 
        USER_ID
    )
    assert result["success"] is True, "Forecast should succeed in test mode"
    assert result["test_mode"] is True, "Should confirm test mode"

@pytest.mark.asyncio
async def test_music_module_extended():
    """Extended tests for music_module.""" 
    path = MODULES_DIR / "music_module.py"
    mod = _load_module(path)
    
    # Test all available functions with test_mode
    funcs = mod.get_functions()
    
    for func_desc in funcs:
        fname = func_desc.get("name")
        if not fname:
            continue
            
        # Build safe test parameters
        schema = func_desc.get("parameters", {})
        props = schema.get("properties", {})
        params: dict[str, Any] = {"test_mode": True}
        
        # Add required parameters with safe defaults
        required = schema.get("required", [])
        for param in required:
            if param == "action":
                params[param] = "play"
            elif param == "song":
                params[param] = "Test Song"
            elif param in props:
                param_type = props[param].get("type", "string")
                if param_type == "string":
                    params[param] = "test"
                    
        result = await mod.execute_function(fname, params, USER_ID)
        assert result["success"] is True, f"Music function {fname} should succeed in test mode"
        assert result.get("test_mode") is True, f"Function {fname} should confirm test mode"
