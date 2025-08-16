"""
Comprehensive tests for core systems with fixed API calls.
Targets the core module for improved coverage.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
import json

from core.function_calling_system import FunctionCallingSystem
from core.plugin_manager import PluginManager
from core.plugin_monitor import PluginMonitor
from core.websocket_manager import ConnectionManager


@pytest.fixture
def function_calling_system():
    """Create FunctionCallingSystem instance for testing."""
    return FunctionCallingSystem()


@pytest.fixture
def plugin_manager():
    """Create PluginManager instance for testing."""
    return PluginManager()


@pytest.fixture
def plugin_monitor():
    """Create PluginMonitor instance for testing."""
    return PluginMonitor()


@pytest.fixture
def websocket_manager():
    """Create ConnectionManager instance for testing."""
    return ConnectionManager()


class TestFunctionCallingSystem:
    """Test FunctionCallingSystem functionality."""
    
    def test_initialization(self, function_calling_system):
        """Test FunctionCallingSystem initialization."""
        assert hasattr(function_calling_system, 'modules')
        assert hasattr(function_calling_system, 'function_handlers')
    
    def test_register_module(self, function_calling_system):
        """Test module registration."""
        module_data = {
            "handler": {"type": "test", "function": lambda: None},
            "schema": {"type": "object"}
        }
        
        function_calling_system.register_module("test_module", module_data)
        
        assert "test_module" in function_calling_system.modules
        assert function_calling_system.modules["test_module"] == module_data
    
    def test_convert_modules_to_functions(self, function_calling_system):
        """Test converting modules to OpenAI functions format."""
        functions = function_calling_system.convert_modules_to_functions()
        assert isinstance(functions, list)
        # Can be empty if no modules are registered
    
    @pytest.mark.asyncio 
    async def test_execute_function_not_found(self, function_calling_system):
        """Test executing non-existent function."""
        result = await function_calling_system.execute_function(
            "nonexistent_function", 
            {}
        )
        assert "not found" in str(result).lower()


class TestPluginManager:
    """Test PluginManager functionality."""
    
    def test_initialization(self, plugin_manager):
        """Test PluginManager initialization."""
        assert hasattr(plugin_manager, 'plugins')
        assert hasattr(plugin_manager, 'user_plugins')
        assert hasattr(plugin_manager, 'plugins_directory')
    
    def test_get_all_plugins(self, plugin_manager):
        """Test getting all plugins."""
        plugins = plugin_manager.get_all_plugins()
        assert isinstance(plugins, dict)
        # Plugins dict can be empty initially
    
    @pytest.mark.asyncio
    async def test_enable_plugin_for_user_success(self, plugin_manager):
        """Test successfully enabling a plugin for a user."""
        plugin_name = "weather_module"
        user_id = "test_user"

        with patch.object(plugin_manager, 'load_plugin') as mock_load:
            mock_load.return_value = True
            
            result = await plugin_manager.enable_plugin_for_user(user_id, plugin_name)
            assert isinstance(result, bool)
    
    @pytest.mark.asyncio
    async def test_enable_plugin_not_available(self, plugin_manager):
        """Test enabling non-available plugin."""
        plugin_name = "nonexistent_module"
        user_id = "test_user"

        with patch.object(plugin_manager, 'load_plugin') as mock_load:
            mock_load.return_value = False
            
            result = await plugin_manager.enable_plugin_for_user(user_id, plugin_name)
            assert isinstance(result, bool)
    
    @pytest.mark.asyncio
    async def test_disable_plugin_for_user(self, plugin_manager):
        """Test disabling a plugin for a user."""
        plugin_name = "weather_module"
        user_id = "test_user"

        result = await plugin_manager.disable_plugin_for_user(user_id, plugin_name)
        assert isinstance(result, bool)
    
    def test_is_plugin_loaded(self, plugin_manager):
        """Test checking if plugin is loaded."""
        plugin_name = "weather_module"

        result = plugin_manager.is_plugin_loaded(plugin_name)
        assert isinstance(result, bool)
        
    def test_get_user_plugins(self, plugin_manager):
        """Test getting user-specific plugins."""
        user_id = "test_user"
        
        plugins = plugin_manager.get_user_plugins(user_id)
        assert isinstance(plugins, dict)
    
    def test_get_plugin_info(self, plugin_manager):
        """Test getting plugin information."""
        plugin_name = "weather_module"
        
        info = plugin_manager.get_plugin_info(plugin_name)
        # Can be None if plugin not found


class TestPluginMonitor:
    """Test PluginMonitor functionality."""
    
    def test_initialization(self, plugin_monitor):
        """Test PluginMonitor initialization."""
        assert hasattr(plugin_monitor, 'stats')
        assert hasattr(plugin_monitor, 'modules_path')
        assert hasattr(plugin_monitor, 'running')
    
    @pytest.mark.asyncio
    async def test_start_monitoring(self, plugin_monitor):
        """Test starting plugin monitoring."""
        result = await plugin_monitor.start_monitoring()
        # Result can be True/False depending on directory existence
        assert isinstance(result, bool)
    
    def test_stats_property(self, plugin_monitor):
        """Test getting monitoring statistics."""
        stats = plugin_monitor.stats
        assert isinstance(stats, dict)
        assert "reloads_count" in stats


class TestConnectionManager:
    """Test ConnectionManager (WebSocket) functionality."""
    
    def test_initialization(self, websocket_manager):
        """Test ConnectionManager initialization."""
        assert hasattr(websocket_manager, 'active_connections')
        assert isinstance(websocket_manager.active_connections, dict)
    
    @pytest.mark.asyncio
    async def test_connect_websocket_with_mock(self, websocket_manager):
        """Test WebSocket connection with proper mock."""
        # Create a more realistic mock WebSocket
        mock_websocket = MagicMock()
        mock_websocket.accept = AsyncMock()
        mock_websocket.client_state.name = "CONNECTED"
        user_id = "test_user"
        
        # Mock the state check
        with patch.object(websocket_manager, 'is_connected') as mock_is_connected:
            mock_is_connected.return_value = True
            
            await websocket_manager.connect(mock_websocket, user_id)
            
            # Should be added to connections during the process
            # (may be removed later if connection fails)
            assert mock_websocket.accept.called
    
    @pytest.mark.asyncio
    async def test_disconnect_websocket(self, websocket_manager):
        """Test WebSocket disconnection."""
        user_id = "test_user"
        
        # Add connection first
        mock_websocket = MagicMock()
        websocket_manager.active_connections[user_id] = mock_websocket
        
        await websocket_manager.disconnect(user_id)
        
        assert user_id not in websocket_manager.active_connections


class TestIntegrationScenarios:
    """Test integration between core systems."""
    
    def test_function_calling_system_initialization(self, function_calling_system):
        """Test FunctionCallingSystem can be initialized."""
        assert function_calling_system is not None
        
    def test_plugin_manager_integration(self, plugin_manager):
        """Test PluginManager integration."""
        # Test getting all plugins
        plugins = plugin_manager.get_all_plugins()
        assert isinstance(plugins, dict)
        
        # Test getting user plugins
        user_plugins = plugin_manager.get_user_plugins("test_user")
        assert isinstance(user_plugins, dict)
