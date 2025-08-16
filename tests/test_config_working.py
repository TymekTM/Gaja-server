"""Working config tests that match the actual ConfigLoader and ConfigManager APIs."""
from __future__ import annotations

import pytest
import os
import json
from unittest.mock import patch, MagicMock, mock_open

from config.config_loader import ConfigLoader, load_config, save_config, create_default_config
from config.config_manager import ConfigManager


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "app": {
            "name": "GAJA Assistant",
            "debug": False
        },
        "server": {
            "host": "127.0.0.1",
            "port": 8001
        },
        "ai": {
            "model": "gpt-5-nano",
            "provider": "openai"
        }
    }


@pytest.fixture
def config_loader():
    """Create ConfigLoader instance for testing."""
    with patch('config.config_loader.load_config') as mock_load:
        mock_load.return_value = {
            "app": {"name": "Test App"},
            "server": {"host": "localhost", "port": 8000}
        }
        return ConfigLoader("test_config.json")


class TestConfigLoader:
    """Test ConfigLoader functionality with actual API."""
    
    def test_initialization(self, config_loader):
        """Test ConfigLoader initialization."""
        assert hasattr(config_loader, '_config')
        assert hasattr(config_loader, 'config_file')
        assert config_loader.config_file == "test_config.json"
    
    def test_get_config(self, config_loader):
        """Test getting configuration."""
        config = config_loader.get_config()
        assert isinstance(config, dict)
        assert "app" in config
    
    def test_get_method(self, config_loader):
        """Test get method with key and default."""
        # Test existing key
        app_name = config_loader.get("app", {})
        assert isinstance(app_name, dict)
        
        # Test with default
        nonexistent = config_loader.get("nonexistent", "default")
        assert nonexistent == "default"
    
    def test_update_config(self, config_loader):
        """Test updating configuration."""
        updates = {"test_key": "test_value"}
        
        with patch('config.config_loader.save_config') as mock_save:
            config_loader.update_config(updates)
            mock_save.assert_called_once()
    
    def test_save_config(self, config_loader):
        """Test saving configuration."""
        with patch('config.config_loader.save_config') as mock_save:
            config_loader.save_config()
            mock_save.assert_called_once()
    
    def test_load(self, config_loader):
        """Test loading configuration."""
        with patch('config.config_loader.load_config') as mock_load:
            mock_load.return_value = {"test": "config"}
            
            result = config_loader.load()
            assert isinstance(result, dict)
            assert result == {"test": "config"}


class TestConfigFunctions:
    """Test module-level config functions."""
    
    def test_load_config_function(self, sample_config):
        """Test load_config function."""
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=json.dumps(sample_config))):
                config = load_config("test.json")
                
                assert isinstance(config, dict)
                assert "app" in config
    
    def test_load_config_not_exists(self):
        """Test load_config with non-existent file."""
        with patch('pathlib.Path.exists', return_value=False):
            with patch('config.config_loader.create_default_config') as mock_default:
                mock_default.return_value = {"default": "config"}
                
                config = load_config("nonexistent.json")
                assert isinstance(config, dict)
                mock_default.assert_called_once()
    
    def test_save_config_function(self, sample_config):
        """Test save_config function."""
        with patch('builtins.open', mock_open()) as mock_file:
            save_config(sample_config, "test.json")
            mock_file.assert_called_once_with("test.json", "w", encoding="utf-8")
    
    def test_create_default_config(self):
        """Test creating default configuration."""
        config = create_default_config()
        
        assert isinstance(config, dict)
        assert "server" in config
        assert "ai" in config


class TestConfigManager:
    """Test ConfigManager functionality with actual API."""
    
    def test_initialization(self):
        """Test ConfigManager initialization."""
        with patch('config.config_manager.EnvironmentManager'):
            with patch('config.config_manager.DatabaseManager'):
                manager = ConfigManager()
                
                assert hasattr(manager, 'environment')
                assert hasattr(manager, 'database')
    
    def test_get_server_config(self):
        """Test getting server configuration."""
        with patch('config.config_manager.EnvironmentManager') as mock_env:
            with patch('config.config_manager.DatabaseManager'):
                mock_env_instance = MagicMock()
                mock_env_instance.get_server_config.return_value = {"test": "config"}
                mock_env.return_value = mock_env_instance
                
                manager = ConfigManager()
                config = manager.get_server_config()
                
                assert config == {"test": "config"}
    
    def test_get_api_key(self):
        """Test getting API key."""
        with patch('config.config_manager.EnvironmentManager') as mock_env:
            with patch('config.config_manager.DatabaseManager') as mock_db:
                mock_env_instance = MagicMock()
                mock_db_instance = MagicMock()
                
                mock_env_instance.get_api_key.return_value = "test_key"
                mock_db_instance.get_user_api_key.return_value = None
                
                mock_env.return_value = mock_env_instance
                mock_db.return_value = mock_db_instance
                
                manager = ConfigManager()
                api_key = manager.get_api_key("openai")
                
                assert api_key == "test_key"
    
    def test_sanitize_config_for_logging(self):
        """Test config sanitization."""
        with patch('config.config_manager.EnvironmentManager') as mock_env:
            with patch('config.config_manager.DatabaseManager'):
                mock_env_instance = MagicMock()
                mock_env_instance.sanitize_config_for_logging.return_value = {"safe": "config"}
                mock_env.return_value = mock_env_instance
                
                manager = ConfigManager()
                result = manager.sanitize_config_for_logging({"api_key": "secret"})
                
                assert result == {"safe": "config"}


class TestEnvironmentOverrides:
    """Test environment variable overrides."""
    
    def test_host_override(self):
        """Test GAJA_HOST environment override."""
        with patch.dict(os.environ, {'GAJA_HOST': '0.0.0.0'}):
            with patch('pathlib.Path.exists', return_value=False):
                config = load_config()
                
                assert config["server"]["host"] == "0.0.0.0"
    
    def test_port_override(self):
        """Test GAJA_PORT environment override."""
        with patch.dict(os.environ, {'GAJA_PORT': '9000'}):
            with patch('pathlib.Path.exists', return_value=False):
                config = load_config()
                
                assert config["server"]["port"] == 9000


class TestConfigValidation:
    """Test configuration validation."""
    
    def test_valid_config_structure(self, sample_config):
        """Test that config has required structure."""
        required_sections = ["server", "ai"]
        
        # Add missing sections to sample config for test
        sample_config.update({
            "server": {"host": "localhost", "port": 8000},
            "ai": {"model": "gpt-5-nano"}
        })
        
        for section in required_sections:
            assert section in sample_config
    
    def test_default_values(self):
        """Test that default config has reasonable values."""
        config = create_default_config()
        
        # Check server section
        assert isinstance(config["server"]["port"], int)
        assert config["server"]["port"] > 0
        
        # Check ai section
        assert config["ai"]["model"] == "gpt-5-nano"
        assert isinstance(config["ai"]["temperature"], (int, float))
