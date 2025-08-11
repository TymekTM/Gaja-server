"""
Extended Web UI for GAJA Server - FastAPI only implementation
Provides data for web administration endpoints.
"""

import os
import json
import time
import psutil
from datetime import datetime
from typing import Optional, Any, Dict


class ExtendedWebUI:
    """Web UI data provider for FastAPI endpoints."""
    
    def __init__(self, config_loader=None, db_manager=None):
        """Initialize web UI with configuration and database manager."""
        self.config_loader = config_loader
        self.db_manager = db_manager
        
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get dashboard data."""
        try:
            # Server status
            process = psutil.Process()
            memory_info = process.memory_info()
            
            # Basic metrics
            data = {
                'status': 'running',
                'uptime': time.time() - process.create_time(),
                'memory_usage': memory_info.rss / 1024 / 1024,  # MB
                'cpu_percent': process.cpu_percent(),
                'active_connections': 0,  # Will be updated by connection manager
                'plugins_loaded': 0,      # Will be updated by plugin manager
                'timestamp': datetime.now().isoformat()
            }
            
            return data
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_config_data(self) -> Dict[str, Any]:
        """Get server configuration data."""
        try:
            config_data = {}
            
            # Try to load config from config_loader
            if self.config_loader and hasattr(self.config_loader, 'get_config'):
                config_data = self.config_loader.get_config()
            else:
                # Fallback: try to read config file directly
                config_path = 'server_config.json'
                if os.path.exists(config_path):
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config_data = json.load(f)
                        
            return {
                'config': config_data,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'config': {},
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_plugin_data(self) -> Dict[str, Any]:
        """Get plugin information."""
        try:
            # This will be populated by the plugin manager
            plugins_data = {
                'total_plugins': 0,
                'active_plugins': 0,
                'plugins': []
            }
            
            return {
                'plugins': plugins_data,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'plugins': {'total_plugins': 0, 'active_plugins': 0, 'plugins': []},
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_memory_stats(self, user_id: str = "1") -> Dict[str, Any]:
        """Get memory statistics."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'memory': {
                    'rss': memory_info.rss,
                    'vms': memory_info.vms,
                    'percent': process.memory_percent(),
                    'available': psutil.virtual_memory().available,
                    'total': psutil.virtual_memory().total
                },
                'user_id': user_id,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'memory': {},
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_logs_data(self) -> Dict[str, Any]:
        """Get logs information."""
        try:
            logs_data = {
                'total_logs': 0,
                'recent_logs': [],
                'log_levels': {'debug': 0, 'info': 0, 'warning': 0, 'error': 0}
            }
            
            # Try to read recent logs from logs directory
            logs_dir = 'logs'
            if os.path.exists(logs_dir):
                log_files = [f for f in os.listdir(logs_dir) if f.endswith('.log')]
                logs_data['log_files'] = log_files
                logs_data['total_logs'] = len(log_files)
            
            return {
                'logs': logs_data,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'logs': {'total_logs': 0, 'recent_logs': [], 'log_levels': {}},
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
