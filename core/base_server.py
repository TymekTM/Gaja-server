"""
Core server application class for GAJA Assistant.
Provides common server functionality without framework-specific code.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Add server path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.ai_module import AIModule
from config.config_loader import load_config, ConfigLoader
from config.config_manager import initialize_database_manager
from extended_webui import ExtendedWebUI
from core.function_calling_system import get_function_calling_system
from modules.onboarding_module import OnboardingModule
from core.plugin_manager import plugin_manager
from core.plugin_monitor import plugin_monitor
from proactive_assistant_simple import get_proactive_assistant

logger = logging.getLogger(__name__)


class BaseServerApp:
    """Base server application class with common functionality."""
    
    def __init__(self):
        self.config = None
        self.db_manager = None
        self.ai_module = None
        self.function_system = None
        self.onboarding_module = None
        self.web_ui = None
        self.plugin_monitor = plugin_monitor
        self.proactive_assistant = None
        self.start_time = None
        self._initialized = False

    async def initialize(self):
        """Initialize all server components."""
        if self._initialized:
            return
            
        try:
            self.start_time = datetime.now()
            logger.info("Starting server initialization...")

            # Load configuration
            self.config = load_config("server_config.json")
            logger.info("✅ Configuration loaded")

            # Initialize database
            self.db_manager = initialize_database_manager("server_data.db")
            await self.db_manager.initialize()
            logger.info("✅ Database initialized")

            # Initialize modules
            config_loader = ConfigLoader("server_config.json")
            self.onboarding_module = OnboardingModule(config_loader, self.db_manager)
            logger.info("✅ Onboarding module initialized")

            # Initialize Web UI
            self.web_ui = ExtendedWebUI(config_loader, self.db_manager)
            if hasattr(self.web_ui, 'set_server_app'):
                self.web_ui.set_server_app(self)
            logger.info("✅ Extended Web UI initialized")

            # Initialize AI module
            self.ai_module = AIModule(self.config)
            logger.info("✅ AI module initialized")

            # Initialize function calling system (use singleton)
            self.function_system = get_function_calling_system()
            await self.function_system.initialize()
            logger.info("✅ Function calling system initialized")

            # Initialize plugin manager
            await plugin_manager.discover_plugins()

            # Load default plugins
            default_plugins = self.config.get("plugins", {}).get("default_enabled", [])
            for plugin_name in default_plugins:
                try:
                    await plugin_manager.load_plugin(plugin_name)
                    logger.info(f"✅ Default plugin {plugin_name} loaded")
                except Exception as e:
                    logger.error(f"❌ Failed to load default plugin {plugin_name}: {e}")

            await self.load_all_user_plugins()
            logger.info("✅ Plugin manager initialized")

            # Initialize plugin monitoring
            await self.plugin_monitor.start_monitoring()
            logger.info("✅ Plugin monitoring started")

            # Initialize proactive assistant
            self.proactive_assistant = get_proactive_assistant()
            self.proactive_assistant.start()
            logger.info("✅ Proactive assistant started")

            self._initialized = True
            logger.info("🚀 Server initialization completed successfully")

        except Exception as e:
            logger.error(f"❌ Server initialization failed: {e}")
            raise

    async def load_all_user_plugins(self):
        """Load plugins for all users from database."""
        try:
            if not self.db_manager:
                logger.warning("Database manager not available, skipping user plugin loading")
                return
                
            users = await self.db_manager.get_all_users()
            for user in users:
                plugins = await self.db_manager.get_user_plugins(user["user_id"])
                for plugin in plugins:
                    if plugin["enabled"]:
                        await plugin_manager.enable_plugin_for_user(
                            user["user_id"], plugin["plugin_name"]
                        )
                        logger.info(
                            f"Plugin {plugin['plugin_name']} enabled for user {user['user_id']}"
                        )
        except Exception as e:
            logger.debug(f"Error loading user plugins: {e}")

    async def cleanup(self):
        """Clean up server resources."""
        try:
            logger.info("Starting server cleanup...")
            
            if self.proactive_assistant:
                self.proactive_assistant.stop()
                logger.info("✅ Proactive assistant stopped")
                
            if self.plugin_monitor:
                await self.plugin_monitor.stop_monitoring()
                logger.info("✅ Plugin monitoring stopped")
                
            if self.db_manager and hasattr(self.db_manager, "close"):
                try:
                    close_method = getattr(self.db_manager, "close")
                    if close_method:
                        if hasattr(close_method, "__await__"):
                            await close_method()
                        else:
                            close_method()
                    logger.info("✅ Database closed")
                except Exception as close_err:
                    logger.warning(f"Error closing database: {close_err}")
                    
            self._initialized = False
            logger.info("🏁 Server cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def get_server_info(self) -> Dict[str, Any]:
        """Get basic server information."""
        return {
            "status": "running" if self._initialized else "initializing",
            "version": "1.0.0",
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "uptime": str(datetime.now() - self.start_time) if self.start_time else None,
        }

    def setup_logging(self, log_level: str = "INFO"):
        """Setup logging configuration."""
        # Import loguru here to avoid circular imports
        from loguru import logger as loguru_logger
        
    # Configure logger
        loguru_logger.remove()
        
        # Add file logging
        loguru_logger.add(
            "logs/server_{time:YYYY-MM-DD}.log",
            rotation="1 day",
            retention="30 days",
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        )
        
        # Add console logging
        loguru_logger.add(
            sys.stderr,
            level=log_level,
            format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}:{function}:{line}</cyan> | {message}",
        )

        # Create logs directory
        os.makedirs("logs", exist_ok=True)

    def get_server_config(self) -> Dict[str, Any]:
        """Get server configuration with environment variable priority."""
        if not self.config:
            self.config = load_config("server_config.json")
            
        # Priority for environment variables
        host = os.getenv("SERVER_HOST", self.config.get("server", {}).get("host", "localhost"))
        port = int(os.getenv("SERVER_PORT", self.config.get("server", {}).get("port", 8001)))
        
        return {
            "host": host,
            "port": port,
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
        }
