#!/usr/bin/env python3
"""GAJA Assistant Server - Working version without lifespan issues."""

import os
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

# Add server path
sys.path.insert(0, str(Path(__file__).parent))

# Import server components
from ai_module import AIModule

# Import API routes
from api.routes import router as api_router
from api.routes import set_server_app
from config_loader import load_config
from config_manager import initialize_database_manager
from extended_webui import ExtendedWebUI
from function_calling_system import FunctionCallingSystem
from onboarding_module import OnboardingModule
from plugin_manager import plugin_manager
from plugin_monitor import plugin_monitor
from proactive_assistant_simple import get_proactive_assistant

# Global server instance
server_app = None


class ServerApp:
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

    async def initialize(self):
        """Initialize all server components synchronously."""
        try:
            from datetime import datetime

            self.start_time = datetime.now()

            # Load configuration
            self.config = load_config("server_config.json")
            logger.info("Configuration loaded")

            # Initialize database
            self.db_manager = initialize_database_manager("server_data.db")
            await self.db_manager.initialize()
            logger.info("Database initialized")

            # Initialize modules
            from config_loader import ConfigLoader

            config_loader = ConfigLoader("server_config.json")
            self.onboarding_module = OnboardingModule(config_loader, self.db_manager)
            logger.info("Onboarding module initialized")

            # Initialize Web UI
            self.web_ui = ExtendedWebUI(config_loader, self.db_manager)
            self.web_ui.set_server_app(self)
            logger.info("Extended Web UI initialized")

            # Initialize AI module
            self.ai_module = AIModule(self.config)
            logger.info("AI module initialized")

            # Initialize function calling system
            self.function_system = FunctionCallingSystem()
            await self.function_system.initialize()
            logger.info("Function calling system initialized")

            # Initialize plugin manager
            await plugin_manager.discover_plugins()

            # Load default plugins
            default_plugins = self.config.get("plugins", {}).get("default_enabled", [])
            for plugin_name in default_plugins:
                try:
                    await plugin_manager.load_plugin(plugin_name)
                    logger.info(f"Default plugin {plugin_name} loaded")
                except Exception as e:
                    logger.error(f"Failed to load default plugin {plugin_name}: {e}")

            await self.load_all_user_plugins()
            logger.info("Plugin manager initialized")

            # Initialize plugin monitoring
            await self.plugin_monitor.start_monitoring()
            logger.info("Plugin monitoring started")

            # Initialize proactive assistant
            self.proactive_assistant = get_proactive_assistant()
            self.proactive_assistant.start()
            logger.info("Proactive assistant started")

            logger.success("Server initialization completed")

        except Exception as e:
            logger.error(f"Server initialization failed: {e}")
            raise

    async def load_all_user_plugins(self):
        """Load plugins for all users from database."""
        try:
            users = await self.db_manager.get_all_users()
            for user in users:
                plugins = await self.db_manager.get_user_plugins(user["user_id"])
                for plugin in plugins:
                    if plugin["enabled"]:
                        await plugin_manager.enable_plugin_for_user(
                            plugin["plugin_name"], user["user_id"]
                        )
                        logger.info(
                            f"Plugin {plugin['plugin_name']} enabled for user {user['user_id']}"
                        )
        except Exception as e:
            logger.debug(f"Error loading user plugins: {e}")

    async def cleanup(self):
        """Clean up server resources."""
        if self.proactive_assistant:
            self.proactive_assistant.stop()
        if self.plugin_monitor:
            await self.plugin_monitor.stop_monitoring()
        if self.db_manager:
            await self.db_manager.close()


# Create FastAPI app
app = FastAPI(
    title="GAJA Assistant Server",
    description="Server obsługujący AI Assistant dla wielu użytkowników",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8000",
        "http://localhost:8001",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "GAJA Assistant Server", "status": "running"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": "2025-01-16T18:57:00Z"}


@app.on_event("startup")
async def startup_event():
    """Initialize server on startup."""
    global server_app
    server_app = ServerApp()
    await server_app.initialize()
    set_server_app(server_app)
    logger.info("Server startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    if server_app:
        await server_app.cleanup()
    logger.info("Server shutdown complete")


if __name__ == "__main__":
    # Load configuration
    config = load_config("server_config.json")

    # Configure logging
    logger.remove()
    logger.add(
        "logs/server_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="30 days",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
    )
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}:{function}:{line}</cyan> | {message}",
    )

    # Create logs directory
    os.makedirs("logs", exist_ok=True)

    logger.info("Starting GAJA Assistant Server...")

    # Start server
    uvicorn.run(
        app,
        host=config.get("server", {}).get("host", "localhost"),
        port=config.get("server", {}).get("port", 8001),
        log_level="warning",  # Zmieniono z "info" na "warning" żeby ukryć zbędne logi
        access_log=False,  # Wyłączono logi żądań HTTP
        reload=False,
    )
