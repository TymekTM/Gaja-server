#!/usr/bin/env python3
"""GAJA Assistant Server Główny serwer obsługujący wielu użytkowników, zarządzający AI,
bazą danych i pluginami."""

import json
import os
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

# Add server path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    # Try to load .env from project root (parent of server directory) - but only if not in container
    is_docker = os.getenv("PRODUCTION", "false").lower() == "true"

    if not is_docker:
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            logger.info(f"✅ Loaded environment variables from {env_path}")
        else:
            logger.warning(f"⚠️ No .env file found at {env_path}")
    else:
        logger.info("🐳 Running in Docker - using environment variables")

except ImportError:
    logger.warning("⚠️ python-dotenv not installed, trying manual .env loading")
    # Manual fallback for .env loading (only if not in Docker)
    is_docker = os.getenv("PRODUCTION", "false").lower() == "true"
    if not is_docker:
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            try:
                with open(env_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            key, value = line.split("=", 1)
                            os.environ[key.strip()] = (
                                value.strip().strip('"').strip("'")
                            )
                logger.info(f"✅ Manually loaded environment variables from {env_path}")
            except Exception as e:
                logger.error(f"❌ Error loading .env file: {e}")
        else:
            logger.warning(f"⚠️ No .env file found at {env_path}")
    else:
        logger.info("🐳 Running in Docker - using environment variables")
except Exception as e:
    logger.error(f"❌ Error loading .env file: {e}")

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
from websocket_manager import WebSocketMessage, connection_manager

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
        self.connection_manager = connection_manager

    async def handle_websocket_message(self, user_id: str, message_data: dict) -> None:
        """Obsługuje wiadomości WebSocket."""
        try:
            message_type = message_data.get("type", "unknown")
            logger.info(f"WebSocket message from {user_id}: {message_type}")

            if message_type == "handshake":
                # Handshake jest obsługiwany automatycznie w connection_manager
                logger.info(f"Handshake completed for user {user_id}")
                return

            elif message_type == "query" or message_type == "ai_query":
                # Zapytanie AI
                query = message_data.get("query", "")
                context = message_data.get("context", {})

                if not query:
                    await self.connection_manager.send_to_user(
                        user_id, WebSocketMessage("error", {"message": "Empty query"})
                    )
                    return

                # Przetwórz zapytanie przez AI
                try:
                    ai_result = await self.ai_module.process_query(query, context)
                    logger.info(
                        f"AI module returned result type: {ai_result.get('type', 'unknown')}"
                    )

                    # Handle clarification requests
                    if ai_result.get("type") == "clarification_request":
                        # Send clarification request to client
                        clarification_data = ai_result.get("clarification_data", {})
                        await self.connection_manager.send_to_user(
                            user_id,
                            WebSocketMessage(
                                "clarification_request",
                                {
                                    "question": clarification_data.get("question", ""),
                                    "context": clarification_data.get("context", ""),
                                    "actions": clarification_data.get("actions", {}),
                                    "timestamp": clarification_data.get("timestamp"),
                                    "original_query": query,
                                },
                            ),
                        )
                        return

                    # Normal AI response
                    response = ai_result.get("response", "")
                    await self.connection_manager.send_to_user(
                        user_id,
                        WebSocketMessage(
                            "ai_response",
                            {
                                "response": response,
                                "query": query,
                                "timestamp": message_data.get("timestamp"),
                            },
                        ),
                    )

                except Exception as e:
                    logger.error(f"AI query error for user {user_id}: {e}")
                    await self.connection_manager.send_to_user(
                        user_id,
                        WebSocketMessage(
                            "error",
                            {"message": "Failed to process AI query", "error": str(e)},
                        ),
                    )

            elif message_type == "plugin_toggle":
                # Przełączanie pluginu
                plugin_data = message_data.get("data", {})
                plugin_name = plugin_data.get("plugin")
                action = plugin_data.get("action")

                if not plugin_name or action not in ["enable", "disable"]:
                    await self.connection_manager.send_to_user(
                        user_id,
                        WebSocketMessage(
                            "error", {"message": "Invalid plugin toggle request"}
                        ),
                    )
                    return

                try:
                    if action == "enable":
                        await plugin_manager.enable_plugin_for_user(
                            plugin_name, user_id
                        )
                        await self.db_manager.set_user_plugin_status(
                            user_id, plugin_name, True
                        )
                        status = "enabled"
                    else:
                        await plugin_manager.disable_plugin_for_user(
                            plugin_name, user_id
                        )
                        await self.db_manager.set_user_plugin_status(
                            user_id, plugin_name, False
                        )
                        status = "disabled"

                    await self.connection_manager.send_to_user(
                        user_id,
                        WebSocketMessage(
                            "plugin_toggled",
                            {"plugin": plugin_name, "status": status, "success": True},
                        ),
                    )

                except Exception as e:
                    logger.error(f"Plugin toggle error for user {user_id}: {e}")
                    await self.connection_manager.send_to_user(
                        user_id,
                        WebSocketMessage(
                            "error",
                            {
                                "message": f"Failed to {action} plugin {plugin_name}",
                                "error": str(e),
                            },
                        ),
                    )

            elif message_type == "plugin_list":
                # Lista pluginów
                try:
                    all_plugins = plugin_manager.get_all_plugins()
                    user_plugins = await self.db_manager.get_user_plugins(user_id)

                    # Połącz informacje o pluginach
                    plugins_info = []
                    user_plugin_status = {
                        p["plugin_name"]: p["enabled"] for p in user_plugins
                    }

                    for plugin_name, plugin_info in all_plugins.items():
                        plugins_info.append(
                            {
                                "name": plugin_name,
                                "enabled": user_plugin_status.get(plugin_name, False),
                                "description": plugin_info.description or "",
                                "version": plugin_info.version or "1.0.0",
                            }
                        )

                    await self.connection_manager.send_to_user(
                        user_id,
                        WebSocketMessage("plugin_list", {"plugins": plugins_info}),
                    )

                except Exception as e:
                    logger.error(f"Plugin list error for user {user_id}: {e}")
                    await self.connection_manager.send_to_user(
                        user_id,
                        WebSocketMessage(
                            "error",
                            {"message": "Failed to get plugin list", "error": str(e)},
                        ),
                    )

            elif message_type == "startup_briefing":
                # Briefing poranny
                try:
                    if self.proactive_assistant:
                        notifications = (
                            await self.proactive_assistant.get_notifications(user_id)
                        )
                        briefing = f"Daily briefing for user {user_id}: {len(notifications)} notifications"
                        await self.connection_manager.send_to_user(
                            user_id,
                            WebSocketMessage(
                                "startup_briefing", {"briefing": briefing}
                            ),
                        )
                    else:
                        await self.connection_manager.send_to_user(
                            user_id,
                            WebSocketMessage(
                                "startup_briefing",
                                {"briefing": "Proactive assistant nie jest dostępny"},
                            ),
                        )
                except Exception as e:
                    logger.error(f"Briefing error for user {user_id}: {e}")
                    await self.connection_manager.send_to_user(
                        user_id,
                        WebSocketMessage(
                            "error",
                            {"message": "Failed to get briefing", "error": str(e)},
                        ),
                    )

            elif message_type == "proactive_check":
                # Sprawdzenie proaktywnych powiadomień
                try:
                    if self.proactive_assistant:
                        notifications = (
                            await self.proactive_assistant.get_notifications(user_id)
                        )
                        await self.connection_manager.send_to_user(
                            user_id,
                            WebSocketMessage(
                                "proactive_notifications",
                                {"notifications": notifications},
                            ),
                        )
                    else:
                        await self.connection_manager.send_to_user(
                            user_id,
                            WebSocketMessage(
                                "proactive_notifications", {"notifications": []}
                            ),
                        )
                except Exception as e:
                    logger.error(f"Proactive check error for user {user_id}: {e}")

            elif message_type == "user_context_update":
                # Handle user context updates (for analytics, debugging, etc.)
                context_data = message_data.get("data", {})
                logger.debug(f"User context update from {user_id}: {context_data}")
                # For now, just acknowledge the update - could be stored in database later
                await self.connection_manager.send_to_user(
                    user_id, WebSocketMessage("context_ack", {"status": "received"})
                )

            else:
                logger.warning(
                    f"Unknown WebSocket message type: {message_type} from user {user_id}"
                )
                await self.connection_manager.send_to_user(
                    user_id,
                    WebSocketMessage(
                        "error", {"message": f"Unknown message type: {message_type}"}
                    ),
                )

        except Exception as e:
            logger.error(f"WebSocket message handling error for user {user_id}: {e}")
            await self.connection_manager.send_to_user(
                user_id,
                WebSocketMessage(
                    "error", {"message": "Internal server error", "error": str(e)}
                ),
            )

    async def initialize(self):
        """Initialize all server components."""
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
        try:
            if self.proactive_assistant:
                self.proactive_assistant.stop()
            if self.plugin_monitor:
                await self.plugin_monitor.stop_monitoring()
            if self.db_manager and hasattr(self.db_manager, "close"):
                await self.db_manager.close()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


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


@app.get("/health")
async def health_check():
    """Health check endpoint for Docker."""
    return {"status": "healthy", "timestamp": "2025-07-16T19:25:00Z"}


# Legacy status endpoint for compatibility with client
@app.get("/api/status")
async def legacy_status():
    """Legacy status endpoint for client compatibility."""
    return {
        "status": "running",
        "version": "1.0.0",
        "timestamp": "2025-07-16T19:25:00Z",
    }


# SSE endpoint for overlay status updates
@app.get("/status/stream")
async def status_stream(request: Request):
    """Server-Sent Events endpoint for overlay status updates."""
    import asyncio

    from fastapi.responses import StreamingResponse

    async def event_stream():
        try:
            while True:
                # Send basic status for now - can be enhanced when client connects
                status_data = {
                    "status": "running",
                    "text": "",
                    "is_listening": False,
                    "is_speaking": False,
                    "wake_word_detected": False,
                    "overlay_visible": False,
                }

                yield f"data: {json.dumps(status_data)}\n\n"
                await asyncio.sleep(1)  # Send updates every second

        except Exception as e:
            logger.error(f"SSE stream error: {e}")
            return

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket endpoint dla komunikacji w czasie rzeczywistym."""
    logger.info(f"WebSocket connection attempt for user: {user_id}")

    try:
        # Nawiąż połączenie
        await connection_manager.connect(
            websocket,
            user_id,
            {
                "client_type": "web_ui",
                "user_agent": websocket.headers.get("user-agent", "unknown"),
            },
        )

        logger.info(f"WebSocket connected for user: {user_id}")

        while True:
            # Odbierz wiadomość
            try:
                data = await websocket.receive_text()
                message_data = json.loads(data)

                # Przetwórz wiadomość przez connection manager
                processed_message = await connection_manager.handle_message(
                    user_id, message_data
                )

                if processed_message and server_app:
                    # Przekaż do aplikacji serwera
                    await server_app.handle_websocket_message(
                        user_id, processed_message
                    )

            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected for user: {user_id}")
                break
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON from user {user_id}: {e}")
                # Tylko spróbuj wysłać błąd jeśli użytkownik jest połączony
                if user_id in connection_manager.active_connections:
                    try:
                        await connection_manager.send_to_user(
                            user_id,
                            WebSocketMessage(
                                "error", {"message": "Invalid JSON format"}
                            ),
                        )
                    except Exception as send_error:
                        logger.debug(
                            f"Could not send JSON error message to {user_id}: {send_error}"
                        )
            except Exception as e:
                logger.error(f"WebSocket message error for user {user_id}: {e}")
                # Tylko spróbuj wysłać błąd jeśli użytkownik jest połączony
                if user_id in connection_manager.active_connections:
                    try:
                        await connection_manager.send_to_user(
                            user_id,
                            WebSocketMessage(
                                "error", {"message": "Message processing error"}
                            ),
                        )
                    except Exception as send_error:
                        logger.debug(
                            f"Could not send error message to {user_id}: {send_error}"
                        )
                        # Nie loguj tego jako ERROR żeby uniknąć spam logów

    except Exception as e:
        logger.error(f"WebSocket connection error for user {user_id}: {e}")
    finally:
        # Wyczyść połączenie
        await connection_manager.disconnect(user_id, "connection_closed")


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "GAJA Assistant Server", "status": "running"}


@app.get("/health")
async def health_check_endpoint():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": "2025-07-16T19:25:00Z"}


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


def main():
    """Main server entry point."""
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
    # Priorytet dla zmiennych środowiskowych
    host = os.getenv("SERVER_HOST", config.get("server", {}).get("host", "localhost"))
    port = int(os.getenv("SERVER_PORT", config.get("server", {}).get("port", 8001)))

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="warning",  # Przywrócone do warning po naprawie overlay
        access_log=False,  # Wyłączone logi żądań HTTP - overlay naprawiony
        reload=False,
    )


if __name__ == "__main__":
    main()
