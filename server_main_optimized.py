#!/usr/bin/env python3
"""GAJA Assistant Server - Optimized main server application."""

import json
import os
import sys
import time
import base64
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from websockets.exceptions import ConnectionClosed

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

# Import OpenAI for TTS
try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

# Import server components
from core.base_server import BaseServerApp
from modules.server_performance_monitor import (
    start_query_tracking, start_server_timer, end_server_timer,
    record_server_response_data, record_server_error, finish_server_query
)

# Import API routes
from api.routes import router as api_router
from api.routes import set_server_app
from core.websocket_manager import WebSocketMessage, connection_manager

# Global server instance
server_app = None


class GAJAServerApp(BaseServerApp):
    """GAJA Server application with WebSocket support."""
    
    def __init__(self):
        super().__init__()
        self.connection_manager = connection_manager

    async def handle_websocket_message(self, user_id: str, message_data: dict) -> None:
        """Handle WebSocket messages with optimized performance tracking."""
        try:
            # Validate message data
            if message_data is None:
                logger.error(f"Received None message_data from user {user_id}")
                await self.connection_manager.send_to_user(
                    user_id, WebSocketMessage("error", {"message": "Invalid message format"})
                )
                return
                
            message_type = message_data.get("type", "unknown")
            logger.info(f"WebSocket message from {user_id}: {message_type}")

            # Handle different message types
            if message_type == "handshake":
                logger.info(f"Handshake completed for user {user_id}")
                return

            elif message_type in ["query", "ai_query", "voice_command"]:
                await self._handle_ai_query(user_id, message_data)

            elif message_type == "plugin_toggle":
                await self._handle_plugin_toggle(user_id, message_data)

            elif message_type == "plugin_list":
                await self._handle_plugin_list(user_id)

            elif message_type == "startup_briefing":
                await self._handle_startup_briefing(user_id)

            elif message_type == "clarification_response":
                await self._handle_clarification_response(user_id, message_data)

            elif message_type == "proactive_check":
                await self._handle_proactive_check(user_id)

            elif message_type == "user_context_update":
                await self._handle_context_update(user_id, message_data)

            else:
                logger.warning(f"Unknown WebSocket message type: {message_type} from user {user_id}")
                await self.connection_manager.send_to_user(
                    user_id,
                    WebSocketMessage("error", {"message": f"Unknown message type: {message_type}"}),
                )

        except Exception as e:
            logger.error(f"WebSocket message handling error for user {user_id}: {e}")
            await self.connection_manager.send_to_user(
                user_id,
                WebSocketMessage("error", {"message": "Internal server error", "error": str(e)}),
            )

    async def _handle_ai_query(self, user_id: str, message_data: dict) -> None:
        """Handle AI query messages."""
        data = message_data.get("data", {})
        query = data.get("query", "")
        context = data.get("context", {})

        if not query:
            await self.connection_manager.send_to_user(
                user_id, WebSocketMessage("error", {"message": "Empty query"})
            )
            return

        # Start performance tracking
        query_id = f"{user_id}_{int(time.time() * 1000)}"
        start_query_tracking(query_id, user_id, query)
        start_server_timer(query_id, "total_server")

        try:
            start_server_timer(query_id, "ai_processing")
            
            # Get conversation history
            history = []
            if self.db_manager:
                try:
                    history = await self.db_manager.get_user_history(user_id, limit=20)
                    logger.debug(f"Retrieved {len(history)} messages from history for user {user_id}")
                except Exception as hist_err:
                    logger.warning(f"Failed to get history for user {user_id}: {hist_err}")
            
            # Add context
            context["history"] = history
            context["user_id"] = user_id
            
            ai_result = await self.ai_module.process_query(query, context)
            end_server_timer(query_id, "ai_processing")

            # Handle clarification requests
            if ai_result.get("type") == "clarification_request":
                await self._handle_clarification(user_id, ai_result, query, query_id)
                return

            # Handle normal AI response
            await self._handle_ai_response(user_id, ai_result, query, message_data, query_id)

        except Exception as e:
            logger.error(f"AI query error for user {user_id}: {e}")
            record_server_error(query_id, f"AI query error: {e}")
            finish_server_query(query_id)
            await self.connection_manager.send_to_user(
                user_id,
                WebSocketMessage("error", {"message": "Failed to process AI query", "error": str(e)}),
            )

    async def _handle_clarification(self, user_id: str, ai_result: dict, query: str, query_id: str) -> None:
        """Handle clarification requests."""
        clarification_data = ai_result.get("clarification_data", {})
        question = clarification_data.get("question", "")
        
        # Generate TTS for clarification
        tts_audio = None
        if question:
            logger.info(f"Generating TTS for clarification: {question[:50]}...")
            try:
                start_server_timer(query_id, "tts_generation")
                tts_audio = await self._generate_tts_audio(question)
                end_server_timer(query_id, "tts_generation")
                logger.info(f"Generated TTS audio for clarification: {len(tts_audio) if tts_audio else 0} bytes")
            except Exception as e:
                end_server_timer(query_id, "tts_generation")
                logger.error(f"Failed to generate TTS for clarification: {e}")
        
        response_data = {
            "question": question,
            "context": clarification_data.get("context", ""),
            "actions": clarification_data.get("actions", {}),
            "timestamp": clarification_data.get("timestamp"),
            "original_query": query,
        }
        
        if tts_audio:
            response_data["tts_audio"] = base64.b64encode(tts_audio).decode('utf-8')
            tts_config = self.config.get("tts", {}) if self.config else {}
            response_data["tts_config"] = {"volume": tts_config.get("volume", 1.0), "tts_metadata": {"format": "mp3"}}
        
        record_server_response_data(query_id, question, len(tts_audio) if tts_audio else None)
        
        start_server_timer(query_id, "websocket_send")
        await self.connection_manager.send_to_user(
            user_id, WebSocketMessage("clarification_request", response_data)
        )
        end_server_timer(query_id, "websocket_send")
        finish_server_query(query_id)

    async def _handle_ai_response(self, user_id: str, ai_result: dict, query: str, message_data: dict, query_id: str) -> None:
        """Handle normal AI responses."""
        response = ai_result.get("response", "")
        
        # Save conversation to history
        if self.db_manager:
            try:
                await self.db_manager.save_interaction(user_id, query, response)
                logger.debug(f"Saved conversation to history for user {user_id}")
            except Exception as save_err:
                logger.warning(f"Failed to save conversation history: {save_err}")
        
        # Generate TTS audio
        tts_audio = await self._generate_tts_for_response(response, query_id)
        
        # Record performance data
        record_server_response_data(query_id, response, len(tts_audio) if tts_audio else None)
        
        # Send response
        response_data = {
            "response": response,
            "query": query,
            "timestamp": message_data.get("timestamp"),
        }
        
        if tts_audio:
            response_data["tts_audio"] = base64.b64encode(tts_audio).decode('utf-8')
            tts_config = self.config.get("tts", {}) if self.config else {}
            response_data["tts_config"] = {"volume": tts_config.get("volume", 1.0), "tts_metadata": {"format": "mp3"}}
        
        start_server_timer(query_id, "websocket_send")
        await self.connection_manager.send_to_user(
            user_id, WebSocketMessage("ai_response", response_data)
        )
        end_server_timer(query_id, "websocket_send")
        end_server_timer(query_id, "total_server")
        finish_server_query(query_id)

    async def _generate_tts_for_response(self, response: str, query_id: str) -> Optional[bytes]:
        """Generate TTS audio for response."""
        tts_audio = None
        try:
            # Parse JSON response
            try:
                parsed_response = json.loads(response)
                if isinstance(parsed_response, dict) and "text" in parsed_response:
                    text_to_speak = parsed_response["text"]
                    if text_to_speak:
                        start_server_timer(query_id, "tts_generation")
                        tts_audio = await self._generate_tts_audio(text_to_speak)
                        end_server_timer(query_id, "tts_generation")
            except json.JSONDecodeError:
                # Use response directly if it's simple text
                if isinstance(response, str) and response.strip():
                    start_server_timer(query_id, "tts_generation")
                    tts_audio = await self._generate_tts_audio(response.strip())
                    end_server_timer(query_id, "tts_generation")
        except Exception as tts_err:
            end_server_timer(query_id, "tts_generation")
            logger.error(f"TTS generation failed: {tts_err}")
        
        return tts_audio

    async def _handle_plugin_toggle(self, user_id: str, message_data: dict) -> None:
        """Handle plugin toggle requests."""
        plugin_data = message_data.get("data", {})
        plugin_name = plugin_data.get("plugin")
        action = plugin_data.get("action")

        if not plugin_name or action not in ["enable", "disable"]:
            await self.connection_manager.send_to_user(
                user_id, WebSocketMessage("error", {"message": "Invalid plugin toggle request"})
            )
            return

        try:
            from core.plugin_manager import plugin_manager
            
            if action == "enable":
                await plugin_manager.enable_plugin_for_user(user_id, plugin_name)
                if self.db_manager:
                    await self.db_manager.update_user_plugin_status(user_id, plugin_name, True)
                status = "enabled"
            else:
                await plugin_manager.disable_plugin_for_user(user_id, plugin_name)
                if self.db_manager:
                    await self.db_manager.update_user_plugin_status(user_id, plugin_name, False)
                status = "disabled"

            await self.connection_manager.send_to_user(
                user_id,
                WebSocketMessage("plugin_toggled", {"plugin": plugin_name, "status": status, "success": True}),
            )

        except Exception as e:
            logger.error(f"Plugin toggle error for user {user_id}: {e}")
            await self.connection_manager.send_to_user(
                user_id,
                WebSocketMessage("error", {"message": f"Failed to {action} plugin {plugin_name}", "error": str(e)}),
            )

    async def _handle_plugin_list(self, user_id: str) -> None:
        """Handle plugin list requests."""
        try:
            from core.plugin_manager import plugin_manager
            
            all_plugins = plugin_manager.get_all_plugins()
            user_plugins = []
            if self.db_manager:
                user_plugins = await self.db_manager.get_user_plugins(user_id)

            user_plugin_status = {p["plugin_name"]: p["enabled"] for p in user_plugins}

            plugins_info = []
            for plugin_name, plugin_info in all_plugins.items():
                plugins_info.append({
                    "name": plugin_name,
                    "enabled": user_plugin_status.get(plugin_name, False),
                    "description": plugin_info.description or "",
                    "version": plugin_info.version or "1.0.0",
                })

            await self.connection_manager.send_to_user(
                user_id, WebSocketMessage("plugin_list", {"plugins": plugins_info})
            )

        except Exception as e:
            logger.error(f"Plugin list error for user {user_id}: {e}")
            await self.connection_manager.send_to_user(
                user_id, WebSocketMessage("error", {"message": "Failed to get plugin list", "error": str(e)})
            )

    async def _handle_clarification_response(self, user_id: str, message_data: dict) -> None:
        """Handle clarification response from client."""
        data = message_data.get("data", {})
        response = data.get("response", "")
        context = data.get("context", {})
        original_query = context.get("original_query", "")
        
        logger.info(f"Received clarification response from {user_id}: {response}")
        
        if not response:
            await self.connection_manager.send_to_user(
                user_id, WebSocketMessage("error", {"message": "Empty clarification response"})
            )
            return
        
        # Process clarification response as a new AI query
        query_id = f"{user_id}_{int(time.time() * 1000)}"
        start_query_tracking(query_id, user_id, response)
        start_server_timer(query_id, "total_server")
        
        try:
            start_server_timer(query_id, "ai_processing")
            
            # Get conversation history
            history = []
            if self.db_manager:
                try:
                    history = await self.db_manager.get_user_history(user_id, limit=20)
                except Exception as hist_err:
                    logger.warning(f"Failed to get history for user {user_id}: {hist_err}")
            
            # Add clarification context
            clarification_context = {
                "history": history,
                "user_id": user_id,
                "is_clarification_response": True,
                "original_query": original_query,
                "clarification_answer": response
            }
            
            ai_result = await self.ai_module.process_query(response, clarification_context)
            end_server_timer(query_id, "ai_processing")
            
            # Handle clarification requests
            if ai_result.get("type") == "clarification_request":
                await self._handle_clarification(user_id, ai_result, response, query_id)
                return

            # Handle normal AI response
            await self._handle_ai_response(user_id, ai_result, response, message_data, query_id)
            
        except Exception as e:
            logger.error(f"Clarification response error for user {user_id}: {e}")
            record_server_error(query_id, f"Clarification response error: {e}")
            finish_server_query(query_id)
            await self.connection_manager.send_to_user(
                user_id,
                WebSocketMessage("error", {"message": "Failed to process clarification response", "error": str(e)}),
            )

    async def _handle_startup_briefing(self, user_id: str) -> None:
        """Handle startup briefing requests."""
        try:
            if self.proactive_assistant:
                notifications = await self.proactive_assistant.get_notifications(user_id)
                briefing = f"Daily briefing for user {user_id}: {len(notifications)} notifications"
                await self.connection_manager.send_to_user(
                    user_id, WebSocketMessage("startup_briefing", {"briefing": briefing})
                )
            else:
                await self.connection_manager.send_to_user(
                    user_id, WebSocketMessage("startup_briefing", {"briefing": "Proactive assistant nie jest dostępny"})
                )
        except Exception as e:
            logger.error(f"Briefing error for user {user_id}: {e}")
            await self.connection_manager.send_to_user(
                user_id, WebSocketMessage("error", {"message": "Failed to get briefing", "error": str(e)})
            )

    async def _handle_proactive_check(self, user_id: str) -> None:
        """Handle proactive notifications check."""
        try:
            if self.proactive_assistant:
                notifications = await self.proactive_assistant.get_notifications(user_id)
                await self.connection_manager.send_to_user(
                    user_id, WebSocketMessage("proactive_notifications", {"notifications": notifications})
                )
            else:
                await self.connection_manager.send_to_user(
                    user_id, WebSocketMessage("proactive_notifications", {"notifications": []})
                )
        except Exception as e:
            logger.error(f"Proactive check error for user {user_id}: {e}")

    async def _handle_context_update(self, user_id: str, message_data: dict) -> None:
        """Handle user context updates."""
        context_data = message_data.get("data", {})
        logger.debug(f"User context update from {user_id}: {context_data}")
        await self.connection_manager.send_to_user(
            user_id, WebSocketMessage("context_ack", {"status": "received"})
        )

    async def _generate_tts_audio(self, text: str) -> Optional[bytes]:
        """Generate TTS audio from text using OpenAI API."""
        try:
            if not AsyncOpenAI:
                logger.warning("OpenAI library not available for TTS")
                return None
                
            # Get OpenAI API key
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key and self.config:
                api_key = self.config.get("openai", {}).get("api_key")
            
            if not api_key:
                logger.warning("No OpenAI API key available for TTS")
                return None
            
            # Initialize client
            client = AsyncOpenAI(api_key=api_key)
            
            # Get TTS configuration
            tts_config = self.config.get("tts", {}) if self.config else {}
            model = "gpt-4o-mini-tts"
            voice = "sage"
            speed = tts_config.get("speed", 1.0)
            
            # Generate speech
            response = await client.audio.speech.create(
                model=model,
                voice=voice,
                input=text,
                response_format="mp3",
                speed=speed
            )
            
            audio_content = response.content
            logger.info(f"Generated TTS audio: {len(audio_content)} bytes")
            return audio_content
            
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            return None


# Create FastAPI app
app = FastAPI(
    title="GAJA Assistant Server",
    description="Optimized server for GAJA AI Assistant with multi-user support",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000", "http://localhost:8001"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router)


@app.get("/health")
async def health_check():
    """Health check endpoint for Docker and monitoring."""
    return {
        "status": "healthy",
        "timestamp": "2025-08-11T19:25:00Z",
        **(server_app.get_server_info() if server_app else {})
    }


@app.get("/api/status")
async def legacy_status():
    """Legacy status endpoint for client compatibility."""
    return {
        "status": "running",
        "version": "1.0.0",
        "timestamp": "2025-08-11T19:25:00Z",
    }


@app.get("/status/stream")
async def status_stream(request: Request):
    """Server-Sent Events endpoint for overlay status updates."""
    import asyncio
    from fastapi.responses import StreamingResponse

    async def event_stream():
        try:
            while True:
                status_data = {
                    "status": "running",
                    "text": "",
                    "is_listening": False,
                    "is_speaking": False,
                    "wake_word_detected": False,
                    "overlay_visible": False,
                }
                yield f"data: {json.dumps(status_data)}\n\n"
                await asyncio.sleep(1)
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
    """WebSocket endpoint for real-time communication."""
    logger.info(f"WebSocket connection attempt for user: {user_id}")

    try:
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
            try:
                data = await websocket.receive_text()
                message_data = json.loads(data)

                processed_message = await connection_manager.handle_message(user_id, message_data)

                if processed_message and server_app:
                    await server_app.handle_websocket_message(user_id, processed_message)

            except (WebSocketDisconnect, ConnectionClosed):
                logger.info(f"WebSocket disconnected for user: {user_id}")
                break
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON from user {user_id}: {e}")
                if user_id in connection_manager.active_connections:
                    try:
                        await connection_manager.send_to_user(
                            user_id, WebSocketMessage("error", {"message": "Invalid JSON format"})
                        )
                    except Exception:
                        break
            except Exception as e:
                error_str = str(e).lower()
                if any(phrase in error_str for phrase in [
                    "websocket is not connected", "need to call \"accept\" first",
                    "connection closed", "websocket disconnected"
                ]):
                    logger.info(f"WebSocket connection lost for user {user_id}: {e}")
                    break
                else:
                    logger.error(f"WebSocket message error for user {user_id}: {e}")
                
                if user_id in connection_manager.active_connections:
                    try:
                        await connection_manager.send_to_user(
                            user_id, WebSocketMessage("error", {"message": "Message processing error"})
                        )
                    except Exception:
                        break

    except Exception as e:
        logger.error(f"WebSocket connection error for user {user_id}: {e}")
    finally:
        await connection_manager.disconnect(user_id, "connection_closed")


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "GAJA Assistant Server", "status": "running"}


@app.on_event("startup")
async def startup_event():
    """Initialize server on startup."""
    global server_app
    server_app = GAJAServerApp()
    server_app.setup_logging()
    await server_app.initialize()
    set_server_app(server_app)
    logger.info("🚀 Server startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    if server_app:
        await server_app.cleanup()
    logger.info("🏁 Server shutdown complete")


def main():
    """Main server entry point."""
    global server_app
    
    # Create temporary server instance for configuration
    temp_server = GAJAServerApp()
    temp_server.setup_logging()
    
    logger.info("🚀 Starting GAJA Assistant Server...")
    
    # Get server configuration
    server_config = temp_server.get_server_config()
    
    # Start server
    uvicorn.run(
        app,
        host=server_config["host"],
        port=server_config["port"],
        log_level="warning",
        access_log=False,
        reload=False,
    )


if __name__ == "__main__":
    main()
