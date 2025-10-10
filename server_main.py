#!/usr/bin/env python3
"""GAJA Assistant Server Główny serwer obsługujący wielu użytkowników, zarządzający AI,
bazą danych i pluginami."""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
import base64
from typing import Optional

import uvicorn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger
from websockets.exceptions import ConnectionClosed

# Add server path
sys.path.insert(0, str(Path(__file__).parent))

# Add OpenAI import for TTS
try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

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
from modules.ai_module import AIModule
from modules.server_performance_monitor import (
    start_query_tracking, start_server_timer, end_server_timer,
    record_server_response_data, record_server_error, finish_server_query
)

# Import API routes
from api.routes import router as api_router
from api.routes import set_server_app
from config.config_loader import load_config
from config.config_manager import initialize_database_manager
from extended_webui import ExtendedWebUI
from core.function_calling_system import get_function_calling_system
from modules.onboarding_module import OnboardingModule
from core.plugin_manager import plugin_manager
from core.plugin_monitor import plugin_monitor
from proactive_assistant_simple import get_proactive_assistant
from core.websocket_manager import WebSocketMessage, connection_manager

from integrations.telegram import TelegramBotService, load_telegram_config

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
        self.telegram_service = None

    async def handle_websocket_message(self, user_id: str, message_data: dict) -> None:
        """Obsługuje wiadomości WebSocket."""
        try:
            # Sprawdź czy message_data nie jest None
            if message_data is None:
                logger.error(f"Received None message_data from user {user_id}")
                await self.connection_manager.send_to_user(
                    user_id, WebSocketMessage("error", {"message": "Invalid message format"})
                )
                return
                
            message_type = message_data.get("type", "unknown")
            logger.info(f"WebSocket message from {user_id}: {message_type}")

            if message_type == "handshake":
                # Handshake jest obsługiwany automatycznie w connection_manager
                logger.info(f"Handshake completed for user {user_id}")
                return

            elif message_type == "query" or message_type == "ai_query" or message_type == "voice_command":
                # Zapytanie AI (tekstowe lub głosowe)
                data = message_data.get("data", {})
                query = data.get("query", "")
                context = data.get("context", {})
                assistant_text: Optional[str] = None

                # DEBUG: Log actual user query to debug inappropriate tool calls
                logger.warning(f"🎯 USER QUERY: '{query}' (type: {message_type})")

                # Jeśli oczekujemy na doprecyzowanie od użytkownika, potraktuj tę
                # zwykłą wiadomość jako odpowiedź doprecyślającą (fallback zgodny z klientem)
                try:
                    pending = self.connection_manager.connection_metadata.get(user_id, {}).get("pending_clarification")
                except Exception:
                    pending = None

                # Ustal tekst odpowiedzi z możliwych miejsc
                plain_query = query or message_data.get("query") or ""
                if pending and plain_query:
                    logger.info("Pending clarification detected; treating incoming query as clarification response")
                    original_query = pending.get("original_query", "") if isinstance(pending, dict) else ""
                    await self._process_clarification_response(user_id, plain_query, original_query, message_data)
                    return

                if not query:
                    await self.connection_manager.send_to_user(
                        user_id, WebSocketMessage("error", {"message": "Empty query"})
                    )
                    return

                # Start server performance tracking
                query_id = f"{user_id}_{int(time.time() * 1000)}"
                start_query_tracking(query_id, user_id, query)
                start_server_timer(query_id, "total_server")

                # Przetwórz zapytanie przez AI
                try:
                    start_server_timer(query_id, "ai_processing")
                    
                    # Get conversation history from database
                    history = []
                    if self.db_manager:
                        try:
                            history = await self.db_manager.get_user_history(user_id, limit=20)
                            logger.debug(f"Retrieved {len(history)} messages from history for user {user_id}")
                        except Exception as hist_err:
                            logger.warning(f"Failed to get history for user {user_id}: {hist_err}")
                    
                    # Add history to context
                    context["history"] = history
                    context["user_id"] = user_id
                    
                    ai_result = await self.ai_module.process_query(query, context)
                    end_server_timer(query_id, "ai_processing")
                    
                    logger.info(
                        f"AI module returned result type: {ai_result.get('type', 'unknown')}"
                    )
                    
                    # DEBUG: Log ai_result keys for fast-path debugging
                    logger.warning(f"🔍 AI RESULT KEYS: {list(ai_result.keys()) if isinstance(ai_result, dict) else 'not dict'}")
                    if ai_result.get("fast_tool_path"):
                        logger.warning(f"🚀 AI RESULT HAS FAST_TOOL_PATH: {ai_result.get('fast_tool_path')}")

                    # ------------------------------------------------------------------
                    # (Eksperymentalne) Wczesne strumieniowanie audio – tryb stub
                    # Jeśli ustawiono GAJA_AUDIO_STREAM_STUB=1 wysyłamy natychmiast
                    # po otrzymaniu tekstu AI mały chunk audio (sztuczny), aby klient
                    # mógł zmierzyć 'time-to-first-audio-token' zanim właściwe TTS
                    # zostanie wygenerowane. Finalna odpowiedź przyjdzie później jako
                    # standardowe "ai_response" (z pełnym tts_audio) – ten fragment nie
                    # zmienia istniejącego zachowania gdy zmienna nie jest ustawiona.
                    # ------------------------------------------------------------------
                    audio_stream_stub_enabled = os.getenv("GAJA_AUDIO_STREAM_STUB", "0").lower() in ("1", "true", "yes")
                    logger.warning(f"🎵 AUDIO STREAM STUB: enabled={audio_stream_stub_enabled}")

                    # Tekst będzie potrzebny – spróbuj wydobyć z ai_result.response
                    early_text = None
                    if audio_stream_stub_enabled and ai_result.get("type") == "normal_response":
                        try:
                            resp_payload = ai_result.get("response")
                            if isinstance(resp_payload, str):
                                # Check if it's a fast-path plain text response
                                if ai_result.get("fast_tool_path"):
                                    early_text = resp_payload  # Fast-path returns plain text
                                    logger.debug("Using fast-path response as early_text")
                                else:
                                    # Regular JSON response parsing
                                    parsed_payload = json.loads(resp_payload)
                                    if isinstance(parsed_payload, dict):
                                        early_text = parsed_payload.get("text")
                        except Exception:
                            early_text = None
                    if audio_stream_stub_enabled and early_text:
                        try:
                            # Krótki fałszywy chunk (można później zastąpić prawdziwym strumieniem)
                            fake_bytes = b"GAJA_STUB_AUDIO_CHUNK"
                            fake_b64 = base64.b64encode(fake_bytes).decode("utf-8")
                            await self.connection_manager.send_to_user(
                                user_id,
                                WebSocketMessage(
                                    "tts_chunk",
                                    {
                                        "chunk": fake_b64,
                                        "index": 0,
                                        "is_final": False,
                                        "format": "mp3",
                                        "stub": True,
                                    },
                                ),
                            )
                            logger.debug("Sent early stub tts_chunk (index=0, stub=True)")
                        except Exception as stub_err:
                            logger.debug(f"Failed to send stub tts_chunk: {stub_err}")

                    # Handle clarification requests
                    if ai_result.get("type") == "clarification_request":
                        # Send clarification request to client
                        clarification_data = ai_result.get("clarification_data", {})
                        
                        # Generate TTS for clarification question
                        tts_audio = None
                        question = clarification_data.get("question", "")
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
                        
                        # Zapisz stan oczekującej prośby o doprecyzowanie po stronie serwera
                        try:
                            meta = self.connection_manager.connection_metadata.setdefault(user_id, {})
                            meta["pending_clarification"] = {
                                "question": question,
                                "original_query": query,
                                "timestamp": time.time(),
                                # Przenieś informacje z clarification_data (jeśli model je dostarczył)
                                "function": clarification_data.get("function"),
                                "parameter": clarification_data.get("parameter"),
                            }
                            # Heurystyka: jeśli wygląda na pogodę, zapisz spodziewaną funkcję
                            if self._looks_like_weather_intent(query) and not meta["pending_clarification"].get("function"):
                                meta["pending_clarification"]["function"] = "weather_get_forecast"
                                meta["pending_clarification"]["parameter"] = "location"
                            logger.debug("Stored pending_clarification in connection metadata")
                        except Exception as e:
                            logger.debug(f"Could not store pending_clarification: {e}")

                        if tts_audio:
                            response_data["tts_audio"] = base64.b64encode(tts_audio).decode('utf-8')
                            # Add TTS configuration for client
                            tts_config = self.config.get("tts", {}) if self.config else {}
                            response_data["tts_config"] = {
                                "volume": tts_config.get("volume", 1.0),
                                "tts_metadata": {"format": "mp3"}
                            }
                            logger.info("TTS audio encoded and added to clarification response")
                        
                        # Record performance data
                        record_server_response_data(query_id, question, len(tts_audio) if tts_audio else None)
                        
                        start_server_timer(query_id, "websocket_send")
                        await self.connection_manager.send_to_user(
                            user_id,
                            WebSocketMessage("clarification_request", response_data),
                        )
                        end_server_timer(query_id, "websocket_send")
                        finish_server_query(query_id)
                        return

                    # Normal AI response
                    response = ai_result.get("response", "")
                    
                    # Save conversation to history
                    if self.db_manager:
                        try:
                            await self.db_manager.save_interaction(user_id, query, response)
                            logger.debug(f"Saved conversation to history for user {user_id}")
                        except Exception as save_err:
                            logger.warning(f"Failed to save conversation history: {save_err}")
                    
                    # Generate TTS audio if response contains text
                    tts_audio = None
                    try:
                        parsed_response = json.loads(response)
                        if isinstance(parsed_response, dict) and "text" in parsed_response:
                            text_to_speak = parsed_response["text"]
                            if text_to_speak:
                                assistant_text = text_to_speak.strip() or None
                                logger.debug("Generating TTS audio...")
                                start_server_timer(query_id, "tts_generation")
                                tts_audio = await self._generate_tts_audio(text_to_speak)
                                end_server_timer(query_id, "tts_generation")
                                logger.debug(f"Generated TTS audio: {len(tts_audio) if tts_audio else 0} bytes")
                    except json.JSONDecodeError:
                        # Try to use the response directly if it's a simple string
                        if isinstance(response, str) and response.strip():
                            logger.debug("Using response as direct text for TTS")
                            start_server_timer(query_id, "tts_generation")
                            tts_audio = await self._generate_tts_audio(response.strip())
                            end_server_timer(query_id, "tts_generation")
                            assistant_text = response.strip()
                    except Exception as tts_err:
                        end_server_timer(query_id, "tts_generation")
                        logger.error(f"TTS generation failed: {tts_err}")
                    
                    # Record performance data
                    record_server_response_data(query_id, response, len(tts_audio) if tts_audio else None)
                    
                    # Send AI response with optional TTS audio
                    response_data = {
                        "query": query,
                        "timestamp": message_data.get("timestamp"),
                    }
                    
                    # Handle fast-path responses - wrap plain text in JSON format
                    if ai_result.get("fast_tool_path") and isinstance(response, str):
                        # Fast-path returns plain text, wrap it for client compatibility
                        response_data["response"] = json.dumps({"text": response})
                        logger.debug(f"🚀 FAST PATH: Wrapped plain text response for client")
                    else:
                        # Regular response (already JSON formatted)
                        response_data["response"] = response
                    
                    # Include fast_tool_path if present in ai_result
                    if ai_result.get("fast_tool_path"):
                        response_data["fast_tool_path"] = True
                    
                    if tts_audio:
                        response_data["tts_audio"] = base64.b64encode(tts_audio).decode('utf-8')
                        # Add TTS configuration for client
                        tts_config = self.config.get("tts", {}) if self.config else {}
                        response_data["tts_config"] = {
                            "volume": tts_config.get("volume", 1.0),
                            "tts_metadata": {"format": "mp3"}
                        }
                        logger.info("TTS audio encoded and added to response")
                    else:
                        logger.info("No TTS audio to include in response")

                    final_response_text = assistant_text or (
                        response.strip() if isinstance(response, str) else ""
                    )
                    if (
                        message_type == "voice_command"
                        and final_response_text
                        and self.telegram_service
                    ):
                        asyncio.create_task(
                            self._forward_voice_result_to_telegram(
                                str(user_id), query or "", final_response_text
                            )
                        )

                    start_server_timer(query_id, "websocket_send")
                    await self.connection_manager.send_to_user(
                        user_id,
                        WebSocketMessage("ai_response", response_data),
                    )
                    # Opcjonalnie – jeśli stub chunk został wysłany, można (nieobowiązkowo)
                    # wysłać kończący chunk sygnalizujący koniec strumienia. Na razie jeśli
                    # istnieje pełne tts_audio, klient dostaje je w ai_response i może sam
                    # uznać strumień za zakończony.
                    end_server_timer(query_id, "websocket_send")
                    end_server_timer(query_id, "total_server")
                    finish_server_query(query_id)

                except Exception as e:
                    logger.error(f"AI query error for user {user_id}: {e}")
                    record_server_error(query_id, f"AI query error: {e}")
                    finish_server_query(query_id)
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
                            user_id, plugin_name
                        )
                        if self.db_manager:
                            await self.db_manager.update_user_plugin_status(
                                user_id, plugin_name, True
                            )
                        status = "enabled"
                    else:
                        await plugin_manager.disable_plugin_for_user(
                            user_id, plugin_name
                        )
                        if self.db_manager:
                            await self.db_manager.update_user_plugin_status(
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
                    user_plugins = []
                    if self.db_manager:
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

            elif message_type == "clarification_response":
                # Obsłuż odpowiedź doprecyślającą z klienta (również gdy data==None)
                raw_data = message_data.get("data")
                response = ""
                context = {}
                if isinstance(raw_data, dict):
                    response = raw_data.get("response", "")
                    context = raw_data.get("context", {}) or {}
                elif isinstance(raw_data, str):
                    response = raw_data
                
                # Fallback: klienci mogą wysyłać sam tekst jako zwykłe pole 'query'
                if not response:
                    response = message_data.get("query") or message_data.get("response") or ""
                
                # Ustal oryginalne pytanie: z kontekstu lub z metadanych połączenia
                original_query = ""
                if isinstance(context, dict):
                    original_query = context.get("original_query", "")
                if not original_query:
                    try:
                        meta = self.connection_manager.connection_metadata.get(user_id, {})
                        pending = meta.get("pending_clarification")
                        if isinstance(pending, dict):
                            original_query = pending.get("original_query", "")
                    except Exception:
                        pass

                logger.info(f"Received clarification response from {user_id}: {response}")

                await self._process_clarification_response(user_id, response, original_query, message_data)
                
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

    async def _process_clarification_response(self, user_id: str, response: str, original_query: str, message_data: dict | None = None) -> None:
        """Wspólny przebieg obsługi odpowiedzi doprecyślającej.
        Akceptuje odpowiedź podaną jako zwykła wiadomość lub w strukturze clarification_response.
        """
        # Walidacja
        if not response:
            await self.connection_manager.send_to_user(
                user_id, WebSocketMessage("error", {"message": "Empty clarification response"})
            )
            return

        # Odczytaj pending_clarification zanim go wyczyścisz (potrzebny do fast‑path)
        pending_copy = None
        try:
            meta = self.connection_manager.connection_metadata.setdefault(user_id, {})
            if "pending_clarification" in meta:
                pending_copy = dict(meta["pending_clarification"]) if isinstance(meta["pending_clarification"], dict) else meta["pending_clarification"]
                del meta["pending_clarification"]
        except Exception:
            pending_copy = None

        query_id = f"{user_id}_{int(time.time() * 1000)}"
        start_query_tracking(query_id, user_id, response)
        start_server_timer(query_id, "total_server")

        try:
            start_server_timer(query_id, "ai_processing")

            # Historia
            history = []
            if self.db_manager:
                try:
                    history = await self.db_manager.get_user_history(user_id, limit=20)
                    logger.debug(f"Retrieved {len(history)} messages from history for user {user_id}")
                except Exception as hist_err:
                    logger.warning(f"Failed to get history for user {user_id}: {hist_err}")

            clarification_context = {
                "history": history,
                "user_id": user_id,
                "is_clarification_response": True,
                "original_query": original_query or "",
                "clarification_answer": response,
            }

            # Fast-path: jeśli oczekujemy lokalizacji do pogody, spróbuj od razu wywołać narzędzie zamiast pytać model
            # Użyj kopii pending z początku metody
            expected_fn = None
            if isinstance(pending_copy, dict):
                expected_fn = pending_copy.get("function")
            if not expected_fn and self._looks_like_weather_intent(original_query):
                expected_fn = "weather_get_forecast"

            if expected_fn == "weather_get_forecast" and response.strip():
                # Spróbuj znormalizować lokalizację i ustalić liczbę dni
                norm_loc = self._normalize_location(response.strip())
                days = self._extract_days_from_text(original_query) or 2
                fast_text = await self._try_fast_weather_summary(user_id, norm_loc, days=days)
                if fast_text:
                    end_server_timer(query_id, "ai_processing")
                    # Zapisz do historii z pełnym pytaniem
                    if self.db_manager:
                        try:
                            query_to_save = original_query or response
                            if original_query and response and response.strip() != original_query.strip():
                                query_to_save = f"{original_query}\n[uzupełnienie: {response}]"
                            wrapped = json.dumps({"text": fast_text, "command": "", "params": {"fast_tool_path": True}}, ensure_ascii=False)
                            await self.db_manager.save_interaction(user_id, query_to_save, wrapped)
                        except Exception as save_err:
                            logger.warning(f"Failed to save conversation history: {save_err}")

                    # TTS i odpowiedź do klienta
                    tts_audio = None
                    try:
                        start_server_timer(query_id, "tts_generation")
                        tts_audio = await self._generate_tts_audio(fast_text)
                        end_server_timer(query_id, "tts_generation")
                    except Exception:
                        end_server_timer(query_id, "tts_generation")
                        tts_audio = None

                    response_data = {
                        "query": response,
                        "timestamp": (message_data or {}).get("timestamp") if isinstance(message_data, dict) else None,
                        "response": json.dumps({"text": fast_text, "command": "", "params": {}}, ensure_ascii=False),
                    }
                    if tts_audio:
                        response_data["tts_audio"] = base64.b64encode(tts_audio).decode("utf-8")
                        tts_config = self.config.get("tts", {}) if self.config else {}
                        response_data["tts_config"] = {"volume": tts_config.get("volume", 1.0), "tts_metadata": {"format": "mp3"}}

                    await self.connection_manager.send_to_user(user_id, WebSocketMessage("ai_response", response_data))
                    finish_server_query(query_id)
                    return

            # Brak fast‑path – przekaż do modelu
            ai_result = await self.ai_module.process_query(response, clarification_context)
            end_server_timer(query_id, "ai_processing")

            # Jeśli kolejna prośba o doprecyzowanie – wyślij i zaktualizuj pending
            if ai_result.get("type") == "clarification_request":
                clarification_data = ai_result.get("clarification_data", {})

                # TTS
                tts_audio = None
                question = clarification_data.get("question", "")
                if question:
                    try:
                        start_server_timer(query_id, "tts_generation")
                        tts_audio = await self._generate_tts_audio(question)
                        end_server_timer(query_id, "tts_generation")
                    except Exception as e:
                        end_server_timer(query_id, "tts_generation")
                        logger.error(f"Failed to generate TTS for follow-up clarification: {e}")

                response_payload = {
                    "question": question,
                    "context": clarification_data.get("context", ""),
                    "actions": clarification_data.get("actions", {}),
                    "timestamp": clarification_data.get("timestamp"),
                    "original_query": original_query or "",
                }

                # Zapisz nowy pending_clarification
                try:
                    meta = self.connection_manager.connection_metadata.setdefault(user_id, {})
                    meta["pending_clarification"] = {
                        "question": question,
                        "original_query": original_query or "",
                        "timestamp": time.time(),
                    }
                except Exception:
                    pass

                if tts_audio:
                    response_payload["tts_audio"] = base64.b64encode(tts_audio).decode("utf-8")
                    tts_config = self.config.get("tts", {}) if self.config else {}
                    response_payload["tts_config"] = {
                        "volume": tts_config.get("volume", 1.0),
                        "tts_metadata": {"format": "mp3"},
                    }

                await self.connection_manager.send_to_user(
                    user_id, WebSocketMessage("clarification_request", response_payload)
                )
                finish_server_query(query_id)
                return

            # W przeciwnym razie – normalna odpowiedź
            ai_response = ai_result.get("response", "")

            if self.db_manager:
                try:
                    # Zapisz pełne pytanie wraz z doprecyzowaniem, aby historia była czytelna
                    query_to_save = original_query or response or ""
                    if original_query and response and response.strip() and response.strip() != original_query.strip():
                        query_to_save = f"{original_query}\n[uzupełnienie: {response}]"
                    await self.db_manager.save_interaction(user_id, query_to_save, ai_response)
                    logger.debug(f"Saved clarification conversation to history for user {user_id}")
                except Exception as save_err:
                    logger.warning(f"Failed to save conversation history: {save_err}")

            # Wygeneruj TTS (opcjonalnie)
            tts_audio = None
            try:
                parsed_response = json.loads(ai_response)
                if isinstance(parsed_response, dict) and "text" in parsed_response:
                    text_to_speak = parsed_response["text"]
                    if text_to_speak:
                        start_server_timer(query_id, "tts_generation")
                        tts_audio = await self._generate_tts_audio(text_to_speak)
                        end_server_timer(query_id, "tts_generation")
            except (json.JSONDecodeError, TypeError):
                pass

            response_data = {
                "query": response,
                "timestamp": (message_data or {}).get("timestamp") if isinstance(message_data, dict) else None,
                "response": ai_response,
            }

            if tts_audio:
                response_data["tts_audio"] = base64.b64encode(tts_audio).decode("utf-8")
                tts_config = self.config.get("tts", {}) if self.config else {}
                response_data["tts_config"] = {
                    "volume": tts_config.get("volume", 1.0),
                    "tts_metadata": {"format": "mp3"},
                }

            await self.connection_manager.send_to_user(
                user_id, WebSocketMessage("ai_response", response_data)
            )
            finish_server_query(query_id)

        except Exception as e:
            logger.error(f"Clarification response error for user {user_id}: {e}")
            finish_server_query(query_id)
            await self.connection_manager.send_to_user(
                user_id,
                WebSocketMessage(
                    "error",
                    {"message": "Failed to process clarification response", "error": str(e)},
                ),
            )

    def _looks_like_weather_intent(self, text: str | None) -> bool:
        try:
            if not text:
                return False
            t = text.lower()
            return any(kw in t for kw in ["pogod", "pada", "deszcz", "prognoz", "temperatur"]) \
                or ("jutro" in t and ("czy" in t or "będzie" in t))
        except Exception:
            return False

    def _normalize_location(self, text: str) -> str:
        """Prosta normalizacja polskich fraz lokalizacyjnych i literówek.
        Usuwa przyimki i zbędne słowa grzecznościowe; poprawia wybrane formy przypadków.
        """
        t = (text or "").strip().lower()
        # Usuń przecinki i kropki na końcach
        t = t.replace(",", " ").replace(".", " ")
        # Usuń częste przyimki i wtrącenia na początku
        prefixes = ["w ", "we ", "na ", "do ", "dla ", "z ", "ze ", "od ", "edla "]
        for p in prefixes:
            if t.startswith(p):
                t = t[len(p):]
                break
        # Usuń słowa grzecznościowe / wtrącenia
        fillers = {"poprosze", "poproszę", "prosze", "proszę", "tam", "mi", "by", "bym", "poprasił", "poprosił", "prosił"}
        words = [w for w in t.split() if w not in fillers]
        t = " ".join(words).strip()
        # Słownik literówek / przypadków (minimalny)
        fixes = {
            "sosnowcu": "sosnowiec",
            "sosnowca": "sosnowiec",
            "sosnofca": "sosnowiec",
            "sosnofiec": "sosnowiec",
        }
        tokens = t.split()
        tokens = [fixes.get(tok, tok) for tok in tokens]
        norm = " ".join(tokens).strip()
        # Tytułowy zapis (pierwsza wielka litera)
        return norm.capitalize()

    def _extract_days_from_text(self, text: str | None) -> int | None:
        try:
            if not text:
                return None
            t = text.lower()
            if "jutro" in t:
                return 1
            return None
        except Exception:
            return None

    async def _try_fast_weather_summary(self, user_id: str, location: str, days: int = 2) -> str | None:
        """Szybka ścieżka: wywołaj narzędzie pogody i zbuduj zwięzłe podsumowanie bez modelu.
        Zwraca tekst podsumowania lub None jeśli coś poszło nie tak.
        """
        try:
            if not hasattr(self, "function_system") or not self.function_system:
                return None
            args = {"location": location, "days": max(1, min(days, 3))}
            result = await self.function_system.execute_function("weather_get_forecast", args)
            if not isinstance(result, dict) or not result.get("success"):
                return None
            data = result.get("data") or {}
            loc = data.get("location") or {}
            name = (loc.get("name") or location).strip()
            country = (loc.get("country") or "").strip()
            parts = []
            if country:
                header = f"Prognoza dla {name}, {country}:"
            else:
                header = f"Prognoza dla {name}:"
            parts.append(header)
            fc = data.get("forecast") or []
            # Formatuj maks. dwa najbliższe dni
            for i, day in enumerate(fc[:2]):
                d = day.get("date") or ""
                desc = (day.get("description") or "").strip()
                tmin = day.get("min_temp")
                tmax = day.get("max_temp")
                seg = f"{d}: {desc}".strip()
                if tmin is not None and tmax is not None:
                    seg += f", min {tmin}°C, max {tmax}°C"
                parts.append(seg)
            return "\n".join(parts)
        except Exception as e:
            logger.debug(f"Fast weather path failed: {e}")
            return None

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
            from config.config_loader import ConfigLoader

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

            # Initialize function calling system (use singleton)
            from core.function_calling_system import get_function_calling_system
            self.function_system = get_function_calling_system()
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

            await self._initialize_telegram_integration()

            logger.success("Server initialization completed")

        except Exception as e:
            logger.error(f"Server initialization failed: {e}")
            raise

    async def _initialize_telegram_integration(self) -> None:
        try:
            integrations_config = self.config.get("integrations", {}) if self.config else {}
            telegram_config_raw = integrations_config.get("telegram", {})
            config = load_telegram_config(telegram_config_raw)
            if not config.enabled:
                logger.info("Telegram integration disabled in configuration")
                return

            self.telegram_service = TelegramBotService(self, config)
            await self.telegram_service.start()
        except Exception as exc:
            logger.error(f"Telegram integration failed to start: {exc}")

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

    async def _forward_voice_result_to_telegram(
        self, user_id: str, query: str, response_text: str
    ) -> None:
        if not self.telegram_service:
            return

        try:
            await self.telegram_service.forward_voice_interaction(
                user_id, query, response_text
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.debug(f"Voice → Telegram bridge failed for user {user_id}: {exc}")

    async def _generate_tts_audio(self, text: str) -> Optional[bytes]:
        """Generate TTS audio from text using OpenAI API."""
        try:
            if not AsyncOpenAI:
                logger.warning("OpenAI library not available for TTS")
                return None
                
            # Get OpenAI API key from config or environment
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key and self.config:
                api_key = self.config.get("openai", {}).get("api_key")
            
            if not api_key:
                logger.warning("No OpenAI API key available for TTS")
                return None
            
            # Initialize OpenAI client
            client = AsyncOpenAI(api_key=api_key)
            
            # Get TTS configuration from config or use defaults
            tts_config = self.config.get("tts", {}) if self.config else {}
            # Standardize to gpt-4o-mini-tts and voice sage
            model = "gpt-4o-mini-tts"
            voice = "sage"
            speed = tts_config.get("speed", 1.0)      # 0.25 to 4.0
            volume = tts_config.get("volume", 1.0)    # Will be used by client for playback
            
            # Generate speech
            response = await client.audio.speech.create(
                model=model,
                voice=voice,
                input=text,
                response_format="mp3",
                speed=speed
            )
            
            # Get audio content - response content is available directly
            audio_content = response.content
            
            logger.info(f"Generated TTS audio: {len(audio_content)} bytes")
            return audio_content
            
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            return None

    async def cleanup(self):
        """Clean up server resources."""
        try:
            if self.proactive_assistant:
                self.proactive_assistant.stop()
            if self.plugin_monitor:
                await self.plugin_monitor.stop_monitoring()
            if self.telegram_service:
                await self.telegram_service.stop()
            if self.db_manager and hasattr(self.db_manager, "close"):
                try:
                    close_method = getattr(self.db_manager, "close")
                    if close_method:
                        if hasattr(close_method, "__await__"):
                            await close_method()
                        else:
                            close_method()
                except Exception as close_err:
                    logger.warning(f"Error closing database: {close_err}")
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

# Serve Admin Panel static assets
admin_panel_path = Path(__file__).parent / "admin_panel"
if admin_panel_path.exists():
    app.mount("/admin/static", StaticFiles(directory=str(admin_panel_path)), name="admin_static")

@app.get("/admin")
async def admin_index():
    """Serve Admin Panel index HTML."""
    index_file = admin_panel_path / "index.html"
    if not index_file.exists():
        return {"error": "Admin panel not found"}
    from fastapi.responses import HTMLResponse
    return HTMLResponse(index_file.read_text(encoding="utf-8"))


# Backward compatibility: legacy direct asset paths (old index referenced panel.css/panel.js at root)
@app.get("/panel.css")
async def legacy_panel_css():
    file_path = admin_panel_path / "panel.css"
    if file_path.exists():
        from fastapi.responses import FileResponse
        return FileResponse(str(file_path), media_type="text/css")
    from fastapi import Response
    return Response(status_code=404)

@app.get("/panel.js")
async def legacy_panel_js():
    file_path = admin_panel_path / "panel.js"
    if file_path.exists():
        from fastapi.responses import FileResponse
        return FileResponse(str(file_path), media_type="application/javascript")
    from fastapi import Response
    return Response(status_code=404)

# Debug UI (Test Bench) static assets
debug_ui_path = Path(__file__).parent / "debug_ui"
if debug_ui_path.exists():
    app.mount("/debug/static", StaticFiles(directory=str(debug_ui_path)), name="debug_static")

@app.get("/debug")
async def debug_index():
    """Serve Debug Center index HTML."""
    index_file = debug_ui_path / "index.html"
    if not index_file.exists():
        return {"error": "Debug UI not found"}
    from fastapi.responses import HTMLResponse
    return HTMLResponse(index_file.read_text(encoding="utf-8"))



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
            except ConnectionClosed:
                logger.info(f"WebSocket connection closed for user: {user_id}")
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
                # Sprawdź czy to błąd związany z WebSocket
                error_str = str(e).lower()
                if any(phrase in error_str for phrase in [
                    "websocket is not connected", 
                    "need to call \"accept\" first",
                    "connection closed",
                    "websocket disconnected"
                ]):
                    logger.info(f"WebSocket connection lost for user {user_id}: {e}")
                    break  # Wyjdź z pętli gdy WebSocket jest rozłączony
                else:
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
                        # Wyjdź z pętli jeśli nie można wysłać wiadomości
                        break

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
