#!/usr/bin/env python3
"""
GAJA Assistant - WebSocket Connection Manager
System zarządzania połączeniami WebSocket w czasie rzeczywistym.
"""

import json
import time
from typing import Any, Optional

from fastapi import WebSocket
from loguru import logger


class WebSocketMessage:
    """Struktura wiadomości WebSocket."""

    def __init__(self, type_: str, data: Any = None, **kwargs):
        self.type = type_
        self.data = data
        self.timestamp = time.time()
        self.kwargs = kwargs

    def to_dict(self) -> dict:
        """Konwertuje wiadomość do słownika."""
        return {
            "type": self.type,
            "data": self.data,
            "timestamp": self.timestamp,
            **self.kwargs,
        }


class ConnectionManager:
    """Manager połączeń WebSocket."""

    def __init__(self):
        # Aktywne połączenia: user_id -> WebSocket
        self.active_connections: dict[str, WebSocket] = {}
        # Metadane połączeń: user_id -> dict
        self.connection_metadata: dict[str, dict] = {}
        # Rate limiting dla logów błędów
        self._last_warning_time: dict[str, float] = {}
        self._warning_cooldown = 5.0  # 5 sekund między ostrzeżeniami dla tego samego użytkownika
        # Statystyki
        self.stats = {
            "total_connections": 0,
            "messages_sent": 0,
            "messages_received": 0,
            "connection_errors": 0,
        }

    async def connect(self, websocket: WebSocket, user_id: str, metadata: dict = None):
        """Nawiązuje połączenie WebSocket."""
        try:
            await websocket.accept()

            # Rozłącz poprzednie połączenie jeśli istnieje
            if user_id in self.active_connections:
                await self.disconnect(user_id, reason="new_connection")

            # Dodaj nowe połączenie
            self.active_connections[user_id] = websocket
            self.connection_metadata[user_id] = {
                "connected_at": time.time(),
                "last_activity": time.time(),
                "messages_sent": 0,
                "messages_received": 0,
                **(metadata or {}),
            }

            self.stats["total_connections"] += 1

            logger.info(f"WebSocket connected for user: {user_id}")

            # Wyślij potwierdzenie połączenia
            await self.send_to_user(
                user_id,
                WebSocketMessage(
                    "handshake_response",
                    data={
                        "success": True,
                        "user_id": user_id,
                        "server_version": "1.0.0",
                        "timestamp": time.time(),
                    },
                ),
            )

        except Exception as e:
            self.stats["connection_errors"] += 1
            logger.error(f"Error connecting WebSocket for user {user_id}: {e}")
            raise

    async def disconnect(self, user_id: str, reason: str = "normal"):
        """Rozłącza połączenie WebSocket."""
        try:
            if user_id in self.active_connections:
                websocket = self.active_connections[user_id]

                # Wyślij wiadomość o rozłączeniu jeśli możliwe
                try:
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "disconnect",
                                "reason": reason,
                                "timestamp": time.time(),
                            }
                        )
                    )
                except:
                    pass  # Połączenie może być już zamknięte

                # Zamknij połączenie
                try:
                    await websocket.close()
                except:
                    pass

                # Usuń z zarządzania
                del self.active_connections[user_id]
                if user_id in self.connection_metadata:
                    del self.connection_metadata[user_id]

                logger.info(
                    f"WebSocket disconnected for user {user_id} (reason: {reason})"
                )

        except Exception as e:
            logger.error(f"Error disconnecting WebSocket for user {user_id}: {e}")

    async def send_to_user(self, user_id: str, message: WebSocketMessage) -> bool:
        """Wysyła wiadomość do konkretnego użytkownika."""
        try:
            if user_id not in self.active_connections:
                # Rate limiting dla logów ostrzeżeń
                current_time = time.time()
                last_warning = self._last_warning_time.get(user_id, 0)
                
                if current_time - last_warning > self._warning_cooldown:
                    logger.warning(f"User {user_id} not connected via WebSocket")
                    self._last_warning_time[user_id] = current_time
                    
                return False

            websocket = self.active_connections[user_id]
            message_dict = message.to_dict()

            await websocket.send_text(json.dumps(message_dict))

            # Aktualizuj statystyki
            self.stats["messages_sent"] += 1
            if user_id in self.connection_metadata:
                self.connection_metadata[user_id]["messages_sent"] += 1
                self.connection_metadata[user_id]["last_activity"] = time.time()

            logger.debug(f"Message sent to user {user_id}: {message.type}")
            return True

        except Exception as e:
            logger.error(f"Error sending message to user {user_id}: {e}")
            # Usuń połączenie jeśli jest uszkodzone
            await self.disconnect(user_id, reason="send_error")
            return False

    async def broadcast(
        self, message: WebSocketMessage, exclude_users: set[str] = None
    ):
        """Wysyła wiadomość do wszystkich połączonych użytkowników."""
        exclude_users = exclude_users or set()
        sent_count = 0

        for user_id in list(self.active_connections.keys()):
            if user_id not in exclude_users:
                if await self.send_to_user(user_id, message):
                    sent_count += 1

        logger.info(f"Broadcast message sent to {sent_count} users")
        return sent_count

    async def send_to_role(self, role: str, message: WebSocketMessage) -> int:
        """Wysyła wiadomość do użytkowników o określonej roli."""
        # TODO: Integracja z systemem ról użytkowników
        sent_count = 0

        for user_id in list(self.active_connections.keys()):
            # Tutaj należy sprawdzić rolę użytkownika z bazy danych
            # Na razie wysyłamy do wszystkich
            if await self.send_to_user(user_id, message):
                sent_count += 1

        return sent_count

    def is_connected(self, user_id: str) -> bool:
        """Sprawdza czy użytkownik jest połączony."""
        return user_id in self.active_connections

    def get_connected_users(self) -> list[str]:
        """Zwraca listę połączonych użytkowników."""
        return list(self.active_connections.keys())

    def get_connection_count(self) -> int:
        """Zwraca liczbę aktywnych połączeń."""
        return len(self.active_connections)

    def get_user_stats(self, user_id: str) -> dict | None:
        """Zwraca statystyki dla konkretnego użytkownika."""
        return self.connection_metadata.get(user_id)

    def get_stats(self) -> dict:
        """Zwraca ogólne statystyki."""
        return {
            **self.stats,
            "active_connections": len(self.active_connections),
            "connected_users": list(self.active_connections.keys()),
        }

    async def handle_message(self, user_id: str, message_data: dict) -> dict | None:
        """Obsługuje wiadomość od użytkownika."""
        try:
            # Aktualizuj statystyki
            self.stats["messages_received"] += 1
            if user_id in self.connection_metadata:
                self.connection_metadata[user_id]["messages_received"] += 1
                self.connection_metadata[user_id]["last_activity"] = time.time()

            message_type = message_data.get("type", "unknown")
            logger.debug(f"Received message from user {user_id}: {message_type}")

            # Zwróć wiadomość dla dalszego przetwarzania
            return {
                "user_id": user_id,
                "type": message_type,
                "data": message_data.get("data"),
                "query": message_data.get("query"),
                "context": message_data.get("context"),
                "timestamp": time.time(),
            }

        except Exception as e:
            logger.error(f"Error handling message from user {user_id}: {e}")
            return None

    async def cleanup_stale_connections(self, max_idle_time: int = 3600):
        """Czyści nieaktywne połączenia."""
        current_time = time.time()
        stale_users = []

        for user_id, metadata in self.connection_metadata.items():
            if current_time - metadata.get("last_activity", 0) > max_idle_time:
                stale_users.append(user_id)

        for user_id in stale_users:
            await self.disconnect(user_id, reason="idle_timeout")

        if stale_users:
            logger.info(f"Cleaned up {len(stale_users)} stale WebSocket connections")


# Globalny manager połączeń
connection_manager = ConnectionManager()
