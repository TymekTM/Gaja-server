"""
config_manager.py - Unified environment and database management system
Combines functionality from environment_manager.py and database_manager.py
"""

import asyncio
import json
import logging
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from database_models import APIUsage, MemoryContext, Message, User, UserSession

logger = logging.getLogger(__name__)


class EnvironmentManager:
    """Zarządza zmiennymi środowiskowymi i bezpiecznym ładowaniem konfiguracji."""

    def __init__(self, env_file: str = ".env"):
        self.env_file = Path(env_file)
        self._load_env_file()

    def _load_env_file(self):
        """Ładuje zmienne z pliku .env jeśli istnieje."""
        if self.env_file.exists():
            try:
                with open(self.env_file, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            key, value = line.split("=", 1)
                            # Don't override if already set in environment
                            if key not in os.environ:
                                os.environ[key] = value
                logger.info(f"Loaded environment variables from {self.env_file}")
            except Exception as e:
                logger.error(f"Error loading .env file: {e}")

    def get_api_key(self, service: str) -> str | None:
        """Pobiera klucz API dla danej usługi z zmiennych środowiskowych.

        Args:
            service: Nazwa usługi (openai, anthropic, etc.)

        Returns:
            Klucz API lub None jeśli nie znaleziono
        """
        # Standardize service name to uppercase
        env_key = f"{service.upper()}_API_KEY"

        # Special cases for legacy compatibility
        if service.lower() == "openai":
            return os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
        elif service.lower() == "azure":
            return os.getenv("AZURE_SPEECH_KEY") or os.getenv("AZURE_API_KEY")

        return os.getenv(env_key)

    def get_database_url(self) -> str:
        """Pobiera URL bazy danych z zmiennych środowiskowych."""
        return os.getenv("DATABASE_URL", "sqlite:///./gaja_assistant.db")

    def get_server_config(self) -> dict[str, Any]:
        """Pobiera konfigurację serwera z zmiennych środowiskowych."""
        return {
            "host": os.getenv("SERVER_HOST", "localhost"),
            "port": int(os.getenv("SERVER_PORT", "8001")),
            "secret_key": os.getenv("SECRET_KEY"),
            "cors_origins": os.getenv("CORS_ORIGINS", "http://localhost:3000").split(
                ","
            ),
            "max_connections_per_user": int(os.getenv("MAX_CONNECTIONS_PER_USER", "5")),
            "session_timeout_hours": int(os.getenv("SESSION_TIMEOUT_HOURS", "24")),
        }

    def sanitize_config_for_logging(self, config: dict[str, Any]) -> dict[str, Any]:
        """Czyści konfigurację z wrażliwych danych przed logowaniem.

        Args:
            config: Słownik konfiguracji

        Returns:
            Oczyszczony słownik bez wrażliwych danych
        """
        sanitized: dict[str, Any] = {}
        sensitive_keys = {
            "api_key",
            "password",
            "secret",
            "token",
            "auth",
            "credential",
            "openai_api_key",
            "anthropic_api_key",
            "deepseek_api_key",
            "azure_speech_key",
            "together_api_key",
            "groq_api_key",
        }

        for key, value in config.items():
            key_lower = key.lower()

            # Check if key contains sensitive information
            if any(sensitive in key_lower for sensitive in sensitive_keys):
                # Show only first and last 4 characters for API keys
                if isinstance(value, str) and len(value) > 8:
                    sanitized[key] = f"{value[:4]}***{value[-4:]}"
                else:
                    sanitized[key] = "***HIDDEN***"
            elif isinstance(value, dict):
                # Recursively sanitize nested dictionaries
                sanitized[key] = self.sanitize_config_for_logging(value)
            else:
                sanitized[key] = value

        return sanitized

    def validate_required_keys(self, required_services: list[str]) -> dict[str, bool]:
        """Sprawdza czy wymagane klucze API są dostępne.

        Args:
            required_services: Lista wymaganych usług

        Returns:
            Słownik z wynikami walidacji
        """
        validation_results = {}

        for service in required_services:
            api_key = self.get_api_key(service)
            validation_results[service] = (
                api_key is not None and len(api_key.strip()) > 0
            )

            if not validation_results[service]:
                logger.warning(f"Missing or empty API key for {service}")

        return validation_results


class DatabaseManager:
    """Zarządza bazą danych SQLite dla systemu Gaja."""

    def __init__(self, db_path: str = "server_data.db"):
        """Inicjalizuje menedżer bazy danych.

        Args:
            db_path: Ścieżka do pliku bazy danych
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._local = threading.local()

        # Inicjalizuj bazę danych
        self._init_database()
        logger.info(f"DatabaseManager initialized with database: {self.db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """Pobiera połączenie do bazy danych dla bieżącego wątku."""
        if not hasattr(self._local, "connection"):
            self._local.connection = sqlite3.connect(
                str(self.db_path), check_same_thread=False, timeout=30.0
            )
            self._local.connection.row_factory = sqlite3.Row
            # Włącz foreign keys - WAŻNE: to musi być zawsze włączone
            self._local.connection.execute("PRAGMA foreign_keys = ON")
            self._local.connection.commit()
        return self._local.connection

    @contextmanager
    def get_db_connection(self):
        """Context manager dla połączenia z bazą danych."""
        conn = self._get_connection()
        # Upewnij się, że foreign keys są włączone
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.commit()

    def _init_database(self):
        """Inicjalizuje strukturę bazy danych."""
        with self.get_db_connection() as conn:
            # Tabela użytkowników
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE,
                    password_hash TEXT,
                    is_active BOOLEAN DEFAULT TRUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    settings TEXT DEFAULT '{}',
                    api_keys TEXT DEFAULT '{}'
                )
            """
            )

            # Tabela sesji użytkowników
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    session_token TEXT UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    is_active BOOLEAN DEFAULT TRUE,
                    client_info TEXT DEFAULT '{}',
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                )
            """
            )

            # Tabela wiadomości/konwersacji
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    session_id INTEGER,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
                    content TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    parent_message_id INTEGER,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
                    FOREIGN KEY (session_id) REFERENCES user_sessions (id) ON DELETE SET NULL,
                    FOREIGN KEY (parent_message_id) REFERENCES messages (id) ON DELETE SET NULL
                )
            """
            )

            # Tabela kontekstu pamięci
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_contexts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    context_type TEXT NOT NULL,
                    key_name TEXT NOT NULL,
                    value TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
                    UNIQUE(user_id, context_type, key_name)
                )
            """
            )

            # Tabela wykorzystania API
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS api_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    api_provider TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    method TEXT NOT NULL,
                    tokens_used INTEGER DEFAULT 0,
                    cost REAL DEFAULT 0.0,
                    success BOOLEAN DEFAULT TRUE,
                    response_time REAL,
                    error_message TEXT,
                    metadata TEXT DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                )
            """
            )

            # Tabela logów systemowych
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS system_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    level TEXT NOT NULL,
                    module TEXT NOT NULL,
                    message TEXT NOT NULL,
                    user_id INTEGER,
                    session_id INTEGER,
                    metadata TEXT DEFAULT '{}',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE SET NULL,
                    FOREIGN KEY (session_id) REFERENCES user_sessions (id) ON DELETE SET NULL
                )
            """
            )

            # Tabela preferencji użytkowników
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_preferences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    category TEXT NOT NULL,
                    key_name TEXT NOT NULL,
                    value TEXT NOT NULL,
                    value_type TEXT DEFAULT 'string',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
                    UNIQUE(user_id, category, key_name)
                )
            """
            )

            # Indeksy dla wydajności
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_user_id ON messages (user_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages (session_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages (created_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_contexts_user_id ON memory_contexts (user_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_contexts_type ON memory_contexts (context_type)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_api_usage_user_id ON api_usage (user_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_api_usage_created_at ON api_usage (created_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_user_sessions_token ON user_sessions (session_token)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_system_logs_created_at ON system_logs (created_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_user_preferences_user_id ON user_preferences (user_id)"
            )

    # === ZARZĄDZANIE UŻYTKOWNIKAMI ===

    def create_user(
        self,
        username: str,
        email: str = None,
        password_hash: str = None,
        settings: dict = None,
        api_keys: dict = None,
    ) -> int:
        """Tworzy nowego użytkownika."""
        with self.get_db_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO users (username, email, password_hash, settings, api_keys)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    username,
                    email,
                    password_hash,
                    json.dumps(settings or {}),
                    json.dumps(api_keys or {}),
                ),
            )
            user_id = cursor.lastrowid
            logger.info(f"Created user: {username} (ID: {user_id})")
            return user_id

    def get_user(self, user_id: int = None, username: str = None) -> User | None:
        """Pobiera użytkownika po ID lub nazwie."""
        with self.get_db_connection() as conn:
            if user_id:
                cursor = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,))
            elif username:
                cursor = conn.execute(
                    "SELECT * FROM users WHERE username = ?", (username,)
                )
            else:
                return None

            row = cursor.fetchone()
            if row:
                return User.from_db_row(row)
            return None

    def update_user_settings(self, user_id: int, settings: dict):
        """Aktualizuje ustawienia użytkownika."""
        with self.get_db_connection() as conn:
            conn.execute(
                """
                UPDATE users SET settings = ?, last_login = CURRENT_TIMESTAMP
                WHERE id = ?
            """,
                (json.dumps(settings), user_id),
            )

    def update_user_api_keys(self, user_id: int, api_keys: dict):
        """Aktualizuje klucze API użytkownika."""
        with self.get_db_connection() as conn:
            conn.execute(
                """
                UPDATE users SET api_keys = ? WHERE id = ?
            """,
                (json.dumps(api_keys), user_id),
            )

    def get_user_level(self, user_id: int) -> str:
        """Return user subscription level (free/plus/pro)."""
        user = self.get_user(user_id=user_id)
        if user and isinstance(user.settings, dict):
            return user.settings.get("level", "free")
        return "free"

    def set_user_level(self, user_id: int, level: str) -> None:
        settings = (
            self.get_user(user_id=user_id).settings
            if self.get_user(user_id=user_id)
            else {}
        )
        settings["level"] = level
        self.update_user_settings(user_id, settings)

    # === ZARZĄDZANIE SESJAMI ===

    def create_session(
        self,
        user_id: int,
        session_token: str,
        expires_at: datetime,
        client_info: dict = None,
    ) -> int:
        """Tworzy nową sesję użytkownika."""
        with self.get_db_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO user_sessions (user_id, session_token, expires_at, client_info)
                VALUES (?, ?, ?, ?)
            """,
                (user_id, session_token, expires_at, json.dumps(client_info or {})),
            )
            return cursor.lastrowid

    def get_session(self, session_token: str) -> UserSession | None:
        """Pobiera sesję po tokenie."""
        with self.get_db_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM user_sessions
                WHERE session_token = ? AND is_active = TRUE AND expires_at > CURRENT_TIMESTAMP
            """,
                (session_token,),
            )
            row = cursor.fetchone()
            if row:
                return UserSession.from_db_row(row)
            return None

    def invalidate_session(self, session_token: str):
        """Dezaktywuje sesję."""
        with self.get_db_connection() as conn:
            conn.execute(
                """
                UPDATE user_sessions SET is_active = FALSE WHERE session_token = ?
            """,
                (session_token,),
            )

    # === ZARZĄDZANIE WIADOMOŚCIAMI ===

    def save_message(
        self,
        user_id: int,
        role: str,
        content: str,
        session_id: int = None,
        metadata: dict = None,
        parent_message_id: int = None,
    ) -> int:
        """Zapisuje wiadomość do bazy danych."""
        with self.get_db_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO messages (user_id, session_id, role, content, metadata, parent_message_id)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    user_id,
                    session_id,
                    role,
                    content,
                    json.dumps(metadata or {}),
                    parent_message_id,
                ),
            )
            return cursor.lastrowid

    def get_user_messages(
        self, user_id: int, limit: int = 100, session_id: int = None
    ) -> list[Message]:
        """Pobiera wiadomości użytkownika."""
        with self.get_db_connection() as conn:
            if session_id:
                cursor = conn.execute(
                    """
                    SELECT * FROM messages
                    WHERE user_id = ? AND session_id = ?
                    ORDER BY created_at DESC LIMIT ?
                """,
                    (user_id, session_id, limit),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT * FROM messages
                    WHERE user_id = ?
                    ORDER BY created_at DESC LIMIT ?
                """,
                    (user_id, limit),
                )

            return [Message.from_db_row(row) for row in cursor.fetchall()]

    def get_conversation_context(
        self, user_id: int, session_id: int = None, limit: int = 10
    ) -> list[Message]:
        """Pobiera kontekst konwersacji dla AI."""
        messages = self.get_user_messages(user_id, limit, session_id)
        return list(reversed(messages))  # Odwróć aby mieć chronologiczny porządek

    # === ZARZĄDZANIE PAMIĘCIĄ ===

    def save_memory_context(
        self,
        user_id: int,
        context_type: str,
        key_name: str,
        value: str,
        metadata: dict = None,
        expires_at: datetime = None,
    ):
        """Zapisuje kontekst pamięci."""
        with self.get_db_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO memory_contexts
                (user_id, context_type, key_name, value, metadata, expires_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
                (
                    user_id,
                    context_type,
                    key_name,
                    value,
                    json.dumps(metadata or {}),
                    expires_at,
                ),
            )

    def get_memory_context(
        self, user_id: int, context_type: str, key_name: str = None
    ) -> list[MemoryContext]:
        """Pobiera kontekst pamięci."""
        with self.get_db_connection() as conn:
            if key_name:
                cursor = conn.execute(
                    """
                    SELECT * FROM memory_contexts
                    WHERE user_id = ? AND context_type = ? AND key_name = ?
                    AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
                """,
                    (user_id, context_type, key_name),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT * FROM memory_contexts
                    WHERE user_id = ? AND context_type = ?
                    AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
                    ORDER BY updated_at DESC
                """,
                    (user_id, context_type),
                )

            return [MemoryContext.from_db_row(row) for row in cursor.fetchall()]

    def cleanup_expired_memory(self):
        """Usuwa wygasłe wpisy pamięci."""
        with self.get_db_connection() as conn:
            cursor = conn.execute(
                """
                DELETE FROM memory_contexts
                WHERE expires_at IS NOT NULL AND expires_at <= CURRENT_TIMESTAMP
            """
            )
            if cursor.rowcount > 0:
                logger.info(f"Cleaned up {cursor.rowcount} expired memory entries")

    # === ZARZĄDZANIE PREFERENCJAMI ===

    def set_user_preference(
        self,
        user_id: int,
        category: str,
        key_name: str,
        value: Any,
        value_type: str = "string",
    ):
        """Ustawia preferencję użytkownika."""
        with self.get_db_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO user_preferences
                (user_id, category, key_name, value, value_type, updated_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
                (user_id, category, key_name, str(value), value_type),
            )

    def get_user_preferences(
        self, user_id: int, category: str = None
    ) -> dict[str, Any]:
        """Pobiera preferencje użytkownika."""
        with self.get_db_connection() as conn:
            if category:
                cursor = conn.execute(
                    """
                    SELECT key_name, value, value_type FROM user_preferences
                    WHERE user_id = ? AND category = ?
                """,
                    (user_id, category),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT key_name, value, value_type FROM user_preferences
                    WHERE user_id = ?
                """,
                    (user_id,),
                )

            preferences = {}
            for row in cursor.fetchall():
                key_name, value, value_type = row
                # Konwertuj wartość na odpowiedni typ
                if value_type == "int":
                    preferences[key_name] = int(value)
                elif value_type == "float":
                    preferences[key_name] = float(value)
                elif value_type == "bool":
                    preferences[key_name] = value.lower() in ("true", "1", "yes")
                elif value_type == "json":
                    preferences[key_name] = json.loads(value)
                else:
                    preferences[key_name] = value

            return preferences

    # === STATYSTYKI I LOGI ===

    def log_api_usage(
        self,
        user_id: int,
        api_provider: str,
        endpoint: str,
        method: str,
        tokens_used: int = 0,
        cost: float = 0.0,
        success: bool = True,
        response_time: float = None,
        error_message: str = None,
        metadata: dict = None,
    ):
        """Loguje wykorzystanie API."""
        with self.get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO api_usage
                (user_id, api_provider, endpoint, method, tokens_used, cost,
                 success, response_time, error_message, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    user_id,
                    api_provider,
                    endpoint,
                    method,
                    tokens_used,
                    cost,
                    success,
                    response_time,
                    error_message,
                    json.dumps(metadata or {}),
                ),
            )

    def get_user_api_usage(self, user_id: int, days: int = 30) -> list[APIUsage]:
        """Pobiera statystyki użycia API dla użytkownika."""
        with self.get_db_connection() as conn:
            cursor = conn.execute(
                f"""
                SELECT * FROM api_usage
                WHERE user_id = ? AND created_at >= date('now', '-{days} days')
                ORDER BY created_at DESC
            """,
                (user_id,),
            )

            return [APIUsage.from_db_row(row) for row in cursor.fetchall()]

    def count_api_calls(self, user_id: int, days: int = 30) -> int:
        """Return number of API calls by user within given days."""
        with self.get_db_connection() as conn:
            since = (datetime.now() - timedelta(days=days)).isoformat()
            cursor = conn.execute(
                "SELECT COUNT(*) FROM api_usage WHERE user_id = ? AND created_at >= ?",
                (user_id, since),
            )
            row = cursor.fetchone()
            return row[0] if row else 0

    def log_system_event(
        self,
        level: str,
        module: str,
        message: str,
        user_id: int = None,
        session_id: int = None,
        metadata: dict = None,
    ):
        """Loguje zdarzenie systemowe."""
        with self.get_db_connection() as conn:
            conn.execute(
                """
                INSERT INTO system_logs (level, module, message, user_id, session_id, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    level,
                    module,
                    message,
                    user_id,
                    session_id,
                    json.dumps(metadata or {}),
                ),
            )

    # === CZYSZCZENIE I KONSERWACJA ===

    def cleanup_old_data(self, days: int = 90):
        """Czyści stare dane z bazy."""
        with self.get_db_connection() as conn:
            # Usuń stare logi systemowe
            cursor = conn.execute(
                f"""
                DELETE FROM system_logs
                WHERE created_at <= date('now', '-{days} days')
            """
            )
            logs_deleted = cursor.rowcount

            # Usuń stare sesje
            cursor = conn.execute(
                f"""
                DELETE FROM user_sessions
                WHERE created_at <= date('now', '-{days} days') AND is_active = FALSE
            """
            )
            sessions_deleted = cursor.rowcount

            # Usuń stare statystyki API
            cursor = conn.execute(
                f"""
                DELETE FROM api_usage
                WHERE created_at <= date('now', '-{days} days')
            """
            )
            api_deleted = cursor.rowcount

            logger.info(
                f"Cleanup completed: {logs_deleted} logs, {sessions_deleted} sessions, {api_deleted} API records deleted"
            )

    def get_database_stats(self) -> dict[str, int]:
        """Pobiera statystyki bazy danych."""
        with self.get_db_connection() as conn:
            stats = {}

            tables = [
                "users",
                "user_sessions",
                "messages",
                "memory_contexts",
                "api_usage",
                "system_logs",
                "user_preferences",
            ]

            for table in tables:
                cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                stats[table] = cursor.fetchone()[0]

            return stats

    def close(self):
        """Zamyka połączenia z bazą danych."""
        if hasattr(self._local, "connection"):
            self._local.connection.close()
            delattr(self._local, "connection")

    # === NOWE METODY DLA SERWERA ===

    async def initialize(self):
        """Asynchroniczna inicjalizacja dla FastAPI."""
        # W tym przypadku synchroniczna inicjalizacja jest wystarczająca
        pass

    async def get_all_users(self) -> list[dict]:
        """Pobiera wszystkich użytkowników z ich ustawieniami pluginów."""
        with self.get_db_connection() as conn:
            cursor = conn.execute(
                """
                SELECT u.id, u.username, u.settings, p.value as enabled_plugins
                FROM users u
                LEFT JOIN user_preferences p ON u.id = p.user_id
                    AND p.category = 'plugins' AND p.key_name = 'enabled'
                WHERE u.is_active = TRUE
            """
            )

            users = []
            for row in cursor.fetchall():
                enabled_plugins = []
                if row["enabled_plugins"]:
                    try:
                        enabled_plugins = json.loads(row["enabled_plugins"])
                    except json.JSONDecodeError:
                        enabled_plugins = []

                users.append(
                    {
                        "id": str(row["id"]),
                        "username": row["username"],
                        "settings": json.loads(row["settings"] or "{}"),
                        "enabled_plugins": enabled_plugins,
                    }
                )

            return users

    async def get_user_history(self, user_id: str, limit: int = 50) -> list[dict]:
        """Pobiera historię wiadomości użytkownika."""
        # Convert string user_id to integer for database
        try:
            if isinstance(user_id, str) and user_id.startswith("client"):
                db_user_id = int(user_id.replace("client", ""))
            else:
                db_user_id = int(user_id)
        except (ValueError, TypeError):
            # If conversion fails, use consistent hash-based ID
            if isinstance(user_id, str):
                db_user_id = abs(hash(user_id)) % (
                    10**8
                )  # Smaller range for consistency
                logger.debug(
                    f"String user_id '{user_id}' converted to numeric: {db_user_id}"
                )
            else:
                db_user_id = 1
                logger.warning(
                    f"Invalid user_id type: {type(user_id)}, using default: 1"
                )

        with self.get_db_connection() as conn:
            cursor = conn.execute(
                """
                SELECT role, content, metadata, created_at
                FROM messages
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            """,
                (db_user_id, limit),
            )

            history = []
            for row in cursor.fetchall():
                history.append(
                    {
                        "role": row["role"],
                        "content": row["content"],
                        "metadata": json.loads(row["metadata"] or "{}"),
                        "timestamp": row["created_at"],
                    }
                )

            return list(reversed(history))  # Odwróć żeby najstarsze były pierwsze

    async def save_interaction(self, user_id: str, query: str, response: str):
        """Zapisuje interakcję użytkownika z AI."""
        # Convert string user_id to integer for database
        # For now, use a simple mapping: "client1" -> 1, "client2" -> 2, etc.
        try:
            if isinstance(user_id, str) and user_id.startswith("client"):
                db_user_id = int(user_id.replace("client", ""))
            else:
                db_user_id = int(user_id)
        except (ValueError, TypeError):
            # If conversion fails, use consistent hash-based ID
            if isinstance(user_id, str):
                db_user_id = abs(hash(user_id)) % (
                    10**8
                )  # Smaller range for consistency
                logger.debug(
                    f"String user_id '{user_id}' converted to numeric: {db_user_id}"
                )
            else:
                db_user_id = 1
                logger.warning(
                    f"Invalid user_id type: {type(user_id)}, using default: 1"
                )

        with self.get_db_connection() as conn:
            # Najpierw upewnij się że użytkownik istnieje
            conn.execute(
                """
                INSERT OR IGNORE INTO users (id, username, email)
                VALUES (?, ?, ?)
            """,
                (db_user_id, f"user_{db_user_id}", f"user_{db_user_id}@gaja.local"),
            )

            # Zapisz zapytanie użytkownika
            conn.execute(
                """
                INSERT INTO messages (user_id, role, content, metadata)
                VALUES (?, 'user', ?, '{}')
            """,
                (db_user_id, query),
            )

            # Zapisz odpowiedź asystenta
            conn.execute(
                """
                INSERT INTO messages (user_id, role, content, metadata)
                VALUES (?, 'assistant', ?, '{}')
            """,
                (db_user_id, response),
            )

    async def update_user_plugins(self, user_id: str, plugin_name: str, enabled: bool):
        """Aktualizuje ustawienia pluginów użytkownika."""
        with self.get_db_connection() as conn:
            # Pobierz obecne ustawienia pluginów
            cursor = conn.execute(
                """
                SELECT value FROM user_preferences
                WHERE user_id = ? AND category = 'plugins' AND key_name = 'enabled'
            """,
                (int(user_id),),
            )

            row = cursor.fetchone()
            if row:
                try:
                    enabled_plugins = json.loads(row["value"])
                except json.JSONDecodeError:
                    enabled_plugins = []
            else:
                enabled_plugins = []

            # Aktualizuj listę pluginów
            if enabled and plugin_name not in enabled_plugins:
                enabled_plugins.append(plugin_name)
            elif not enabled and plugin_name in enabled_plugins:
                enabled_plugins.remove(plugin_name)

            # Zapisz zaktualizowane ustawienia
            conn.execute(
                """                INSERT OR REPLACE INTO user_preferences
                (user_id, category, key_name, value, updated_at)
                VALUES (?, 'plugins', 'enabled', ?, CURRENT_TIMESTAMP)
            """,
                (int(user_id), json.dumps(enabled_plugins)),
            )

    async def get_user_plugins(self, user_id: str) -> list[dict]:
        """Pobiera ustawienia pluginów użytkownika."""
        try:
            with self.get_db_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT value FROM user_preferences
                    WHERE user_id = ? AND category = 'plugins' AND key_name = 'enabled'
                """,
                    (int(user_id),),
                )

                row = cursor.fetchone()
                if row:
                    try:
                        enabled_plugins = json.loads(row["value"])
                    except json.JSONDecodeError:
                        enabled_plugins = []
                else:
                    enabled_plugins = []

                # Konwertuj do formatu oczekiwanego przez WebSocket
                plugins = []
                for plugin_name in enabled_plugins:
                    plugins.append(
                        {
                            "plugin_name": plugin_name,
                            "enabled": True,
                            "updated_at": None,
                        }
                    )

                return plugins

        except Exception as e:
            logger.error(f"Failed to get user plugins: {e}")
            return []

    async def set_user_plugin_status(
        self, user_id: str, plugin_name: str, enabled: bool
    ):
        """Ustawia status pluginu użytkownika (alias dla update_user_plugins)."""
        await self.update_user_plugins(user_id, plugin_name, enabled)

    def get_user_api_key(self, user_id: int, provider: str) -> str | None:
        """Pobiera klucz API użytkownika dla danego providera.

        Args:
            user_id: ID użytkownika
            provider: Nazwa providera (np. 'openweather', 'newsapi', 'google_search')

        Returns:
            Klucz API lub None jeśli nie istnieje
        """
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT api_keys FROM users WHERE id = ?", (user_id,))

                row = cursor.fetchone()
                if row and row["api_keys"]:
                    api_keys = json.loads(row["api_keys"])
                    return api_keys.get(provider)

                return None
        except Exception as e:
            logger.error(
                f"Error getting API key for user {user_id}, provider {provider}: {e}"
            )
            return None

    def set_user_api_key(self, user_id: int, provider: str, api_key: str) -> bool:
        """Ustawia klucz API użytkownika dla danego providera.

        Args:
            user_id: ID użytkownika
            provider: Nazwa providera
            api_key: Klucz API

        Returns:
            True jeśli zapisano pomyślnie
        """
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()

                # Pobierz aktualne klucze API
                cursor.execute("SELECT api_keys FROM users WHERE id = ?", (user_id,))

                row = cursor.fetchone()
                if not row:
                    logger.error(f"User {user_id} not found")
                    return False

                # Parsuj aktualne klucze
                api_keys = json.loads(row["api_keys"]) if row["api_keys"] else {}

                # Dodaj/zaktualizuj klucz
                api_keys[provider] = api_key

                # Zapisz z powrotem
                cursor.execute(
                    "UPDATE users SET api_keys = ? WHERE id = ?",
                    (json.dumps(api_keys), user_id),
                )
                conn.commit()
                logger.info(f"Set API key for user {user_id}, provider {provider}")
                return True
        except Exception as e:
            logger.error(
                f"Error setting API key for user {user_id}, provider {provider}: {e}"
            )
            return False

    def remove_user_api_key(self, user_id: int, provider: str) -> bool:
        """Usuwa klucz API użytkownika dla danego providera.

        Args:
            user_id: ID użytkownika
            provider: Nazwa providera

        Returns:
            True jeśli usunięto pomyślnie
        """
        try:
            with self.get_db_connection() as conn:
                cursor = conn.cursor()

                # Pobierz aktualne klucze API
                cursor.execute("SELECT api_keys FROM users WHERE id = ?", (user_id,))

                row = cursor.fetchone()
                if not row:
                    return False

                # Parsuj aktualne klucze
                api_keys = json.loads(row["api_keys"]) if row["api_keys"] else {}

                # Usuń klucz jeśli istnieje
                if provider in api_keys:
                    del api_keys[provider]

                    # Zapisz z powrotem
                    cursor.execute(
                        "UPDATE users SET api_keys = ? WHERE id = ?",
                        (json.dumps(api_keys), user_id),
                    )
                    conn.commit()
                    logger.info(
                        f"Removed API key for user {user_id}, provider {provider}"
                    )
                    return True

                return True  # Klucz nie istniał, ale można uznać to za sukces
        except Exception as e:
            logger.error(
                f"Error removing API key for user {user_id}, provider {provider}: {e}"
            )
            return False

    async def update_user_plugin_status(
        self, user_id: str, plugin_name: str, enabled: bool
    ):
        """Updates plugin status for a user in the database."""
        try:
            # Convert string user_id to integer for database
            if user_id.startswith("client"):
                db_user_id = int(user_id.replace("client", ""))
            else:
                db_user_id = int(user_id)
        except ValueError:
            # If conversion fails, use 1 as default
            db_user_id = 1

        try:
            with self.get_db_connection() as conn:
                # Check if user exists, create if not
                cursor = conn.execute(
                    "SELECT id FROM users WHERE id = ?", (db_user_id,)
                )
                if not cursor.fetchone():
                    # Create user
                    conn.execute(
                        """
                        INSERT INTO users (id, username, settings, api_keys)
                        VALUES (?, ?, '{}', '{}')
                    """,
                        (db_user_id, f"user_{db_user_id}"),
                    )

                # Get current enabled plugins
                cursor = conn.execute(
                    """
                    SELECT value FROM user_preferences
                    WHERE user_id = ? AND category = 'plugins' AND key_name = 'enabled'
                """,
                    (db_user_id,),
                )

                row = cursor.fetchone()
                if row:
                    try:
                        enabled_plugins = json.loads(row["value"])
                    except json.JSONDecodeError:
                        enabled_plugins = []
                else:
                    enabled_plugins = []

                # Update plugin list
                if enabled and plugin_name not in enabled_plugins:
                    enabled_plugins.append(plugin_name)
                elif not enabled and plugin_name in enabled_plugins:
                    enabled_plugins.remove(plugin_name)

                # Save updated settings
                conn.execute(
                    """
                    INSERT OR REPLACE INTO user_preferences
                    (user_id, category, key_name, value, updated_at)
                    VALUES (?, 'plugins', 'enabled', ?, CURRENT_TIMESTAMP)
                """,
                    (db_user_id, json.dumps(enabled_plugins)),
                )

                logger.info(
                    f"Updated plugin {plugin_name} status for user {user_id}: {enabled}"
                )
                return True
        except Exception as e:
            logger.error(
                f"Error updating plugin status for user {user_id}, plugin {plugin_name}: {e}"
            )
            return False


class ConfigManager:
    """Unified configuration manager combining environment and database
    functionality."""

    def __init__(self, env_file: str = ".env", db_path: str = "server_data.db"):
        self.environment = EnvironmentManager(env_file)
        self.database = DatabaseManager(db_path)
        logger.info(
            "ConfigManager initialized with unified environment and database management"
        )

    async def initialize(self):
        """Asynchroniczna inicjalizacja dla FastAPI."""
        await self.database.initialize()

    def get_api_key(self, service: str, user_id: int = None) -> str | None:
        """Pobiera klucz API - najpierw z ustawień użytkownika, potem z environment."""
        if user_id:
            user_key = self.database.get_user_api_key(user_id, service)
            if user_key:
                return user_key

        return self.environment.get_api_key(service)

    def get_server_config(self) -> dict[str, Any]:
        """Pobiera konfigurację serwera."""
        return self.environment.get_server_config()

    def sanitize_config_for_logging(self, config: dict[str, Any]) -> dict[str, Any]:
        """Czyści konfigurację z wrażliwych danych przed logowaniem."""
        return self.environment.sanitize_config_for_logging(config)

    def close(self):
        """Zamyka połączenia."""
        self.database.close()


# Global instances
_config_manager = None
_env_manager = EnvironmentManager()  # For backward compatibility
_db_manager = None


def get_config_manager() -> ConfigManager:
    """Pobiera globalną instancję unified config managera."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_environment_manager() -> EnvironmentManager:
    """Pobiera globalną instancję environment managera (backward compatibility)."""
    return _env_manager


def get_database_manager() -> DatabaseManager:
    """Pobiera globalną instancję database managera (backward compatibility)."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def initialize_database_manager(db_path: str = "server_data.db") -> DatabaseManager:
    """Inicjalizuje globalną instancję database managera (backward compatibility)."""
    global _db_manager
    _db_manager = DatabaseManager(db_path)
    return _db_manager


# Backward compatibility aliases
env_manager = _env_manager
