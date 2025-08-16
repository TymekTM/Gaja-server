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

from database.database_models import APIUsage, MemoryContext, Message, User, UserSession

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

            # --- New feature tables (shopping list, notes, tasks, vector memories) ---
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS shopping_list_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    item TEXT NOT NULL,
                    quantity TEXT DEFAULT '1',
                    status TEXT DEFAULT 'pending', -- pending | bought | removed
                    is_persistent BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_shopping_user_id ON shopping_list_items (user_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_shopping_status ON shopping_list_items (status)"
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    title TEXT,
                    content TEXT NOT NULL,
                    tags TEXT DEFAULT '', -- comma separated
                    is_persistent BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_notes_user_id ON notes (user_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_notes_tags ON notes (tags)"
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    priority INTEGER DEFAULT 3, -- 1(high) - 5(low)
                    due_at TIMESTAMP,
                    status TEXT DEFAULT 'open', -- open | in_progress | done | cancelled
                    is_persistent BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_user_id ON tasks (user_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_due_at ON tasks (due_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks (status)"
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_vectors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    key TEXT,
                    content TEXT NOT NULL,
                    embedding BLOB NOT NULL, -- JSON encoded list[float]
                    similarity_hint REAL DEFAULT 0.0,
                    is_persistent BOOLEAN DEFAULT 1,
                    expires_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_vectors_user_id ON memory_vectors (user_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_vectors_key ON memory_vectors (key)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_vectors_persistent ON memory_vectors (is_persistent)"
            )

            # --- Incremental schema upgrades / migrations (lightweight) ---
            # Add list_name column to shopping_list_items if missing (for multiple named lists support)
            try:
                cursor = conn.execute("PRAGMA table_info(shopping_list_items)")
                existing_cols = {row[1] for row in cursor.fetchall()}
                if "list_name" not in existing_cols:
                    conn.execute(
                        "ALTER TABLE shopping_list_items ADD COLUMN list_name TEXT DEFAULT 'shopping'"
                    )
                    # Backfill existing rows with default list name
                    conn.execute(
                        "UPDATE shopping_list_items SET list_name = 'shopping' WHERE list_name IS NULL OR list_name = ''"
                    )
                    conn.execute(
                        "CREATE INDEX IF NOT EXISTS idx_shopping_list_name ON shopping_list_items (list_name)"
                    )
                    logger.info(
                        "Database migration: added list_name column to shopping_list_items"
                    )
            except Exception as e:
                logger.error(f"Schema migration (list_name) failed: {e}")

    # === ZARZĄDZANIE UŻYTKOWNIKAMI ===

    def create_user(
        self,
        username: str,
        email: str | None = None,
        password_hash: str | None = None,
        settings: dict | None = None,
        api_keys: dict | None = None,
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
            user_id = cursor.lastrowid or 0
            logger.info(f"Created user: {username} (ID: {user_id})")
            return user_id

    def get_user(self, user_id: int | None = None, username: str | None = None) -> User | None:
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
        if user:
            settings = user.settings if isinstance(user.settings, dict) else {}
            if isinstance(settings, dict):
                return settings.get("level", "free")
        return "free"

    def set_user_level(self, user_id: int, level: str) -> None:
        user = self.get_user(user_id=user_id)
        if user and isinstance(user.settings, dict):
            settings = dict(user.settings)
        else:
            settings = {}
        settings["level"] = level
        self.update_user_settings(user_id, settings)

    # === ZARZĄDZANIE SESJAMI ===

    def create_session(
        self,
        user_id: int,
        session_token: str,
        expires_at: datetime,
        client_info: dict | None = None,
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
            return cursor.lastrowid or 0

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
        session_id: int | None = None,
        metadata: dict | None = None,
        parent_message_id: int | None = None,
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
            return cursor.lastrowid or 0

    def get_user_messages(
        self, user_id: int, limit: int = 100, session_id: int | None = None
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
        self, user_id: int, session_id: int | None = None, limit: int = 10
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
        metadata: dict | None = None,
        expires_at: datetime | None = None,
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
        self, user_id: int, context_type: str, key_name: str | None = None
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

    # === WEKTOROWA PAMIĘĆ SEMANTYCZNA ===

    def add_memory_vector(
        self,
        user_id: int,
        content: str,
        embedding: list[float],
        key: str | None = None,
        similarity_hint: float = 0.0,
        is_persistent: bool = True,
        expires_at: datetime | None = None,
    ) -> int:
        """Dodaje wektor pamięci semantycznej.

        Args:
            user_id: ID użytkownika
            content: Treść pamięci
            embedding: Wektor osadzający (lista float)
            key: Opcjonalny klucz dla grupowania
            similarity_hint: Wstępny hint podobieństwa (może być 0)
            is_persistent: Czy pamięć jest trwała
            expires_at: Czas wygaśnięcia dla pamięci nietrwałych

        Returns:
            ID wstawionego rekordu
        """
        try:
            with self.get_db_connection() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO memory_vectors (user_id, key, content, embedding, similarity_hint, is_persistent, expires_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        user_id,
                        key,
                        content,
                        json.dumps(embedding),
                        similarity_hint,
                        1 if is_persistent else 0,
                        expires_at.isoformat() if expires_at else None,
                    ),
                )
                return cursor.lastrowid or -1
        except Exception as e:
            logger.error(f"Error inserting memory vector: {e}")
            return -1

    def get_memory_vectors(self, user_id: int, limit: int = 500) -> list[dict[str, Any]]:
        """Pobiera wektorowe pamięci użytkownika (aktywnie ważne)."""
        with self.get_db_connection() as conn:
            cursor = conn.execute(
                """
                SELECT id, key, content, embedding, similarity_hint, is_persistent, expires_at, created_at
                FROM memory_vectors
                WHERE user_id = ? AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
                ORDER BY created_at DESC
                LIMIT ?
            """,
                (user_id, limit),
            )
            rows = cursor.fetchall()
            results: list[dict[str, Any]] = []
            for r in rows:
                try:
                    embedding = json.loads(r[3]) if r[3] else []
                except json.JSONDecodeError:
                    embedding = []
                results.append(
                    {
                        "id": r[0],
                        "key": r[1],
                        "content": r[2],
                        "embedding": embedding,
                        "similarity_hint": r[4],
                        "is_persistent": bool(r[5]),
                        "expires_at": r[6],
                        "created_at": r[7],
                    }
                )
            return results

    def cleanup_expired_vectors(self) -> int:
        """Czyści wygasłe wektorowe pamięci. Zwraca liczbę usuniętych rekordów."""
        with self.get_db_connection() as conn:
            cursor = conn.execute(
                """
                DELETE FROM memory_vectors
                WHERE expires_at IS NOT NULL AND expires_at <= CURRENT_TIMESTAMP
            """
            )
            deleted = cursor.rowcount or 0
            if deleted:
                logger.info(f"Cleaned up {deleted} expired vector memories")
            return deleted

    # === SHOPPING LIST (structured, multi-list with statuses) ===

    def add_shopping_item(
        self,
        user_id: int,
        item: str,
        quantity: str | None = None,
        list_name: str = "shopping",
        is_persistent: bool = True,
    ) -> int:
        """Add an item to a (possibly named) shopping list.

        Deduplicates (case-insensitive) by (user_id, list_name, item) when status != removed.
        If item already exists and was marked bought/ pending, increments quantity numeric part if both quantities numeric.
        Returns ID of inserted or updated row.
        """
        clean_item = item.strip()
        clean_list = list_name.strip() or "shopping"
        qty = (quantity or "1").strip() or "1"
        try:
            with self.get_db_connection() as conn:
                # Try existing
                cursor = conn.execute(
                    """
                    SELECT id, quantity FROM shopping_list_items
                    WHERE user_id = ? AND LOWER(list_name)=LOWER(?) AND LOWER(item)=LOWER(?) AND status != 'removed'
                    ORDER BY id DESC LIMIT 1
                    """,
                    (user_id, clean_list, clean_item),
                )
                row = cursor.fetchone()
                if row:
                    # Potentially merge quantities if numeric
                    existing_id, existing_qty = row
                    try:
                        new_total = int(existing_qty) + int(qty)
                        merged_qty = str(new_total)
                    except (ValueError, TypeError):
                        merged_qty = existing_qty  # Keep original if not both ints
                    conn.execute(
                        """
                        UPDATE shopping_list_items SET quantity = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                        """,
                        (merged_qty, existing_id),
                    )
                    return existing_id
                cursor = conn.execute(
                    """
                    INSERT INTO shopping_list_items (user_id, item, quantity, status, is_persistent, list_name)
                    VALUES (?, ?, ?, 'pending', ?, ?)
                    """,
                    (user_id, clean_item, qty, 1 if is_persistent else 0, clean_list),
                )
                return cursor.lastrowid or 0
        except Exception as e:
            logger.error(f"Error add_shopping_item: {e}")
            return 0

    def list_shopping_items(
        self,
        user_id: int,
        list_name: str | None = None,
        status: str | None = None,
        include_removed: bool = False,
    ) -> list[dict[str, Any]]:
        """List shopping items with optional filters."""
        clauses = ["user_id = ?"]
        params: list[Any] = [user_id]
        if list_name:
            clauses.append("LOWER(list_name)=LOWER(?)")
            params.append(list_name)
        if status:
            clauses.append("status = ?")
            params.append(status)
        if not include_removed:
            clauses.append("status != 'removed'")
        where_sql = " AND ".join(clauses)
        query = f"""
            SELECT id, item, quantity, status, is_persistent, created_at, updated_at, list_name
            FROM shopping_list_items
            WHERE {where_sql}
            ORDER BY created_at ASC
        """
        with self.get_db_connection() as conn:
            try:
                cursor = conn.execute(query, tuple(params))
                rows = cursor.fetchall()
                return [
                    {
                        "id": r[0],
                        "item": r[1],
                        "quantity": r[2],
                        "status": r[3],
                        "is_persistent": bool(r[4]),
                        "created_at": r[5],
                        "updated_at": r[6],
                        "list_name": r[7],
                    }
                    for r in rows
                ]
            except Exception as e:
                logger.error(f"Error list_shopping_items: {e}")
                return []

    def update_shopping_item_status(
        self, user_id: int, item_id: int, status: str
    ) -> bool:
        """Update status of a shopping item (pending|bought|removed)."""
        if status not in {"pending", "bought", "removed"}:
            return False
        try:
            with self.get_db_connection() as conn:
                cursor = conn.execute(
                    """
                    UPDATE shopping_list_items
                    SET status = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ? AND user_id = ?
                    """,
                    (status, item_id, user_id),
                )
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error update_shopping_item_status: {e}")
            return False

    def remove_shopping_item(self, user_id: int, item_id: int, hard: bool = False) -> bool:
        """Remove shopping item. Soft removal sets status=removed unless hard specified."""
        try:
            with self.get_db_connection() as conn:
                if hard:
                    cursor = conn.execute(
                        "DELETE FROM shopping_list_items WHERE id = ? AND user_id = ?",
                        (item_id, user_id),
                    )
                else:
                    cursor = conn.execute(
                        """
                        UPDATE shopping_list_items SET status='removed', updated_at=CURRENT_TIMESTAMP
                        WHERE id = ? AND user_id = ?
                        """,
                        (item_id, user_id),
                    )
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error remove_shopping_item: {e}")
            return False

    def clear_shopping_list(
        self, user_id: int, list_name: str, include_bought: bool = True
    ) -> int:
        """Clear items from a named list. If include_bought is False, only remove pending."""
        try:
            with self.get_db_connection() as conn:
                if include_bought:
                    cursor = conn.execute(
                        """
                        UPDATE shopping_list_items SET status='removed', updated_at=CURRENT_TIMESTAMP
                        WHERE user_id = ? AND LOWER(list_name)=LOWER(?) AND status != 'removed'
                        """,
                        (user_id, list_name),
                    )
                else:
                    cursor = conn.execute(
                        """
                        UPDATE shopping_list_items SET status='removed', updated_at=CURRENT_TIMESTAMP
                        WHERE user_id = ? AND LOWER(list_name)=LOWER(?) AND status = 'pending'
                        """,
                        (user_id, list_name),
                    )
                return cursor.rowcount or 0
        except Exception as e:
            logger.error(f"Error clear_shopping_list: {e}")
            return 0

    # === NOTES MANAGEMENT ===

    def add_note(
        self,
        user_id: int,
        title: str | None,
        content: str,
        tags: list[str] | None = None,
        is_persistent: bool = True,
    ) -> int:
        """Create a new note with optional tags."""
        tag_str = ",".join(sorted({t.strip() for t in (tags or []) if t.strip()}))
        try:
            with self.get_db_connection() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO notes (user_id, title, content, tags, is_persistent)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (user_id, title, content, tag_str, 1 if is_persistent else 0),
                )
                return cursor.lastrowid or 0
        except Exception as e:
            logger.error(f"Error add_note: {e}")
            return 0

    def list_notes(
        self,
        user_id: int,
        tag: str | None = None,
        search: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List notes with optional tag or search filter."""
        clauses = ["user_id = ?"]
        params: list[Any] = [user_id]
        if tag:
            clauses.append("tags LIKE ?")
            params.append(f"%{tag}%")
        if search:
            clauses.append("(content LIKE ? OR title LIKE ?)")
            params.extend([f"%{search}%", f"%{search}%"])
        where_sql = " AND ".join(clauses)
        query = f"""
            SELECT id, title, content, tags, is_persistent, created_at, updated_at
            FROM notes
            WHERE {where_sql}
            ORDER BY updated_at DESC
            LIMIT ?
        """
        params.append(limit)
        try:
            with self.get_db_connection() as conn:
                cursor = conn.execute(query, tuple(params))
                rows = cursor.fetchall()
                return [
                    {
                        "id": r[0],
                        "title": r[1],
                        "content": r[2],
                        "tags": r[3].split(",") if r[3] else [],
                        "is_persistent": bool(r[4]),
                        "created_at": r[5],
                        "updated_at": r[6],
                    }
                    for r in rows
                ]
        except Exception as e:
            logger.error(f"Error list_notes: {e}")
            return []

    def get_note(self, user_id: int, note_id: int) -> dict[str, Any] | None:
        try:
            with self.get_db_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT id, title, content, tags, is_persistent, created_at, updated_at
                    FROM notes WHERE id = ? AND user_id = ?
                    """,
                    (note_id, user_id),
                )
                row = cursor.fetchone()
                if not row:
                    return None
                return {
                    "id": row[0],
                    "title": row[1],
                    "content": row[2],
                    "tags": row[3].split(",") if row[3] else [],
                    "is_persistent": bool(row[4]),
                    "created_at": row[5],
                    "updated_at": row[6],
                }
        except Exception as e:
            logger.error(f"Error get_note: {e}")
            return None

    def update_note(
        self,
        user_id: int,
        note_id: int,
        title: str | None = None,
        content: str | None = None,
        tags: list[str] | None = None,
    ) -> bool:
        fields = []
        params: list[Any] = []
        if title is not None:
            fields.append("title = ?")
            params.append(title)
        if content is not None:
            fields.append("content = ?")
            params.append(content)
        if tags is not None:
            tag_str = ",".join(sorted({t.strip() for t in tags if t.strip()}))
            fields.append("tags = ?")
            params.append(tag_str)
        if not fields:
            return False
        params.extend([note_id, user_id])
        sql = f"UPDATE notes SET {', '.join(fields)}, updated_at = CURRENT_TIMESTAMP WHERE id = ? AND user_id = ?"
        try:
            with self.get_db_connection() as conn:
                cursor = conn.execute(sql, tuple(params))
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error update_note: {e}")
            return False

    def delete_note(self, user_id: int, note_id: int) -> bool:
        try:
            with self.get_db_connection() as conn:
                cursor = conn.execute(
                    "DELETE FROM notes WHERE id = ? AND user_id = ?",
                    (note_id, user_id),
                )
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error delete_note: {e}")
            return False

    # === TASKS MANAGEMENT (advanced) ===

    def add_task_db(
        self,
        user_id: int,
        title: str,
        description: str = "",
        priority: int = 3,
        due_at: datetime | None = None,
        is_persistent: bool = True,
    ) -> int:
        """Add a structured task (priority 1-5)."""
        p = min(5, max(1, priority))
        try:
            with self.get_db_connection() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO tasks (user_id, title, description, priority, due_at, is_persistent)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user_id,
                        title,
                        description,
                        p,
                        due_at.strftime("%Y-%m-%d %H:%M:%S") if due_at else None,
                        1 if is_persistent else 0,
                    ),
                )
                return cursor.lastrowid or 0
        except Exception as e:
            logger.error(f"Error add_task_db: {e}")
            return 0

    def list_tasks(
        self,
        user_id: int,
        status: str | None = None,
        include_done: bool = True,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        clauses = ["user_id = ?"]
        params: list[Any] = [user_id]
        if status:
            clauses.append("status = ?")
            params.append(status)
        if not include_done and not status:
            clauses.append("status != 'done'")
        where_sql = " AND ".join(clauses)
        query = f"""
            SELECT id, title, description, priority, due_at, status, is_persistent, created_at, updated_at
            FROM tasks
            WHERE {where_sql}
            ORDER BY 
                CASE status WHEN 'open' THEN 0 WHEN 'in_progress' THEN 1 WHEN 'done' THEN 2 ELSE 3 END,
                priority ASC,
                COALESCE(due_at,'9999-12-31') ASC
            LIMIT ?
        """
        params.append(limit)
        try:
            with self.get_db_connection() as conn:
                cursor = conn.execute(query, tuple(params))
                rows = cursor.fetchall()
                return [
                    {
                        "id": r[0],
                        "title": r[1],
                        "description": r[2],
                        "priority": r[3],
                        "due_at": r[4],
                        "status": r[5],
                        "is_persistent": bool(r[6]),
                        "created_at": r[7],
                        "updated_at": r[8],
                    }
                    for r in rows
                ]
        except Exception as e:
            logger.error(f"Error list_tasks: {e}")
            return []

    def update_task_status(self, user_id: int, task_id: int, status: str) -> bool:
        if status not in {"open", "in_progress", "done", "cancelled"}:
            return False
        try:
            with self.get_db_connection() as conn:
                cursor = conn.execute(
                    """
                    UPDATE tasks SET status = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = ? AND user_id = ?
                    """,
                    (status, task_id, user_id),
                )
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error update_task_status: {e}")
            return False

    def update_task(
        self,
        user_id: int,
        task_id: int,
        title: str | None = None,
        description: str | None = None,
        priority: int | None = None,
        due_at: datetime | None = None,
    ) -> bool:
        fields = []
        params: list[Any] = []
        if title is not None:
            fields.append("title = ?")
            params.append(title)
        if description is not None:
            fields.append("description = ?")
            params.append(description)
        if priority is not None:
            fields.append("priority = ?")
            params.append(min(5, max(1, priority)))
        if due_at is not None:
            fields.append("due_at = ?")
            params.append(due_at.strftime("%Y-%m-%d %H:%M:%S"))
        if not fields:
            return False
        params.extend([task_id, user_id])
        sql = f"UPDATE tasks SET {', '.join(fields)}, updated_at = CURRENT_TIMESTAMP WHERE id = ? AND user_id = ?"
        try:
            with self.get_db_connection() as conn:
                cursor = conn.execute(sql, tuple(params))
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error update_task: {e}")
            return False

    def delete_task(self, user_id: int, task_id: int) -> bool:
        try:
            with self.get_db_connection() as conn:
                cursor = conn.execute(
                    "DELETE FROM tasks WHERE id = ? AND user_id = ?",
                    (task_id, user_id),
                )
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error delete_task: {e}")
            return False

    def get_overdue_tasks(self, user_id: int) -> list[dict[str, Any]]:
        try:
            with self.get_db_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT id, title, description, priority, due_at, status FROM tasks
                    WHERE user_id = ? AND due_at IS NOT NULL AND due_at < CURRENT_TIMESTAMP AND status != 'done'
                    ORDER BY due_at ASC
                    """,
                    (user_id,),
                )
                rows = cursor.fetchall()
                return [
                    {
                        "id": r[0],
                        "title": r[1],
                        "description": r[2],
                        "priority": r[3],
                        "due_at": r[4],
                        "status": r[5],
                    }
                    for r in rows
                ]
        except Exception as e:
            logger.error(f"Error get_overdue_tasks: {e}")
            return []

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
        self, user_id: int, category: str | None = None
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
        response_time: float | None = None,
        error_message: str | None = None,
        metadata: dict | None = None,
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
        user_id: int | None = None,
        session_id: int | None = None,
        metadata: dict | None = None,
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

    def get_api_key(self, service: str, user_id: int | None = None) -> str | None:
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
