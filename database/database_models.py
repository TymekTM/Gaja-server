import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class User:
    """Model użytkownika."""

    id: int
    username: str
    email: str | None
    password_hash: str | None
    is_active: bool
    created_at: datetime
    last_login: datetime | None
    settings: dict[str, Any]
    api_keys: dict[str, str]

    @classmethod
    def from_db_row(cls, row: sqlite3.Row) -> "User":
        """Tworzy instancję User z wiersza bazy danych."""
        return cls(
            id=row["id"],
            username=row["username"],
            email=row["email"],
            password_hash=row["password_hash"],
            is_active=bool(row["is_active"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            last_login=(
                datetime.fromisoformat(row["last_login"]) if row["last_login"] else None
            ),
            settings=json.loads(row["settings"]) if row["settings"] else {},
            api_keys=json.loads(row["api_keys"]) if row["api_keys"] else {},
        )

    def to_dict(self) -> dict[str, Any]:
        """Konwertuje do słownika (bez wrażliwych danych)."""
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "settings": self.settings,
            # api_keys i password_hash celowo pominięte
        }


@dataclass
class UserSession:
    """Model sesji użytkownika."""

    id: int
    user_id: int
    session_token: str
    created_at: datetime
    expires_at: datetime
    is_active: bool
    client_info: dict[str, Any]

    @classmethod
    def from_db_row(cls, row: sqlite3.Row) -> "UserSession":
        """Tworzy instancję UserSession z wiersza bazy danych."""
        return cls(
            id=row["id"],
            user_id=row["user_id"],
            session_token=row["session_token"],
            created_at=datetime.fromisoformat(row["created_at"]),
            expires_at=datetime.fromisoformat(row["expires_at"]),
            is_active=bool(row["is_active"]),
            client_info=json.loads(row["client_info"]) if row["client_info"] else {},
        )

    def to_dict(self) -> dict[str, Any]:
        """Konwertuje do słownika."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "session_token": self.session_token,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "is_active": self.is_active,
            "client_info": self.client_info,
        }

    @property
    def is_expired(self) -> bool:
        """Sprawdza czy sesja wygasła."""
        return datetime.now() > self.expires_at


@dataclass
class Message:
    """Model wiadomości/konwersacji."""

    id: int
    user_id: int
    session_id: int | None
    role: str  # 'user', 'assistant', 'system'
    content: str
    metadata: dict[str, Any]
    created_at: datetime
    parent_message_id: int | None

    @classmethod
    def from_db_row(cls, row: sqlite3.Row) -> "Message":
        """Tworzy instancję Message z wiersza bazy danych."""
        return cls(
            id=row["id"],
            user_id=row["user_id"],
            session_id=row["session_id"],
            role=row["role"],
            content=row["content"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            created_at=datetime.fromisoformat(row["created_at"]),
            parent_message_id=row["parent_message_id"],
        )

    def to_dict(self) -> dict[str, Any]:
        """Konwertuje do słownika."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "role": self.role,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "parent_message_id": self.parent_message_id,
        }

    def to_chat_format(self) -> dict[str, str]:
        """Konwertuje do formatu dla modeli chat (OpenAI, Anthropic, etc.)."""
        return {"role": self.role, "content": self.content}


@dataclass
class MemoryContext:
    """Model kontekstu pamięci."""

    id: int
    user_id: int
    context_type: str
    key_name: str
    value: str
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    expires_at: datetime | None

    @classmethod
    def from_db_row(cls, row: sqlite3.Row) -> "MemoryContext":
        """Tworzy instancję MemoryContext z wiersza bazy danych."""
        return cls(
            id=row["id"],
            user_id=row["user_id"],
            context_type=row["context_type"],
            key_name=row["key_name"],
            value=row["value"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            expires_at=(
                datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Konwertuje do słownika."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "context_type": self.context_type,
            "key_name": self.key_name,
            "value": self.value,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }

    @property
    def is_expired(self) -> bool:
        """Sprawdza czy kontekst wygasł."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at


@dataclass
class APIUsage:
    """Model wykorzystania API."""

    id: int
    user_id: int
    api_provider: str
    endpoint: str
    method: str
    tokens_used: int
    cost: float
    success: bool
    response_time: float | None
    error_message: str | None
    metadata: dict[str, Any]
    created_at: datetime

    @classmethod
    def from_db_row(cls, row: sqlite3.Row) -> "APIUsage":
        """Tworzy instancję APIUsage z wiersza bazy danych."""
        return cls(
            id=row["id"],
            user_id=row["user_id"],
            api_provider=row["api_provider"],
            endpoint=row["endpoint"],
            method=row["method"],
            tokens_used=row["tokens_used"],
            cost=row["cost"],
            success=bool(row["success"]),
            response_time=row["response_time"],
            error_message=row["error_message"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def to_dict(self) -> dict[str, Any]:
        """Konwertuje do słownika."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "api_provider": self.api_provider,
            "endpoint": self.endpoint,
            "method": self.method,
            "tokens_used": self.tokens_used,
            "cost": self.cost,
            "success": self.success,
            "response_time": self.response_time,
            "error_message": self.error_message,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class SystemLog:
    """Model logu systemowego."""

    id: int
    level: str
    module: str
    message: str
    user_id: int | None
    session_id: int | None
    metadata: dict[str, Any]
    created_at: datetime

    @classmethod
    def from_db_row(cls, row: sqlite3.Row) -> "SystemLog":
        """Tworzy instancję SystemLog z wiersza bazy danych."""
        return cls(
            id=row["id"],
            level=row["level"],
            module=row["module"],
            message=row["message"],
            user_id=row["user_id"],
            session_id=row["session_id"],
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def to_dict(self) -> dict[str, Any]:
        """Konwertuje do słownika."""
        return {
            "id": self.id,
            "level": self.level,
            "module": self.module,
            "message": self.message,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class UserPreference:
    """Model preferencji użytkownika."""

    id: int
    user_id: int
    category: str
    key_name: str
    value: str
    value_type: str
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_db_row(cls, row: sqlite3.Row) -> "UserPreference":
        """Tworzy instancję UserPreference z wiersza bazy danych."""
        return cls(
            id=row["id"],
            user_id=row["user_id"],
            category=row["category"],
            key_name=row["key_name"],
            value=row["value"],
            value_type=row["value_type"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def to_dict(self) -> dict[str, Any]:
        """Konwertuje do słownika."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "category": self.category,
            "key_name": self.key_name,
            "value": self.value,
            "value_type": self.value_type,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @property
    def parsed_value(self) -> Any:
        """Zwraca wartość skonwertowaną do odpowiedniego typu."""
        if self.value_type == "int":
            return int(self.value)
        elif self.value_type == "float":
            return float(self.value)
        elif self.value_type == "bool":
            return self.value.lower() in ("true", "1", "yes")
        elif self.value_type == "json":
            return json.loads(self.value)
        else:
            return self.value
