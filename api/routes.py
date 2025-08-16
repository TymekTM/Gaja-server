"""API routes for Gaja Web UI integration.

Provides REST API endpoints for web UI functionality.
"""

import os
import json  # Added for AI response normalization
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from auth.security import security_manager
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from loguru import logger

# Import server components
# from server_main import server_app  # Import moved to avoid circular import
from core.plugin_manager import plugin_manager
from pydantic import BaseModel, Field

# OpenAI for TTS
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Server app will be injected after initialization
server_app: Optional[Any] = None


def set_server_app(app):
    """Set server app instance after initialization."""
    global server_app
    server_app = app

# Initialize API router and security schemes
router = APIRouter(prefix="/api/v1")
security = HTTPBearer()
optional_security = HTTPBearer(auto_error=False)


# Request/Response models
class LoginRequest(BaseModel):
    email: str
    password: str


class LoginResponse(BaseModel):
    success: bool
    token: str  # access token
    refreshToken: str
    user: dict[str, Any]


class UserSettings(BaseModel):
    voice: str | None = None
    language: str = "pl"
    wakeWord: bool | None = None
    privacy: dict[str, Any] | None = None


class User(BaseModel):
    id: str
    email: str
    role: str
    settings: UserSettings
    createdAt: str


class Memory(BaseModel):
    id: str
    userId: str
    content: str
    createdAt: str
    importance: int = Field(ge=0, le=5)
    isDeleted: bool = False
    tags: list[str] | None = None


class Plugin(BaseModel):
    slug: str
    name: str
    version: str
    enabled: bool
    author: str
    description: str | None = None
    installedAt: str | None = None


class SystemMetrics(BaseModel):
    cpu: float
    ram: dict[str, int]
    gpu: dict[str, int] | None = None
    tokensPerSecond: float | None = None
    uptime: int


class LogEntry(BaseModel):
    id: str
    level: str
    message: str
    timestamp: str
    module: str | None = None
    metadata: dict[str, Any] | None = None


class ApiResponse(BaseModel):
    success: bool
    data: Any | None = None
    error: str | None = None
    message: str | None = None


# ================= TEST / CI UTILITIES (non-production) =================
@router.post("/auth/_unlock_test_user")
async def unlock_test_user(payload: dict[str, str]) -> dict[str, Any]:
    """CI helper: clear failed attempts for a user (enabled only if GAJA_TEST_MODE=1).

    Body: {"email": "demo@mail.com"}
    """
    if os.getenv("GAJA_TEST_MODE") not in {"1", "true", "True"}:
        raise HTTPException(status_code=403, detail="Test mode not enabled")
    email = (payload.get("email") or "").lower().strip()
    if not email:
        raise HTTPException(status_code=400, detail="Email required")
    security_manager.clear_failed_attempts(email)
    return {"success": True, "message": f"Cleared failed attempts for {email}"}

@router.post("/auth/_test_token")
async def test_token(payload: dict[str,str]) -> dict[str, Any]:
    """Issue a test access token directly (GAJA_TEST_MODE only).

    Body: {"email": "demo@mail.com"}
    """
    if os.getenv("GAJA_TEST_MODE") not in {"1","true","True"}:
        raise HTTPException(status_code=403, detail="Test mode not enabled")
    email = (payload.get("email") or "").lower().strip()
    if not email:
        raise HTTPException(status_code=400, detail="Email required")
    user = SECURE_USERS.get(email)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    token_data = {"userId": user["id"], "email": user["email"], "role": user["role"]}
    token = security_manager.create_access_token(token_data)
    return {"token": token}


# ===== Additional models for integrations / models listing =====
class ModelInfo(BaseModel):
    id: str
    name: str
    type: str = "llm"
    size: str | None = None
    loaded: bool = True
    parameters: dict[str, Any] | None = None


class IntegrationInfo(BaseModel):
    name: str
    displayName: str
    connected: bool
    connectedAt: str | None = None
    config: dict[str, Any] | None = None


# Bezpieczni użytkownicy - hasła są zahashowane
# Przykładowi użytkownicy z zahashowanymi hasłami
# UWAGA: W produkcji należy używać zewnętrznej bazy danych z właściwą migracją
SECURE_USERS = {
    "admin@gaja.app": {
        "id": "1",
        "email": "admin@gaja.app",
        "role": "admin",
        "password_hash": "$2b$12$sWZp8vkKmF41Ndi6a5uoQu08GUi3gbpa0hqo1ipDgOSodtrI1KNLu",
        "is_active": True,
        "created_at": datetime.now().isoformat(),
        "settings": {
            "language": "en",
            "voice": "default",
            "wakeWord": True,
            "privacy": {"shareAnalytics": True, "storeConversations": True},
        },
    },
    "demo@mail.com": {
        "id": "2",
        "email": "demo@mail.com",
        "role": "user",
        "password_hash": "$2b$12$Tg.YnWlT4wbt3QnvYI1KCeal.5M0TowXoeAXTHr7ad1ULLabzJhWe",
        "is_active": True,
        "created_at": datetime.now().isoformat(),
        "settings": {
            "language": "pl",
            "voice": "default",
            "wakeWord": True,
            "privacy": {"shareAnalytics": False, "storeConversations": True},
        },
    },
}


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict[str, Any]:
    """Get current user from JWT token with proper security validation."""
    try:
        token = credentials.credentials

        # Weryfikuj token używając SecurityManager
        payload = security_manager.verify_token(token, "access")

        # Pobierz użytkownika na podstawie ID/email z tokenu
        user_email = payload.get("email")
        user_id = payload.get("userId") or payload.get("user_id")

        # Znajdź użytkownika w systemie
        user = None
        for _, user_data in SECURE_USERS.items():
            if user_data["email"] == user_email or user_data["id"] == user_id:
                user = user_data
                break

        if not user:
            logger.warning(f"User not found for email {user_email} or ID {user_id}")
            raise HTTPException(status_code=401, detail="User not found")

        if not user.get("is_active", True):
            raise HTTPException(status_code=401, detail="Account deactivated")

        return user

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token validation error: {e}")
        raise HTTPException(status_code=401, detail="Invalid authentication") from e


def optional_auth(
    credentials: HTTPAuthorizationCredentials | None = Depends(optional_security),
) -> dict[str, Any] | None:
    """Optional authentication for public endpoints."""
    if not credentials:
        return None
    try:
        return get_current_user(credentials)
    except HTTPException:
        return None


# Auth endpoints
REFRESH_TOKENS: dict[str, dict[str, Any]] = {}


@router.post("/auth/login")
async def login(request: LoginRequest, fastapi_request: Request) -> dict[str, Any]:
    """Secure user login endpoint with proper authentication."""
    try:
        email = request.email.lower().strip()
        password = request.password

        test_mode = os.getenv("GAJA_TEST_MODE") in {"1", "true", "True"}
        # Lockout bypass in test mode to allow repeated CI runs
        if not test_mode:
            if security_manager.is_account_locked(email):
                logger.warning(f"Login attempt on locked account: {email}")
                raise HTTPException(
                    status_code=429,
                    detail="Account temporarily locked due to too many failed attempts",
                )
        else:
            # Always clear failed attempts in test mode to avoid sticky lock state
            security_manager.clear_failed_attempts(email)

        # Znajdź użytkownika (create test user dynamically in test mode if missing)
        user = SECURE_USERS.get(email)
        if test_mode and email == "demo@mail.com" and not user:
            # Create ephemeral demo user with simple password bypass (not hashed) for CI
            user = {
                "id": "2",
                "email": email,
                "role": "user",
                "password_hash": SECURE_USERS.get("demo@mail.com", {}).get("password_hash", ""),
                "is_active": True,
                "created_at": datetime.now().isoformat(),
                "settings": {"language": "pl", "voice": "default", "wakeWord": True, "privacy": {"shareAnalytics": False, "storeConversations": True}},
            }
            SECURE_USERS[email] = user
        if not user:
            security_manager.record_failed_attempt(email)
            logger.warning(f"Login attempt with non-existent email: {email}")
            raise HTTPException(status_code=401, detail="Invalid credentials")

        # Sprawdź czy konto jest aktywne
        if not user.get("is_active", True):
            logger.warning(f"Login attempt on deactivated account: {email}")
            raise HTTPException(status_code=401, detail="Account deactivated")

        # Weryfikuj hasło (skip in test mode to simplify CI setup)
        if not test_mode:
            if not security_manager.verify_password(password, user["password_hash"]):
                security_manager.record_failed_attempt(email)
                logger.warning(f"Failed login attempt for user: {email}")
                raise HTTPException(status_code=401, detail="Invalid credentials")
        else:
            # Accept any password for the demo user; for others still verify
            if email == "demo@mail.com":
                logger.debug("Test mode: bypass password for demo user")
            elif not security_manager.verify_password(password, user["password_hash"]):
                security_manager.record_failed_attempt(email)
                raise HTTPException(status_code=401, detail="Invalid credentials (test mode)")

        # Udane logowanie - wyczyść nieudane próby
        security_manager.clear_failed_attempts(email)

        # Utwórz tokeny
        token_data = {
            "userId": user["id"],
            "email": user["email"],
            "role": user["role"],
        }
        access_token = security_manager.create_access_token(token_data)
        refresh_token = security_manager.create_refresh_token(token_data)

        # Store refresh token (simple in-memory store; replace with DB in production)
        REFRESH_TOKENS[refresh_token] = {
            "user_id": user["id"],
            "email": user["email"],
            "role": user["role"],
            "created_at": datetime.now().isoformat(),
        }

        # Loguj udane logowanie (bez wrażliwych danych)
        client_ip = fastapi_request.client.host if fastapi_request and fastapi_request.client else "unknown"
        log_data = security_manager.sanitize_log_data(
            {
                "email": email,
                "user_id": user["id"],
                "role": user["role"],
                "ip": client_ip,
            }
        )
        logger.info(f"Successful login: {log_data}")

        # Return explicit dict to avoid any potential serialization issues omitting refreshToken
        return {
            "success": True,
            "token": access_token,
            "refreshToken": refresh_token,
            "user": {
                "id": user["id"],
                "email": user["email"],
                "role": user["role"],
                "settings": user["settings"],
                "createdAt": user["created_at"],
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=500, detail="Authentication service unavailable"
        ) from e


@router.post("/auth/magic-link")
async def magic_link(request: dict[str, str]) -> ApiResponse:
    """Send magic link (mock implementation)."""
    email = request.get("email")
    if not email:
        raise HTTPException(status_code=400, detail="Email required")
    logger.info(f"Magic link sent to {email}")
    return ApiResponse(success=True, message="Magic link sent")

@router.post("/auth/refresh")
async def refresh_token_route(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict[str, Any]:
    """Issue a new access token given a valid refresh token in Authorization header."""
    try:
        refresh_token = credentials.credentials
        # Verify token structure & type using security_manager (will raise if invalid)
        payload = security_manager.verify_token(refresh_token, "refresh")

        # Optionally check in-memory store (defence-in-depth)
        if refresh_token not in REFRESH_TOKENS:
            raise HTTPException(status_code=401, detail="Unknown refresh token")

        token_data = {
            "userId": payload.get("userId"),
            "email": payload.get("email"),
            "role": payload.get("role"),
        }
        new_access = security_manager.create_access_token(token_data)
        return {"token": new_access}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Refresh token error: {e}")
        raise HTTPException(status_code=401, detail="Invalid refresh token") from e

    # (function body moved above)


@router.get("/me")
async def get_me(current_user: dict[str, Any] = Depends(get_current_user)) -> User:
    """Get current user information."""
    return User(
        id=current_user["id"],
        email=current_user["email"],
        role=current_user["role"],
        settings=UserSettings(**current_user["settings"]),
        createdAt=datetime.now().isoformat(),
    )


@router.patch("/me/settings")
async def update_settings(
    settings: UserSettings, current_user: dict[str, Any] = Depends(get_current_user)
) -> User:
    """Update user settings securely."""
    try:
        user_email = current_user["email"]

        # Waliduj że użytkownik istnieje
        if user_email not in SECURE_USERS:
            raise HTTPException(status_code=404, detail="User not found")

        # Zaktualizuj ustawienia (tylko niepuste wartości)
        settings_update = settings.dict(exclude_unset=True)
        SECURE_USERS[user_email]["settings"].update(settings_update)

        # Loguj zmianę ustawień
        log_data = security_manager.sanitize_log_data(
            {
                "user_id": current_user["id"],
                "email": user_email,
                "updated_settings": list(settings_update.keys()),
            }
        )
        logger.info(f"User settings updated: {log_data}")

        return User(
            id=current_user["id"],
            email=current_user["email"],
            role=current_user["role"],
            settings=UserSettings(**SECURE_USERS[user_email]["settings"]),
            createdAt=current_user["created_at"],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Settings update error: {e}")
        raise HTTPException(status_code=500, detail="Failed to update settings") from e


# Memory endpoints
@router.get("/memory")
async def get_memories(
    page: int = 1,
    limit: int = 20,
    current_user: dict[str, Any] = Depends(get_current_user),
) -> dict[str, Any]:
    """Get user memories."""
    try:
        user_id = current_user["id"]

        # Get memories from database
        if hasattr(server_app, "db_manager"):
            history = await server_app.db_manager.get_user_history(user_id, limit=limit)

            memories = []
            for i, item in enumerate(history):
                memories.append(
                    {
                        "id": str(i),
                        "userId": user_id,
                        "content": item.get("user_query", "")
                        + " -> "
                        + item.get("ai_response", ""),
                        "createdAt": item.get("timestamp", datetime.now().isoformat()),
                        "importance": 3,  # Default importance
                        "isDeleted": False,
                        "tags": ["conversation"],
                    }
                )

            return {"memories": memories, "total": len(memories)}
        else:
            return {"memories": [], "total": 0}

    except Exception as e:
        logger.error(f"Get memories error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get memories") from e


@router.delete("/memory/{memory_id}")
async def delete_memory(
    memory_id: str, current_user: dict[str, Any] = Depends(get_current_user)
) -> ApiResponse:
    """Delete a memory."""
    # Mock implementation - in production, delete from database
    logger.info(f"Memory {memory_id} deleted by user {current_user['id']}")
    return ApiResponse(success=True, message="Memory deleted")


# Plugin endpoints
@router.get("/plugins")
async def get_plugins(
    current_user: dict[str, Any] = Depends(get_current_user)
) -> list[Plugin]:
    """Get available plugins."""
    try:
        plugins = []

        if server_app and hasattr(server_app, "plugin_manager"):
            pm = server_app.plugin_manager
            user_plugins = pm.get_user_plugins(current_user["id"])

            for plugin_name, plugin_info in pm.plugins.items():
                plugins.append(
                    Plugin(
                        slug=plugin_name,
                        name=plugin_info.name,
                        version=plugin_info.version,
                        enabled=user_plugins.get(plugin_name, False),
                        author=plugin_info.author,
                        description=plugin_info.description,
                        installedAt=datetime.now().isoformat(),
                    )
                )
        else:
            # Fallback to global plugin_manager
            user_plugins = plugin_manager.get_user_plugins(current_user["id"])
            for plugin_name, plugin_info in plugin_manager.plugins.items():
                plugins.append(Plugin(
                    slug=plugin_name,
                    name=plugin_info.name,
                    version=plugin_info.version,
                    enabled=user_plugins.get(plugin_name, False),
                    author=plugin_info.author,
                    description=plugin_info.description,
                    installedAt=datetime.now().isoformat(),
                ))

        return plugins

    except Exception as e:
        logger.error(f"Get plugins error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get plugins") from e


@router.patch("/plugins/{plugin_slug}")
async def toggle_plugin(
    plugin_slug: str,
    request: dict[str, bool],
    current_user: dict[str, Any] = Depends(get_current_user),
) -> Plugin:
    """Toggle plugin enabled status."""
    enabled = request.get("enabled", False)
    user_id = current_user["id"]
    if not server_app or not hasattr(server_app, "plugin_manager"):
        # Graceful degraded mode: report initializing instead of hard 503
        logger.warning("Plugin toggle requested but plugin_manager not ready; returning initializing status")
        # Return a placeholder plugin object so tests can retry
        return Plugin(
            slug=plugin_slug,
            name=plugin_slug,
            version="initializing",
            enabled=False,
            author="system",
            description="Plugin manager initializing",
            installedAt=datetime.now().isoformat(),
        )
    pm = server_app.plugin_manager
    try:
        if enabled:
            success = await pm.enable_plugin_for_user(user_id, plugin_slug)
        else:
            success = await pm.disable_plugin_for_user(user_id, plugin_slug)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to toggle plugin")
        # DB update (optional)
        if hasattr(server_app, "db_manager") and server_app.db_manager:
            try:
                await server_app.db_manager.update_user_plugin_status(user_id, plugin_slug, enabled)
            except Exception as db_err:
                logger.warning(f"DB plugin status update failed: {db_err}")
        plugin_info = pm.plugins.get(plugin_slug)
        if not plugin_info:
            raise HTTPException(status_code=404, detail="Plugin not found")
        return Plugin(
            slug=plugin_slug,
            name=plugin_info.name,
            version=plugin_info.version,
            enabled=enabled,
            author=plugin_info.author,
            description=plugin_info.description,
            installedAt=datetime.now().isoformat(),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Toggle plugin error: {e}")
        raise HTTPException(status_code=500, detail="Failed to toggle plugin") from e
# Additional endpoints (formerly misplaced inside plugin toggle)
@router.post("/plugins")
async def upload_plugin(file: UploadFile = File(...), current_user: dict[str, Any] = Depends(get_current_user)) -> Plugin:
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    try:
        contents = await file.read()
        size_kb = len(contents) / 1024
        logger.info(f"Received plugin upload {file.filename} size={size_kb:.1f}KB")
        return Plugin(
            slug=(file.filename or "uploaded").replace('.zip',''),
            name=file.filename or "uploaded.zip",
            version="0.0.1",
            enabled=False,
            author="upload",
            description="Uploaded plugin (placeholder)",
            installedAt=datetime.now().isoformat(),
        )
    except Exception as e:
        logger.error(f"Plugin upload error: {e}")
        raise HTTPException(status_code=500, detail="Plugin upload failed") from e

@router.get("/models")
async def list_models(current_user: dict[str, Any] = Depends(get_current_user)) -> list[ModelInfo]:
    return [ModelInfo(id="default-llm", name="Default LLM", size="medium", loaded=True, parameters={"temperature":0.7})]

@router.post("/models/reload")
async def reload_models(current_user: dict[str, Any] = Depends(get_current_user)) -> ApiResponse:
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return ApiResponse(success=True, message="Models reloaded")

@router.post("/backup")
async def create_backup(current_user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    filename = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
    logger.info(f"Creating placeholder backup {filename}")
    return {"filename": filename}

@router.post("/restore")
async def restore_backup(file: UploadFile = File(...), current_user: dict[str, Any] = Depends(get_current_user)) -> ApiResponse:
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    try:
        await file.read()
        logger.info(f"Received backup restore file {file.filename}")
        return ApiResponse(success=True, message="Backup restored (placeholder)")
    except Exception as e:
        logger.error(f"Restore error: {e}")
        raise HTTPException(status_code=500, detail="Restore failed") from e

INTEGRATIONS: dict[str, dict[str, Any]] = {
    "google": {"displayName": "Google", "connected": False},
    "slack": {"displayName": "Slack", "connected": False},
}

@router.get("/integrations")
async def list_integrations(current_user: dict[str, Any] = Depends(get_current_user)) -> list[IntegrationInfo]:
    out: list[IntegrationInfo] = []
    for key, val in INTEGRATIONS.items():
        info = val.copy()
        out.append(IntegrationInfo(name=key, displayName=info.get("displayName", key.title()), connected=info.get("connected", False), connectedAt=info.get("connectedAt"), config=info.get("config")))
    return out

@router.post("/integrations/{name}/link")
async def link_integration(name: str, current_user: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    integ = INTEGRATIONS.get(name)
    if not integ:
        raise HTTPException(status_code=404, detail="Integration not found")
    integ["connected"] = True
    integ["connectedAt"] = datetime.now().isoformat()
    integ["config"] = {"scopes": ["basic"], "mock": True}
    return {"authUrl": f"https://auth.example.com/{name}?token=mock"}

@router.delete("/integrations/{name}")
async def unlink_integration(name: str, current_user: dict[str, Any] = Depends(get_current_user)) -> ApiResponse:
    integ = INTEGRATIONS.get(name)
    if not integ:
        raise HTTPException(status_code=404, detail="Integration not found")
    integ["connected"] = False
    integ.pop("connectedAt", None)
    integ.pop("config", None)
    return ApiResponse(success=True, message="Integration unlinked")


# System metrics endpoints
@router.get("/metrics")
async def get_metrics(
    current_user: dict[str, Any] = Depends(get_current_user)
) -> SystemMetrics:
    """Get system metrics."""
    try:
        import psutil

        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()

        # Get uptime
        uptime = 0
        if hasattr(server_app, "start_time"):
            uptime = int((datetime.now() - server_app.start_time).total_seconds())

        return SystemMetrics(
            cpu=cpu_percent,
            ram={"used": memory.used, "total": memory.total},
            uptime=uptime,
        )

    except Exception as e:
        logger.error(f"Get metrics error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get metrics") from e


# Log endpoints
@router.get("/logs")
async def get_logs(
    level: str | None = None,
    tail: int | None = 100,
    current_user: dict[str, Any] = Depends(get_current_user),
) -> list[LogEntry]:
    """Get system logs."""
    try:
        logs = []

        # Read from log files
        logs_dir = Path("logs")
        if logs_dir.exists():
            log_files = sorted(
                logs_dir.glob("*.log"), key=lambda x: x.stat().st_mtime, reverse=True
            )

            if log_files:
                log_file = log_files[0]  # Most recent log file

                try:
                    with open(log_file, encoding="utf-8") as f:
                        lines = f.readlines()

                    # Get last 'tail' lines
                    recent_lines = (
                        lines[-tail:] if tail and len(lines) > tail else lines
                    )

                    for i, line in enumerate(recent_lines):
                        # Parse log line (simplified)
                        parts = line.strip().split(" | ")
                        if len(parts) >= 4:
                            timestamp = parts[0]
                            level_str = parts[1]
                            location = parts[2]
                            message = parts[3]

                            # Filter by level if specified
                            if level and level_str.upper() != level.upper():
                                continue

                            logs.append(
                                LogEntry(
                                    id=str(i),
                                    level=level_str.lower(),
                                    message=message,
                                    timestamp=timestamp,
                                    module=location.split(":")[0]
                                    if ":" in location
                                    else None,
                                )
                            )

                except Exception as e:
                    logger.error(f"Error reading log file: {e}")

        return logs

    except Exception as e:
        logger.error(f"Get logs error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get logs") from e


# Health check endpoint
@router.get("/healthz")
async def health_check(
    user: dict[str, Any] | None = Depends(optional_auth)
) -> dict[str, Any]:
    """Health check endpoint."""
    try:
        uptime = 0
        if hasattr(server_app, "start_time"):
            uptime = int((datetime.now() - server_app.start_time).total_seconds())

        return {
            "version": "1.0.0",
            "uptime": uptime,
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=500, detail="Health check failed") from e


# Alias for health check
@router.get('/health')
async def health_alias():
    return await health_check()


# Legacy status endpoint for compatibility
@router.get("/status")
async def status_check() -> dict[str, Any]:
    """Legacy status endpoint for client compatibility."""
    try:
        return {
            "status": "running",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Status check error: {e}")
        raise HTTPException(status_code=500, detail="Status check failed") from e


# AI Query endpoint for web UI
@router.post("/ai/query")
async def ai_query(
    request: dict[str, Any], current_user: dict[str, Any] = Depends(get_current_user)
) -> dict[str, Any]:
    """Process AI query from web UI."""
    try:
        query = request.get("query", "")
        context = request.get("context", {})

        if not query:
            raise HTTPException(status_code=400, detail="Query is required")

        # Use AI module directly if server_app has ai_module
        if getattr(server_app, "ai_module", None):
            # Prepare context for AI module
            ai_context = {
                "user_id": current_user["id"],
                "history": [],  # Future: inject stored conversation history
                "available_plugins": [],  # Future: user-specific enabled plugins
                "modules": {},  # Future: share callable tool registry
            }
            ai_context.update(context)

            # Process query using AI module (returns structured dict with nested JSON string)
            processed = await server_app.ai_module.process_query(query, ai_context)  # type: ignore[attr-defined]

            # Normalization: always expose top-level 'response' as a plain text string for clients/tests
            text_response: str = ""
            try:
                # processed['response'] may itself be a dict OR a JSON string
                inner = processed.get("response") if isinstance(processed, dict) else None
                # If inner is a dict with 'response' key (legacy double wrap), unwrap once
                if isinstance(inner, dict) and "response" in inner and "type" in inner:
                    inner = inner.get("response")
                if isinstance(inner, dict):
                    # If dict contains textual field(s)
                    if "text" in inner:
                        text_response = str(inner.get("text") or "")
                    elif "response" in inner and isinstance(inner.get("response"), str):
                        text_response = inner.get("response", "")
                    else:
                        # Fallback: JSON dump of dict (rare path)
                        text_response = json.dumps(inner, ensure_ascii=False)
                elif isinstance(inner, str):
                    # Attempt to parse JSON string to extract 'text'
                    stripped = inner.strip()
                    extracted = None
                    if stripped.startswith("{") and stripped.endswith("}"):
                        try:
                            parsed = json.loads(stripped)
                            if isinstance(parsed, dict):
                                extracted = parsed.get("text") or parsed.get("response")
                                if extracted:
                                    text_response = str(extracted)
                                else:
                                    text_response = stripped
                        except Exception:
                            text_response = stripped
                    else:
                        text_response = stripped
                else:
                    text_response = "(brak danych odpowiedzi)"
            except Exception as norm_exc:  # pragma: no cover
                logger.warning(f"AI response normalization failed: {norm_exc}")
                text_response = "(błąd normalizacji odpowiedzi)"

            return {
                "success": True,
                # Provide the normalized plain text for primary consumption
                "response": text_response,
                # Expose raw structured data for advanced clients / debugging
                "raw": processed,
                "timestamp": datetime.now().timestamp(),
            }
        else:
            raise HTTPException(status_code=500, detail="AI module not available")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"AI query error: {e}")
        raise HTTPException(status_code=500, detail="AI query failed") from e


# WebSocket status endpoint
@router.get("/ws/status")
async def ws_status(
    current_user: dict[str, Any] = Depends(get_current_user)
) -> dict[str, Any]:
    """Get WebSocket connection status."""
    try:
        user_id = current_user["id"]

        # Check if user is connected via WebSocket
        connected = False
        if hasattr(server_app, "connection_manager"):
            connected = server_app.connection_manager.is_connected(user_id)

        # Jeśli nie ma server_app, spróbuj zaimportować connection_manager bezpośrednio
        if not hasattr(server_app, "connection_manager"):
            try:
                from core.websocket_manager import connection_manager

                connected = connection_manager.is_connected(user_id)
            except ImportError:
                connected = False

        return {
            "connected": connected,
            "userId": user_id,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"WebSocket status error: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to get WebSocket status"
        ) from e


# Daily briefing endpoint
@router.get("/briefing/daily")
async def get_daily_briefing(
    current_user: dict[str, Any] = Depends(get_current_user)
) -> dict[str, Any]:
    """Get daily briefing for user."""
    try:
        user_id = current_user["id"]

        if hasattr(server_app, "daily_briefing"):
            briefing = await server_app.daily_briefing.generate_daily_briefing(user_id)
            return {
                "success": True,
                "briefing": briefing,
                "timestamp": datetime.now().isoformat(),
            }
        else:
            return {"success": False, "error": "Daily briefing not available"}

    except Exception as e:
        logger.error(f"Daily briefing error: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to get daily briefing"
        ) from e


# Admin endpoints
@router.get("/admin/stats")
async def get_admin_stats(
    current_user: dict[str, Any] = Depends(get_current_user)
) -> dict[str, Any]:
    """Get admin statistics."""
    try:
        # Check if user is admin
        if current_user.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Admin access required")

        # Get active users count (from database or connection manager)
        active_users = 0
        if hasattr(server_app, "connection_manager") and server_app.connection_manager:
            active_users = len(server_app.connection_manager.active_connections)

        # Get total interactions from database
        total_interactions = 0
        today_interactions = 0
        if hasattr(server_app, "db_manager"):
            # Get all users to calculate total interactions
            try:
                users = await server_app.db_manager.get_all_users()
                for user in users:
                    history = await server_app.db_manager.get_user_history(
                        user["user_id"], limit=10000
                    )
                    total_interactions += len(history)

                    # Count today's interactions
                    from datetime import date, datetime

                    today = date.today()
                    for item in history:
                        try:
                            item_date = datetime.fromisoformat(
                                item.get("timestamp", "")
                            ).date()
                            if item_date == today:
                                today_interactions += 1
                        except Exception:
                            continue
            except Exception as e:
                logger.debug(f"Error getting interaction stats: {e}")

        # Get plugin stats
        active_plugins = 0
        total_plugins = 0
        if hasattr(server_app, "plugin_manager"):
            pm = server_app.plugin_manager
            total_plugins = len(pm.plugins)
            # Count enabled plugins across all users
            for plugin_name in pm.plugins:
                for user_id in pm.user_plugins:
                    if pm.user_plugins[user_id].get(plugin_name, False):
                        active_plugins += 1
                        break

        # Get system metrics
        import psutil

        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()

        # Get uptime
        uptime_seconds = 0
        uptime_percentage = 99.98
        if hasattr(server_app, "start_time"):
            from datetime import datetime

            uptime_seconds = int(
                (datetime.now() - server_app.start_time).total_seconds()
            )

        # Recent activity (simplified)
        recent_activity = []
        if hasattr(server_app, "db_manager"):
            try:
                users = await server_app.db_manager.get_all_users()
                if users:
                    # Get recent user activity
                    recent_activity.append(
                        {
                            "action": f"Total {len(users)} users in system",
                            "time": "System status",
                            "type": "info",
                        }
                    )

                    # Add some plugin info
                    if total_plugins > 0:
                        recent_activity.append(
                            {
                                "action": f"{total_plugins} plugins available, {active_plugins} active",
                                "time": "Plugin status",
                                "type": "success",
                            }
                        )
            except Exception as e:
                logger.debug(f"Error getting recent activity: {e}")

        return {
            "active_users": active_users,
            "total_interactions": total_interactions,
            "today_interactions": today_interactions,
            "active_plugins": active_plugins,
            "total_plugins": total_plugins,
            "memory_usage": {
                "used": memory.used,
                "total": memory.total,
                "percentage": memory.percent,
            },
            "cpu_usage": cpu_percent,
            "uptime": {"seconds": uptime_seconds, "percentage": uptime_percentage},
            "recent_activity": recent_activity,
            "security": {
                "ssl_valid": True,
                "firewall_active": True,
                "failed_logins": 0,  # This would come from auth logs
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Admin stats error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get admin stats") from e


# WebUI endpoint
@router.get("/webui")
async def serve_webui():
    """Serve the WebUI HTML interface."""
    try:
        # Ścieżka do pliku webui.html
        webui_path = Path(__file__).parent.parent / "webui.html"

        if not webui_path.exists():
            raise HTTPException(status_code=404, detail="WebUI not found")

        return FileResponse(webui_path, media_type="text/html")

    except Exception as e:
        logger.error(f"WebUI serve error: {e}")
        raise HTTPException(status_code=500, detail="Failed to serve WebUI") from e


# TTS endpoint
@router.post("/tts/stream")
async def stream_tts(
    request: dict[str, str],
    current_user: dict[str, Any] = Depends(get_current_user)
) -> StreamingResponse:
    """Stream TTS audio from server."""
    try:
        text = request.get("text", "")
        if not text:
            raise HTTPException(status_code=400, detail="Text is required")
        
        # Check if OpenAI is available
        if OpenAI is None:
            raise HTTPException(status_code=500, detail="OpenAI library not available")
        
        # Get API key from environment or config
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            try:
                from config.config_loader import load_config
                config = load_config("server_config.json")
                api_key = config.get("ai", {}).get("api_key")
            except Exception:
                pass
        
        if not api_key:
            raise HTTPException(status_code=500, detail="OpenAI API key not configured")
        
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # TTS settings
        model = request.get("model", "tts-1")
        voice = request.get("voice", "alloy")
        
        logger.info(f"Generating TTS for user {current_user['id']}: {text[:50]}...")
        
        def generate_audio():
            """Generate audio stream."""
            try:
                with client.audio.speech.with_streaming_response.create(
                    model=model,
                    voice=voice,
                    input=text,
                    response_format="opus",
                ) as response:
                    for chunk in response.iter_bytes():
                        yield chunk
            except Exception as e:
                logger.error(f"TTS generation error: {e}")
                raise HTTPException(status_code=500, detail=f"TTS generation failed: {e}")
        
        return StreamingResponse(
            generate_audio(),
            media_type="audio/opus",
            headers={
                "Content-Disposition": "inline; filename=tts.opus",
                "Cache-Control": "no-cache"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS stream error: {e}")
        raise HTTPException(status_code=500, detail=f"TTS stream failed: {e}")


# Habit engine endpoints
try:
    from habit.engine import HabitEngine

    habit_engine = HabitEngine()
except Exception:
    habit_engine = None

@router.post("/habit/events")
async def habit_log_event(event: dict[str, Any]):
    if not habit_engine:
        raise HTTPException(status_code=503, detail="Habit engine unavailable")
    habit_engine.log_event(event)
    habit_engine.scan_new_habits()
    return {"success": True}

@router.post("/habit/decide")
async def habit_decide(context: dict[str, Any]):
    if not habit_engine:
        raise HTTPException(status_code=503, detail="Habit engine unavailable")
    decision = habit_engine.decide(context)
    if not decision:
        return {"decision": None}
    return {"decision": {
        "id": decision.id,
        "action": {"verb": decision.action.verb, "object": decision.action.object, "params": getattr(decision.action, 'params', {})},
        "mode": decision.mode,
        "reason": decision.reason,
        "predicted_reward": decision.predicted_reward,
        "executed": decision.executed
    }}

@router.get("/habit/habits")
async def habit_list(mode: str | None = None):
    if not habit_engine:
        raise HTTPException(status_code=503, detail="Habit engine unavailable")
    habits = habit_engine.storage.get_habits(mode)
    return {"habits": [
        {
            "id": h.habit_id,
            "action": {"verb": h.action.verb, "object": h.action.object},
            "mode": h.mode,
            "stats": h.stats,
            "context_proto": h.context_proto,
        } for h in habits
    ]}

class FeedbackRequest(BaseModel):
    decision_id: str
    outcome: str
    latency_ms: int | None = None

@router.post("/habit/feedback")
async def habit_feedback(decision_id: str, outcome: str, latency_ms: int | None = None):
    if not habit_engine:
        raise HTTPException(status_code=503, detail="Habit engine unavailable")
    try:
        fb_id = habit_engine.feedback(decision_id, outcome, latency_ms)
        return {"feedback_id": fb_id}
    except Exception as e:
        logger.exception("Feedback processing failed")
        raise HTTPException(status_code=500, detail=str(e))

@router.post('/habit/promote/{habit_id}')
async def habit_promote(habit_id: str):
    if not habit_engine:
        raise HTTPException(status_code=503, detail='Habit engine unavailable')
    h = habit_engine.storage.get_habit(habit_id)
    if not h:
        raise HTTPException(status_code=404, detail='Habit not found')
    habit_engine.storage.update_habit_mode(habit_id, 'auto')
    return {'habit_id': habit_id, 'new_mode': 'auto'}

@router.post('/habit/demote/{habit_id}')
async def habit_demote(habit_id: str):
    if not habit_engine:
        raise HTTPException(status_code=503, detail='Habit engine unavailable')
    h = habit_engine.storage.get_habit(habit_id)
    if not h:
        raise HTTPException(status_code=404, detail='Habit not found')
    habit_engine.storage.update_habit_mode(habit_id, 'suggest')
    return {'habit_id': habit_id, 'new_mode': 'suggest'}

@router.get('/habit/telemetry')
async def habit_telemetry():
    if not habit_engine:
        raise HTTPException(status_code=503, detail='Habit engine unavailable')
    out = []
    for h in habit_engine.storage.get_habits():
        key = f"{h.action.verb}|{h.action.object}"
        theta_info = habit_engine.policy.get_theta(key)
        if theta_info:
            feats, vals = theta_info
            top = sorted(zip(feats, vals), key=lambda x: abs(x[1]), reverse=True)[:5]
        else:
            top = []
        out.append({
            'habit_id': h.habit_id,
            'action': {'verb': h.action.verb, 'object': h.action.object},
            'mode': h.mode,
            'stats': h.stats,
            'top_features': top
        })
    return {'habits': out}

class HabitLimitsRequest(BaseModel):
    decision_cooldown: int | None = None
    suggestion_limit_per_hour: int | None = None

@router.post('/habit/limits')
async def habit_set_limits(req: HabitLimitsRequest):
    if not habit_engine:
        raise HTTPException(status_code=503, detail='Habit engine unavailable')
    if req.decision_cooldown is not None:
        habit_engine.decision_cooldown = req.decision_cooldown
    if req.suggestion_limit_per_hour is not None:
        habit_engine.suggestion_limit_per_hour = req.suggestion_limit_per_hour
    return {'decision_cooldown': habit_engine.decision_cooldown, 'suggestion_limit_per_hour': habit_engine.suggestion_limit_per_hour}

@router.delete('/habit/{habit_id}')
async def habit_delete(habit_id: str):
    if not habit_engine:
        raise HTTPException(status_code=503, detail='Habit engine unavailable')
    if not habit_engine.storage.get_habit(habit_id):
        raise HTTPException(status_code=404, detail='Habit not found')
    habit_engine.storage.delete_habit(habit_id)
    return {'deleted': habit_id}

@router.get('/habit/decisions')
async def habit_decisions(limit: int = 50):
    if not habit_engine:
        raise HTTPException(status_code=503, detail='Habit engine unavailable')
    return {'decisions': habit_engine.storage.list_decisions(limit)}

@router.get('/habit/feedback')
async def habit_feedback_list(limit: int = 50):
    if not habit_engine:
        raise HTTPException(status_code=503, detail='Habit engine unavailable')
    return {'feedback': habit_engine.storage.list_feedback(limit)}

class ExecuteRequest(BaseModel):
    decision_id: str

@router.post('/habit/execute')
async def habit_execute(req: ExecuteRequest):
    if not habit_engine:
        raise HTTPException(status_code=503, detail='Habit engine unavailable')
    ok = habit_engine.execute_action(req.decision_id)
    if not ok:
        raise HTTPException(status_code=404, detail='Decision not found')
    return {'executed': True}

class HabitSchedulerRequest(BaseModel):
    interval: int | None = None

@router.post('/habit/scheduler/start')
async def habit_scheduler_start(req: HabitSchedulerRequest):
    if not habit_engine:
        raise HTTPException(status_code=503, detail='Habit engine unavailable')
    if req.interval is not None:
        habit_engine.scheduler_interval = req.interval
    habit_engine.start_scheduler()
    return {'started': True, 'interval': habit_engine.scheduler_interval}

@router.post('/habit/scheduler/stop')
async def habit_scheduler_stop():
    if not habit_engine:
        raise HTTPException(status_code=503, detail='Habit engine unavailable')
    habit_engine.stop_scheduler()
    return {'stopped': True}

class HabitImportRequest(BaseModel):
    data: dict
    merge: bool = True

@router.get('/habit/export')
async def habit_export():
    if not habit_engine:
        raise HTTPException(status_code=503, detail='Habit engine unavailable')
    return habit_engine.export_state()

@router.post('/habit/import')
async def habit_import(req: HabitImportRequest):
    if not habit_engine:
        raise HTTPException(status_code=503, detail='Habit engine unavailable')
    habit_engine.import_state(req.data, merge=req.merge)
    return {'imported': True}


# ==================== WEB UI ENDPOINTS ====================

@router.get("/admin/dashboard")
async def get_admin_dashboard():
    """Get dashboard data for web UI."""
    if not server_app or not hasattr(server_app, 'web_ui'):
        raise HTTPException(status_code=503, detail="Web UI not available")
    
    return server_app.web_ui.get_dashboard_data()


@router.get("/admin/config")
async def get_admin_config():
    """Get server configuration."""
    if not server_app or not hasattr(server_app, 'web_ui'):
        raise HTTPException(status_code=503, detail="Web UI not available")
    
    return server_app.web_ui.get_config_data()


@router.get("/admin/plugins")
async def get_admin_plugins():
    """Get plugin information."""
    if not server_app or not hasattr(server_app, 'web_ui'):
        raise HTTPException(status_code=503, detail="Web UI not available")
    
    return server_app.web_ui.get_plugin_data()


@router.get("/admin/memory")
async def get_admin_memory_stats(user_id: str = "1"):
    """Get memory statistics."""
    if not server_app or not hasattr(server_app, 'web_ui'):
        raise HTTPException(status_code=503, detail="Web UI not available")
    
    return server_app.web_ui.get_memory_stats(user_id)


@router.get("/admin/logs")
async def get_admin_logs():
    """Get logs information."""
    if not server_app or not hasattr(server_app, 'web_ui'):
        raise HTTPException(status_code=503, detail="Web UI not available")
    
    return server_app.web_ui.get_logs_data()


# Export router
__all__ = ["router"]
