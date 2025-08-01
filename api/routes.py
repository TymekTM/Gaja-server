"""API routes for Gaja Web UI integration.

Provides REST API endpoints for web UI functionality.
"""

import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from auth.security import security_manager
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from loguru import logger

# Import server components
# from server_main import server_app  # Import moved to avoid circular import
from plugin_manager import plugin_manager
from pydantic import BaseModel, Field

# OpenAI for TTS
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Server app will be injected after initialization
server_app = None


def set_server_app(app):
    """Set server app instance after initialization."""
    global server_app
    server_app = app


# Security
security = HTTPBearer()
optional_security = HTTPBearer(auto_error=False)

# Router
router = APIRouter(prefix="/api/v1", tags=["api"])


# Request/Response models
class LoginRequest(BaseModel):
    email: str
    password: str


class LoginResponse(BaseModel):
    success: bool
    token: str
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
@router.post("/auth/login")
async def login(request: LoginRequest) -> LoginResponse:
    """Secure user login endpoint with proper authentication."""
    try:
        email = request.email.lower().strip()
        password = request.password

        # Sprawdź czy konto nie jest zablokowane
        if security_manager.is_account_locked(email):
            logger.warning(f"Login attempt on locked account: {email}")
            raise HTTPException(
                status_code=429,
                detail="Account temporarily locked due to too many failed attempts",
            )

        # Znajdź użytkownika
        user = SECURE_USERS.get(email)
        if not user:
            security_manager.record_failed_attempt(email)
            logger.warning(f"Login attempt with non-existent email: {email}")
            raise HTTPException(status_code=401, detail="Invalid credentials")

        # Sprawdź czy konto jest aktywne
        if not user.get("is_active", True):
            logger.warning(f"Login attempt on deactivated account: {email}")
            raise HTTPException(status_code=401, detail="Account deactivated")

        # Weryfikuj hasło
        if not security_manager.verify_password(password, user["password_hash"]):
            security_manager.record_failed_attempt(email)
            logger.warning(f"Failed login attempt for user: {email}")
            raise HTTPException(status_code=401, detail="Invalid credentials")

        # Udane logowanie - wyczyść nieudane próby
        security_manager.clear_failed_attempts(email)

        # Utwórz tokeny
        token_data = {
            "userId": user["id"],
            "email": user["email"],
            "role": user["role"],
        }
        access_token = security_manager.create_access_token(token_data)
        # refresh_token = security_manager.create_refresh_token(token_data)

        # Loguj udane logowanie (bez wrażliwych danych)
        log_data = security_manager.sanitize_log_data(
            {
                "email": email,
                "user_id": user["id"],
                "role": user["role"],
                "ip": request.client.host if hasattr(request, "client") and request.client else "unknown",
            }
        )
        logger.info(f"Successful login: {log_data}")

        return LoginResponse(
            success=True,
            token=access_token,
            user={
                "id": user["id"],
                "email": user["email"],
                "role": user["role"],
                "settings": user["settings"],
                "createdAt": user["created_at"],
            },
        )

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

    # Mock magic link sending
    logger.info(f"Magic link sent to {email}")
    return ApiResponse(success=True, message="Magic link sent")


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
    try:
        enabled = request.get("enabled", False)
        user_id = current_user["id"]

        if hasattr(server_app, "plugin_manager"):
            pm = server_app.plugin_manager

            if enabled:
                success = await pm.enable_plugin_for_user(user_id, plugin_slug)
            else:
                success = await pm.disable_plugin_for_user(user_id, plugin_slug)

            if not success:
                raise HTTPException(status_code=400, detail="Failed to toggle plugin")

            # Update database
            if hasattr(server_app, "db_manager"):
                await server_app.db_manager.update_user_plugin_status(
                    user_id, plugin_slug, enabled
                )

            # Return updated plugin info
            plugin_info = pm.plugins.get(plugin_slug)
            if plugin_info:
                return Plugin(
                    slug=plugin_slug,
                    name=plugin_info.name,
                    version=plugin_info.version,
                    enabled=enabled,
                    author=plugin_info.author,
                    description=plugin_info.description,
                    installedAt=datetime.now().isoformat(),
                )

        raise HTTPException(status_code=404, detail="Plugin not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Toggle plugin error: {e}")
        raise HTTPException(status_code=500, detail="Failed to toggle plugin") from e


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
        if hasattr(server_app, "ai_module") and server_app.ai_module:
            # Prepare context for AI module
            ai_context = {
                "user_id": current_user["id"],
                "history": [],  # Can be enhanced with actual history
                "available_plugins": [],  # Can be enhanced with user plugins
                "modules": {},  # Can be enhanced with available modules
            }
            ai_context.update(context)

            # Process query using AI module
            response = await server_app.ai_module.process_query(query, ai_context)

            return {
                "success": True,
                "response": response,
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
                from websocket_manager import connection_manager

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
                from config_loader import load_config
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


# Export router
__all__ = ["router"]
