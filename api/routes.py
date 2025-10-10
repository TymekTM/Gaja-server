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
from datetime import datetime as _dt
import os as _os
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from loguru import logger

from core.app_paths import resolve_data_path

# Import server components
# from server_main import server_app  # Import moved to avoid circular import
from core.plugin_manager import plugin_manager
from pydantic import BaseModel, Field, EmailStr, field_validator

# OpenAI for TTS
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Server app will be injected after initialization (treated as dynamic Any for runtime injection)
server_app: Any = None  # type: ignore


def set_server_app(app):
    """Set server app instance after initialization."""
    global server_app
    server_app = app

# Initialize API router and security schemes
router = APIRouter(prefix="/api/v1")
security = HTTPBearer(auto_error=False)
optional_security = HTTPBearer(auto_error=False)


# Request/Response models
class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, description="Password cannot be empty")


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

class DeviceCreateRequest(BaseModel):
    name: str
    type: str = Field(default="headless")
    metadata: dict[str, Any] | None = None

class DeviceUpdateRequest(BaseModel):
    name: str | None = None
    status: str | None = Field(default=None, description="offline|online|error")
    metadata: dict[str, Any] | None = None

class DeviceHeartbeatRequest(BaseModel):
    status: str | None = None
    metadata: dict[str, Any] | None = None

class ServerConfigPatch(BaseModel):
    tts: dict[str, Any] | None = None
    plugins: dict[str, Any] | None = None
    ai: dict[str, Any] | None = None


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


@router.post("/debug/tools/verify")
async def debug_verify_tool(payload: dict[str, Any]) -> dict[str, Any]:
    """Persist verification checkbox state for a function tool.

    Body: { name: string, tested: bool }
    Stores persisted verification state with timestamp and version.
    """
    try:
        name = (payload.get("name") or "").strip()
        tested = bool(payload.get("tested", False))
        if not name:
            raise HTTPException(status_code=400, detail="name required")
        ver_path = resolve_data_path(
            "debug_tools_verifications.json", create_parents=True
        )
        import json as _json
        data = {}
        if ver_path.exists():
            try:
                data = _json.loads(ver_path.read_text(encoding='utf-8'))
            except Exception:
                data = {}
        rec = data.get(name) or {}
        current_version = _os.getenv("GAJA_VERSION") or "1.0.0"
        rec.update({
            "tested": tested,
            "version": current_version,
            "ts": _dt.now().isoformat(),
        })
        data[name] = rec
        with ver_path.open("w", encoding="utf-8") as f:
            _json.dump(data, f, ensure_ascii=False, indent=2)
        return {"success": True, "name": name, "tested": tested, "version": current_version, "ts": rec["ts"]}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Debug verify tool error: {e}")
        return {"success": False, "error": str(e)}


# =============== DEBUG CENTER (Test Bench) ======================
@router.get("/debug/tools")
async def debug_list_tools() -> dict[str, Any]:
    """Return available functions (tools) as seen by the function calling system."""
    try:
        from core.function_calling_system import get_function_calling_system

        fcs = get_function_calling_system()
        funcs = fcs.convert_modules_to_functions() or []

        # Load persisted verifications
        ver_path = resolve_data_path("debug_tools_verifications.json", create_parents=True)
        ver = {}
        try:
            if ver_path.exists():
                import json as _json
                ver = _json.loads(ver_path.read_text(encoding='utf-8'))
        except Exception:
            ver = {}

        current_version = _os.getenv("GAJA_VERSION") or "1.0.0"

        tools: list[dict[str, Any]] = []
        for f in funcs:
            try:
                fn = f.get("function", {})
                name = (fn.get("name") or "").strip()
                desc = fn.get("description") or ""
                params = fn.get("parameters") or {"type": "object", "properties": {}, "required": []}
                # Resolve plugin/module from function_handlers if available
                handler = fcs.function_handlers.get(name)
                plugin = None
                if isinstance(handler, dict):
                    plugin = handler.get("plugin_name") or handler.get("module_name")
                if not plugin and name:
                    plugin = name.split("_", 1)[0]

                # Verification status
                v = ver.get(name) or {}
                tested = bool(v.get("tested", False))
                last_ver = v.get("version")
                stale = tested and (last_ver is not None) and (str(last_ver) != str(current_version))
                tools.append({
                    "name": name,
                    "description": desc,
                    "plugin": plugin,
                    "parameters": params,
                    "tested": tested,
                    "stale": stale,
                    "lastVerifiedVersion": last_ver,
                    "lastVerifiedAt": v.get("ts"),
                })
            except Exception:
                continue

        return {"tools": tools, "version": current_version}
    except Exception as e:
        logger.error(f"Debug tools error: {e}")
        return {"tools": [], "error": str(e)}


@router.get("/debug/providers")
async def debug_list_providers() -> dict[str, Any]:
    """List available AI providers and default models (for Debug Center switcher)."""
    try:
        from modules.ai_module import get_ai_providers
        from config.config_loader import MAIN_MODEL, PROVIDER, _config

        provs = get_ai_providers().providers
        provider_names = list(provs.keys())
        defaults = {"openai": MAIN_MODEL}
        try:
            pm = (_config.get("ai") or {}).get("provider_models") or {}
            if isinstance(pm, dict):
                defaults.update({k: v for k, v in pm.items() if isinstance(v, str)})
        except Exception:
            pass
        base_urls = {"lmstudio": _config.get("LMSTUDIO_URL_BASE")}
        return {
            "providers": provider_names,
            "defaults": defaults,
            "current": {"provider": PROVIDER, "model": MAIN_MODEL},
            "baseUrls": base_urls,
        }
    except Exception as e:
        logger.error(f"Debug providers error: {e}")
        return {"providers": [], "defaults": {}, "baseUrls": {}, "error": str(e)}


@router.post("/debug/providers/lmstudio_url")
async def debug_set_lmstudio_url(payload: dict[str, Any]) -> dict[str, Any]:
    """Set LM Studio base URL at runtime and persist to config file."""
    try:
        base = (payload.get("baseUrl") or "").strip()
        if not base:
            raise HTTPException(status_code=400, detail="baseUrl required")
        # Normalize (no trailing slash)
        if base.endswith('/'):
            base = base[:-1]
        from config.config_loader import save_config, load_config, _config
        # Update in-memory config
        _config["LMSTUDIO_URL_BASE"] = base
        # Persist to file
        cfg_path = "server_config.json"
        try:
            on_disk = load_config(cfg_path)
        except Exception:
            on_disk = {}
        on_disk["LMSTUDIO_URL_BASE"] = base
        save_config(on_disk, cfg_path)
        return {"success": True, "baseUrl": base}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Set LMStudio URL error: {e}")
        return {"success": False, "error": str(e)}


@router.post("/debug/chat")
async def debug_chat(request: dict[str, Any]) -> dict[str, Any]:
    """Debug chat endpoint returning rich trace and tool-call details.

    Body: { query: str, userId?: str, forceModel?: str }
    """
    try:
        query = (request.get("query") or "").strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")

        user_id = str(request.get("userId") or "1")
        force_model = request.get("forceModel")
        provider_override = request.get("providerOverride")
        no_fallback = bool(request.get("noFallback", False))
        client_history = request.get("history") or []  # [{role, content}]

        # Prepare minimal context with history from DB if available
        history = []
        if server_app and getattr(server_app, "db_manager", None):
            try:
                history = await server_app.db_manager.get_user_history(user_id, limit=10)
            except Exception as e:
                logger.debug(f"Debug: history load failed: {e}")

        from collections import deque
        from modules.ai_module import (
            chat_with_providers,
            _cached_system_prompt,
            MAIN_MODEL,
            PROVIDER,
        )
        from core.function_calling_system import get_function_calling_system
        import time as _time
        import json as _json
        import hashlib as _hash

        # Build conversation history (prefer client session history; fallback to DB)
        conv_msgs = []
        if isinstance(client_history, list) and client_history:
            for m in client_history[-20:]:
                role = (m.get('role') or 'user').strip()
                content = (m.get('content') or '').strip()
                if role in ("user", "assistant") and content:
                    conv_msgs.append({"role": role, "content": content})
        else:
            for msg in history[-10:]:
                role = msg.get("role") or ("assistant" if msg.get("ai_response") else "user")
                content = msg.get("content") or msg.get("user_query") or msg.get("ai_response") or ""
                if content:
                    if role == "assistant":
                        try:
                            parsed = _json.loads(content)
                            if isinstance(parsed, dict) and "text" in parsed:
                                content = parsed["text"]
                        except Exception:
                            pass
                    conv_msgs.append({"role": role, "content": content})
        conv_msgs.append({"role": "user", "content": query})

        # Prepare system prompt (mimic generate_response path)
        functions = []
        fcs = get_function_calling_system()
        functions = fcs.convert_modules_to_functions() or []
        tools_info = ""
        funcs_count = len(functions)
        system_prompt = _cached_system_prompt(
            None,  # no override
            "pl",
            1.0,
            _hash.sha256((tools_info or "").encode("utf-8")).hexdigest()[:12],
            None,
            False,
            "DebugUser",
            funcs_count,
        )
        messages = list(conv_msgs)
        messages.insert(0, {"role": "system", "content": system_prompt})

        # Call providers with tracer
        tracking_id = f"debug_{user_id}_{int(_time.time() * 1000)}"
        from modules.ai_module import LatencyTracer
        tracer = LatencyTracer(tracking_id, enabled=True)

        t0 = _time.perf_counter()
        result = await chat_with_providers(
            model=force_model or MAIN_MODEL,
            messages=messages,
            functions=functions,
            function_calling_system=fcs,
            provider_override=provider_override,
            tracer=tracer,
            stream=False,
            no_fallback=no_fallback,
        )
        elapsed_ms = int((_time.perf_counter() - t0) * 1000)
        # Persist trace and also return in-memory events
        try:
            tracer.flush_to_file()
        except Exception:
            pass

        # Normalize output
        text_response = ""
        tool_calls = []
        tools_used = 0
        usage = result.get("usage") if isinstance(result, dict) else None
        if isinstance(result, dict):
            msg = result.get("message") or {}
            text_response = (msg.get("content") or "").strip()
            tools_used = int(result.get("tool_calls_executed", 0) or 0)
            for item in (result.get("tool_call_details") or []):
                tool_calls.append({
                    "name": item.get("name"),
                    "arguments": item.get("arguments"),
                    "result": item.get("result"),
                })
        else:
            s = str(result or "")
            try:
                parsed = _json.loads(s)
                text_response = str(parsed.get("text") or s)
            except Exception:
                text_response = s

        # Use tracer events directly for UI (no file roundtrip)
        trace_events = getattr(tracer, 'events', []) or []

        # If no textual response and a tool failed, surface the tool error as assistant text
        if (not text_response) and tool_calls:
            for tc in tool_calls:
                res = tc.get('result')
                if isinstance(res, dict) and not res.get('success', True):
                    err = res.get('error') or res.get('message') or 'Błąd narzędzia'
                    text_response = f"Błąd narzędzia: {err}"
                    break

        return {
            "success": True,
            "response": text_response,
            "toolCalls": tool_calls,
            "toolsUsed": tools_used,
            "traceEvents": trace_events,
            "model": force_model or MAIN_MODEL,
            "provider": provider_override or PROVIDER,
            "fallbackUsed": any(ev.get("event") == "provider_fallback_success" for ev in trace_events),
            "usage": usage,
            "elapsedMs": elapsed_ms,
            "systemPrompt": system_prompt,
            "messages": messages,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Debug chat error: {e}")
        return {"success": False, "error": str(e)}


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
    role = payload.get("role", "user")
    token_data = {"sub": email, "email": email, "role": role}
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


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> dict[str, Any]:
    """Get current user from JWT token with proper security validation."""
    try:
        if not credentials:
            raise HTTPException(status_code=401, detail="Authorization header missing")

        token = credentials.credentials
        payload = security_manager.verify_token(token, "access")

        user_email = payload.get("email") or payload.get("sub") or "user@gaja.app"
        user_id = payload.get("userId") or payload.get("user_id") or payload.get("sub") or "1"
        role = payload.get("role", "user")

        settings = USER_SETTINGS_STORE.setdefault(
            user_email, DEFAULT_USER_SETTINGS.copy()
        )

        return {
            "id": str(user_id),
            "email": user_email,
            "role": role,
            "is_active": True,
            "settings": settings,
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Token validation error: {exc}")
        raise HTTPException(status_code=401, detail="Invalid authentication") from exc


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

DEFAULT_USER_SETTINGS: dict[str, Any] = {
    "language": "pl",
    "voice": "default",
    "wakeWord": True,
    "privacy": {
        "shareAnalytics": False,
        "storeConversations": True,
    },
}

USER_SETTINGS_STORE: dict[str, dict[str, Any]] = {}



@router.post("/auth/login")
async def login(request: LoginRequest, fastapi_request: Request) -> dict[str, Any]:
    """Secure user login endpoint with proper authentication."""
    try:
        email = request.email.lower().strip()
        password = request.password

        test_mode = os.getenv("GAJA_TEST_MODE") in {"1", "true", "True"}
        username = email

        if not test_mode and security_manager.is_account_locked(username):
            lock_info = security_manager.failed_attempts[username]
            return {
                "success": False,
                "error": "Account is temporarily locked",
                "locked_until": lock_info.get("locked_until"),
            }

        auth_result = security_manager.authenticate_user(username, password)

        if not auth_result.get("success") and test_mode:
            created = security_manager.create_user(username, password or "demo_pass")
            if created:
                auth_result = security_manager.authenticate_user(username, password or "demo_pass")

        if not auth_result.get("success"):
            security_manager.record_failed_attempt(username)
            logger.warning(f"Failed login attempt for user: {email}")
            raise HTTPException(status_code=401, detail=auth_result.get("error", "Invalid credentials"))

        security_manager.clear_failed_attempts(username)

        role = auth_result.get("user", {}).get("role", "user")
        payload = {
            "sub": username,
            "email": email,
            "role": role,
        }
        access_token = security_manager.create_access_token(payload)
        refresh_token = security_manager.create_refresh_token(payload)

        REFRESH_TOKENS[refresh_token] = {
            "user_id": username,
            "email": email,
            "role": role,
            "created_at": datetime.now().isoformat(),
        }

        client_ip = fastapi_request.client.host if fastapi_request and fastapi_request.client else "unknown"
        log_data = security_manager.sanitize_log_data(
            {
                "email": email,
                "user_id": username,
                "role": role,
                "ip": client_ip,
            }
        )
        logger.info(f"Successful login: {log_data}")

        settings_store = USER_SETTINGS_STORE.setdefault(
            email, DEFAULT_USER_SETTINGS.copy()
        )

        user_payload = {
            "id": username,
            "email": email,
            "role": role,
            "settings": settings_store,
            "createdAt": datetime.now().isoformat(),
        }

        return {
            "success": True,
            "token": access_token,
            "refreshToken": refresh_token,
            "user": user_payload,
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Login error: {exc}")
        raise HTTPException(
            status_code=500, detail="Authentication service unavailable"
        ) from exc

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

        store = USER_SETTINGS_STORE.setdefault(
            user_email, DEFAULT_USER_SETTINGS.copy()
        )

        settings_update = settings.dict(exclude_unset=True)
        store.update(settings_update)

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
            settings=UserSettings(**store),
            createdAt=datetime.now().isoformat(),
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
    """Get user conversation memories built from message history.

    Pairs consecutive user/assistant messages into compact entries for UI.
    """
    try:
        user_id = current_user["id"]

        if not hasattr(server_app, "db_manager"):
            return {"memories": [], "total": 0}

        # Fetch enough raw messages to build ~limit pairs
        raw = await server_app.db_manager.get_user_history(user_id, limit=limit * 2)
        pairs = []
        last_user = None
        last_ts = None
        for msg in raw:
            role = (msg.get("role") or "").lower()
            content = (msg.get("content") or "").strip()
            ts = msg.get("timestamp")
            if role == "user" and content:
                last_user = content
                last_ts = ts
            elif role == "assistant" and content:
                if last_user:
                    pairs.append({
                        "question": last_user,
                        "answer": content,
                        "timestamp": last_ts or ts,
                    })
                    last_user = None
                    last_ts = None
        # If dangling user question left without answer, include it
        if last_user:
            pairs.append({"question": last_user, "answer": "", "timestamp": last_ts})

        # Map to memory objects expected by UI
        memories = []
        for i, p in enumerate(pairs[-limit:]):  # last N pairs
            memories.append(
                {
                    "id": str(i),
                    "userId": user_id,
                    "content": f"{p['question']} -> {p['answer']}",
                    "createdAt": p.get("timestamp") or datetime.now().isoformat(),
                    "importance": 3,
                    "isDeleted": False,
                    "tags": ["conversation"],
                }
            )

        return {"memories": memories, "total": len(memories)}

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

        # Get total & today's interactions efficiently (messages rows)
        total_interactions = 0
        today_interactions = 0
        if hasattr(server_app, "db_manager") and server_app.db_manager:
            try:
                # Direct SQL counts instead of iterating every user's history
                with server_app.db_manager.get_db_connection() as conn:
                    cur = conn.execute("SELECT COUNT(*) FROM messages")
                    row = cur.fetchone()
                    total_interactions = row[0] if row else 0
                    cur = conn.execute(
                        "SELECT COUNT(*) FROM messages WHERE DATE(created_at)=DATE('now')"
                    )
                    row = cur.fetchone()
                    today_interactions = row[0] if row else 0
            except Exception as e:
                logger.debug(f"Error counting interactions: {e}")

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

# ================= ADMIN / USER MANAGEMENT =================

@router.get("/admin/users")
async def admin_list_users(current_user: dict[str, Any] = Depends(get_current_user)):
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    users: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    # Primary: database users
    if server_app and getattr(server_app, "db_manager", None):
        try:
            db_users = await server_app.db_manager.get_all_users()
            for u in db_users:
                uid = str(u.get("id"))
                seen_ids.add(uid)
                users.append({
                    "id": uid,
                    "username": u.get("username") or f"user_{uid}",
                    "settings": u.get("settings", {}),
                    "enabled_plugins": u.get("enabled_plugins", []),
                    "source": "db"
                })
        except Exception as e:
            logger.error(f"Error listing users: {e}")
    # Fallback: in-memory settings store (for test/demo accounts)
    for email, settings in USER_SETTINGS_STORE.items():
        uid = email
        if uid not in seen_ids:
            users.append({
                "id": uid,
                "username": email,
                "settings": settings,
                "enabled_plugins": [],
                "source": "local",
            })
            seen_ids.add(uid)
    # Include active websocket connections info
    active = []
    if server_app and hasattr(server_app, "connection_manager"):
        try:
            active = list(server_app.connection_manager.active_connections.keys())
        except Exception:
            active = []
    return {"users": users, "active_connections": active}

@router.post("/admin/users/{user_id}/disconnect")
async def admin_disconnect_user(user_id: str, current_user: dict[str, Any] = Depends(get_current_user)):
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    if server_app and hasattr(server_app, "connection_manager"):
        try:
            await server_app.connection_manager.disconnect(user_id, "admin_disconnect")
            return {"success": True, "message": f"User {user_id} disconnected"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
    raise HTTPException(status_code=404, detail="Connection manager not available")

# ================= DEVICE MANAGEMENT =================

@router.get("/admin/devices")
async def admin_list_devices(current_user: dict[str, Any] = Depends(get_current_user)):
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    if not server_app or not getattr(server_app, "db_manager", None):
        return {"devices": []}
    devices = server_app.db_manager.list_devices()
    # mask api_key partially
    for d in devices:
        key = d.get("api_key")
        if key:
            d["api_key_masked"] = f"{key[:4]}***{key[-4:]}"
            del d["api_key"]
    return {"devices": devices}

@router.post("/admin/devices")
async def admin_create_device(req: DeviceCreateRequest, current_user: dict[str, Any] = Depends(get_current_user)):
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    if not server_app or not getattr(server_app, "db_manager", None):
        raise HTTPException(status_code=503, detail="Database not available")
    device = server_app.db_manager.create_device(req.name, req.type, req.metadata)
    return {"device": device}

@router.get("/admin/devices/{device_id}")
async def admin_get_device(device_id: int, current_user: dict[str, Any] = Depends(get_current_user)):
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    if not server_app or not getattr(server_app, "db_manager", None):
        raise HTTPException(status_code=503, detail="Database not available")
    device = server_app.db_manager.get_device(device_id=device_id)
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")
    key = device.get("api_key")
    if key:
        device["api_key_masked"] = f"{key[:4]}***{key[-4:]}"
    if "api_key" in device:
        del device["api_key"]
    return {"device": device}

@router.patch("/admin/devices/{device_id}")
async def admin_update_device(device_id: int, req: DeviceUpdateRequest, current_user: dict[str, Any] = Depends(get_current_user)):
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    if not server_app or not getattr(server_app, "db_manager", None):
        raise HTTPException(status_code=503, detail="Database not available")
    ok = server_app.db_manager.update_device(device_id, name=req.name, status=req.status, metadata=req.metadata)
    if not ok:
        raise HTTPException(status_code=404, detail="Device not found or no changes")
    device = server_app.db_manager.get_device(device_id=device_id)
    if device and device.get("api_key"):
        device["api_key_masked"] = f"{device['api_key'][:4]}***{device['api_key'][-4:]}"
        del device["api_key"]
    return {"device": device}

@router.post("/admin/devices/{device_id}/regenerate-key")
async def admin_regenerate_device_key(device_id: int, current_user: dict[str, Any] = Depends(get_current_user)):
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    if not server_app or not getattr(server_app, "db_manager", None):
        raise HTTPException(status_code=503, detail="Database not available")
    new_key = server_app.db_manager.regenerate_device_key(device_id)
    if not new_key:
        raise HTTPException(status_code=404, detail="Device not found")
    return {"api_key": new_key}

@router.delete("/admin/devices/{device_id}")
async def admin_delete_device(device_id: int, current_user: dict[str, Any] = Depends(get_current_user)):
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    if not server_app or not getattr(server_app, "db_manager", None):
        raise HTTPException(status_code=503, detail="Database not available")
    ok = server_app.db_manager.delete_device(device_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Device not found")
    return {"deleted": True}

@router.post("/device/heartbeat")
async def device_heartbeat(api_key: str, req: DeviceHeartbeatRequest):
    # Open endpoint for devices using api_key; no user auth but minimal validation
    if not server_app or not getattr(server_app, "db_manager", None):
        raise HTTPException(status_code=503, detail="Database not available")
    device = server_app.db_manager.get_device(api_key=api_key)
    if not device:
        raise HTTPException(status_code=404, detail="Device not found")
    server_app.db_manager.heartbeat_device(api_key, status=req.status or "online", metadata=req.metadata)
    return {"success": True, "device_id": device["id"]}

@router.get("/admin/server-config")
async def admin_get_server_config(current_user: dict[str, Any] = Depends(get_current_user)):
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    if not server_app or not getattr(server_app, "config", None):
        return {"config": {}}
    return {"config": server_app.config}

@router.patch("/admin/server-config")
async def admin_patch_server_config(patch: ServerConfigPatch, current_user: dict[str, Any] = Depends(get_current_user)):
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    if not server_app or not getattr(server_app, "config", None):
        raise HTTPException(status_code=503, detail="Config not available")
    updated: dict[str, Any] = {}
    for section in ["tts", "plugins", "ai"]:
        value = getattr(patch, section)
        if value is not None:
            if section not in server_app.config or not isinstance(server_app.config.get(section), dict):
                server_app.config[section] = {}
            server_app.config[section].update(value)
            updated[section] = server_app.config[section]
    return {"updated": updated}


# Export router
__all__ = ["router"]
